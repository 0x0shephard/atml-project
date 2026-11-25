"""Turn a setting into a model checkpoint with simulated compression.

Every ``nn.Linear`` inside the decoder blocks is compressed; embeddings and ``lm_head``
are left untouched. The compressed values are written back into ordinary weights and the
model is saved as a normal HF checkpoint, which vLLM loads with no special support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from ..config import ModelConfig, Setting, calibration
from ..record import iter_records
from .awq import iter_awq
from .lowrank import asvd_truncate, svd_truncate
from .prune import collect_input_norms, magnitude_prune, wanda_prune
from .quant import fake_quant_rtn

log = logging.getLogger(__name__)

NEEDS_CALIBRATION = {"wanda", "asvd", "awq"}
NEEDS_NORMS = {"wanda", "asvd"}


def target_linears(model: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in n}


def calibration_batches(tokenizer, model_key: str, results_dir: Path | None, n_samples: int, seq_len: int,  # noqa: ANN001
                        source: str = "baseline_chains") -> list[torch.Tensor]:
    texts: list[str] = []
    if source == "baseline_chains" and results_dir is not None:
        run = Path(results_dir) / model_key / "base" / "none"
        texts = [r["think_text"] + r["answer_text"] for r in iter_records(run)]
        if not texts:
            log.warning("no baseline chains at %s; falling back to wikitext for calibration", run)
    if not texts:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = ["\n\n".join(ds["text"])]
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
    n = min(n_samples, len(ids) // seq_len)
    if n == 0:
        return [ids[None, :seq_len]]
    return [ids[i * seq_len:(i + 1) * seq_len][None] for i in range(n)]


@torch.no_grad()
def compress_model(model: nn.Module, spec: dict, calib: list[torch.Tensor] | None = None) -> dict:
    """Apply ``spec`` (a setting's ``weight`` block) in place. Returns summary stats."""
    method = spec["method"]
    linears = target_linears(model)
    norms, extra = None, {}
    if method in NEEDS_CALIBRATION and calib is None:
        raise ValueError(f"{method} needs calibration batches")
    if method in NEEDS_NORMS:
        norms = collect_input_norms(model, linears, calib)
    if method == "awq":  # works a group of layers at a time, so walk its output instead
        order = iter_awq(model, linears, calib, spec["bits"], spec.get("group_size", 128),  # type: ignore[arg-type]
                         n_tokens=spec.get("n_tokens", 4096), clip=spec.get("clip", True), stats=extra)
    else:
        order = ((n, None) for n in linears)
    rel_err, zeros, total = [], 0, 0
    for name, awq_w in order:
        lin = linears[name]
        w = lin.weight.data
        wf = w.float()
        if awq_w is not None:
            new = awq_w.float()
        elif method == "rtn":
            new = fake_quant_rtn(wf, spec["bits"], spec.get("group_size", 128), spec.get("sym", False))
        elif method == "magnitude":
            new = magnitude_prune(wf, spec["sparsity"])
        elif method == "wanda":
            new = wanda_prune(wf, norms[name], spec["sparsity"])  # type: ignore[index]
        elif method == "svd":
            new = svd_truncate(wf, spec["rank_frac"])
        elif method == "asvd":
            new = asvd_truncate(wf, norms[name], spec["rank_frac"], spec.get("alpha", 0.5))  # type: ignore[index]
        else:
            raise ValueError(f"unknown method {method}")
        rel_err.append(float((new - wf).norm() / wf.norm().clamp_min(1e-12)))
        zeros += int((new == 0).sum())
        total += new.numel()
        lin.weight.data.copy_(new.to(w.dtype))
    return {
        "n_linear": len(linears),
        "mean_rel_frob_error": sum(rel_err) / max(len(rel_err), 1),
        "max_rel_frob_error": max(rel_err, default=0.0),
        "zero_fraction": zeros / max(total, 1),
        **extra,
    }


def build_checkpoint(model_cfg: ModelConfig, setting: Setting, out_root: Path, results_dir: Path | None = None,
                     device: str | None = None) -> str:
    """Return a path (or hub id) vLLM/HF can load for this setting, building it if needed."""
    if not setting.changes_weights:
        return model_cfg.hf_id  # type: ignore[return-value]
    out = Path(out_root) / model_cfg.key / setting.name
    if (out / "DONE").exists():
        return str(out)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dtype = getattr(torch, model_cfg.dtype)
    work_dtype = save_dtype if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, torch_dtype=work_dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_cfg.tokenizer_path)
    spec = dict(setting.weight)  # type: ignore[arg-type]
    calib = None
    if spec["method"] in NEEDS_CALIBRATION:
        c = calibration()
        # AWQ uses AutoAWQ's calibration size (128 x 512 tokens) unless the setting says otherwise
        n_samples = spec.get("calib_samples", 128 if spec["method"] == "awq" else c.get("n_samples", 128))
        seq_len = spec.get("calib_seq_len", 512 if spec["method"] == "awq" else c.get("seq_len", 2048))
        calib = calibration_batches(tok, model_cfg.key, results_dir, n_samples, seq_len, c.get("source", "baseline_chains"))
    stats = compress_model(model, spec, calib)
    log.info("compressed %s/%s: %s", model_cfg.key, setting.name, stats)
    out.mkdir(parents=True, exist_ok=True)
    model.to(save_dtype).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "compress_stats.json").write_text(json.dumps({"setting": setting.name, **spec, **stats}, indent=2))
    (out / "DONE").touch()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return str(out)
