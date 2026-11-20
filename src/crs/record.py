"""Per-generation records and the on-disk run layout.

A run directory ``results/<model>/<setting>/<fix>/`` holds

* ``records.jsonl``  - one JSON line per (problem, sample): text, lengths, hesitation
  positions, finish reason, and (after ``crs.run score``) extraction results.
* ``arrays/part-XXXX.npz`` - per-token arrays for each record, keyed ``<record_id>/<name>``:
  ``ids`` (int32 token ids), ``lp`` (float16 log-prob of the chosen token under the raw
  model distribution) and ``ent`` (float16 top-k entropy estimate, nats).
* ``meta.json`` - configs, versions and throughput of the run.

Records are appended one chunk at a time, so a run killed mid-way resumes from the last
completed chunk.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


# --------------------------------------------------------------------------- outputs


@dataclass
class GenResult:
    """What a backend returns for one sample."""

    token_ids: list[int]
    logprobs: np.ndarray | None = None  # chosen-token log-prob per position
    entropy: np.ndarray | None = None  # top-k entropy estimate per position
    finish_reason: str = "length"  # "stop" | "length"


def topk_entropy(topk_logprobs: Sequence[float] | np.ndarray) -> float:
    """Entropy (nats) estimated from the top-k log-probs of a distribution.

    The probability mass outside the top-k is treated as a single bucket, which gives a
    lower bound on the true entropy. Both backends use this same estimator so values are
    comparable across vLLM and HF runs.
    """
    lp = np.asarray(topk_logprobs, dtype=np.float64)
    lp = lp[np.isfinite(lp)]
    p = np.exp(lp)
    h = float(-(p * lp).sum())
    tail = 1.0 - float(p.sum())
    if tail > 1e-12:
        h -= tail * math.log(tail)
    return h


def topk_entropy_torch(topk_logprobs):  # type: ignore[no-untyped-def]
    """Batched torch version of :func:`topk_entropy` over the last dim."""
    import torch

    lp = topk_logprobs.float()
    p = lp.exp()
    h = -(p * torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))).sum(-1)
    tail = (1.0 - p.sum(-1)).clamp_min(0.0)
    h = h - torch.where(tail > 1e-12, tail * tail.clamp_min(1e-12).log(), torch.zeros_like(tail))
    return h


# --------------------------------------------------------------------------- records


@dataclass
class TokenInfo:
    """Tokenizer-derived constants needed to build records."""

    think_end_ids: list[int]
    stop_ids: set[int]
    hesitation_ids: set[int] = field(default_factory=set)


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> int:
    n = len(sub)
    if n == 0:
        return -1
    first = sub[0]
    for i in range(len(seq) - n + 1):
        if seq[i] == first and list(seq[i : i + n]) == list(sub):
            return i
    return -1


def build_record(
    problem: dict,
    model: str,
    setting: str,
    fix: str,
    sample_idx: int,
    gen: GenResult,
    tokenizer,  # noqa: ANN001
    tinfo: TokenInfo,
) -> tuple[dict, dict[str, np.ndarray]]:
    ids = list(gen.token_ids)
    lp = gen.logprobs
    ent = gen.entropy
    # strip trailing stop / pad tokens so lengths count real tokens only
    n = len(ids)
    while n > 0 and ids[n - 1] in tinfo.stop_ids:
        n -= 1
    stopped = gen.finish_reason == "stop" or n < len(ids)
    ids = ids[:n]
    if lp is not None:
        lp = np.asarray(lp)[:n]
    if ent is not None:
        ent = np.asarray(ent)[:n]

    te = find_subsequence(ids, tinfo.think_end_ids)
    think_closed = te >= 0
    n_think = te if think_closed else n
    think_text = tokenizer.decode(ids[:n_think], skip_special_tokens=False)
    answer_text = tokenizer.decode(ids[n_think + len(tinfo.think_end_ids):], skip_special_tokens=True) if think_closed else ""
    # think_text is kept verbatim (Qwen3 emits "<think>\n" itself; R1 templates put it in
    # the prompt) so that character offsets map back onto token positions.

    hes_pos = [i for i in range(n_think) if ids[i] in tinfo.hesitation_ids]
    rid = f"{problem['id']}__s{sample_idx}"
    rec: dict[str, Any] = {
        "rid": rid,
        "problem_id": problem["id"],
        "source": problem["source"],
        "level": problem.get("level"),
        "gold": problem["answer"],
        "model": model,
        "setting": setting,
        "fix": fix,
        "sample_idx": sample_idx,
        "finish_reason": "stop" if stopped else "length",
        "think_closed": think_closed,
        "n_tokens": n,
        "n_think_tokens": n_think,
        "hesitation_positions": hes_pos,
        "hesitation_count": len(hes_pos),
        "hesitation_per_1k": 1000.0 * len(hes_pos) / max(n_think, 1),
        "mean_entropy_think": float(np.mean(ent[:n_think])) if ent is not None and n_think > 0 else None,
        "mean_entropy_at_hesitation": float(np.mean(ent[hes_pos])) if ent is not None and hes_pos else None,
        "think_text": think_text,
        "answer_text": answer_text,
    }
    arrays = {"ids": np.asarray(ids, dtype=np.int32)}
    if lp is not None:
        arrays["lp"] = lp.astype(np.float16)
    if ent is not None:
        arrays["ent"] = ent.astype(np.float16)
    return rec, arrays


# --------------------------------------------------------------------------- run dirs


def run_dir(results: Path, model: str, setting: str, fix: str, samples: int = 1) -> Path:
    name = fix if samples == 1 else f"{fix}-vote{samples}"
    return Path(results) / model / setting / name


class RunWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        (self.path / "arrays").mkdir(parents=True, exist_ok=True)
        self.records_file = self.path / "records.jsonl"

    def existing_parts(self) -> int:
        return len(list((self.path / "arrays").glob("part-*.npz")))

    def done(self) -> dict[str, int]:
        """problem_id -> number of samples already written."""
        out: dict[str, int] = {}
        for rec in iter_records(self.path):
            out[rec["problem_id"]] = out.get(rec["problem_id"], 0) + 1
        return out

    def write_chunk(self, records: list[dict], arrays: list[dict[str, np.ndarray]]) -> None:
        part = f"arrays/part-{self.existing_parts():04d}.npz"
        flat = {f"{r['rid']}/{k}": v for r, a in zip(records, arrays) for k, v in a.items()}
        tmp = self.path / (part + ".tmp.npz")
        np.savez_compressed(tmp, **flat)
        tmp.rename(self.path / part)
        with open(self.records_file, "a") as f:
            for r in records:
                r["array_file"] = part
                f.write(json.dumps(r) + "\n")

    def write_meta(self, meta: dict) -> None:
        meta_file = self.path / "meta.json"
        old = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        sessions = old.get("sessions", []) + [{**meta, "time": time.strftime("%Y-%m-%d %H:%M:%S")}]
        meta_file.write_text(json.dumps({"sessions": sessions}, indent=2))


def iter_records(path: Path, name: str = "records.jsonl") -> Iterator[dict]:
    f = Path(path) / name
    if not f.exists():
        return
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # a line cut off by a killed session
                    continue


def load_arrays(path: Path, records: Iterable[dict], names: Iterable[str] = ("ids", "lp", "ent")) -> dict[str, dict[str, np.ndarray]]:
    """Load per-token arrays for the given records, grouped by part file."""
    by_part: dict[str, list[str]] = {}
    for r in records:
        by_part.setdefault(r["array_file"], []).append(r["rid"])
    out: dict[str, dict[str, np.ndarray]] = {}
    for part, rids in by_part.items():
        with np.load(Path(path) / part) as z:
            for rid in rids:
                out[rid] = {k: z[f"{rid}/{k}"] for k in names if f"{rid}/{k}" in z.files}
    return out


def write_records(path: Path, records: Iterable[dict], name: str) -> None:
    tmp = Path(path) / (name + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.rename(Path(path) / name)
