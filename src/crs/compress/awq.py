"""AWQ (activation-aware weight quantization, Lin et al., 2024), as simulated quantization.

Plain round-to-nearest (``quant.fake_quant_rtn``) treats every input channel alike. AWQ notices
that a few input channels carry much larger activations, so rounding errors in their weights
cost far more. It scales those weight columns up before rounding and back down afterwards:

    W' = Q(W * s) / s,   s_j = mean|x_j| ** alpha   (normalised so sqrt(max * min) = 1)

``alpha`` is searched over a grid of 20 values for each set of layers that share an input; the
value with the smallest output error on calibration tokens wins (``alpha = 0`` is plain RTN, so
AWQ is never worse than RTN on the calibration data). Then, for every layer except q/k, the
range each weight group is rounded over is clipped: shrink factors 1.0, 0.95, ..., 0.55 are
tried per output row and group, keeping the one with the smallest output error.

This follows AutoAWQ's defaults (n_grid 20, max_shrink 0.5, asymmetric groups of 128, clip
skipped for q/k, o_proj scaled only when v_proj's output matches o_proj's input, which it does
not for grouped-query models like R1 and Qwen3). As in AutoAWQ, the statistics come from one
pass through the full-precision model. Simplifications:
- the scale search scores each layer's own output error (summed over the layers that share an
  input), not the output of the whole attention or MLP block;
- the error is measured on ``n_tokens`` calibration tokens sampled at random, not all of them.
Because the weights are written back de-quantized, the scales need not be folded into the
previous layer norm or linear; the result is the same.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn

from .quant import fake_quant_rtn

# Linears that read the same input inside one decoder layer (Qwen2 / Qwen3 / Llama naming).
INPUT_GROUPS = [("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",)]
NO_CLIP = {"q_proj", "k_proj"}


def layer_groups(linears: dict[str, nn.Linear]) -> list[list[str]]:
    """Group linear names that share an input; unknown names get a group of their own."""
    groups, seen = [], set()
    for name in linears:
        if name in seen:
            continue
        parent, short = name.rsplit(".", 1)
        members = [name]
        for g in INPUT_GROUPS:
            if short in g:
                members = [f"{parent}.{s}" for s in g if f"{parent}.{s}" in linears]
                break
        seen.update(members)
        groups.append(members)
    return groups


@torch.no_grad()
def collect_inputs(model: nn.Module, linears: dict[str, nn.Linear], groups: list[list[str]],
                   batches: list[torch.Tensor], n_tokens: int, seed: int = 0) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """For the first linear of every group: mean |x| per input channel over all calibration
    tokens, and ``n_tokens`` randomly sampled input rows (float16, on the CPU)."""
    gen = torch.Generator().manual_seed(seed)
    per_batch = math.ceil(n_tokens / max(len(batches), 1))
    picks = [torch.randperm(b.numel(), generator=gen)[:per_batch] for b in batches]
    abs_sum: dict[str, torch.Tensor] = {}
    rows: dict[str, list[torch.Tensor]] = {g[0]: [] for g in groups}
    count = [0]
    cur = [0]

    def make_hook(name: str):  # noqa: ANN202
        def hook(_m, inp, _out):  # noqa: ANN001, ANN202
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            a = x.float().abs().sum(0)
            abs_sum[name] = abs_sum[name] + a if name in abs_sum else a
            rows[name].append(x[picks[cur[0]].to(x.device)].to("cpu", torch.float16))
        return hook

    hooks = [linears[g[0]].register_forward_hook(make_hook(g[0])) for g in groups]
    try:
        device = next(model.parameters()).device
        for i, ids in enumerate(batches):
            cur[0] = i
            model(input_ids=ids.to(device), use_cache=False)
            count[0] += ids.numel()
    finally:
        for h in hooks:
            h.remove()
    return {k: (abs_sum[k] / count[0], torch.cat(rows[k])) for k in abs_sum}


def _scales(absmean: torch.Tensor, alpha: float) -> torch.Tensor:
    s = absmean.float().pow(alpha).clamp_min(1e-4)
    return s / (s.max() * s.min()).sqrt()


@torch.no_grad()
def search_scale(x: torch.Tensor, absmean: torch.Tensor, w: torch.Tensor, bits: int, group_size: int,
                 n_grid: int = 20) -> tuple[torch.Tensor, float]:
    """Best per-input-channel scale for weights ``w`` (out, in) reading inputs ``x`` (tokens, in)."""
    ref = x @ w.T
    best_err, best_s, best_a = math.inf, None, 0.0
    for i in range(n_grid):
        a = i / n_grid
        s = _scales(absmean, a)
        wq = fake_quant_rtn(w * s, bits, group_size) / s
        err = float((x @ wq.T - ref).pow(2).mean())
        if err < best_err:
            best_err, best_s, best_a = err, s, a
    return best_s, best_a  # type: ignore[return-value]


@torch.no_grad()
def search_clip(x: torch.Tensor, w: torch.Tensor, bits: int, group_size: int, n_grid: int = 20,
                max_shrink: float = 0.5, row_chunk: int = 256) -> torch.Tensor:
    """Per (row, group) clip value for ``w`` (out, in), scored by the output error on ``x``.

    The error of a group with weight change d over inputs x_g is mean_t (x_g[t] . d)^2 =
    d^T C_g d with C_g = x_g^T x_g / T, so only the per-group covariances are needed."""
    out_f, in_f = w.shape
    g = group_size if group_size and group_size > 0 else in_f
    n_g = in_f // g
    xg = x.reshape(x.shape[0], n_g, g)
    cov = torch.einsum("tgi,tgj->gij", xg, xg) / x.shape[0]
    wg = w.reshape(out_f, n_g, g)
    org_max = wg.abs().amax(-1, keepdim=True)
    best_max = org_max.clone()
    for r0 in range(0, out_f, row_chunk):
        rows = slice(r0, min(r0 + row_chunk, out_f))
        wr, mr = wg[rows], org_max[rows]
        best_err = torch.full(mr.shape[:2], math.inf, device=w.device)
        for i in range(int(max_shrink * n_grid)):
            mx = mr * (1 - i / n_grid)
            cur = torch.clamp(wr, -mx, mx)
            q = fake_quant_rtn(cur.reshape(-1, in_f), bits, g).reshape(wr.shape)
            d = q - wr
            err = torch.einsum("rgi,gij,rgj->rg", d, cov, d)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_max[rows] = torch.where(better[..., None], mx, best_max[rows])
    return best_max


@torch.no_grad()
def iter_awq(model: nn.Module, linears: dict[str, nn.Linear], batches: list[torch.Tensor], bits: int,
             group_size: int = 128, n_grid: int = 20, max_shrink: float = 0.5, n_tokens: int = 4096,
             clip: bool = True, stats: dict | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (name, new de-quantized weight in the layer's dtype) one input group at a time, so
    the caller can write each back before the next is built. Each group's search reads only its
    own weights and the statistics gathered up front, so writing back early changes nothing.
    ``stats`` (if given) receives the mean alpha and mean clip ratio at the end."""
    groups = layer_groups(linears)
    feats = collect_inputs(model, linears, groups, batches, n_tokens)
    alphas: list[float] = []
    shrink: list[float] = []
    for members in groups:
        absmean, rows = feats.pop(members[0])
        dev = linears[members[0]].weight.device
        x = rows.to(dev, torch.float32)
        ws = [linears[m].weight.data.float() for m in members]
        short = members[0].rsplit(".", 1)[-1]
        scale_ok = True
        if short == "o_proj":  # AutoAWQ scales o_proj through v_proj only when the shapes line up
            v = linears.get(members[0].rsplit(".", 1)[0] + ".v_proj")
            scale_ok = v is not None and v.weight.shape[0] == ws[0].shape[1]
        if scale_ok:
            s, a = search_scale(x, absmean.to(dev), torch.cat(ws), bits, group_size, n_grid)
            alphas.append(a)
        else:
            s = torch.ones(ws[0].shape[1], device=dev)
        xs = x / s
        out = []
        for m, w in zip(members, ws):
            wsc = w * s
            if clip and m.rsplit(".", 1)[-1] not in NO_CLIP:
                mx = search_clip(xs, wsc, bits, group_size, n_grid, max_shrink)
                o, i_ = wsc.shape
                g = group_size if group_size and group_size > 0 else i_
                org = wsc.reshape(o, i_ // g, g).abs().amax(-1, keepdim=True)
                shrink.append(float((mx / org.clamp_min(1e-12)).mean()))
                wsc = torch.clamp(wsc.reshape(o, i_ // g, g), -mx, mx).reshape(o, i_)
            out.append((m, (fake_quant_rtn(wsc, bits, group_size) / s).to(linears[m].weight.dtype)))
        del x, xs, ws
        yield from out
    if stats is not None:
        stats["awq_mean_alpha"] = sum(alphas) / max(len(alphas), 1)
        stats["awq_mean_clip"] = sum(shrink) / max(len(shrink), 1)


def awq_quantize(model: nn.Module, linears: dict[str, nn.Linear], batches: list[torch.Tensor], bits: int,
                 **kw) -> tuple[dict[str, torch.Tensor], dict]:
    """All new weights at once (for tests); ``iter_awq`` is the memory-friendly version."""
    stats: dict = {}
    new = dict(iter_awq(model, linears, batches, bits, stats=stats, **kw))
    return new, stats
