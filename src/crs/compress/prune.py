"""Unstructured pruning: magnitude and Wanda (Sun et al., 2023).

Wanda scores each weight by |W_ij| * ||X_j||_2 (the L2 norm of input feature j over
calibration tokens) and removes the lowest-scoring fraction within each output row.
Simplification: activation norms are collected in one pass through the dense model
instead of Wanda's layer-by-layer pass through already-pruned layers.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


@torch.no_grad()
def prune_rows(w: torch.Tensor, score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Zero the ``sparsity`` fraction of lowest-``score`` entries in every row."""
    k = int(w.shape[1] * sparsity)
    if k <= 0:
        return w
    idx = torch.topk(score, k, dim=1, largest=False).indices
    out = w.clone()
    out.scatter_(1, idx, 0)
    return out


@torch.no_grad()
def magnitude_prune(w: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Per-matrix magnitude pruning (exactly ``sparsity`` of the whole matrix)."""
    k = int(w.numel() * sparsity)
    if k <= 0:
        return w
    thresh_idx = torch.topk(w.abs().flatten().float(), k, largest=False).indices
    out = w.clone().flatten()
    out[thresh_idx] = 0
    return out.view_as(w)


@torch.no_grad()
def wanda_prune(w: torch.Tensor, act_norm: torch.Tensor, sparsity: float) -> torch.Tensor:
    score = w.abs().float() * act_norm.float().to(w.device)[None, :]
    return prune_rows(w, score, sparsity)


@torch.no_grad()
def collect_input_norms(model: nn.Module, linears: dict[str, nn.Linear], batches: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
    """L2 norm of each linear layer's input features over all calibration tokens."""
    sq: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):  # noqa: ANN202
        def hook(_m, inp, _out):  # noqa: ANN001, ANN202
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s = (x * x).sum(0)
            sq[name] = sq[name] + s if name in sq else s
        return hook

    for name, lin in linears.items():
        hooks.append(lin.register_forward_hook(make_hook(name)))
    try:
        device = next(model.parameters()).device
        for ids in batches:
            model(input_ids=ids.to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {k: v.sqrt() for k, v in sq.items()}
