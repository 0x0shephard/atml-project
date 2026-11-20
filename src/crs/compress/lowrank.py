"""Low-rank truncation: keep the top ``rank_frac`` of each matrix's singular values."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def svd_truncate(w: torch.Tensor, rank_frac: float) -> torch.Tensor:
    full = min(w.shape)
    r = max(1, math.ceil(full * rank_frac))
    if r >= full:
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    return ((u[:, :r] * s[:r]) @ vh[:r]).to(w.dtype)
