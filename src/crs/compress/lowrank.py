"""Low-rank truncation: keep the top ``rank_frac`` of each matrix's singular values.

``svd`` truncates the weight itself. In the 2026-11-21 main run, keeping 80% of the rank
this way gave R1 0% accuracy and pure garbage text, because plain SVD treats every input
channel as equally important.

``asvd`` (activation-aware SVD, Yuan et al., 2023) first scales each input column by how
large that channel's activations are, ``S = diag(||X_j||^alpha)``. It truncates ``W S`` and
then undoes the scaling, so the error lands on channels the model barely uses.
Simplification: one ``rank_frac`` for every matrix, with no per-layer rank search. The
activation statistic is the L2 norm over calibration tokens (the same pass Wanda uses),
not the mean absolute value.
"""

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


@torch.no_grad()
def asvd_truncate(w: torch.Tensor, act_norm: torch.Tensor, rank_frac: float, alpha: float = 0.5) -> torch.Tensor:
    s = act_norm.float().to(w.device).clamp_min(1e-6) ** alpha
    s = s / s.mean()
    return (svd_truncate(w.float() * s[None, :], rank_frac) / s[None, :]).to(w.dtype)
