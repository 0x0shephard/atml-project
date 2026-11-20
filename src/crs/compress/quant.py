"""Simulated (fake) weight quantization: round-to-nearest, asymmetric min-max.

Weights are quantized per output row in groups of ``group_size`` input channels
(``group_size=-1``: one group per row, i.e. per-channel) and dequantized straight back,
so the model keeps its ordinary dtype but only carries the quantized values.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def fake_quant_rtn(w: torch.Tensor, bits: int, group_size: int = 128, sym: bool = False) -> torch.Tensor:
    out_f, in_f = w.shape
    g = in_f if group_size is None or group_size <= 0 else group_size
    if in_f % g != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {g}")
    x = w.float().reshape(out_f, in_f // g, g)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = (x.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-10)
        q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
        deq = q * scale
    else:
        qmax = 2**bits - 1
        lo = x.amin(-1, keepdim=True)
        hi = x.amax(-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-10)
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(x / scale) + zero, 0, qmax)
        deq = (q - zero) * scale
    return deq.reshape(out_f, in_f).to(w.dtype)
