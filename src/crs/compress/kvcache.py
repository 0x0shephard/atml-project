"""Simulated KV-cache quantization for the HF backend.

Keys and values are fake-quantized (asymmetric min-max, per token, in groups of
``group_size`` channels along head_dim) as they enter the cache, then stored in the model
dtype. Every cached token, including the prompt and the current step, is quantized.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def fake_quant_lastdim(x: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if bits >= 16:
        return x
    d = x.shape[-1]
    g = d if group_size <= 0 or group_size > d else group_size
    assert d % g == 0, f"head_dim {d} not divisible by group {g}"
    xf = x.float().reshape(*x.shape[:-1], d // g, g)
    lo = xf.amin(-1, keepdim=True)
    hi = xf.amax(-1, keepdim=True)
    qmax = 2**bits - 1
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round((xf - lo) / scale), 0, qmax)
    return (q * scale + lo).reshape(x.shape).to(x.dtype)


class FakeQuantKVCache(DynamicCache):
    def __init__(self, bits: int, group_size: int = 64, **kwargs):  # noqa: ANN003
        super().__init__(**kwargs)
        self.kv_bits = bits
        self.kv_group = group_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # noqa: ANN001, ANN201
        k = fake_quant_lastdim(key_states, self.kv_bits, self.kv_group)
        v = fake_quant_lastdim(value_states, self.kv_bits, self.kv_group)
        return super().update(k, v, layer_idx, cache_kwargs)
