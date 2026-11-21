"""Simulated KV-cache quantization for the HF backend, following KIVI (Liu et al., 2024).

Keys are fake-quantized **per channel**: each channel of a block of ``key_group`` consecutive
tokens shares one min/max. Values are fake-quantized **per token**, in groups of ``group_size``
channels along head_dim. The most recent ``residual`` tokens stay in full precision until
they leave that window, then are quantized as a block. Quantized values are stored in the
model dtype.

Per-channel keys are what make low-bit KV usable: Qwen-family keys have a few outlier
channels. Per-token min/max lets one outlier set the scale for every other channel in its
group, and in the 2026-11-21 smoke run that turned int4 output into gibberish from the
first token. ``key_mode: per_token`` keeps that older scheme for comparison.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def _fake_quant(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Asymmetric min-max fake-quant; one (min, scale) pair per slice along all dims but `dim`."""
    xf = x.float()
    lo = xf.amin(dim, keepdim=True)
    hi = xf.amax(dim, keepdim=True)
    qmax = 2**bits - 1
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round((xf - lo) / scale), 0, qmax)
    return (q * scale + lo).to(x.dtype)


def fake_quant_lastdim(x: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Per-token quantization: groups of `group_size` channels along the last dim."""
    if bits >= 16:
        return x
    d = x.shape[-1]
    g = d if group_size <= 0 or group_size > d else group_size
    assert d % g == 0, f"head_dim {d} not divisible by group {g}"
    xg = x.reshape(*x.shape[:-1], d // g, g)
    return _fake_quant(xg, bits, -1).reshape(x.shape)


def fake_quant_tokens(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel quantization over the token axis of x[batch, heads, tokens, head_dim]."""
    if bits >= 16:
        return x
    return _fake_quant(x, bits, -2)


class FakeQuantKVCache(DynamicCache):
    def __init__(self, bits: int, group_size: int = 32, key_mode: str = "per_channel",
                 key_group: int = 32, residual: int = 128, **kwargs):  # noqa: ANN003
        super().__init__(**kwargs)
        assert key_mode in ("per_channel", "per_token"), key_mode
        self.kv_bits = bits
        self.kv_group = group_size
        self.key_mode = key_mode
        self.key_group = key_group
        self.residual = residual if key_mode == "per_channel" else 0
        self.n_quant: dict[int, int] = {}  # tokens already quantized, per layer

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
        if self.key_mode == "per_token":
            k = fake_quant_lastdim(key_states, self.kv_bits, self.kv_group)
            v = fake_quant_lastdim(value_states, self.kv_bits, self.kv_group)
            return super().update(k, v, layer_idx, *args, **kwargs)

        # Store raw, then quantize, in place, each whole block of `key_group` tokens that
        # has left the full-precision window of the newest `residual` tokens.
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        done = self.n_quant.get(layer_idx, 0)
        total = keys.shape[-2]
        while total - done - self.residual >= self.key_group:
            s = slice(done, done + self.key_group)
            keys[:, :, s, :] = fake_quant_tokens(keys[:, :, s, :], self.kv_bits)
            values[:, :, s, :] = fake_quant_lastdim(values[:, :, s, :], self.kv_bits, self.kv_group)
            done += self.key_group
        self.n_quant[layer_idx] = done
        return keys, values
