"""Backend interface shared by the vLLM and HF generation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..record import GenResult


@dataclass
class DecodeSpec:
    max_tokens: int
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    n: int = 1
    greedy: bool = False
    seeds: list[int] | None = None  # one per prompt, for reproducible sampling
    logprobs_k: int = 20  # 0 disables per-token stats
    logit_bias: dict[int, float] | None = None
    # {"token_ids": [...], "alpha_max": float, "h0": float, "tau": float, "think_end_ids": [...]}
    entropy_penalty: dict[str, Any] | None = None
    ignore_eos: bool = False  # benchmarks only: every sequence runs to max_tokens
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt_ids: list[list[int]], spec: DecodeSpec) -> list[list[GenResult]]:
        """Return ``spec.n`` samples for each prompt, in prompt order."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def make_backend(kind: str, **kwargs: Any) -> Backend:
    if kind == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(**kwargs)
    if kind == "hf":
        from .hf_backend import HFBackend

        return HFBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r}")
