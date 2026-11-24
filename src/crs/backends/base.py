"""Backend interface shared by the vLLM and HF generation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

    def generate_stream(self, prompt_ids: list[list[int]], spec: DecodeSpec,
                        on_done: Callable[[int, list[GenResult]], None],
                        stop_check: Callable[[list[int]], bool] | None = None, chunk: int = 20) -> None:
        """Generate for every prompt, calling ``on_done(prompt_index, samples)`` as each one finishes.

        ``stop_check`` sees a chain's tokens so far; when it returns True the chain is stopped and
        reported with finish_reason "loop". This default works in chunks of ``chunk`` prompts and
        ignores ``stop_check``; the vLLM backend overrides it to keep the GPU full and stop early.
        """
        for c in range(0, len(prompt_ids), chunk):
            sub = replace(spec, seeds=spec.seeds[c:c + chunk] if spec.seeds else None)
            for i, out in enumerate(self.generate(prompt_ids[c:c + chunk], sub)):
                on_done(c + i, out)

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
