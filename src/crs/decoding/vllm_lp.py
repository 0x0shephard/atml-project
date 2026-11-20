"""vLLM V1 logits processor for the entropy-adaptive hesitation penalty (RQ5).

Uses vLLM's ``AdapterLogitsProcessor`` (vllm >= 0.11), which wraps a per-request callable
``(output_ids, logits_row) -> logits_row``. Requests opt in through
``SamplingParams.extra_args["crs_entropy_penalty"]``; all other requests are untouched.
"""

from __future__ import annotations

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .penalties import EntropyPenalty


class EntropyPenaltyLogitsProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):  # noqa: ANN001, ANN201
        cfg = (params.extra_args or {}).get("crs_entropy_penalty")
        if not cfg:
            return None
        return EntropyPenalty(**cfg)
