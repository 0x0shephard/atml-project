"""Thinking-budget evaluation by truncation (RQ6 and the budget fix).

A chain generated with the full 8,192-token budget is cut after ``budget`` thinking
tokens, a short closing text ending the thinking is appended, and the model completes
only the boxed answer (greedy, few tokens). Chains that already finished their thinking
within the budget keep their original answer.
"""

from __future__ import annotations

FORCE_TEXT = "\n\nI have to give my final answer now.\n</think>\n\nThe final answer is $\\boxed{"
ANSWER_PREFIX = "The final answer is $\\boxed{"
FORCED_MAX_TOKENS = 48


def needs_forcing(rec: dict, budget: int) -> bool:
    return not (rec["think_closed"] and rec["n_think_tokens"] <= budget)


def forced_prompt(prompt_ids: list[int], out_ids: list[int], budget: int, tokenizer) -> list[int]:  # noqa: ANN001
    return list(prompt_ids) + list(out_ids[:budget]) + tokenizer.encode(FORCE_TEXT, add_special_tokens=False)


def budget_record(rec: dict, budget: int, out_ids: list[int], tokenizer, continuation: str | None) -> dict:  # noqa: ANN001
    """The record as it would look had thinking been capped at ``budget`` tokens."""
    r = dict(rec)
    r["budget"] = budget
    r["fix"] = f"{rec['fix']}@{budget}"
    if continuation is None:  # finished within budget: unchanged
        r["forced"] = False
        return r
    r["forced"] = True
    r["finish_reason"] = "forced"
    r["think_closed"] = True
    r["n_think_tokens"] = min(budget, rec["n_think_tokens"])
    r["n_tokens"] = r["n_think_tokens"]
    r["think_text"] = tokenizer.decode(out_ids[:budget], skip_special_tokens=False)
    r["answer_text"] = ANSWER_PREFIX + continuation
    r["hesitation_positions"] = [p for p in rec["hesitation_positions"] if p < budget]
    r["hesitation_count"] = len(r["hesitation_positions"])
    r["hesitation_per_1k"] = 1000.0 * r["hesitation_count"] / max(r["n_think_tokens"], 1)
    r["mean_entropy_think"] = r["mean_entropy_at_hesitation"] = None  # full-chain values no longer apply
    return r
