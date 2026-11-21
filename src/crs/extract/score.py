"""Score a run: final answer, correctness, scan of the reasoning, failure label.

Writes ``scored.jsonl`` next to ``records.jsonl``: one row per record with every numeric
field and the extraction results, but without the long texts.

Outcome labels
    correct        final answer equals the gold answer
    truncated      no final answer because generation hit the token limit; running out of
                   room is not abandoning an answer, so these never count as commit_fail
                   (``had_correct_scan`` still says whether the answer had appeared)
    commit_fail    wrong (or no) final answer, but the correct answer appeared in the
                   reasoning first ("had it, then abandoned it")
    never_correct  wrong (or no) final answer and the correct answer never appeared
"""

from __future__ import annotations

from pathlib import Path

from ..record import iter_records, write_records
from .answers import final_answer, is_equiv
from .reasoning_scan import scan

TEXT_FIELDS = ("think_text", "answer_text")


def is_truncated(rec: dict, answered: bool) -> bool:
    return not answered and rec.get("finish_reason") == "length"


def outcome(correct: bool, had_correct: bool, truncated: bool = False) -> str:
    if correct:
        return "correct"
    if truncated:
        return "truncated"
    return "commit_fail" if had_correct else "never_correct"


def score_record(rec: dict, tokenizer=None) -> dict:  # noqa: ANN001
    pred = final_answer(rec["answer_text"])
    correct = is_equiv(pred, rec["gold"])
    sc = scan(rec["think_text"], rec["gold"], tokenizer)
    row = {k: v for k, v in rec.items() if k not in TEXT_FIELDS}
    row.update(sc)
    trunc = is_truncated(rec, pred is not None)
    row.update({
        "pred": pred,
        "answered": pred is not None,
        "truncated": trunc,
        "correct": correct,
        "outcome": outcome(correct, sc["had_correct_scan"], trunc),
        "outcome_strict": outcome(correct, sc["had_correct_strict"], trunc),
    })
    if sc["first_correct_token"] is not None and rec["n_think_tokens"]:
        row["tokens_after_first_correct"] = rec["n_think_tokens"] - sc["first_correct_token"]
        row["first_correct_frac"] = sc["first_correct_token"] / rec["n_think_tokens"]
    return row


def score_run(run: Path, tokenizer=None, records_name: str = "records.jsonl", out_name: str = "scored.jsonl") -> list[dict]:  # noqa: ANN001
    rows = [score_record(r, tokenizer) for r in iter_records(run, records_name)]
    write_records(run, rows, out_name)
    return rows
