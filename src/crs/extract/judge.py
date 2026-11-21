"""Judge-model confirmation of scan hits.

For every record where the scan found the correct answer inside the reasoning, the judge
sees a window of reasoning around each hit (earliest first, up to ``max_hits``) and says
whether the solver states that value as its answer to the whole problem rather than as an
intermediate quantity. The first confirmed hit becomes ``first_correct_token_judged``.

    python -m crs.run judge --runs results/r1-1.5b/*/*      (GPU, vLLM)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..record import iter_records, write_records
from .score import outcome

JUDGE_MODEL = "Qwen/Qwen3-4B"
WINDOW = 500  # characters each side of the hit

PROMPT = """You are checking a transcript of a model solving a math problem.

Problem:
{problem}

Reference answer: {gold}

Excerpt from the solver's reasoning. The span to check is marked <<< >>>:
...{before}<<<{hit}>>>{after}...

Question: at the marked span, does the solver state the reference answer (or an
equivalent value) as its answer to the whole problem, even tentatively, rather than as
an intermediate quantity or a value it is merely testing? Reply with exactly YES or NO."""


def excerpt(think: str, char: int, hit_text: str) -> tuple[str, str, str]:
    end = char + len(hit_text)
    return think[max(0, char - WINDOW):char], think[char:end], think[end:end + WINDOW // 2]


def build_prompt(problem: str, gold: str, think: str, char: int, hit_text: str) -> str:
    before, hit, after = excerpt(think, char, hit_text)
    return PROMPT.format(problem=problem.strip(), gold=gold, before=before, hit=hit, after=after)


def judge_runs(runs: list[Path], problems: dict[str, dict], model: str = JUDGE_MODEL, max_hits: int = 3,
               backend=None) -> None:  # noqa: ANN001
    from transformers import AutoTokenizer

    from ..backends.base import DecodeSpec, make_backend

    tok = AutoTokenizer.from_pretrained(model)
    jobs = []  # (run, rid, hit_idx, prompt)
    for run in runs:
        texts = {r["rid"]: r["think_text"] for r in iter_records(run)}
        for row in iter_records(run, "scored.jsonl"):
            for j, h in enumerate(row.get("correct_hits", [])[:max_hits]):
                p = build_prompt(problems[row["problem_id"]]["problem"], row["gold"], texts[row["rid"]], h["char"], h["text"])
                chat = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
                jobs.append((run, row["rid"], j, tok(chat, add_special_tokens=False).input_ids))
    if not jobs:
        return
    be = backend or make_backend("vllm", model_path=model, dtype="float16", max_model_len=4096)
    outs = be.generate([j[3] for j in jobs], DecodeSpec(max_tokens=4, greedy=True, logprobs_k=0))
    verdicts: dict[tuple[str, str], dict[int, bool]] = {}
    for (run, rid, j, _), o in zip(jobs, outs):
        ans = tok.decode(o[0].token_ids, skip_special_tokens=True).strip().upper()
        verdicts.setdefault((str(run), rid), {})[j] = ans.startswith("YES")
    for run in runs:
        rows = list(iter_records(run, "scored.jsonl"))
        for row in rows:
            v = verdicts.get((str(run), row["rid"]), {})
            hits = row.get("correct_hits", [])
            row["judge_verdicts"] = [v.get(j) for j in range(min(len(hits), max_hits))]
            first = next((hits[j] for j in sorted(v) if v[j]), None)
            row["had_correct_judged"] = first is not None
            row["first_correct_token_judged"] = first["token"] if first else None
            row["outcome_judged"] = outcome(row["correct"], first is not None, row.get("truncated", False))
            row["judge_model"] = model
        write_records(run, rows, "scored.jsonl")
    (Path(runs[0]).parents[2] / "judge_log.json").write_text(json.dumps({"model": model, "n_prompts": len(jobs)}))
