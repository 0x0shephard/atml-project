"""Command-line entry point.

    python -m crs.run gen      --model r1-1.5b --setting rtn-w3-g128 --fix none [--samples 4] [--limit 10]
    python -m crs.run budget   --model r1-1.5b --setting rtn-w3-g128 --fix none
    python -m crs.run score    --runs 'results/r1-1.5b/*/*'
    python -m crs.run judge    --runs 'results/*/*/*'
    python -m crs.run wordlists --model r1-1.5b
    python -m crs.run manual   export|agreement ...
    python -m crs.run jobs     --priority configs/priority.yaml
    python -m crs.run report   --results results --out reports
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config, data
from .backends.base import Backend, DecodeSpec, make_backend
from .decoding.budget import FORCED_MAX_TOKENS, budget_record, forced_prompt, needs_forcing
from .decoding.penalties import expand_wordlist, logit_bias, read_wordlist
from .prompts import load_tokenizer, prompt_ids, token_info
from .record import RunWriter, build_record, iter_records, load_arrays, run_dir, write_records

log = logging.getLogger("crs")

DEFAULT_RESULTS = Path("results")
DEFAULT_CKPT = Path("checkpoints")


def stable_seed(seed: int, pid: str) -> int:
    return int(hashlib.sha256(f"{seed}-{pid}".encode()).hexdigest()[:8], 16)


def decode_spec(model_key: str, fix_name: str, tokenizer, max_tokens: int, samples: int) -> DecodeSpec:  # noqa: ANN001
    mcfg = config.models()[model_key]
    fx = config.fixes()[fix_name]
    gd = config.generation_defaults()
    s = mcfg.sampling
    spec = DecodeSpec(max_tokens=max_tokens, temperature=s["temperature"], top_p=s["top_p"], top_k=s.get("top_k", -1),
                      n=samples, logprobs_k=gd["logprobs_k"])
    if fx.type == "logit_bias":
        wl = expand_wordlist(read_wordlist(fx.params["wordlist"], model_key), tokenizer)
        spec.logit_bias = logit_bias(wl.token_ids, fx.params["alpha"])
    elif fx.type == "entropy_adaptive":
        wl = expand_wordlist(read_wordlist(fx.params["wordlist"], model_key), tokenizer)
        spec.entropy_penalty = {
            "token_ids": wl.token_ids,
            "alpha_max": fx.params["alpha_max"],
            "h0": fx.params["h0"],
            "tau": fx.params["tau"],
            "think_end_ids": tokenizer.encode(mcfg.think_end, add_special_tokens=False),
        }
    elif fx.type == "budget":
        raise SystemExit(f"fix {fix_name!r} is derived from the 'none' run: use `crs.run budget` then `score`")
    return spec


def open_backend(model_key: str, setting_name: str, max_model_len: int, results: Path, ckpt_dir: Path,
                 entropy_penalty: bool = False, backend: str | None = None,
                 attention_backend: str | None = None) -> Backend:
    from .compress.apply import build_checkpoint

    mcfg = config.models()[model_key]
    st = config.settings()[setting_name]
    path = build_checkpoint(mcfg, st, ckpt_dir, results)
    return make_backend(
        backend or st.backend,
        model_path=path,
        tokenizer_path=mcfg.tokenizer_path,
        dtype=mcfg.dtype,
        max_model_len=max_model_len,
        seed=config.generation_defaults()["seed"],
        enable_entropy_penalty=entropy_penalty,
        kv=st.kv,
        batch_size=mcfg.hf_batch_size,
        attention_backend=attention_backend or config.generation_defaults().get("vllm_attention_backend"),
    )


def bench_run(model_key: str, attention_backend: str, concurrency: int, gen_tokens: int, results: Path,
              ckpt_dir: Path = DEFAULT_CKPT) -> dict:
    """vLLM throughput: `concurrency` problems at once, each generating exactly `gen_tokens`
    tokens (EOS ignored) with the real sampling and logprob settings. Writes one JSON file."""
    mcfg = config.models()[model_key]
    gd = config.generation_defaults()
    tok = load_tokenizer(mcfg.tokenizer_path)
    problems = data.load(limit=concurrency)
    prompts = [prompt_ids(p, mcfg, tok) for p in problems]
    out = {"model": model_key, "attention_backend": attention_backend, "concurrency": len(prompts),
           "gen_tokens": gen_tokens}
    t0 = time.time()
    be = open_backend(model_key, "base", max(len(p) for p in prompts) + gen_tokens + 16, results, ckpt_dir,
                      backend="vllm", attention_backend=attention_backend)
    out["startup_s"] = time.time() - t0
    spec = decode_spec(model_key, "none", tok, gen_tokens, 1)  # the real sampling and logprob settings
    spec.ignore_eos = True
    spec.seeds = [stable_seed(gd["seed"], p["id"]) for p in problems]
    t1 = time.time()
    res = be.generate(prompts, spec)
    dt = time.time() - t1
    n_tok = sum(len(r[0].token_ids) for r in res)
    out.update({"gen_s": dt, "tokens": n_tok, "tokens_per_s": n_tok / dt, "per_seq_tokens_per_s": n_tok / dt / len(prompts),
                "nonfinite": int(sum(not np.isfinite(r[0].logprobs).all() for r in res if r[0].logprobs is not None))})
    d = results / "bench"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{model_key}__{attention_backend}__c{len(prompts)}.json").write_text(json.dumps(out, indent=1))
    log.info("bench %s", out)
    return out


def generate_run(model_key: str, setting: str, fix: str, samples: int = 1, limit: int | None = None,
                 max_tokens: int | None = None, shard: tuple[int, int] = (0, 1), results: Path = DEFAULT_RESULTS,
                 ckpt_dir: Path = DEFAULT_CKPT, backend: Backend | None = None, backend_kind: str | None = None,
                 problems_file: Path | None = None) -> Path:
    mcfg = config.models()[model_key]
    gd = config.generation_defaults()
    max_tokens = max_tokens or gd["max_tokens"]
    problems = data.load(problems_file or data.PROBLEMS_FILE, limit=limit)
    problems = problems[shard[0]::shard[1]]
    rd = run_dir(results, model_key, setting, fix, samples)
    writer = RunWriter(rd)
    meta_file = rd / "meta.json"
    if meta_file.exists():
        # resume skips finished problems, so never mix chains generated with a different length limit
        # (e.g. 1,024-token smoke results inside an 8,192-token main run)
        old = {s.get("max_tokens") for s in json.loads(meta_file.read_text()).get("sessions", [])}
        if old - {max_tokens}:
            raise ValueError(f"{rd} holds results generated with max_tokens={sorted(old)}, not {max_tokens}; "
                             "move them aside before resuming")
    done = writer.done()
    todo = [p for p in problems if done.get(p["id"], 0) < samples]
    log.info("%s: %d/%d problems to do", rd, len(todo), len(problems))
    if not todo:
        return rd

    tok = load_tokenizer(mcfg.tokenizer_path)
    hand = expand_wordlist(read_wordlist("handwritten"), tok)
    tinfo = token_info(mcfg, tok, set(hand.token_ids))
    spec = decode_spec(model_key, fix, tok, max_tokens, samples)
    own_backend = backend is None
    if backend is None:
        backend = open_backend(model_key, setting, max_tokens + 2048, results, ckpt_dir,
                               entropy_penalty=bool(spec.entropy_penalty), backend=backend_kind)

    t_start, n_tok, n_nonfinite = time.time(), 0, 0
    kind = backend_kind or config.settings()[setting].backend
    chunk = gd.get("chunk_size_hf", gd["chunk_size"]) if kind == "hf" else gd["chunk_size"]
    for c in range(0, len(todo), chunk):
        batch = todo[c:c + chunk]
        spec.seeds = [stable_seed(gd["seed"], p["id"]) for p in batch]
        t0 = time.time()
        outs = backend.generate([prompt_ids(p, mcfg, tok) for p in batch], spec)
        recs, arrs = [], []
        for p, samples_out in zip(batch, outs):
            for si, g in enumerate(samples_out):
                if g.logprobs is not None and not np.all(np.isfinite(g.logprobs)):
                    n_nonfinite += 1
                r, a = build_record(p, model_key, setting, fix, si, g, tok, tinfo)
                recs.append(r)
                arrs.append(a)
                n_tok += len(g.token_ids)
        writer.write_chunk(recs, arrs)
        log.info("chunk %d: %d records, %.0f s", c // chunk, len(recs), time.time() - t0)
    dt = time.time() - t_start
    writer.write_meta({
        "model": model_key, "setting": setting, "fix": fix, "samples": samples, "max_tokens": max_tokens,
        "n_problems": len(todo), "tokens": n_tok, "seconds": dt, "tokens_per_s": n_tok / max(dt, 1e-9),
        "records_with_nonfinite_logprobs": n_nonfinite, "hesitation_skipped_words": hand.skipped,
        "logit_bias_tokens": len(spec.logit_bias or {}),
    })
    if n_nonfinite:
        log.warning("%d records had non-finite logprobs: check fp16 numerics (models.yaml dtype)", n_nonfinite)
    if own_backend:
        backend.close()
    return rd


def budget_run(model_key: str, setting: str, fix: str, samples: int = 1, budgets: list[int] | None = None,
               results: Path = DEFAULT_RESULTS, ckpt_dir: Path = DEFAULT_CKPT, backend: Backend | None = None,
               backend_kind: str | None = None, problems_file: Path | None = None) -> Path:
    mcfg = config.models()[model_key]
    budgets = budgets or config.budgets()
    rd = run_dir(results, model_key, setting, fix, samples)
    recs = list(iter_records(rd))
    arrs = load_arrays(rd, recs, ("ids",))
    probs = {p["id"]: p for p in data.load(problems_file or data.PROBLEMS_FILE)}
    tok = load_tokenizer(mcfg.tokenizer_path)
    jobs: list[tuple[int, int, list[int]]] = []  # (budget, record idx, prompt)
    for b in budgets:
        for i, r in enumerate(recs):
            if needs_forcing(r, b):
                jobs.append((b, i, forced_prompt(prompt_ids(probs[r["problem_id"]], mcfg, tok), arrs[r["rid"]]["ids"].tolist(), b, tok)))
    conts: dict[tuple[int, int], str] = {}
    if jobs:
        own = backend is None
        if backend is None:
            backend = open_backend(model_key, setting, max(budgets) + 2048, results, ckpt_dir, backend=backend_kind)
        outs = backend.generate([j[2] for j in jobs], DecodeSpec(max_tokens=FORCED_MAX_TOKENS, greedy=True, logprobs_k=0))
        for (b, i, _), o in zip(jobs, outs):
            conts[(b, i)] = tok.decode(o[0].token_ids, skip_special_tokens=True)
        if own:
            backend.close()
    for b in budgets:
        out = [budget_record(r, b, arrs[r["rid"]]["ids"].tolist(), tok, conts.get((b, i))) for i, r in enumerate(recs)]
        write_records(rd, out, f"budget_{b}.jsonl")
    return rd


def score_runs(runs: list[Path], problems_file: Path | None = None) -> None:
    from .extract.score import score_run

    pf = problems_file or data.PROBLEMS_FILE
    statements = {p["id"]: p["problem"] for p in data.load(pf)} if pf.exists() else {}
    for run in runs:
        recs = list(iter_records(run))
        if not recs:
            continue
        tok = load_tokenizer(config.models()[recs[0]["model"]].tokenizer_path)
        rows = score_run(run, tok, problems=statements)
        for f in sorted(run.glob("budget_*.jsonl")):
            score_run(run, tok, f.name, "scored_" + f.name, problems=statements)
        acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
        cf = sum(r["outcome"] == "commit_fail" for r in rows) / max(len(rows), 1)
        print(f"{run}: n={len(rows)} acc={acc:.3f} commit_fail={cf:.3f}")


def expand_jobs(spec_file: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = config.load_yaml(spec_file)
    jobs = []
    for g in spec["groups"]:
        if "bench" in g:  # speed benchmark: one job per model x attention backend x concurrency
            b = g["bench"]
            for m in g["models"]:
                for attn in b["attention_backends"]:
                    for c in b["concurrency"]:
                        jobs.append({"group": g["name"], "kind": "bench", "model": m, "attn": attn, "concurrency": c,
                                     "gen_tokens": b["gen_tokens"], "timeout_min": b.get("timeout_min")})
            continue
        fixes = g.get("fixes") or [g.get("fix", "none")]
        for fix in fixes:
            for m in g["models"]:
                for s in g["settings"]:
                    job = {"group": g["name"], "kind": "gen", "model": m, "setting": s, "fix": fix,
                           "samples": g.get("samples", 1)}
                    if g.get("limit"):  # per-group problem limit (first N problems), overrides the file's
                        job["limit"] = g["limit"]
                    jobs.append(job)
    extra = {k: spec[k] for k in ("limit", "max_tokens") if k in spec}
    return jobs, extra


def _runs(patterns: list[str]) -> list[Path]:
    out = []
    for p in patterns:
        out.extend(Path(x) for x in sorted(glob.glob(p)) if (Path(x) / "records.jsonl").exists())
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="crs.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, choices=list(config.models()))
        p.add_argument("--setting", required=True, choices=list(config.settings()))
        p.add_argument("--fix", default="none", choices=list(config.fixes()))
        p.add_argument("--samples", type=int, default=1)
        p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
        p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
        p.add_argument("--backend", choices=["vllm", "hf"], default=None, help="override the setting's backend")
        p.add_argument("--problems", type=Path, default=None)

    g = sub.add_parser("gen", help="generate reasoning chains for one run")
    common(g)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--max-tokens", type=int, default=None)
    g.add_argument("--shard", default="0/1", help="i/n: take every n-th problem starting at i")

    b = sub.add_parser("budget", help="evaluate smaller thinking budgets by truncation")
    common(b)
    b.add_argument("--budgets", type=int, nargs="*", default=None)

    s = sub.add_parser("score", help="extract answers and label failures")
    s.add_argument("--runs", nargs="+", required=True, help="run directories or glob patterns")

    j = sub.add_parser("judge", help="judge-model confirmation of scan hits (GPU)")
    j.add_argument("--runs", nargs="+", required=True)
    j.add_argument("--judge-model", default=None)

    w = sub.add_parser("wordlists", help="mine a word list and build the random control")
    w.add_argument("--model", required=True)
    w.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    w.add_argument("--comp-settings", nargs="+", default=["rtn-w3-g128", "rtn-w4-g128"])
    w.add_argument("--k", type=int, default=50)
    w.add_argument("--out-dir", type=Path, default=config.WORDLIST_DIR)

    m = sub.add_parser("manual", help="10%% manual check: export a CSV or score a labelled one")
    m.add_argument("action", choices=["export", "agreement"])
    m.add_argument("--runs", nargs="*", default=["results/*/*/*"])
    m.add_argument("--csv", type=Path, default=Path("reports/manual_check.csv"))

    be = sub.add_parser("bench", help="vLLM throughput benchmark (GPU)")
    be.add_argument("--model", required=True, choices=list(config.models()))
    be.add_argument("--attn", default="auto", help="VLLM_ATTENTION_BACKEND, or auto")
    be.add_argument("--concurrency", type=int, default=64)
    be.add_argument("--gen-tokens", type=int, default=1024)
    be.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    be.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)

    jb = sub.add_parser("jobs", help="expand a priority/smoke file into jobs")
    jb.add_argument("--priority", default="priority.yaml")
    jb.add_argument("--json", action="store_true")

    r = sub.add_parser("report", help="analysis tables and figures")
    r.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    r.add_argument("--out", type=Path, default=Path("reports"))
    r.add_argument("--judged", action="store_true", help="use judge-confirmed outcomes")

    a = ap.parse_args(argv)
    if a.cmd == "gen":
        i, n = (int(x) for x in a.shard.split("/"))
        generate_run(a.model, a.setting, a.fix, a.samples, a.limit, a.max_tokens, (i, n), a.results, a.ckpt_dir,
                     backend_kind=a.backend, problems_file=a.problems)
    elif a.cmd == "bench":
        bench_run(a.model, a.attn, a.concurrency, a.gen_tokens, a.results, a.ckpt_dir)
    elif a.cmd == "budget":
        budget_run(a.model, a.setting, a.fix, a.samples, a.budgets, a.results, a.ckpt_dir, backend_kind=a.backend,
                   problems_file=a.problems)
    elif a.cmd == "score":
        score_runs(_runs(a.runs))
    elif a.cmd == "judge":
        from .extract.judge import JUDGE_MODEL, judge_runs

        probs = {p["id"]: p for p in data.load()}
        judge_runs(_runs(a.runs), probs, a.judge_model or JUDGE_MODEL)
    elif a.cmd == "wordlists":
        from .wordlists.mine import mine, random_control

        mcfg = config.models()[a.model]
        tok = load_tokenizer(mcfg.tokenizer_path)
        base = [run_dir(a.results, a.model, "base", "none")]
        comp = [run_dir(a.results, a.model, s, "none") for s in a.comp_settings]
        hand_words = read_wordlist("handwritten")
        hand = expand_wordlist(hand_words, tok)
        mined = mine(base, comp, tok, k=a.k)
        rand = random_control(list(hand.variants), set(hand.token_ids), base, tok)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        hdr = f"# generated by `crs.run wordlists --model {a.model}` from {', '.join(a.comp_settings)} vs base\n"
        (a.out_dir / f"mined_{a.model}.txt").write_text(hdr + "\n".join(mined) + "\n")
        (a.out_dir / f"random_{a.model}.txt").write_text(hdr + "\n".join(rand) + "\n")
        print(f"mined ({len(mined)}): {mined}\nrandom ({len(rand)}): {rand}")
    elif a.cmd == "manual":
        from .extract import manual_sample

        if a.action == "export":
            probs = {p["id"]: p for p in data.load()}
            n = manual_sample.export(_runs(a.runs), probs, config.settings(), a.csv)
            print(f"wrote {n} rows to {a.csv}; fill in human_label (YES/NO), then run `manual agreement`")
        else:
            print(json.dumps(manual_sample.agreement(a.csv), indent=2))
    elif a.cmd == "jobs":
        jobs, extra = expand_jobs(a.priority)
        if a.json:
            print(json.dumps({"jobs": jobs, **extra}))
        else:
            for jb_ in jobs:
                if jb_["kind"] == "bench":
                    print(f"{jb_['group']}\t{jb_['model']}\tbench {jb_['attn']}\tc={jb_['concurrency']}")
                else:
                    print(f"{jb_['group']}\t{jb_['model']}\t{jb_['setting']}\t{jb_['fix']}\t{jb_['samples']}")
    elif a.cmd == "report":
        from .analysis.report import report

        report(a.results, a.out, judged=a.judged)


if __name__ == "__main__":
    main()
