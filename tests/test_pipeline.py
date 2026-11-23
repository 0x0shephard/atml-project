"""End-to-end on CPU: tiny random model via the HF backend, 3 problems."""

import pytest
import numpy as np
import torch

from crs.analysis.report import report
from crs.backends.base import DecodeSpec
from crs.backends.hf_backend import HFBackend
from crs.compress.apply import compress_model
from crs.record import iter_records, load_arrays
from crs.run import budget_run, expand_jobs, generate_run, score_runs

from .conftest import make_tiny


def test_stats_use_raw_logits(tiny, tok):
    """Entropy/logprob stats must not depend on temperature (recorded before warpers)."""
    be = HFBackend(model=tiny, tokenizer=tok, dtype="float32", batch_size=2, device="cpu")
    p = [tok.encode("Compute 2+2.")]
    a = be.generate(p, DecodeSpec(max_tokens=1, temperature=0.3, top_p=1.0, top_k=-1, logprobs_k=20, seeds=[1]))[0][0]
    b = be.generate(p, DecodeSpec(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, logprobs_k=20, seeds=[1]))[0][0]
    assert np.allclose(a.entropy[0], b.entropy[0], atol=1e-5)
    logits = tiny(torch.tensor(p)).logits[0, -1]
    exp_lp = torch.log_softmax(logits.float(), -1)[a.token_ids[0]].item()
    assert abs(a.logprobs[0] - exp_lp) < 1e-4


def test_end_to_end(tmp_path, tok, problems_file):
    results = tmp_path / "results"
    be = HFBackend(model=make_tiny(tok), tokenizer=tok, dtype="float32", batch_size=2, device="cpu")
    rd = generate_run("tiny", "base", "none", limit=3, max_tokens=24, results=results, backend=be,
                      problems_file=problems_file)
    recs = list(iter_records(rd))
    assert len(recs) == 3
    arrs = load_arrays(rd, recs)
    for r in recs:
        a = arrs[r["rid"]]
        assert len(a["ids"]) == r["n_tokens"] == len(a["lp"]) == len(a["ent"])
        assert np.isfinite(a["lp"]).all()
    # resume: nothing left to do
    generate_run("tiny", "base", "none", limit=3, max_tokens=24, results=results, backend=be, problems_file=problems_file)
    assert len(list(iter_records(rd))) == 3
    # resuming with another length limit would mix chains generated under different rules
    with pytest.raises(ValueError, match="max_tokens"):
        generate_run("tiny", "base", "none", limit=4, max_tokens=48, results=results, backend=be,
                     problems_file=problems_file)

    # penalised run and a compressed version (compressed in memory)
    generate_run("tiny", "base", "pen-hand", limit=3, max_tokens=24, results=results, backend=be,
                 problems_file=problems_file)
    comp = make_tiny(tok)
    compress_model(comp, {"method": "rtn", "bits": 3, "group_size": 64})
    be_c = HFBackend(model=comp, tokenizer=tok, dtype="float32", batch_size=2, device="cpu")
    generate_run("tiny", "rtn-w3-g128", "none", limit=3, max_tokens=24, results=results, backend=be_c,
                 problems_file=problems_file)
    be_k = HFBackend(model=make_tiny(tok), tokenizer=tok, dtype="float32", batch_size=2, device="cpu",
                     kv={"bits": 4, "group_size": 64})
    generate_run("tiny", "kv-int4", "none", limit=3, max_tokens=24, results=results, backend=be_k,
                 problems_file=problems_file)

    budget_run("tiny", "base", "none", budgets=[8, 16], results=results, backend=be, problems_file=problems_file)
    assert (rd / "budget_8.jsonl").exists()

    score_runs(sorted(p for p in results.glob("*/*/*")))
    assert (rd / "scored.jsonl").exists() and (rd / "scored_budget_8.jsonl").exists()
    row = next(iter_records(rd, "scored.jsonl"))
    assert row["outcome"] in ("correct", "commit_fail", "never_correct", "truncated")

    tables = report(results, tmp_path / "reports")
    assert not tables["vs_baseline"].empty
    assert set(tables["vs_baseline"]["setting"]) >= {"rtn-w3-g128"}
    assert not tables["fix_effects"].empty
    assert (tmp_path / "reports" / "summary.md").exists()


def test_bench_jobs_expand():
    from crs import config

    jobs, extra = expand_jobs(config.CONFIG_DIR / "bench.yaml")
    bench = [j for j in jobs if j["kind"] == "bench"]
    assert len(bench) == 12 and {j["attn"] for j in bench} == {"FLEX_ATTENTION", "TRITON_ATTN", "XFORMERS"}
    assert all(j["timeout_min"] and j["gen_tokens"] for j in bench)
    assert {j["setting"] for j in jobs if j["kind"] == "gen"} == {"kv-int4", "kv-int8"}
    assert extra == {"limit": 10, "max_tokens": 1024}


def test_hf_batch_matches_single_and_drops_finished(tok):
    """Batched greedy decoding (left padding, rows dropped when they finish) gives the same
    tokens as decoding each prompt alone."""
    model = make_tiny(tok)
    be = HFBackend(model=model, tokenizer=tok, dtype="float32", batch_size=3, device="cpu")
    prompts = [[5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15]]
    spec = DecodeSpec(max_tokens=12, greedy=True, logprobs_k=5)
    # make the first prompt's 3rd generated token an EOS so that row stops early
    first = be.generate(prompts[:1], DecodeSpec(max_tokens=12, greedy=True, logprobs_k=0))[0][0].token_ids
    be.eos_ids = [first[2]]
    batched = [r[0] for r in be.generate(prompts, spec)]
    single = [be.generate([p], spec)[0][0] for p in prompts]
    assert batched[0].finish_reason == "stop" and len(batched[0].token_ids) <= 3
    for b, s in zip(batched, single):
        assert b.token_ids == s.token_ids and b.finish_reason == s.finish_reason
        assert np.allclose(b.logprobs, s.logprobs, atol=1e-4) and np.allclose(b.entropy, s.entropy, atol=1e-4)


def test_group_limit(tmp_path):
    f = tmp_path / "jobs.yaml"
    f.write_text("groups:\n  - {name: a, models: [r1-1.5b], settings: [base], fix: none, limit: 7}\n"
                 "  - {name: b, models: [r1-1.5b], settings: [kv-int4], fix: none}\n")
    jobs, extra = expand_jobs(f)
    assert jobs[0]["limit"] == 7 and "limit" not in jobs[1] and extra == {}


def test_cutoff_jobs_expand(tmp_path):
    from crs import config

    jobs, extra = expand_jobs(config.CONFIG_DIR / "cutoff.yaml")
    gen = [j for j in jobs if j["kind"] == "gen"]
    budget = [j for j in jobs if j["kind"] == "budget"]
    assert [j["setting"] for j in gen] == ["rtn-w3-g128", "base"]
    assert all(j["max_tokens"] == 32768 and j["results"] == "results-32k" for j in gen)
    assert all((config.CONFIG_DIR.parent / "data" / j["problems"]).exists() for j in gen)
    assert len(budget) == 5 and all(j["fix"] == "none" and "results" not in j for j in budget)
    assert extra == {}
    # a longer length limit must not share a results folder with the 8,192-token runs
    f = tmp_path / "bad.yaml"
    f.write_text("groups:\n  - {name: x, models: [r1-1.5b], settings: [base], fix: none, max_tokens: 16384}\n")
    with pytest.raises(ValueError, match="results"):
        expand_jobs(f)


def test_extend_continues_cut_off_chains(tmp_path, tok, problems_file):
    import json

    from crs.run import extend_run

    short, long = tmp_path / "results", tmp_path / "results-long"
    be = HFBackend(model=make_tiny(tok), tokenizer=tok, dtype="float32", batch_size=2, device="cpu")
    src = generate_run("tiny", "base", "none", limit=3, max_tokens=12, results=short, backend=be,
                       problems_file=problems_file)
    recs = list(iter_records(src))
    # mark one chain as having stopped on its own, so it must be copied, not continued
    recs[0]["finish_reason"] = "stop"
    (src / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    cut = [r for r in recs if r["finish_reason"] == "length"]
    assert cut, "the tiny model should run into the 12-token limit"

    rd = extend_run("tiny", "base", 30, short, long, chunk_size=1, backend=be, problems_file=problems_file)
    new = {r["problem_id"]: r for r in iter_records(rd)}
    assert set(new) == {r["problem_id"] for r in recs}
    old_a = load_arrays(src, recs)
    new_a = load_arrays(rd, new.values())
    for r in recs:
        o, n = old_a[r["rid"]], new_a[new[r["problem_id"]]["rid"]]
        assert len(n["ids"]) == len(n["lp"]) == len(n["ent"]) == new[r["problem_id"]]["n_tokens"]
        if r["finish_reason"] == "stop":
            assert np.array_equal(o["ids"], n["ids"]) and "extended_from" not in new[r["problem_id"]]
        else:  # continued: starts with the old chain, ran on past the old limit
            assert np.array_equal(o["ids"], n["ids"][:len(o["ids"])])
            assert np.allclose(o["lp"].astype(np.float32), n["lp"][:len(o["lp"])].astype(np.float32), atol=1e-2)
            assert 12 < len(n["ids"]) <= 30 and new[r["problem_id"]]["extended_from"] == 12
    # resume: nothing left; a different limit in the same folder is refused; never extend in place
    extend_run("tiny", "base", 30, short, long, backend=be, problems_file=problems_file)
    assert len(list(iter_records(rd))) == len(recs)
    with pytest.raises(ValueError, match="max_tokens"):
        extend_run("tiny", "base", 40, short, long, backend=be, problems_file=problems_file)
    with pytest.raises(ValueError, match="different results folder"):
        extend_run("tiny", "base", 30, short, short, backend=be, problems_file=problems_file)
