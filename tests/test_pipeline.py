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
