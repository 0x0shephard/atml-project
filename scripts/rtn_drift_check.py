"""How far does RTN weight rounding move each model from its full-size self?

Run from the repo root: python scripts/rtn_drift_check.py [qwen3-1.7b] [r1-1.5b]
On each model's own base reasoning chains (first N_TOK tokens of N_SEQ chains, fp32 on CPU), it reports
KL(base || rounded) per token, top-1 agreement, NLL, the weight error, and the largest |activation|
going into and out of every Linear (fp16 limit 65504). Needs ~10 GB RAM per model.

2026-11-25: Qwen3 W4 KL 0.133 vs R1 W4 0.027 at the same weight error (~0.105): Qwen3 amplifies
rounding ~5x (activation outliers ~17k vs ~1.5k). No code bug; no fp16 overflow.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from crs.compress.apply import compress_model, target_linears  # noqa: E402

torch.set_num_threads(10)
N_SEQ, N_TOK = 4, 768
MODELS = {"qwen3-1.7b": "Qwen/Qwen3-1.7B", "r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
SPECS = {"w4g128": {"method": "rtn", "bits": 4, "group_size": 128},
         "w3g128": {"method": "rtn", "bits": 3, "group_size": 128}}
only = sys.argv[1:] or list(MODELS)


def texts(key):
    out = []
    for line in open(f"results/{key}/base/none/records.jsonl"):
        r = json.loads(line)
        if r["n_think_tokens"] > N_TOK + 50:
            out.append(r["think_text"])
        if len(out) == N_SEQ:
            break
    return out


def load(hf):
    return AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()


def act_hooks(model):
    stats = defaultdict(lambda: [0.0, 0.0])
    hs = []
    for n, m in target_linears(model).items():
        def h(mod, inp, out, n=n):
            s = stats[n]
            s[0] = max(s[0], float(inp[0].abs().max()))
            s[1] = max(s[1], float(out.abs().max()))
        hs.append(m.register_forward_hook(h))
    return stats, hs


@torch.no_grad()
def run(model, ids_list):
    stats, hs = act_hooks(model)
    lps, resid = [], 0.0
    for ids in ids_list:
        o = model(ids[None], output_hidden_states=True)
        lps.append(F.log_softmax(o.logits[0, :-1].float(), -1).half())
        resid = max(resid, max(float(h.abs().max()) for h in o.hidden_states))
        del o
    for h in hs:
        h.remove()
    return lps, stats, resid


def summarize_acts(stats):
    by = defaultdict(lambda: [0.0, 0.0])
    for n, (i, o) in stats.items():
        t = n.split(".")[-1]
        by[t][0] = max(by[t][0], i)
        by[t][1] = max(by[t][1], o)
    return {k: (round(v[0]), round(v[1])) for k, v in by.items()}


for key in only:
    hf = MODELS[key]
    tok = AutoTokenizer.from_pretrained(hf)
    ids_list = [tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0, :N_TOK] for t in texts(key)]
    t0 = time.time()
    model = load(hf)
    base_lp, base_stats, base_resid = run(model, ids_list)
    base_nll = sum(float(-lp.float().gather(-1, ids[1:, None]).mean()) for lp, ids in zip(base_lp, ids_list)) / N_SEQ
    print(f"\n=== {key}  (base fwd {time.time()-t0:.0f}s)", flush=True)
    print(f"base: nll {base_nll:.3f}  max|resid| {base_resid:.0f}  max|lin in,out| {summarize_acts(base_stats)}", flush=True)
    del model
    for sname, spec in SPECS.items():
        model = load(hf)
        info = compress_model(model, spec)
        lp_q, st_q, resid_q = run(model, ids_list)
        kl, agree, nll = 0.0, 0.0, 0.0
        for bl, ql, ids in zip(base_lp, lp_q, ids_list):
            bl, ql = bl.float(), ql.float()
            kl += float((bl.exp() * (bl - ql)).sum(-1).mean())
            agree += float((bl.argmax(-1) == ql.argmax(-1)).float().mean())
            nll += float(-ql.gather(-1, ids[1:, None]).mean())
        n = N_SEQ
        print(f"{sname}: KL {kl/n:.4f}  top1-agree {agree/n:.3f}  nll {nll/n:.3f} (base {base_nll:.3f})  "
              f"max|resid| {resid_q:.0f}", flush=True)
        print(f"   weight rel err mean {info['mean_rel_frob_error']:.4f} max {info['max_rel_frob_error']:.4f}")
        print(f"   max|lin in,out| {summarize_acts(st_q)}", flush=True)
        del model, lp_q
