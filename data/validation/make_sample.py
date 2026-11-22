"""Draw the detector-validation sample (2026-11-22).

Hit set: error chains (outcome != correct) where the scan found the gold answer; 10 per setting,
up to 3 correct hits each, shown with context. Recall set: error chains with no scan hit in
which the gold value still appears somewhere in the reasoning text.
Writes sample.jsonl (all items) and labeller_A.md / labeller_B.md (A and B share 16 items).
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SETTINGS = ["base", "rtn-w4-g128", "rtn-w3-g128", "wanda-30", "wanda-50", "kv-int4", "asvd-r95", "asvd-r90"]
BEFORE, AFTER = 500, 250
rng = random.Random(0)
problems = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/problems.jsonl")}


def win(t, a, b):
    return t[max(0, a - BEFORE):a].replace("\n", " "), t[a:b], t[b:b + AFTER].replace("\n", " ")


items = []
for s in SETTINGS:
    run = ROOT / "results/r1-1.5b" / s / "none"
    texts = {json.loads(l)["rid"]: json.loads(l)["think_text"] for l in open(run / "records.jsonl")}
    rows = [json.loads(l) for l in open(run / "scored.jsonl")]
    err = [r for r in rows if r["outcome"] != "correct"]
    hit = [r for r in err if r["had_correct_scan"]]
    for r in rng.sample(hit, min(10, len(hit))):
        t = texts[r["rid"]]
        spans = [dict(kind=h["kind"], text=h["text"], ctx=win(t, h["char"], h["char"] + len(h["text"])))
                 for h in r["correct_hits"][:3]]
        items.append(dict(set="hit", setting=s, rid=r["rid"], outcome=r["outcome"], gold=r["gold"],
                          strict=r["had_correct_strict"], pred=r.get("pred"), spans=spans))
    nohit = []
    for r in err:
        if r["had_correct_scan"]:
            continue
        g = re.escape(r["gold"].strip())
        occ = [m for m in re.finditer(rf"(?<![\d.]){g}(?![\d])", texts[r["rid"]])]
        if occ:
            nohit.append((r, occ))
    for r, occ in rng.sample(nohit, min(5, len(nohit))):
        t = texts[r["rid"]]
        spans = [dict(kind="raw_occurrence", text=m.group(), ctx=win(t, m.start(), m.end())) for m in occ[-3:]]
        items.append(dict(set="recall", setting=s, rid=r["rid"], outcome=r["outcome"], gold=r["gold"],
                          strict=False, pred=r.get("pred"), spans=spans))

rng.shuffle(items)
for i, it in enumerate(items):
    it["item"] = i
    it["problem"] = problems[it["rid"].split("__")[0]]["problem"]
(OUT / "sample.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))

# A gets the first half plus 16 shared; B the second half plus the same 16 shared
half = len(items) // 2
shared = items[:16]
groups = {"A": items[:half], "B": items[half:] + shared}


def render(it):
    out = [f"## Item {it['item']}", f"**Problem:** {it['problem'].strip()}", f"**Reference answer:** {it['gold']}", ""]
    for j, sp in enumerate(it["spans"]):
        b, h, a = sp["ctx"]
        out.append(f"**Span {j}:** …{b}<<<{h}>>>{a}…")
        out.append("")
    return "\n".join(out)


for name, grp in groups.items():
    (OUT / f"labeller_{name}.md").write_text("\n\n---\n\n".join(render(it) for it in grp))
print(len(items), {k: len(v) for k, v in groups.items()},
      sum(it["set"] == "hit" for it in items), sum(it["set"] == "recall" for it in items),
      sum(len(it["spans"]) for it in items))
