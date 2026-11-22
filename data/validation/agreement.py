"""Score the detector rules against the two labellers (2026-11-22)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
items = {it["item"]: it for it in map(json.loads, open(D / "sample.jsonl"))}
lab = {}
for name in "AB":
    for x in json.load(open(D / f"labels_{name}.json")):
        lab.setdefault((x["item"], x["span"]), {})[name] = x


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = Counter(a for a, _ in pairs); pb = Counter(b for _, b in pairs)
    pe = sum(pa[k] * pb[k] for k in set(pa) | set(pb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


both = [(v["A"]["label"], v["B"]["label"]) for v in lab.values() if "A" in v and "B" in v]
po, k = kappa(both)
print(f"labeller agreement on {len(both)} shared spans: {po:.0%} raw, Cohen kappa {k:.2f}")
dis = [(i, s) for (i, s), v in lab.items() if "A" in v and "B" in v and v["A"]["label"] != v["B"]["label"]]
print("disagreements:", dis)

# reference = A or B; on disagreement use adjudicated.json if present, else NO (conservative)
adj = json.load(open(D / "adjudicated.json")) if (D / "adjudicated.json").exists() else {}
ref = {}
for (i, s), v in lab.items():
    labels = {x["label"] for x in v.values()}
    ref[(i, s)] = labels.pop() if len(labels) == 1 else adj.get(f"{i}:{s}", "NO")
missing = [(it, j) for it in items for j in range(len(items[it]["spans"])) if (it, j) not in ref]
print("unlabelled spans:", missing)

# span-level precision of each candidate kind
by_kind = defaultdict(Counter)
for (i, s), y in ref.items():
    by_kind[items[i]["spans"][s]["kind"]][y] += 1
print("\nspan level: share of flagged spans that are real statements of the answer")
for kd, c in by_kind.items():
    print(f"  {kd:15} {c['YES']}/{c['YES'] + c['NO']} = {c['YES'] / (c['YES'] + c['NO']):.0%}")

# chain level
print("\nchain level (hit set: the loose rule says YES for all of these)")
tab = defaultdict(Counter)
for i, it in items.items():
    if it["set"] != "hit":
        continue
    real = any(ref.get((i, s)) == "YES" for s in range(len(it["spans"])))
    first_real = ref.get((i, 0)) == "YES"
    tab["all"][(it["strict"], real)] += 1
    tab[it["setting"]][(it["strict"], real)] += 1
    tab["first_hit"][first_real] += 1
for k2, c in tab.items():
    if k2 == "first_hit":
        continue
    n = sum(c.values()); real = c[(True, True)] + c[(False, True)]
    sp = c[(True, True)] / max(1, c[(True, True)] + c[(True, False)])
    sr = c[(True, True)] / max(1, real)
    print(f"  {k2:12} n={n:3} loose precision={real / n:.0%}  strict precision={sp:.0%} strict recall={sr:.0%}")
c = tab["first_hit"]; print(f"  first flagged span is real: {c[True]}/{c[True] + c[False]}")

print("\nrecall set: failed chains with no flag, where the gold value appears in the text")
rc = Counter()
for i, it in items.items():
    if it["set"] == "recall":
        rc[any(ref.get((i, s)) == "YES" for s in range(len(it["spans"])))] += 1
print(f"  actually stated as the answer: {rc[True]}/{rc[True] + rc[False]}")

print("\nlow-confidence labels:", sum(x["confidence"] == "low" for v in lab.values() for x in v.values()))
print("\nNO notes on 'conclusion' spans (sample):")
for (i, s), y in list(ref.items()):
    if y == "NO" and items[i]["spans"][s]["kind"] == "conclusion":
        v = lab[(i, s)]; x = v.get("A") or v.get("B")
        print(f"  {i}:{s} {x['note']}")
