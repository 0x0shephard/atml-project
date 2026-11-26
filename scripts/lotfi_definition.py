"""Had-it share the way Lotfi et al. count it: MATH-500 only, denominator = all wrong answers."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "src")

def norm(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s.strip("$.")

def recs(path):
    return {json.loads(l)["rid"]: json.loads(l) for l in open(path)}

rows = []
for root in ["results", "results-32k-ext"]:
    for scored in sorted(Path(root).glob("r1-1.5b/*/none/scored.jsonl")):
        setting = scored.parts[-3]
        if setting.startswith("svd") or setting in ("asvd-r80", "asvd-r85", "asvd-r90", "base-hf"):
            continue
        sc = recs(scored)
        texts = {}
        for l in open(scored.parent / "records.jsonl"):
            r = json.loads(l); texts[r["rid"]] = r.get("think_text", "")
        m = [r for r in sc.values() if r["source"] == "math500"]
        wrong = [r for r in m if not r["correct"]]
        cut = sum(r["outcome"] == "truncated" for r in wrong)
        loose = sum(bool(r.get("had_correct_scan")) for r in wrong)
        strict = sum(bool(r.get("had_correct_strict")) for r in wrong)
        anym = 0
        for r in wrong:
            g = norm(r["gold"])
            t = norm(texts.get(r["rid"], ""))
            if g and g in t:
                anym += 1
        # had-it among wrong answers that DID finish (not cut off)
        fin = [r for r in wrong if r["outcome"] != "truncated"]
        loose_fin = sum(bool(r.get("had_correct_scan")) for r in fin)
        n = len(m); w = len(wrong)
        rows.append((root.replace("results-32k-ext", "32k").replace("results", "8k"), setting, n, 100 * (n - w) / n, w, cut,
                     100 * strict / w, 100 * loose / w, 100 * anym / w, loose, f"{loose_fin}/{len(fin)}"))
print("| limit | setting | MATH-500 n | right % | errors | of which cut off | had-it strict % | had-it loose (ours) % | had-it any mention % | loose count | loose among finished errors |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| %s | %s | %d | %.1f | %d | %d | %.0f | %.0f | %.0f | %d | %s |" % r)
