"""Build data/calib/pile_val_pool.jsonl: the general-text calibration pool for AWQ.

Lotfi et al. (arXiv 2606.00206, A.1) calibrate AWQ on 128 sequences of 512 tokens from
Pile-uncopyrighted, i.e. AutoAWQ's default recipe. AutoAWQ shuffles the Pile validation split
(seed 42), keeps the first 128 documents that fit in 512 tokens, joins them and cuts the result
into 512-token blocks. Downloading the whole split (~450 MB) on every machine is wasteful, so
this script streams the first STREAM records of monology/pile-uncopyrighted's val.jsonl.zst,
keeps documents short enough to possibly fit in 512 tokens, shuffles them with seed 42 and saves
POOL of them. crs.compress.apply.calibration_batches(source="pile") then applies AutoAWQ's
selection with the model's own tokenizer. Run once; the output is committed.

    python scripts/make_pile_calib.py      # needs `zstandard` and network access
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

STREAM, MAX_CHARS, POOL, SEED = 20000, 3000, 600, 42
OUT = Path(__file__).resolve().parents[1] / "data" / "calib" / "pile_val_pool.jsonl"

ds = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train", streaming=True)
docs = []
for i, r in enumerate(ds):
    if i >= STREAM:
        break
    t = r["text"].strip()
    if t and len(t) <= MAX_CHARS:
        docs.append({"text": t, "set": r.get("meta", {}).get("pile_set_name")})
random.Random(SEED).shuffle(docs)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for d in docs[:POOL]:
        f.write(json.dumps(d) + "\n")
print(f"{len(docs)} short docs in the first {STREAM}; wrote {min(POOL, len(docs))} to {OUT}")
