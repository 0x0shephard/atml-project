"""Merge one Kaggle session's results folder into the local copy without losing anything.

A session's output holds the results it restored at start plus what it added. With several
sessions running at once, a session's restored copy can be older than the local one (another
session added chains since), so copying files over would throw those chains away. This happened
on 2026-11-24: pulling a Qwen3 session replaced results-32k-ext/.../rtn-w3-g128/records.jsonl
(200 chains) with the 184-chain copy that session had restored.

Rules, per run folder:
  *.jsonl         union by "rid" (or the whole line when there is none), local lines first
  arrays/*.npz    copied only when missing locally (two sessions never write the same run)
  meta.json       the version listing more sessions
  anything else   copied only when missing locally

  python kaggle/merge_results.py <session results dir> <local results dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def _key(line: str) -> str:
    try:
        return json.loads(line).get("rid") or line
    except (ValueError, AttributeError):
        return line


def merge_jsonl(src: Path, dst: Path) -> int:
    old = dst.read_text().splitlines() if dst.exists() else []
    seen = {_key(x) for x in old if x.strip()}
    new = [x for x in src.read_text().splitlines() if x.strip() and _key(x) not in seen]
    if new:
        with open(dst, "a") as f:
            if old and not dst.read_text().endswith("\n"):
                f.write("\n")
            f.write("\n".join(new) + "\n")
    return len(new)


def merge_meta(src: Path, dst: Path) -> None:
    n = lambda p: len(json.loads(p.read_text()).get("sessions", []))  # noqa: E731
    if not dst.exists() or n(src) > n(dst):
        shutil.copy2(src, dst)


def merge(src_root: Path, dst_root: Path) -> dict[str, int]:
    added: dict[str, int] = {}
    for f in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = f.relative_to(src_root)
        d = dst_root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".jsonl":
            k = merge_jsonl(f, d)
            if k:
                added[str(rel)] = k
        elif f.name == "meta.json":
            merge_meta(f, d)
        elif not d.exists():
            shutil.copy2(f, d)
    return added


if __name__ == "__main__":
    for path, k in merge(Path(sys.argv[1]), Path(sys.argv[2])).items():
        print(f"  +{k:4d} lines  {path}")
