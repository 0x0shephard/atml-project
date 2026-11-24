"""kaggle/merge_results.py: pulling a session must never drop chains another session added."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("merge_results", Path(__file__).parents[1] / "kaggle" / "merge_results.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)


def _w(p: Path, rids: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps({"rid": r}) + "\n" for r in rids))


def test_older_session_copy_does_not_drop_local_chains(tmp_path):
    src, dst = tmp_path / "session", tmp_path / "local"
    _w(dst / "m/s/none/records.jsonl", ["a", "b", "c"])  # local already has c from another session
    (dst / "m/s/none/arrays").mkdir(parents=True)
    (dst / "m/s/none/arrays/part-0001.npz").write_text("local")
    _w(src / "m/s/none/records.jsonl", ["a", "b", "d"])  # this session restored a, b and added d
    (src / "m/s/none/arrays").mkdir(parents=True)
    (src / "m/s/none/arrays/part-0001.npz").write_text("session")
    (src / "m/s/none/arrays/part-0002.npz").write_text("new")
    added = mr.merge(src, dst)
    rids = [json.loads(x)["rid"] for x in (dst / "m/s/none/records.jsonl").read_text().splitlines()]
    assert rids == ["a", "b", "c", "d"]
    assert added == {"m/s/none/records.jsonl": 1}
    assert (dst / "m/s/none/arrays/part-0001.npz").read_text() == "local"  # never overwritten
    assert (dst / "m/s/none/arrays/part-0002.npz").read_text() == "new"


def test_meta_keeps_the_longer_session_list(tmp_path):
    src, dst = tmp_path / "s", tmp_path / "d"
    src.mkdir(); dst.mkdir()
    (dst / "meta.json").write_text(json.dumps({"sessions": [1, 2]}))
    (src / "meta.json").write_text(json.dumps({"sessions": [1]}))
    mr.merge(src, dst)
    assert json.loads((dst / "meta.json").read_text())["sessions"] == [1, 2]
