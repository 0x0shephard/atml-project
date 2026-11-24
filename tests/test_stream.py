"""Loop detection and the streaming vLLM generator (with a fake engine: vLLM does not run on CPU)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from crs.backends.base import Backend, DecodeSpec
from crs.decoding.loops import LoopRule, count_repeats, first_loop_point, is_looping
from crs.extract.score import is_truncated
from crs.record import GenResult

RULE = LoopRule()  # 64-token window written 8 times


def test_periodic_tail_is_a_loop():
    rng = np.random.default_rng(0)
    head = rng.integers(10, 5000, 1500).tolist()
    sentence = rng.integers(10, 5000, 13).tolist()  # "Therefore, the circle is the incircle ..."
    ids = head + sentence * 60
    assert count_repeats(ids, 64, 8192) >= 8
    assert is_looping(ids, RULE)
    p = first_loop_point(ids, RULE)
    assert p is not None and 1500 < p <= len(ids)


def test_ordinary_text_is_not_a_loop():
    rng = np.random.default_rng(1)
    ids = rng.integers(10, 5000, 9000).tolist()
    assert not is_looping(ids, RULE)
    # a phrase repeated a few times, as in normal reasoning, is not enough
    phrase = rng.integers(10, 5000, 80).tolist()
    ids2 = ids[:3000] + phrase + ids[3000:4000] + phrase + ids[4000:5000] + phrase
    assert not is_looping(ids2, RULE)


def test_overlapping_matches_are_not_double_counted():
    ids = [7] * 2000  # one token repeated: 64-token windows overlap everywhere
    assert count_repeats(ids, 64, 8192) == 2000 // 64


def test_short_chains_are_never_stopped():
    ids = [1, 2, 3, 4] * 200  # loops, but only 800 tokens
    assert not is_looping(ids, RULE)


def test_loop_counts_as_truncated_when_unanswered():
    assert is_truncated({"finish_reason": "loop"}, answered=False)
    assert not is_truncated({"finish_reason": "loop"}, answered=True)


class _Engine:
    """Mimics vLLM V1 LLMEngine with output_kind=DELTA: each step returns a few new tokens."""

    def __init__(self, plans: dict[str, list[int]], per_step: int = 50):
        self.plans, self.per_step = plans, per_step
        self.reqs: dict[str, dict] = {}
        self.aborted: list[str] = []

    def add_request(self, rid, prompt, params):  # noqa: ANN001
        assert params.output_kind == "DELTA"
        i = int(rid.split("-")[1])
        self.reqs[rid] = {"toks": self.plans[i], "pos": 0, "max": params.max_tokens}

    def has_unfinished_requests(self) -> bool:
        return bool(self.reqs)

    def abort_request(self, rids):  # noqa: ANN001
        for r in rids:
            self.aborted.append(r)
            self.reqs.pop(r, None)

    def step(self):  # noqa: ANN201
        outs = []
        for rid, st in list(self.reqs.items()):
            new = st["toks"][st["pos"]:st["pos"] + self.per_step][: st["max"] - st["pos"]]
            st["pos"] += len(new)
            ended = st["pos"] >= len(st["toks"]) or st["pos"] >= st["max"]
            reason = None
            if ended:
                reason = "stop" if st["pos"] >= len(st["toks"]) else "length"
                del self.reqs[rid]
            lps = [{t: SimpleNamespace(logprob=-0.5, rank=1), t + 1: SimpleNamespace(logprob=-1.5, rank=2)} for t in new]
            outs.append(SimpleNamespace(request_id=rid, finished=ended,
                                        outputs=[SimpleNamespace(token_ids=new, logprobs=lps, finish_reason=reason)]))
        return outs


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids  # noqa: E731
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = SimpleNamespace(DELTA="DELTA")
    for name, mod in {"vllm": vllm, "vllm.inputs": inputs, "vllm.sampling_params": sp}.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _backend(engine):  # noqa: ANN001, ANN202
    from crs.backends.vllm_backend import VLLMBackend

    be = VLLMBackend.__new__(VLLMBackend)
    be.llm = SimpleNamespace(llm_engine=engine)
    be._params = lambda spec, seed: SimpleNamespace(max_tokens=spec.max_tokens, output_kind=None, seed=seed)  # noqa: E731
    return be


def test_stream_hands_back_chains_as_they_finish_and_stops_loops(fake_vllm):
    rng = np.random.default_rng(2)
    short = rng.integers(10, 5000, 120).tolist()
    long = rng.integers(10, 5000, 3000).tolist()
    loop = rng.integers(10, 5000, 1200).tolist() + rng.integers(10, 5000, 13).tolist() * 400
    eng = _Engine({0: long, 1: short, 2: loop, 3: long})
    be = _backend(eng)
    got: list[tuple[int, GenResult]] = []
    spec = DecodeSpec(max_tokens=4000, logprobs_k=2, seeds=[1, 2, 3, 4])
    be.generate_stream([[0]] * 4, spec, lambda i, o: got.append((i, o[0])),
                       stop_check=lambda ids: is_looping(ids, RULE), check_every=256)
    order = [i for i, _ in got]
    assert order[0] == 1  # the short chain is handed back first, without waiting for the others
    res = dict(got)
    assert sorted(res) == [0, 1, 2, 3]
    assert res[1].token_ids == short and res[1].finish_reason == "stop"
    assert res[0].token_ids == long and res[3].token_ids == long
    assert res[2].finish_reason == "loop" and len(res[2].token_ids) < 4000
    assert res[2].token_ids == loop[: len(res[2].token_ids)]
    assert eng.aborted == ["crs1-2"]
    assert len(res[0].logprobs) == len(long) and np.allclose(res[0].logprobs, -0.5)
    assert res[0].entropy is not None and len(res[0].entropy) == len(long)


def test_stream_reports_the_length_limit(fake_vllm):
    eng = _Engine({0: list(range(10, 700))})
    got = {}
    _backend(eng).generate_stream([[0]], DecodeSpec(max_tokens=300, logprobs_k=0),
                                   lambda i, o: got.update({i: o[0]}))
    assert got[0].finish_reason == "length" and len(got[0].token_ids) == 300 and got[0].logprobs is None


class _Chunked(Backend):
    def __init__(self):
        self.calls: list[int] = []

    def generate(self, prompt_ids, spec):  # noqa: ANN001, ANN201
        self.calls.append(len(prompt_ids))
        return [[GenResult([s], None, None, "stop")] for s in spec.seeds]


def test_default_stream_works_in_chunks_and_keeps_seeds_aligned():
    be = _Chunked()
    got = {}
    be.generate_stream([[0]] * 5, DecodeSpec(max_tokens=8, seeds=[10, 11, 12, 13, 14]),
                       lambda i, o: got.update({i: o[0].token_ids[0]}), chunk=2)
    assert be.calls == [2, 2, 1]
    assert got == {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}
