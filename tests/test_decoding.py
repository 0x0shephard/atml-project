import torch

from crs.decoding.budget import FORCE_TEXT, budget_record, forced_prompt, needs_forcing
from crs.decoding.penalties import EntropyPenalty, expand_wordlist, logit_bias, read_wordlist


def test_expand_wordlist(tok):
    exp = expand_wordlist(["Wait", "But", "Zqxwvjk"], tok)
    assert "Zqxwvjk" in exp.skipped
    assert tok.encode(" Wait", add_special_tokens=False)[0] in exp.token_ids
    assert tok.encode("Wait", add_special_tokens=False)[0] in exp.token_ids
    for ids in exp.variants.values():
        for t in ids:
            assert len(tok.decode([t]).strip()) > 1
    assert set(logit_bias(exp.token_ids, 3.0).values()) == {-3.0}


def test_handwritten_list_mostly_single_tokens(tok):
    words = read_wordlist("handwritten")
    exp = expand_wordlist(words, tok)
    assert len(words) >= 45
    assert len(exp.skipped) < len(words) / 2


def test_entropy_penalty_gating():
    pen = EntropyPenalty(token_ids=[1, 2], alpha_max=6.0, h0=1.5, tau=0.5, think_end_ids=[9])
    flat = torch.zeros(10)  # uniform: entropy ln(10) = 2.3 > h0
    out = pen([5], flat.clone())
    assert out[1] < -4.0 and out[0] == 0
    peaked = torch.full((10,), -20.0)
    peaked[3] = 20.0  # entropy ~0: only the sigmoid floor, 6 * sigmoid(-3) = 0.28
    assert pen([5], peaked.clone())[1] > -20.5
    pen([5, 9], flat.clone())  # think end emitted
    assert torch.equal(pen([5, 9, 4], flat.clone()), flat)


def test_entropy_penalty_batch():
    pen = EntropyPenalty(token_ids=[1], alpha_max=6.0, h0=1.5, tau=0.5)
    logits = torch.zeros(2, 10)
    out = pen.apply_batch(logits.clone(), torch.tensor([True, False]))
    assert out[0, 1] < -4.0 and out[1, 1] == 0


def test_budget_forcing(tok):
    rec = {"think_closed": True, "n_think_tokens": 100, "n_tokens": 120, "fix": "none",
           "hesitation_positions": [3, 50, 90], "think_text": "x", "answer_text": "y"}
    assert needs_forcing(rec, 64) and not needs_forcing(rec, 128)
    out_ids = list(range(1000, 1120))
    p = forced_prompt([1, 2, 3], out_ids, 64, tok)
    assert p[:3] == [1, 2, 3] and p[3:67] == out_ids[:64]
    assert p[67:] == tok.encode(FORCE_TEXT, add_special_tokens=False)
    r = budget_record(rec, 64, out_ids, tok, "7}$.")
    assert r["n_think_tokens"] == 64 and r["hesitation_positions"] == [3, 50] and r["forced"]
    assert r["answer_text"].endswith(r"\boxed{7}$.") and r["fix"] == "none@64"
    assert budget_record(rec, 128, out_ids, tok, None)["forced"] is False
