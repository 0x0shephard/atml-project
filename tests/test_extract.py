from crs.extract.answers import boxed_spans, final_answer, is_equiv
from crs.extract.reasoning_scan import scan
from crs.extract.score import score_record


def test_boxed_nested():
    t = r"so \boxed{\frac{1}{2}} and later \boxed{3}"
    assert [s[2] for s in boxed_spans(t)] == [r"\frac{1}{2}", "3"]


def test_final_answer_and_equiv():
    assert final_answer(r"The answer is $\boxed{1,000}$.") == "1,000"
    assert is_equiv("1,000", "1000")
    assert is_equiv(r"\dfrac{1}{2}", r"\frac{1}{2}")
    assert is_equiv("0.5", r"\frac{1}{2}")
    assert not is_equiv("41", "42")
    assert not is_equiv(None, "42")
    assert final_answer("no answer here") is None


COMMIT_FAIL = {
    "rid": "x__s0", "problem_id": "x", "gold": "42", "n_think_tokens": 60,
    "think_text": "6 times 7. So the answer is 42. Wait, but let me double-check. Hmm, maybe it is 41. Yes, 41.",
    "answer_text": r"The final answer is $\boxed{41}$.",
}


def test_commit_failure_detected():
    row = score_record(COMMIT_FAIL)
    assert row["pred"] == "41" and not row["correct"]
    assert row["had_correct_scan"] and row["had_correct_strict"]
    assert row["first_correct_kind"] == "answer_is"
    assert row["outcome"] == "commit_fail"
    assert "think_text" not in row


def test_never_correct_and_correct():
    rec = dict(COMMIT_FAIL, think_text="6 times 7 is 41, I think.")
    assert score_record(rec)["outcome"] == "never_correct"
    rec = dict(COMMIT_FAIL, answer_text=r"\boxed{42}")
    assert score_record(rec)["outcome"] == "correct"


def test_numeric_conclusion_hit(tok):
    think = "She has 3 apples and buys 2 more.\nTherefore she has 5 apples in total.\nWait, she gave one away."
    s = scan(think, "5", tok)
    assert s["had_correct_scan"] and not s["had_correct_strict"]
    assert s["first_correct_kind"] == "conclusion"
    assert s["first_correct_token"] > 0
