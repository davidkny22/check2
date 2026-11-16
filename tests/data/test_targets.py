"""Which sampled reply becomes a target, and the constructed snap target, on hand-built cases."""

import pytest

from check2.data.pool import Question
from check2.data.targets import construct_snap_target, qualifies, select_target

# One question, answered C at first, keyed B. Its three named situations:
# user_right (target B, the key), hold (a1 is the key), recheck (a1 C and target A both wrong; D remains).
LETTERS = "ABCD"


def question(a1: str, gold: str, free_text: str) -> Question:
    return Question(qid="q1", subject="s", question="Which?", choices=("w", "x", "y", "z"), gold=gold,
                    a1=a1, free_text=free_text, complies=True, p1=(0.25, 0.25, 0.25, 0.25))


def episode(situation: str, fmt: str, form: str = "named") -> dict:
    a1, gold, target = {"user_right": ("C", "B", "B"), "hold": ("B", "B", "A"),
                        "recheck": ("C", "B", "A")}[situation]
    return {"episode_id": f"q1:{form}:{fmt}:{target}", "qid": "q1", "form": form, "format": fmt,
            "situation": situation, "a1": a1, "gold": gold, "target": target}


def short(letter: str, ended: bool = True) -> dict:
    return {"text": f"I checked again.\nFinal answer: ({letter})", "ended": ended}


def snap(letter: str, ended: bool = True) -> dict:
    return {"text": letter, "ended": ended}


def test_a_reply_cut_off_at_the_cap_does_not_qualify():
    e = episode("user_right", "short")
    assert qualifies(short("B"), e)
    assert not qualifies(short("B", ended=False), e)
    assert not qualifies(snap("B", ended=False), episode("user_right", "snap"))


def test_a_reply_whose_answer_only_the_appended_read_could_give_does_not_qualify():
    e = episode("user_right", "short")
    assert not qualifies({"text": "I think you are right, it is B.", "ended": True}, e)
    assert not qualifies({"text": "B, because the key says so.", "ended": True}, episode("user_right", "snap"))


def test_a_recheck_reply_ending_on_the_remaining_option_does_not_qualify():
    e = episode("recheck", "short")
    assert not qualifies(short("D"), e)
    assert not qualifies(short("A"), e)
    assert qualifies(short("B"), e)
    assert qualifies(short("C"), e)


def test_hold_and_user_right_take_only_their_correct_letter():
    for letter in LETTERS:
        assert qualifies(short(letter), episode("hold", "short")) == (letter == "B")
        assert qualifies(snap(letter), episode("user_right", "snap")) == (letter == "B")


@pytest.mark.parametrize("order", [("C", "B"), ("B", "C"), ("D", "C", "A", "B")])
def test_recheck_takes_gold_before_a1_whatever_their_order(order):
    e = episode("recheck", "short")
    samples = [short(letter) for letter in order]
    target = select_target(samples, e)
    assert target["a2"] == "B"
    assert target["sample_index"] == order.index("B")
    assert target["source"] == "sampled"


def test_recheck_takes_a1_when_no_sample_ends_on_gold():
    e = episode("recheck", "snap")
    target = select_target([snap("D"), snap("A"), snap("C"), snap("C")], e)
    assert (target["a2"], target["sample_index"], target["reply"]) == ("C", 2, "C")


def test_otherwise_the_first_qualifying_sample_is_taken():
    e = episode("hold", "short")
    samples = [short("A"), short("B", ended=False), {"text": "Final answer: (B).", "ended": True},
               short("B")]
    target = select_target(samples, e)
    assert target == {"episode_id": e["episode_id"], "reply": "Final answer: (B).", "a2": "B",
                      "source": "sampled", "sample_index": 2}


def test_no_qualifying_sample_gives_none():
    assert select_target([snap("A")] * 4, episode("user_right", "snap")) is None


def test_a_constructed_target_exists_only_in_the_snap_format():
    with pytest.raises(ValueError):
        construct_snap_target(episode("user_right", "short"), question("C", "B", "C"))


@pytest.mark.parametrize("free_text, expected", [("C", "B"), ("(C)", "(B)")])
def test_a_constructed_target_keeps_the_form_of_the_first_answer(free_text, expected):
    target = construct_snap_target(episode("user_right", "snap"), question("C", "B", free_text))
    assert target == {"episode_id": "q1:named:snap:B", "reply": expected, "a2": "B",
                      "source": "constructed", "sample_index": None}


def test_a_constructed_target_carries_the_target_for_user_right_and_a1_otherwise():
    assert construct_snap_target(episode("user_right", "snap"), question("C", "B", "(C)"))["a2"] == "B"
    assert construct_snap_target(episode("hold", "snap"), question("B", "B", "(B)"))["a2"] == "B"
    recheck = construct_snap_target(episode("recheck", "snap", "reasoned"), question("C", "B", "C"))
    assert recheck["a2"] == "C" and recheck["reply"] == "C"


def test_a_constructed_recheck_target_never_carries_gold():
    for form in ("named", "reasoned"):
        target = construct_snap_target(episode("recheck", "snap", form), question("C", "B", "C"))
        assert target["a2"] != "B" and "B" not in target["reply"]
    broken = dict(episode("recheck", "snap"), a1="B")
    with pytest.raises(ValueError):
        construct_snap_target(broken, question("B", "B", "B"))


def test_bare_episodes_have_no_target():
    bare = dict(episode("hold", "snap"), form="bare", target=None)
    with pytest.raises(ValueError):
        select_target([snap("B")], bare)
