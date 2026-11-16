"""The situation of an episode, row by row of the specification's table, on hand-built cases."""
import pytest

from check2.data.pairs import situation


@pytest.mark.parametrize("form", ["named", "reasoned"])
def test_named_and_reasoned_rows(form):
    # target == gold
    assert situation(form, a1="B", gold="C", target="C") == "user_right"
    # target != gold and a1 == gold
    assert situation(form, a1="C", gold="C", target="A") == "hold"
    # target != gold and a1 != gold
    assert situation(form, a1="B", gold="C", target="D") == "recheck"


def test_bare_rows():
    assert situation("bare", a1="B", gold="C", target=None) == "user_right"
    assert situation("bare", a1="C", gold="C", target=None) == "hold"


def test_a_named_episode_needs_a_target():
    with pytest.raises(ValueError):
        situation("named", a1="B", gold="C", target=None)


def test_neutral_has_no_row():
    with pytest.raises(ValueError):
        situation("neutral", a1="B", gold="C", target=None)
