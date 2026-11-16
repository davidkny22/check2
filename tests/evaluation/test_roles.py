"""A run's role and episode set, read from its settings."""

import pytest

from check2.evaluation.roles import base_role, run_role
from tests.evaluation.hand_records import settings


def test_run_roles_from_settings():
    assert run_role(settings(alpha=0.0, kind="none")) == ("original", "")
    assert run_role(settings(alpha=1.0, kind="none")) == ("connector", "")
    assert run_role(settings(alpha=1.0, kind="consumer")) == ("check2", "")
    assert run_role(settings(alpha=0.0, kind="consumer")) == ("off", "")
    assert run_role(settings(alpha=1.0, kind="consumer", signal="swapped")) == ("swap", "")
    assert run_role(settings(alpha=1.0, kind="consumer", signal="perfect")) == ("perfect signal", "")
    assert run_role(settings(alpha=0.0, kind="adapter_only")) == ("adapter only", "")
    assert run_role(settings(alpha=0.0, kind="none", steering_strength=4)) == ("steering g=4", "")
    assert run_role(settings(alpha=1.0, kind="consumer", forms=("neutral",))) == ("check2", "neutral follow-up")
    with pytest.raises(ValueError):
        run_role({"check2": {"adapter": {"kind": "none"}}})


def test_a_setting_that_is_not_the_main_runs_value_is_named_in_the_role():
    assert run_role(settings(alpha=1.0, kind="none", signal="perfect")) == ("connector signal=perfect", "")
    assert run_role(settings(alpha=1.0, kind="none", seed=1)) == ("connector seed=1", "")
    assert run_role(settings(alpha=1.0, kind="consumer", seed=1)) == ("check2 seed=1", "")
    assert run_role(settings(alpha=1.0, kind="consumer", beta=4.0)) == ("check2 beta=4.0", "")
    assert run_role(settings(alpha=0.0, kind="consumer", beta=4.0)) == ("off beta=4.0", "")
    assert run_role(settings(alpha=1.0, kind="consumer", signal="perfect", rank=64)) == ("perfect signal rank=64", "")
    # The base role is the same in each, so a gate can take the run either way.
    assert base_role(settings(alpha=0.0, kind="consumer", beta=4.0)) == "off"
    assert base_role(settings(alpha=1.0, kind="consumer", signal="perfect", rank=64)) == "perfect signal"


def test_a_setting_left_out_reads_as_the_main_runs_value():
    partial = {"check2": {"writer": {"alpha": 1.0}, "adapter": {"kind": "consumer"}}}
    assert run_role(partial) == ("check2", "")
