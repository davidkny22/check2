"""The runtime check against the pinned versions, and its switch for a development card."""

from __future__ import annotations

import pytest
import torch
import transformers
import triton

from check2.mechanism.loading import DEVELOPMENT_CARD_VARIABLE, RuntimeMismatch, check_runtime


def _pins_of_this_process(**changes) -> dict:
    pins = {"transformers": transformers.__version__, "torch": torch.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    pins.update(changes)
    return pins


def test_another_card_is_refused_without_the_switch(monkeypatch):
    monkeypatch.delenv(DEVELOPMENT_CARD_VARIABLE, raising=False)
    with pytest.raises(RuntimeMismatch, match="gpu"):
        check_runtime(_pins_of_this_process(gpu="a card this machine does not have"))
    check_runtime(_pins_of_this_process())


def test_another_card_is_allowed_with_the_switch_and_says_so(monkeypatch, capsys):
    monkeypatch.setenv(DEVELOPMENT_CARD_VARIABLE, "1")
    check_runtime(_pins_of_this_process(gpu="a card this machine does not have"))
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 1 and "development card" in printed[0]


@pytest.mark.parametrize("switch", [None, "1"])
def test_a_version_difference_is_refused_either_way(monkeypatch, switch):
    if switch is None:
        monkeypatch.delenv(DEVELOPMENT_CARD_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(DEVELOPMENT_CARD_VARIABLE, switch)
    with pytest.raises(RuntimeMismatch, match="transformers"):
        check_runtime(_pins_of_this_process(transformers="0.0.0", gpu="a card this machine does not have"))
