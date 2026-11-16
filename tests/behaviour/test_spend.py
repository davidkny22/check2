import json
from decimal import Decimal

import pytest

from check2.behaviour.spend import PRICES_FILE, CapReached, CounterHeld, Price, Spend, load_prices

OPUS = Price("claude-opus-5-5", Decimal("4"), Decimal("20"), Decimal("5"), Decimal("0.2"), None)
HAIKU = Price("claude-haiku-5-5", Decimal("0.1"), Decimal("0.5"), Decimal("0.125"), Decimal("0.01"),
              100_000)


def test_worst_case_is_counted_input_and_max_tokens_at_list_price():
    # 1,000 input tokens at $4 per million and 2,048 output tokens at $20 per million.
    assert OPUS.worst_case(1000, 2048) == Decimal("0.004") + Decimal("0.04096")


def test_cost_from_reply_counts():
    assert HAIKU.cost(1000, 10) == Decimal("0.0001") + Decimal("0.000005")


def test_haiku_prices_refuse_a_prompt_past_their_tier():
    with pytest.raises(ValueError):
        HAIKU.worst_case(100_001, 16)


def test_prices_file_names_the_cap_and_each_price_with_its_address_and_date():
    data = json.loads(PRICES_FILE.read_text(encoding="utf-8"))
    cap, prices = load_prices()
    assert cap == Decimal("45")
    assert sorted(prices) == ["claude-haiku-5-5", "claude-opus-5-5", "claude-sonnet-5-5"]
    for p in data["prices"].values():
        assert p["address"].startswith("https://") and p["read"]


def test_a_reservation_that_would_reach_the_cap_is_refused_and_leaves_nothing(tmp_path):
    with Spend(tmp_path / "spend.json", Decimal("1")) as spend:
        spend.reserve({"a": Decimal("0.6")})
        before = (tmp_path / "spend.json").read_text(encoding="utf-8")
        with pytest.raises(CapReached):
            spend.reserve({"b": Decimal("0.3"), "c": Decimal("0.1")})  # 0.6 + 0.4 reaches 1
        assert set(spend.reserved) == {"a"}
        assert (tmp_path / "spend.json").read_text(encoding="utf-8") == before
        spend.reserve({"b": Decimal("0.3")})  # 0.9 stays under 1


def test_spent_after_settling_is_the_sum_of_costs(tmp_path):
    with Spend(tmp_path / "spend.json", Decimal("45")) as spend:
        spend.reserve({"a": Decimal("0.5"), "b": Decimal("0.5")})
        spend.settle("a", Decimal("0.0123"), "m", 100, 3)
        spend.settle("b", Decimal("0.0045"), "m", 50, 1)
        assert spend.spent == Decimal("0.0168")
        assert spend.reserved == {}
    data = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))
    assert Decimal(data["spent_usd"]) == Decimal("0.0168")
    assert data["by_model"]["m"]["calls"] == 2


def test_a_second_process_finds_the_counter_held(tmp_path):
    with Spend(tmp_path / "spend.json", Decimal("45")):
        with pytest.raises(CounterHeld):
            Spend(tmp_path / "spend.json", Decimal("45")).open()
    Spend(tmp_path / "spend.json", Decimal("45")).open().close()  # free again once released


def test_a_save_waits_out_a_counter_held_open_for_a_moment(tmp_path, monkeypatch):
    from check2.behaviour import spend as spend_module

    real_replace, refusals = spend_module.os.replace, [PermissionError("held")] * 3

    def replace(src, dst):
        if refusals:
            raise refusals.pop()
        real_replace(src, dst)

    monkeypatch.setattr(spend_module.os, "replace", replace)
    monkeypatch.setattr(spend_module, "SAVE_WAIT_S", 0)
    with Spend(tmp_path / "spend.json", Decimal("45")) as spend:
        spend.reserve({"a": Decimal("0.5")})
        spend.settle("a", Decimal("0.01"), "m")
    assert Decimal(json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))["spent_usd"]) == Decimal("0.01")


def test_reservations_left_by_a_stopped_process_count_as_spent(tmp_path):
    spend = Spend(tmp_path / "spend.json", Decimal("45")).open()
    spend.reserve({"a": Decimal("0.25")})
    spend.lock_path.unlink()  # the process died holding a reservation
    spend._held = False
    with Spend(tmp_path / "spend.json", Decimal("45")) as again:
        assert again.spent == Decimal("0.25")
        assert again.reserved == {}
