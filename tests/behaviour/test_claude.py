from decimal import Decimal

import httpx2
import pytest

import anthropic
from check2.behaviour import claude
from check2.behaviour.claude import ClaudeBackend, SettingRejected
from check2.behaviour.spend import CapReached, Price, Spend, load_prices
from tests.behaviour.fake_claude import SYSTEM, FakeClient, item, response

FORBIDDEN = {"temperature", "top_p", "top_k", "fallbacks", "betas", "cache_control", "speed",
             "service_tier"}


def _backend(tmp_path, model="claude-haiku-5-5", client=None, cap="45", chunk_size=4):
    _, prices = load_prices()
    spend = Spend(tmp_path / "spend.json", Decimal(cap)).open()
    return ClaudeBackend(model, client or FakeClient(), spend, prices[model], workers=2,
                         chunk_size=chunk_size, sleep=lambda s: None), spend


def _status_error(cls, code):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("fake", response=httpx2.Response(code, request=request), body=None)


def test_a_chunk_that_would_pass_the_cap_is_refused_and_nothing_is_sent(tmp_path):
    # Haiku: 40 input tokens and 16 output tokens cost 0.000012 at worst; four of them reach 0.00004.
    client = FakeClient()
    backend, spend = _backend(tmp_path, client=client, cap="0.00004")
    with pytest.raises(CapReached):
        list(backend.generate([item(f"q{i}") for i in range(4)]))
    assert client.messages.created == []
    assert spend.spent == 0 and spend.reserved == {}


def test_spend_after_settling_is_the_sum_of_the_replies_costs(tmp_path):
    sizes = {"q0": (40, 2), "q1": (55, 7), "q2": (61, 16)}

    def answer(fields):
        n_in, n_out = sizes[fields["messages"][0]["content"]]
        return response(input_tokens=n_in, output_tokens=n_out)

    backend, spend = _backend(tmp_path, client=FakeClient(answer))
    replies = list(backend.generate([item(q, user=q) for q in sizes]))
    # Haiku: $0.10 and $0.50 per million input and output tokens.
    expected = sum((Decimal(i) * Decimal("0.10") + Decimal(o) * Decimal("0.50")) / 1_000_000
                   for i, o in sizes.values())
    assert spend.spent == expected
    assert spend.reserved == {}
    assert sum(Decimal(r.detail["cost_usd"]) for r in replies) == expected


def test_a_refusal_is_recorded_as_one_and_not_retried(tmp_path):
    client = FakeClient(lambda f: response(text="", stop_reason="refusal", model=f["model"],
                                           category="general_harms"))
    backend, spend = _backend(tmp_path, model="claude-sonnet-5-5", client=client)
    [reply] = list(backend.generate([item("q0")]))
    assert reply.refused and not reply.ended and reply.text == ""
    assert reply.detail["refusal_category"] == "general_harms"
    assert len(client.messages.created) == 1
    assert client.messages.created[0]["model"] == "claude-sonnet-5-5"


@pytest.mark.parametrize("model", sorted(claude.SETTINGS))
def test_requests_hold_the_turns_and_the_model_setting_and_nothing_else(tmp_path, model):
    client = FakeClient()
    backend, _ = _backend(tmp_path, model=model, client=client)
    list(backend.generate([item("q0", max_new_tokens=256)]))
    [sent] = client.messages.created
    [counted] = client.messages.counted
    assert not FORBIDDEN & set(sent) and not FORBIDDEN & set(counted)
    assert sent["system"] == SYSTEM
    assert [m["role"] for m in sent["messages"]] == ["user"]
    expected = {
        "claude-haiku-5-5": ({"thinking": {"type": "disabled"}}, 256),
        "claude-sonnet-5-5": ({"thinking": {"type": "between_tools"}}, 256),
        "claude-opus-5-5": ({"output_config": {"effort": "low"}}, 2048),
    }[model]
    fields, max_tokens = expected
    assert set(sent) == {"model", "max_tokens", "system", "messages", *fields}
    assert {k: sent[k] for k in fields} == fields
    assert sent["max_tokens"] == max_tokens
    assert counted == {k: v for k, v in sent.items() if k != "max_tokens"}


def test_stop_reasons_set_ended_and_text_blocks_only_make_the_reply(tmp_path):
    def answer(fields):
        r = response(text="Final answer: (B)", stop_reason="max_tokens", model=fields["model"])
        r.content.insert(0, type("B", (), {"type": "thinking", "thinking": ""})())
        return r

    backend, _ = _backend(tmp_path, model="claude-opus-5-5", client=FakeClient(answer))
    [reply] = list(backend.generate([item("q0")]))
    assert not reply.ended and not reply.refused
    assert reply.text == "Final answer: (B)"
    assert reply.detail["thinking_blocks"] == 1 and reply.detail["thinking"] == "thinking on, effort low"


def test_a_rate_limited_call_is_released_and_a_server_error_counted_at_its_worst_case(tmp_path):
    errors = [_status_error(anthropic.RateLimitError, 429),
              _status_error(anthropic.InternalServerError, 500)]

    def answer(fields):
        if errors:
            raise errors.pop(0)
        return response(input_tokens=40, output_tokens=2)

    backend, spend = _backend(tmp_path, client=FakeClient(answer))
    [reply] = list(backend.generate([item("q0")]))
    worst = (Decimal(40) * Decimal("0.10") + Decimal(16) * Decimal("0.50")) / 1_000_000
    cost = (Decimal(40) * Decimal("0.10") + Decimal(2) * Decimal("0.50")) / 1_000_000
    assert reply.detail["attempts"] == 3
    assert spend.spent == worst + cost and spend.reserved == {}


def test_a_rejected_setting_stops_the_run_and_bills_nothing(tmp_path):
    def answer(fields):
        raise _status_error(anthropic.BadRequestError, 400)

    backend, spend = _backend(tmp_path, client=FakeClient(answer))
    with pytest.raises(SettingRejected):
        list(backend.generate([item("q0")]))
    assert spend.spent == 0 and spend.reserved == {}


def test_the_cap_check_fails_when_the_worst_case_leaves_out_max_tokens(tmp_path, monkeypatch):
    """The planted fault: priced without max_tokens, the chunk of the first test is sent."""
    monkeypatch.setattr(Price, "worst_case",
                        lambda self, input_tokens, max_tokens: input_tokens * self.input_per_mtok
                        / 1_000_000)
    client = FakeClient()
    backend, _ = _backend(tmp_path, client=client, cap="0.00004")
    list(backend.generate([item(f"q{i}") for i in range(4)]))
    assert len(client.messages.created) == 4
