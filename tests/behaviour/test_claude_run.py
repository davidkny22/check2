import json
import random

import pytest

from check2 import record
from check2.behaviour import claude_list, claude_run
from check2.behaviour import items as item_lists
from check2.behaviour.backend import Item
from check2.behaviour.spend import CounterHeld
from tests.behaviour.fake_claude import SYSTEM, FakeClient, response

KEY = "sk-ant-fake-0123456789-key-that-must-never-be-written"
QIDS = [f"subject/{n}" for n in range(12)]


def _first(qid):
    return Item(item_id=f"{qid}:first", qid=qid, source="pool", kind="first", fmt=None, target=None,
                writer=None, messages=({"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": f"question {qid}"}), max_new_tokens=24)


def _follow_ups(first_replies):
    out = []
    for qid in QIDS:
        reply = first_replies.get(qid)
        if reply is None or reply.strip() not in "ABCD" or len(reply.strip()) != 1:
            continue
        for fmt, cap in (("snap", 16), ("short", 256)):
            out.append(Item(item_id=f"{qid}:reasoned:{fmt}:C", qid=qid, source="pool", kind="reasoned",
                            fmt=fmt, target="C", writer="opus",
                            messages=({"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": f"question {qid}"},
                                      {"role": "assistant", "content": reply},
                                      {"role": "user", "content": "I think (C)."}), max_new_tokens=cap))
    return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(item_lists, "first_answer_items", lambda: [_first(q) for q in QIDS])
    monkeypatch.setattr(item_lists, "follow_up_items", _follow_ups)
    monkeypatch.setattr(record, "ROOT", tmp_path)
    monkeypatch.setattr(record, "source_state",
                        lambda: {"commit": "0" * 40, "tree_clean": True, "commit_source": "test"})
    monkeypatch.setattr(record, "gpu_name", lambda: None)
    key_file = tmp_path / "keys" / ".env"
    key_file.parent.mkdir()
    key_file.write_text(f"OTHER=1\n{claude_run.KEY_NAME}={KEY}\n", encoding="utf-8")
    clients = []

    def factory(key):
        assert key == KEY
        clients.append(FakeClient(lambda f: response(text="B", model=f["model"])))
        return clients[-1]

    return tmp_path, key_file, factory, clients


def _files(root):
    return [p for p in root.rglob("*") if p.is_file() and "keys" not in p.parts]


def test_a_run_writes_every_reply_and_its_record_and_never_the_key(world):
    root, key_file, factory, clients = world
    out = root / "results" / "behaviour" / "claude-haiku-5-5"
    code = claude_run.main(["--model", "claude-haiku-5-5", "--out", str(out), "--key-file", str(key_file)],
                           client_factory=factory)
    assert code == 0
    rows = [json.loads(line) for line in (out / "replies.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(QIDS) * 3
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete" and run["episodes_written"] == len(QIDS) * 3
    [summary] = [p for p in out.glob("*.json") if p.name != "run.json"]
    values = json.loads(summary.read_text(encoding="utf-8"))["values"]
    assert values["thinking"] == "thinking disabled" and float(values["spend_this_model_usd"]) > 0
    assert not (root / "results" / "behaviour" / "claude-spend.json.lock").exists()
    for path in _files(root):
        assert KEY not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_a_stopped_run_resumes_without_asking_again(world):
    root, key_file, factory, clients = world
    out = root / "results" / "behaviour" / "claude-haiku-5-5"
    args = ["--model", "claude-haiku-5-5", "--out", str(out), "--key-file", str(key_file)]
    assert claude_run.main(args, client_factory=factory) == 0
    assert claude_run.main(args, client_factory=factory) == 0
    assert clients[1].messages.created == []


def test_a_second_process_on_the_counter_refuses_to_start(world):
    root, key_file, factory, _ = world
    lock = root / "results" / "behaviour" / "claude-spend.json.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n", encoding="utf-8")
    out = root / "results" / "behaviour" / "claude-haiku-5-5"
    with pytest.raises(CounterHeld):
        claude_run.main(["--model", "claude-haiku-5-5", "--out", str(out), "--key-file", str(key_file)],
                        client_factory=factory)
    assert not (out / "replies.jsonl").exists()


def test_a_slow_rate_cuts_the_questions_not_yet_asked_and_says_so(world, monkeypatch):
    root, key_file, factory, clients = world
    monkeypatch.setattr(claude_run, "TIMED_CALLS", 4)
    ticks = iter(range(0, 10_000, 10))  # each reading of the clock is 10 seconds later
    out = root / "results" / "behaviour" / "claude-sonnet-5-5"
    client = factory(KEY)
    from check2.behaviour.claude import ClaudeBackend
    from check2.behaviour.spend import Spend, load_prices
    cap, prices = load_prices()
    with Spend(root / "spend.json", cap) as spend:
        backend = ClaudeBackend("claude-sonnet-5-5", client, spend, prices["claude-sonnet-5-5"], workers=1,
                                chunk_size=4, sleep=lambda s: None)
        run = claude_run.Run("claude-sonnet-5-5", out, backend, clock=lambda: next(ticks), deadline=60)
        run.execute()
    # Four calls in 10 seconds; at clock 20, 40 seconds remain: 16 calls. The four questions asked
    # keep their 8 follow-ups, and 8 calls are left: two more questions of three calls each.
    assert run.timing["cut"] and run.timing["questions_kept"] == 4 + 2
    asked_first = {r["qid"] for r in run.done.values() if r["kind"] == "first"}
    assert len(asked_first) == 6 and set(QIDS[:4]) <= asked_first
    assert set(run.expected_ids()) == set(run.done)


def test_opus_gets_a_seeded_third_of_the_sorted_questions(world):
    expected = sorted(random.Random(2052).sample(sorted(QIDS), 4))
    assert claude_list.asked_qids("claude-opus-5-5") == expected
    assert claude_list.asked_qids("claude-haiku-5-5") == sorted(QIDS)


def test_preflight_counts_tokens_and_sends_nothing_billed(world, capsys):
    root, key_file, factory, clients = world
    out = root / "results" / "behaviour" / "claude-preflight"
    code = claude_run.main(["--model", "claude-haiku-5-5", "--model", "claude-opus-5-5", "--out", str(out),
                            "--key-file", str(key_file), "--preflight"], client_factory=factory)
    assert code == 0
    assert clients[0].messages.created == [] and clients[0].messages.counted
    text = capsys.readouterr().out
    assert "no billed call" in text and "claude-opus-5-5" in text and "Total:" in text
    data = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    # Haiku asks every question: 12 first answers, and on the placeholder 'A' two follow-ups each.
    assert data["estimates"][0]["calls"] == 12 * 3
    for path in _files(root):
        assert KEY not in path.read_text(encoding="utf-8", errors="ignore"), path
