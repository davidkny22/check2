"""The record every gate, measurement and run writes."""
import json
import subprocess

import pytest

from check2 import record


@pytest.fixture
def bare_root(tmp_path, monkeypatch):
    """A root with no git repository above it, as on a session."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(record, "ROOT", tmp_path)
    return tmp_path


def test_a_record_with_no_git_carries_the_sent_commit(bare_root, monkeypatch):
    (bare_root / ".sent-commit").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
    monkeypatch.setenv("CHECK2_SESSION", "session-x")
    path = record.write_record("gate", "some_gate", {"value": 1}, verdict="green")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert data["commit_source"] == ".sent-commit"
    assert data["session"] == "session-x"
    assert path.parent == bare_root / "results" / "gates" / "some_gate"
    assert path.name.endswith("-0123456789ab.json")


def test_no_git_and_no_sent_commit_is_refused(bare_root):
    with pytest.raises(RuntimeError):
        record.write_record("measurement", "m", {})


def test_a_record_in_git_carries_head_and_the_tree_state(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "ROOT", tmp_path)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run(git + ["add", "a.txt"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    head = subprocess.run(git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    config = {"check2": {"seed": 0}}
    path = record.write_record("measurement", "timing", {"ms": 3.0}, config=config)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["commit"] == head and data["tree_clean"] is True
    second = json.loads(record.write_record("measurement", "timing", {}).read_text(encoding="utf-8"))
    assert second["tree_clean"] is True  # the first record, untracked under results/, does not count
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    untracked = json.loads(record.write_record("measurement", "timing", {}).read_text(encoding="utf-8"))
    assert untracked["tree_clean"] is False
    (tmp_path / "b.txt").unlink()
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    later = json.loads(record.write_record("measurement", "timing", {}).read_text(encoding="utf-8"))
    assert later["tree_clean"] is False
    assert path.parent == tmp_path / "results" / "measurements" / "timing"
    assert data["config"] == config and len(data["config_sha256"]) == 64
    for key in ("gpu", "versions", "started", "finished", "values", "verdict"):
        assert key in data
    assert set(data["versions"]) == {"python", "torch", "transformers", "flash-linear-attention"}


def test_unknown_kind_is_refused(bare_root):
    with pytest.raises(ValueError):
        record.write_record("note", "x", {})


def test_a_run_record_and_its_completion(bare_root):
    (bare_root / ".sent-commit").write_text("abc\n", encoding="utf-8")
    config = {"check2": {"data": {"eval_split": "heldout"}, "seed": 1}}
    path = record.start_run("original-heldout", config)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path == bare_root / "results" / "original-heldout" / "run.json"
    assert (data["status"], data["split"], data["seed"], data["commit"]) == ("running", "heldout", 1, "abc")
    with pytest.raises(ValueError):
        record.finish_run(path, "complete", episodes_expected=10, episodes_written=9, order_sha256="h")
    record.finish_run(path, "complete", episodes_expected=10, episodes_written=10, order_sha256="h")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["status"], data["episodes_written"], data["order_sha256"]) == ("complete", 10, "h")
    assert data["finished"] is not None
