"""Every run has one config file, complete, loadable, and recognised by the result tables."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from check2.config.schema import SYSTEM_PROMPT_SUFFIX, load_config, parse_config
from check2.evaluation.runs import run_id_for, settings_of
from check2.evaluation.runs.parts import check_executable
from check2.evaluation import swap
from check2.evaluation.tables import _one, _original_for, discover, run_role
from check2.gates.off import require_runs
from tests.evaluation.hand_records import ALL as EVERY_QUESTION
from tests.evaluation.hand_records import reasoned_run
from tests.evaluation.test_tables import write_run

CONFIGS = Path(__file__).resolve().parents[3] / "experiments" / "configs"
ALL = sorted(CONFIGS.glob("*/*.yaml"))

# The run table over test, then the items dropped only with agreement, with the role each must read as.
TEST_RUNS = {
    "original": ("original", ""), "connector": ("connector", ""), "check2": ("check2", ""), "off": ("off", ""),
    "swap": ("swap", ""), "adapter-only": ("adapter only", ""), "perfect-signal": ("perfect signal", ""),
    "system-prompt": ("system prompt", ""), "steering-g1": ("steering g=1", ""), "steering-g2": ("steering g=2", ""),
    "steering-g4": ("steering g=4", ""), "steering-g8": ("steering g=8", ""),
    "check2-alpha0.5": ("dial alpha=0.5", ""), "check2-alpha2": ("dial alpha=2.0", ""),
    "check2-bands-3-6-9-off": ("bands off [3, 6, 9]", ""), "check2-bands-11-15-off": ("bands off [11, 15]", ""),
    "check2-bands-18-21-23-off": ("bands off [18, 21, 23]", ""),
    "formed-model": ("original", "signal forming formed_model_only"),
    "formed-check2": ("check2", "signal forming formed_with_channel"),
    "check2-seed1": ("check2 seed=1", ""),
    "original-neutral": ("original", "neutral follow-up"), "check2-neutral": ("check2", "neutral follow-up"),
    "original-crosswriter": ("original", "cross-writer"), "check2-crosswriter": ("check2", "cross-writer"),
}
HELDOUT_RUNS = ["original", "original-repeat", "connector", "check2", "off", "adapter-only", "perfect-signal",
                "perfect-signal-connector", "steering-g1", "steering-g2", "steering-g4", "steering-g8",
                "formed-model", "formed-connector", "formed-check2", "formed-off", "check2-seed1", "connector-seed1"]


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_run_has_a_config():
    names = {(p.parent.name, p.stem) for p in ALL}
    assert {("test", n) for n in TEST_RUNS} <= names
    assert {("heldout", n) for n in HELDOUT_RUNS} <= names
    assert ("train", "original-train") in names


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_config_is_complete_loads_and_is_recognised(path):
    raw = _raw(path)
    assert settings_of(parse_config(raw)) == raw, "a key is left to its default"
    config = load_config(path)
    assert config.data.eval_split == path.parent.name
    assert config.adapter.implementation == "hooks"
    role, variant = run_role(raw)
    if path.parent.name == "test" and path.stem in TEST_RUNS:
        assert (role, variant) == TEST_RUNS[path.stem]
    check_executable(config, None)
    assert run_id_for(path, config, None) == path.stem + ("-heldout" if path.parent.name == "heldout" else "")


def test_system_prompt_carries_the_suffix_and_steering_its_strength():
    assert load_config(CONFIGS / "test" / "system-prompt.yaml").baseline.system_prompt_suffix == SYSTEM_PROMPT_SUFFIX
    assert load_config(CONFIGS / "heldout" / "steering-g4.yaml").baseline.steering_strength == 4


def test_no_two_config_files_of_one_split_give_the_same_role_and_variant():
    seen: dict = {}
    for path in ALL:
        seen.setdefault((path.parent.name, run_role(_raw(path))), []).append(path.stem)
    shared = {key: sorted(stems) for key, stems in seen.items() if len(stems) > 1}
    # Only a run and its repeat share a role, and the tables tell them apart by the repeat's name.
    assert shared == {(split, ("original", "")): ["original", "original-repeat"] for split in ("test", "heldout")}


def _folder_of_every_config(results: Path) -> dict[str, dict]:
    """One complete run folder per config file, named as the runner names it, each holding the same made-up records."""
    raws = {}
    for path in ALL:
        raw = _raw(path)
        run_id = run_id_for(path, parse_config(raw), None)
        part = "first" if path.parent.name == "test" else None
        write_run(results, run_id, raw, reasoned_run(EVERY_QUESTION, [20] * 10, part=part), split=path.parent.name)
        raws[run_id] = raw
    return raws


def test_every_lookup_finds_the_main_run_in_a_folder_of_every_config(tmp_path):
    raws = _folder_of_every_config(tmp_path)
    runs, left_out = discover(tmp_path)
    assert left_out == ["check2-neutral", "original-neutral", "original-repeat", "original-repeat-heldout"]
    for role, run_id in (("check2", "check2"), ("off", "off"), ("connector", "connector"),
                         ("perfect signal", "perfect-signal"), ("original", "original"), ("swap", "swap"),
                         ("adapter only", "adapter-only")):
        assert _one(runs, role)["id"] == run_id, role
    for run in runs:
        if run["role"] == "original" or run["split"] == "train" or run["variant"].startswith("signal forming"):
            continue
        expected = {"test": "original", "heldout": "original-heldout"}[run["split"]]
        if run["variant"] == "cross-writer":
            expected = "original-crosswriter"
        assert _original_for(run, runs)["id"] == expected, run["id"]
    assert swap.check2_run("test", tmp_path).name == "check2"
    for gate_role, run_id in (("off", "off-heldout"), ("off", "off-beta4-heldout"),
                              ("perfect signal", "perfect-signal-heldout"),
                              ("perfect signal", "perfect-signal-rank64-heldout")):
        require_runs({"run_id": run_id, "config": raws[run_id]},
                     {"run_id": "original-heldout", "config": raws["original-heldout"]}, gate_role)
    with pytest.raises(ValueError):
        require_runs({"run_id": "c", "config": raws["perfect-signal-connector-heldout"]},
                     {"run_id": "original-heldout", "config": raws["original-heldout"]}, "perfect signal")
