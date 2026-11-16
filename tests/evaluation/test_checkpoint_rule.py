"""The checkpoint rule on candidates and runs built by hand, with the expected choice worked out on paper."""

import json

import pytest

from check2.evaluation.checkpoint_rule import candidates_from_runs, choose, main, qualifies, rank
from check2.training import checkpoints
from tests.evaluation import faults
from tests.evaluation.hand_records import adopting, build, settings


def candidate(label, order, wrong_adoption, right_adoption_drop):
    return {"label": label, "order": order, "wrong_adoption": wrong_adoption, "right_adoption_drop": right_adoption_drop}


def test_two_candidates_qualify_and_the_lower_wrong_adoption_wins():
    # Epoch 1 and 2 lose at most 10 points; epoch 3 has the lowest wrong adoption but loses 12.
    candidates = [
        candidate("epoch 1", 1, 0.30, 5.0),
        candidate("epoch 2", 2, 0.20, 8.0),
        candidate("epoch 3", 3, 0.10, 12.0),
    ]
    assert choose(candidates)["label"] == "epoch 2"


def test_none_qualifies_and_the_smallest_drop_wins():
    candidates = [
        candidate("epoch 1", 1, 0.05, 15.0),
        candidate("epoch 2", 2, 0.30, 12.0),
        candidate("epoch 3", 3, 0.01, 20.0),
    ]
    assert choose(candidates)["label"] == "epoch 2"


def test_a_tie_goes_to_the_earlier_epoch():
    candidates = [candidate("epoch 2", 2, 0.20, 3.0), candidate("epoch 1", 1, 0.20, 7.0)]
    assert choose(candidates)["label"] == "epoch 1"
    # A tie among candidates that do not qualify, on the drop.
    candidates = [candidate("epoch 3", 3, 0.10, 14.0), candidate("epoch 2", 2, 0.40, 14.0)]
    assert choose(candidates)["label"] == "epoch 2"


def test_a_tie_goes_to_the_smaller_strength():
    candidates = [candidate("g=8", 8, 0.25, 2.0), candidate("g=2", 2, 0.25, 9.0), candidate("g=4", 4, 0.25, 1.0)]
    assert [c["label"] for c in rank(candidates)] == ["g=2", "g=4", "g=8"]


def test_the_ranking_of_a_mixed_set():
    candidates = [
        candidate("g=1", 1, 0.40, 2.0),    # qualifies, wrong 0.40
        candidate("g=2", 2, 0.30, 11.0),   # does not qualify, drop 11
        candidate("g=4", 4, 0.10, 10.0),   # qualifies at exactly 10, wrong 0.10
        candidate("g=8", 8, 0.05, 25.0),   # does not qualify, drop 25
        candidate("e=3", 3, 0.35, -4.0),   # a gain qualifies, wrong 0.35
    ]
    # Qualifying by wrong adoption: g=4 (0.10), e=3 (0.35), g=1 (0.40); then by drop: g=2 (11), g=8 (25).
    assert [c["label"] for c in rank(candidates)] == ["g=4", "e=3", "g=1", "g=2", "g=8"]


def test_the_limit_is_ten_points_inclusive_and_a_gain_qualifies():
    assert qualifies(candidate("a", 1, 0.1, 10.0))
    assert not qualifies(candidate("a", 1, 0.1, 10.000001))
    assert qualifies(candidate("a", 1, 0.1, -30.0))
    # Ten points, not a proportion of 0.10.
    assert qualifies(candidate("a", 1, 0.1, 0.5))


def test_an_empty_set_is_refused():
    with pytest.raises(ValueError):
        choose([])


def heldout_run(wrong_adopted, right_adopted):
    """Two questions in each format: per question one wrong-target and ten right-target reasoned episodes."""
    records = []
    for fmt in ("snap", "short"):
        records += build(
            {
                f"q{i}": {"wrong": adopting(wrong_adopted[fmt][i], 1), "right": adopting(right_adopted[fmt][i], 10)}
                for i in range(2)
            },
            fmt=fmt,
        )
    return records


def test_candidates_from_runs_pool_both_formats_on_reasoned_episodes():
    original = heldout_run({"snap": [1, 1], "short": [1, 0]}, {"snap": [10, 10], "short": [8, 6]})
    epoch_1 = heldout_run({"snap": [0, 1], "short": [0, 0]}, {"snap": [9, 10], "short": [7, 6]})
    # A named episode in the candidate only: the rule reads reasoned episodes, so it is not paired.
    epoch_1 += build({"q0": {"wrong": [True]}}, form="named")
    epoch_2 = heldout_run({"snap": [0, 0], "short": [0, 0]}, {"snap": [5, 6], "short": [4, 3]})
    candidates = candidates_from_runs(
        [{"label": "epoch 1", "order": 1, "records": epoch_1}, {"label": "epoch 2", "order": 2, "records": epoch_2}],
        original,
    )
    # Original right adoption: (10 + 10 + 8 + 6) / 40 = 34/40 = 85 points.
    # Epoch 1: wrong 1/4; right (9 + 10 + 7 + 6) / 40 = 32/40 = 80 points; drop 5.
    # Epoch 2: wrong 0/4; right (5 + 6 + 4 + 3) / 40 = 18/40 = 45 points; drop 40.
    assert candidates[0] == {"label": "epoch 1", "order": 1, "wrong_adoption": 0.25, "right_adoption": 0.8,
                             "right_adoption_drop": pytest.approx(5.0)}
    assert candidates[1] == {"label": "epoch 2", "order": 2, "wrong_adoption": 0.0, "right_adoption": 0.45,
                             "right_adoption_drop": pytest.approx(40.0)}
    assert choose(candidates)["label"] == "epoch 1"


def test_candidates_from_runs_refuse_a_missing_episode_and_test_episodes():
    original = heldout_run({"snap": [1, 1], "short": [1, 0]}, {"snap": [10, 10], "short": [8, 6]})
    epoch = heldout_run({"snap": [0, 1], "short": [0, 0]}, {"snap": [9, 10], "short": [7, 6]})
    with pytest.raises(ValueError):
        candidates_from_runs([{"label": "e", "order": 1, "records": epoch[1:]}], original)
    on_test = [{**r, "part": "first"} for r in epoch]
    with pytest.raises(ValueError):
        candidates_from_runs([{"label": "e", "order": 1, "records": on_test}], original)


# ---------------------------------------------------------------------------
# The command over heldout run folders
# ---------------------------------------------------------------------------

ORIGINAL_ADOPTIONS = ({"snap": [1, 1], "short": [1, 0]}, {"snap": [10, 10], "short": [8, 6]})
# Worked out on paper against the Original run's right adoption of 34 of 40, 85 points.
# Wrong adoption 1 of 4, right 32 of 40: drop 5, qualifies.
GOOD_WITH_ONE_WRONG = ({"snap": [0, 1], "short": [0, 0]}, {"snap": [9, 10], "short": [7, 6]})
# Wrong adoption 0 of 4, right 18 of 40: drop 40, does not qualify.
LOW_RIGHT = ({"snap": [0, 0], "short": [0, 0]}, {"snap": [5, 6], "short": [4, 3]})
# Wrong adoption 0 of 4, right 34 of 40: drop 0, qualifies.
BEST = ({"snap": [0, 0], "short": [0, 0]}, {"snap": [10, 10], "short": [8, 6]})
# Wrong adoption 2 of 4, right 34 of 40: drop 0, qualifies.
HALF_WRONG = ({"snap": [1, 0], "short": [1, 0]}, {"snap": [10, 10], "short": [8, 6]})


def write_heldout(results, run_id, adoptions, config, split="heldout"):
    from tests.evaluation.test_tables import write_run

    write_run(results, run_id, config, heldout_run(*adoptions), split=split)
    return results / run_id


@pytest.fixture()
def results(tmp_path, monkeypatch):
    from check2 import record

    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))
    folder = tmp_path / "results"
    write_heldout(folder, "original-heldout", ORIGINAL_ADOPTIONS, settings())
    return folder


def recorded(results):
    (path,) = list((results / "measurements" / "checkpoint_rule").glob("*.json"))
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def chosen_root(tmp_path, monkeypatch):
    """The folder the chosen-epoch files are written to and read from, the test's own."""
    monkeypatch.setattr(checkpoints, "CHOSEN_ROOT", tmp_path / "trainings")
    return tmp_path / "trainings"


# The consumer training of `settings(alpha=1.0, kind="consumer")`, named as the checkpoints are:
# `consumer-<signal>-rank<r>-beta<b>-seed<s>`.
CONSUMER_TRAINING = "consumer-reader-rank32-beta1-seed0"
CONNECTOR_TRAINING = "connector-reader-seed0"


def _keeps_the_first_epoch_and_writes_the_chosen_file_of_its_training(results, chosen_root) -> None:
    check2 = settings(alpha=1.0, kind="consumer")
    runs = [write_heldout(results, "check2-heldout-epoch1", GOOD_WITH_ONE_WRONG, check2),
            write_heldout(results, "check2-heldout-epoch2", LOW_RIGHT, check2)]
    assert main(["--original", str(results / "original-heldout"), "--runs", *map(str, runs), "--kind", "epochs",
                 "--write-chosen"]) == 0
    path, rec = recorded(results)
    values = rec["values"]
    assert values["training_id"] == CONSUMER_TRAINING
    assert values["ranking"] == ["check2-heldout-epoch1", "check2-heldout-epoch2"]
    assert values["kept"] == {"label": "check2-heldout-epoch1", "epoch": 1}
    assert values["recorded_check"] == {"check": "the kept checkpoint is the last epoch's", "holds": False}
    first, second = values["candidates"]
    assert (first["wrong_adoption"], first["right_adoption"], first["qualifies"]) == (0.25, 0.8, True)
    assert first["right_adoption_drop"] == pytest.approx(5.0)
    assert (second["wrong_adoption"], second["right_adoption"], second["qualifies"]) == (0.0, 0.45, False)
    assert second["right_adoption_drop"] == pytest.approx(40.0)
    written = sorted(p.relative_to(chosen_root).as_posix() for p in chosen_root.rglob("*") if p.is_file())
    assert written == [f"{CONSUMER_TRAINING}/chosen.json"]
    chosen = chosen_root / CONSUMER_TRAINING / "chosen.json"
    assert json.loads(chosen.read_text(encoding="utf-8")) == {"training_id": CONSUMER_TRAINING, "epoch": 1,
                                                              "record": path.name}
    assert checkpoints.chosen_epoch(CONSUMER_TRAINING) == 1


def test_the_command_keeps_the_first_epoch_of_the_ranking_and_writes_the_chosen_file(results, chosen_root):
    _keeps_the_first_epoch_and_writes_the_chosen_file_of_its_training(results, chosen_root)


# The rule writing the chosen file under the folder of another training: the connector the ranked training
# continues from.
CHOSEN_UNDER_THE_CONNECTOR = "chosen-file-under-the-connector-s-folder"
CHOSEN_UNDER_THE_CONNECTOR_FAULT = ("check2.evaluation.checkpoint_rule", "_training_name",
                                    "return checkpoints.training_id(config)",
                                    "return checkpoints.connector_id(config)")


def test_a_chosen_file_written_under_another_training_s_folder_turns_the_check_red(results, chosen_root,
                                                                                  monkeypatch):
    monkeypatch.setitem(faults.FAULTS, CHOSEN_UNDER_THE_CONNECTOR, CHOSEN_UNDER_THE_CONNECTOR_FAULT)
    faults.fails_under(CHOSEN_UNDER_THE_CONNECTOR,
                       lambda: _keeps_the_first_epoch_and_writes_the_chosen_file_of_its_training(results, chosen_root))


def test_runs_of_two_trainings_are_refused_and_no_chosen_file_is_written(results, chosen_root):
    runs = [write_heldout(results, "check2-heldout-epoch1", GOOD_WITH_ONE_WRONG, settings(alpha=1.0, kind="consumer")),
            write_heldout(results, "connector-heldout-epoch2", BEST, settings(alpha=1.0, kind="none"))]
    with pytest.raises(ValueError, match=f"more than one training.*{CONNECTOR_TRAINING}.*{CONSUMER_TRAINING}"):
        main(["--original", str(results / "original-heldout"), "--runs", *map(str, runs), "--kind", "epochs",
              "--write-chosen"])
    assert not chosen_root.exists()


def test_the_recorded_check_holds_when_the_last_epoch_is_kept(results):
    connector = settings(alpha=1.0, kind="none")
    runs = [write_heldout(results, f"connector-heldout-epoch{k}", adoptions, connector)
            for k, adoptions in ((1, GOOD_WITH_ONE_WRONG), (2, LOW_RIGHT), (3, BEST))]
    assert main(["--original", str(results / "original-heldout"), "--runs", *map(str, runs), "--kind", "epochs"]) == 0
    values = recorded(results)[1]["values"]
    assert values["kept"]["epoch"] == 3 and values["recorded_check"]["holds"] is True


@pytest.mark.parametrize("g4, g8, kept, holds", [
    (LOW_RIGHT, HALF_WRONG, [1, 2, 8], False),
    (HALF_WRONG, LOW_RIGHT, [1, 2, 4], True),
])
def test_the_command_keeps_three_steering_strengths(results, g4, g8, kept, holds):
    # Qualifying by wrong adoption: g=2 (0), g=1 (1 of 4), then the half-wrong strength (2 of 4); the strength
    # with a drop of 40 comes last.
    strengths = {1: GOOD_WITH_ONE_WRONG, 2: BEST, 4: g4, 8: g8}
    runs = [write_heldout(results, f"steering-g{g}-heldout", adoptions, settings(steering_strength=g))
            for g, adoptions in strengths.items()]
    assert main(["--original", str(results / "original-heldout"), "--runs", *map(str, runs), "--kind",
                 "steering"]) == 0
    values = recorded(results)[1]["values"]
    assert values["kept"]["strengths"] == kept
    assert values["recorded_check"] == {"check": "all three kept steering strengths sit at one end of the grid",
                                        "holds": holds}


def test_the_command_refuses_runs_of_another_split_or_of_mixed_kinds(results):
    original = str(results / "original-heldout")
    check2 = settings(alpha=1.0, kind="consumer")
    epoch = write_heldout(results, "check2-heldout-epoch1", GOOD_WITH_ONE_WRONG, check2)
    steering = write_heldout(results, "steering-g1-heldout", BEST, settings(steering_strength=1))
    on_test = write_heldout(results, "check2-epoch2", BEST, check2, split="test")
    other = write_heldout(results, "connector-heldout-epoch2", BEST, settings(alpha=1.0, kind="none"))
    for runs, kind, refused in (([epoch, steering], "epochs", "not one epoch's checkpoint"),
                                ([steering, epoch], "steering", "not a steering run"),
                                ([epoch, on_test], "epochs", "applied to heldout runs"),
                                ([epoch, other], "epochs", "more than one training")):
        with pytest.raises(ValueError, match=refused):
            main(["--original", original, "--runs", *map(str, runs), "--kind", kind])
    with pytest.raises(ValueError, match="not an Original run"):
        main(["--original", str(epoch), "--runs", str(epoch), "--kind", "epochs"])
    with pytest.raises(SystemExit):
        main(["--original", original, "--runs", str(steering), "--kind", "steering", "--write-chosen"])
    assert not (results / "measurements").exists()
