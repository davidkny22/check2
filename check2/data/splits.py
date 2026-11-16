"""The splits by question id, the `heldout` draw and the reader's five folds."""
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from check2.data import paths
from check2.data.pairs import Pair
from check2.data.pool import Question, complying, duplicate_groups, normalise

HELDOUT_SEED = 2027
HELDOUT_SHARE = 0.1
FOLD_SEED = 2031
FOLD_COUNT = 5
# The one id of the saved second part that the whitespace-removing rule puts in a
# duplicate group; the file was drawn under a rule that kept single spaces.
SECOND_PART_TWINS = ("college_physics/93",)


@dataclass(frozen=True)
class Splits:
    test_first: list[str]
    test_second: list[str]
    heldout: list[str]
    train: list[str]
    in_no_split: list[str]
    folds: list[list[str]]

    @property
    def test(self) -> list[str]:
        return sorted(self.test_first + self.test_second)

    def split_of(self) -> dict[str, str]:
        """The split of every id that is in one."""
        out = {}
        for name in ("test", "heldout", "train"):
            for qid in getattr(self, name):
                out[qid] = name
        return out


def _read_test_parts(pool: dict[str, Question]) -> tuple[list[str], list[str]]:
    plan = json.loads(paths.PLAN.read_text(encoding="utf-8"))
    second = json.loads(paths.TEST_SECOND_PART.read_text(encoding="utf-8"))
    first = sorted(plan["test_questions"])
    second_ids = sorted(second["wrong_first"] + second["right_first"])
    missing = [q for q in first + second_ids if q not in pool]
    if missing:
        raise ValueError(f"test ids without a complying first answer: {missing[:5]}")
    overlap = sorted(set(first) & set(second_ids))
    if overlap:
        raise ValueError(f"the saved second part holds ids of the first part: {overlap[:5]}")
    return first, second_ids


def units_of(ids: list[str], groups: list[list[str]]) -> list[tuple[str, ...]]:
    """The ids as units: a duplicate group within `ids` stays whole, any other id stands alone."""
    present = set(ids)
    units, grouped = [], set()
    for g in groups:
        members = tuple(sorted(i for i in g if i in present))
        if members:
            units.append(members)
            grouped.update(members)
    units.extend((i,) for i in ids if i not in grouped)
    return sorted(units)


def _stratum(unit: tuple[str, ...], pool: dict[str, Question], with_pair: set[str]) -> int:
    first = pool[unit[0]]
    if first.wrong_first:
        return 0
    return 1 if first.qid in with_pair else 2


def draw_heldout(units: list[tuple[str, ...]], pool: dict[str, Question],
                 with_pair: set[str]) -> list[tuple[str, ...]]:
    """10% of the units of each stratum, wrong-first, right-first with a pair, right-first
    without one, in that order, from one seeded generator."""
    rng = random.Random(HELDOUT_SEED)
    drawn = []
    for stratum in range(3):
        members = sorted(u for u in units if _stratum(u, pool, with_pair) == stratum)
        drawn.extend(rng.sample(members, round(HELDOUT_SHARE * len(members))))
    return drawn


def draw_folds(units: list[tuple[str, ...]]) -> list[list[str]]:
    """The units in a seeded order, dealt in turn to folds 0 to 4; each fold's ids sorted."""
    order = random.Random(FOLD_SEED).sample(sorted(units), len(units))
    return [sorted(i for u in order[k::FOLD_COUNT] for i in u) for k in range(FOLD_COUNT)]


def build_splits(pool: list[Question], pairs: list[Pair]) -> Splits:
    """Draw the splits from the pool and the jobs' pairs; the result depends on no reason."""
    by_id = complying(pool)
    groups = duplicate_groups(pool)
    test_first, test_second = _read_test_parts(by_id)
    test = set(test_first) | set(test_second)
    in_group = {i for g in groups for i in g}
    twins = sorted(set(test_second) & in_group)
    if tuple(twins) != SECOND_PART_TWINS:
        raise ValueError(f"the saved second part has ids in duplicate groups {twins}, "
                         f"expected {list(SECOND_PART_TWINS)}")
    test_texts = {normalise(by_id[q].question) for q in test}
    in_no_split = sorted(q for q in by_id if q not in test and normalise(by_id[q].question) in test_texts)
    outside = test | set(in_no_split)
    rest = sorted(q for q in by_id if q not in outside)
    units = units_of(rest, groups)
    with_pair = {p.qid for p in pairs}
    heldout_units = draw_heldout(units, by_id, with_pair)
    held = set(heldout_units)
    train_units = [u for u in units if u not in held]
    return Splits(
        test_first=test_first, test_second=test_second,
        heldout=sorted(i for u in heldout_units for i in u),
        train=sorted(i for u in train_units for i in u),
        in_no_split=in_no_split, folds=draw_folds(train_units))


def save_splits(splits: Splits, path: Path = paths.SPLITS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(splits), indent=1) + "\n", encoding="utf-8", newline="\n")


def load_splits(path: Path = paths.SPLITS) -> Splits:
    return Splits(**json.loads(path.read_text(encoding="utf-8")))
