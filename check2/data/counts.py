"""The counts the pinned files must give, and the count table of an assembly."""
from collections import Counter

from check2.data.pairs import Pair
from check2.data.pool import Question, complying, duplicate_groups
from check2.data.splits import Splits

# The specification's counts on the files as given.
POOL_QUESTIONS = 2081
COMPLYING = 2054
WRONG_FIRST = 617
DUPLICATE_GROUPS = 24
DUPLICATE_IDS = 48
GROUPS_SHARING_OPTIONS = 12
TEST_FIRST = 301
TEST_SECOND = 240
FIRST_PART_PAIRS = 437
FIRST_PART_GOLD = 136
SECOND_PART_PAIRS = 360
SECOND_PART_GOLD = 120
OTHER_PAIRS = 1562
OTHER_PAIR_QUESTIONS = 1081
TEST_PAIRS = 797
TEST_GOLD = 256
TEST_EPISODES = 4270


def pool_counts(pool: list[Question]) -> dict[str, int]:
    by_id = complying(pool)
    groups = duplicate_groups(pool)
    return {
        "pool_questions": len(pool),
        "complying": len(by_id),
        "wrong_first": sum(q.wrong_first for q in by_id.values()),
        "duplicate_groups": len(groups),
        "duplicate_ids": sum(len(g) for g in groups),
        "groups_sharing_options": sum(len({by_id[i].choices for i in g}) == 1 for g in groups),
    }


def pair_counts(pool: list[Question], pairs: list[Pair], test_first: list[str],
                test_second: list[str]) -> dict[str, int]:
    """Counts of the writer's pairs, as the jobs define them."""
    by_id = complying(pool)
    first, second = set(test_first), set(test_second)
    opus = [p for p in pairs if p.writer == "opus"]
    in_first = [p for p in opus if p.qid in first]
    in_second = [p for p in opus if p.qid in second]
    other = [p for p in opus if p.qid not in first]

    def gold(ps: list[Pair]) -> int:
        return sum(p.target == by_id[p.qid].gold for p in ps)

    return {
        "test_first": len(first), "test_second": len(second),
        "first_part_pairs": len(in_first), "first_part_gold": gold(in_first),
        "second_part_pairs": len(in_second), "second_part_gold": gold(in_second),
        "other_pairs": len(other), "other_pair_questions": len({p.qid for p in other}),
        "test_pairs": len(in_first) + len(in_second), "test_gold": gold(in_first) + gold(in_second),
    }


EXPECTED = {
    "pool_questions": POOL_QUESTIONS, "complying": COMPLYING, "wrong_first": WRONG_FIRST,
    "duplicate_groups": DUPLICATE_GROUPS, "duplicate_ids": DUPLICATE_IDS,
    "groups_sharing_options": GROUPS_SHARING_OPTIONS,
    "test_first": TEST_FIRST, "test_second": TEST_SECOND,
    "first_part_pairs": FIRST_PART_PAIRS, "first_part_gold": FIRST_PART_GOLD,
    "second_part_pairs": SECOND_PART_PAIRS, "second_part_gold": SECOND_PART_GOLD,
    "other_pairs": OTHER_PAIRS, "other_pair_questions": OTHER_PAIR_QUESTIONS,
    "test_pairs": TEST_PAIRS, "test_gold": TEST_GOLD,
}


def mismatches(actual: dict[str, int]) -> list[str]:
    """Each count that differs from the specification's, with both numbers."""
    return [f"{k}: the files give {actual[k]}, the specification says {v}"
            for k, v in EXPECTED.items() if k in actual and actual[k] != v]


def expected_test_episodes(removed_test_pairs: int, test_pairs_without_reason: int) -> int:
    """Two bare episodes per `test` question and four per `test` pair, less the four of each
    removed pair and the two reasoned episodes of each pair whose reason is not written yet."""
    return TEST_EPISODES - 4 * removed_test_pairs - 2 * test_pairs_without_reason


def episode_table(episodes: list[dict]) -> dict[str, Counter]:
    """Episode counts by file-level grouping: split, form, format and situation."""
    return {
        "split_form_format": Counter(f"{e['split']}/{e['form']}/{e['format']}" for e in episodes
                                     if e["writer"] in (None, "opus")),
        "split_situation": Counter(f"{e['split']}/{e['situation']}" for e in episodes
                                   if e["form"] != "neutral" and e["writer"] in (None, "opus")),
    }


def split_counts(splits: Splits) -> dict[str, int]:
    out = {name: len(getattr(splits, name))
           for name in ("test_first", "test_second", "heldout", "train", "in_no_split")}
    for k, fold in enumerate(splits.folds):
        out[f"fold_{k}"] = len(fold)
    return out
