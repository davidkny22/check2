"""The reason-only reader's `p` on the pairs of the expanded question set.

    python -m check2.evaluation.expanded_reason_only [--per-format N]

On a session with the model: loads the fitted reason-only reader, computes its `p` for every
pair the expanded episodes hold, in one forward pass over the reason-only prompts at the
selected read position, saving no state, and writes `data/expansion/p_reason_only.jsonl` in
the form of the reader's other saved outputs, keyed `qid:target`, with a measurement record.
The reason-only prompt holds the reason alone, so a pair's two formats share one `p`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from check2.data import paths
from check2.data.condition_episodes import EXPANDED_EPISODES, read_jsonl
from check2.reader import files
from check2.reader.reason_only import pair_key, write_test_p
from check2.record import write_record

P_EXPANDED = paths.DATA / "expansion" / "p_reason_only.jsonl"


def expanded_pairs(episodes: Sequence[dict]) -> list[dict]:
    """One `{qid, target, reason}` per pair of the episodes, in key order; a pair whose formats disagree on
    its reason is an error."""
    pairs: dict[str, dict] = {}
    for e in episodes:
        key = pair_key(e["qid"], e["target"])
        pair = {"qid": e["qid"], "target": e["target"], "reason": e["reason"]}
        if pairs.setdefault(key, pair) != pair:
            raise ValueError(f"the episodes of pair {key} hold two reasons")
    return [pairs[k] for k in sorted(pairs)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.expanded_reason_only",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--reader-path", default=str(files.REASON_ONLY_WEIGHTS))
    parser.add_argument("--per-format", type=int, default=None, help="a small draw of N pairs, for a first try")
    args = parser.parse_args(argv)
    from check2.reader.crossfit import load_reader
    from check2.reader.original import load_original, load_tokenizer

    position = files.read_json(files.SELECTION)["read_position"]
    pairs = expanded_pairs(read_jsonl(EXPANDED_EPISODES))
    if args.per_format is not None:
        keys = [{"episode_id": pair_key(p["qid"], p["target"]), "format": "reason_only"} for p in pairs]
        kept = {k["episode_id"] for k in files.small_draw(keys, args.per_format)}
        pairs = [p for p in pairs if pair_key(p["qid"], p["target"]) in kept]
    model, tokenizer = load_original(), load_tokenizer()
    reader = load_reader(args.reader_path).to(next(model.parameters()).device)
    p = write_test_p(model, tokenizer, reader, pairs, position, output=P_EXPANDED)
    values = {"pairs": len(p), "read_position": position, "per_format": args.per_format,
              "output": P_EXPANDED.relative_to(paths.ROOT).as_posix(), "reader_path": args.reader_path}
    record = write_record("measurement", "reason_only_reader_expanded", values)
    print(f"reason-only p for {len(p)} expanded pairs, written to {P_EXPANDED}; record {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
