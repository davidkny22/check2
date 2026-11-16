"""The reader's jobs that run the original model on a session.

    python -m check2.reader.build template            the template assumption on every episode
    python -m check2.reader.build first-turn          first-turn option probabilities
    python -m check2.reader.build cache               the conversation cache, `train` and `heldout`
    python -m check2.reader.build reason-only-cache   the reason-only cache, `train` and `heldout`
    python -m check2.reader.build spot-check          32 saved episodes against a fresh pass

`--per-format N` makes a small run on N episodes per format of each split. Each job writes a
measurement record.
"""
import argparse
import random
import sys

import torch

from check2.data import paths
from check2.data.paths import LETTERS
from check2.data.pool import complying, load_pool
from check2.data.splits import load_splits
from check2.reader import files
from check2.reader.cache import (CACHED_SPLITS, READ_LAYERS, batch_order, cache_states, episode_rows,
                                 read_meta, saved_states, tokenize)
from check2.reader.first_turn import first_turn_reads
from check2.reader.original import load_original, load_tokenizer, pinned_revision
from check2.reader.reason_only import FORMAT, reason_only_items
from check2.reader.template import check_episodes
from check2.record import write_record

SPOT_CHECK_EPISODES = 32


def template_job(args) -> int:
    tokenizer = load_tokenizer()
    meta = files.read_json(paths.EPISODES / "meta.json")
    counts, failing, findings = {}, 0, []
    for path in sorted(paths.EPISODES.glob("*.jsonl")):
        episodes = files.small_draw(files.read_jsonl(path), args.per_format)
        n_bad, found = check_episodes(tokenizer, episodes, meta["tail_ids"], meta["tail_position"])
        counts[path.name] = {"episodes": len(episodes), "failing": n_bad}
        failing += n_bad
        findings += found
        print(f"{path.name}: {len(episodes)} episodes, {n_bad} failing", flush=True)
    verdict = "green" if failing == 0 else "red"
    record = write_record("measurement", "template_tail",
                          {"revision": pinned_revision(), "tail_ids": meta["tail_ids"],
                           "tail_position": meta["tail_position"], "files": counts, "failing": failing,
                           "findings": findings, "per_format": args.per_format}, verdict=verdict)
    for f in findings:
        print(f"FINDING {f}")
    print(f"verdict: {verdict}; record {record}")
    return 0 if failing == 0 else 1


def first_turn_job(args) -> int:
    model, tokenizer = load_original(), load_tokenizer()
    pool = complying(load_pool())
    questions = [pool[q] for q in sorted(pool)]
    if args.per_format is not None:
        questions = questions[:args.per_format]
    reads = first_turn_reads(model, tokenizer, questions)
    files.write_jsonl(files.FIRST_TURN, ({"qid": q, "probabilities": p, "read_as": form}
                                         for q, (p, form) in sorted(reads.items())))
    # Two checks: the most probable option should be the first answer the model wrote, and the
    # pool's own probabilities, measured in design by reading every bracketed answer as a
    # bracket then a letter, are compared form by form.
    by_form = {}
    for q in questions:
        p, form = reads[q.qid]
        row = by_form.setdefault(form, {"questions": 0, "most_probable_is_first_answer": 0,
                                        "max_abs_difference_from_pool": 0.0})
        row["questions"] += 1
        row["most_probable_is_first_answer"] += LETTERS[max(range(4), key=p.__getitem__)] == q.a1
        row["max_abs_difference_from_pool"] = max(row["max_abs_difference_from_pool"],
                                                  max(abs(a - b) for a, b in zip(p, q.p1)))
    values = {"questions": len(questions), "by_form": by_form, "per_format": args.per_format}
    record = write_record("measurement", "first_turn_probabilities", values)
    print(values)
    print(f"wrote {files.FIRST_TURN} and {record}")
    return 0


def _cached_episodes(per_format):
    episodes = []
    for split in CACHED_SPLITS:
        episodes += files.small_draw(files.load_split(split), per_format)
    return episodes


def cache_job(args) -> int:
    model, tokenizer = load_original(), load_tokenizer()
    episodes = _cached_episodes(args.per_format)
    index = cache_states(model, tokenizer, episodes, files.CONVERSATION_CACHE, READ_LAYERS)
    files.write_json(files.CACHE_INDEX, index)
    tokens = sum(v[2] for v in index.values())
    meta = read_meta(files.CONVERSATION_CACHE)
    record = write_record("measurement", "reader_cache",
                          {"episodes": len(index), "saved_tokens": tokens, "shards": meta["shards"],
                           "dtype": meta["dtype"], "layers": meta["layers"], "per_format": args.per_format,
                           "splits": {s: sum(e["split"] == s for e in episodes) for s in CACHED_SPLITS}})
    print(f"cached {len(index)} episodes, {tokens} tokens, in {meta['shards']} shards; record {record}")
    return 0


def reason_only_cache_job(args) -> int:
    model, tokenizer = load_original(), load_tokenizer()
    pool = complying(load_pool())
    items = reason_only_items(tokenizer, files.read_jsonl(paths.PAIRS), pool, load_splits().split_of(),
                              CACHED_SPLITS)
    items = [i for s in CACHED_SPLITS for i in files.small_draw([i for i in items if i["split"] == s],
                                                               args.per_format)]
    conversation_tail = files.read_json(paths.EPISODES / "meta.json")["tail_ids"]
    encoded = tokenizer([i["prompt"] for i in items])["input_ids"]
    if any(list(ids[-len(t):]) != t for ids in encoded for t in conversation_tail.values()):
        print("STOP: a reason-only prompt does not end in the conversations' template tail")
        return 1
    files.write_jsonl(files.REASON_ONLY_ITEMS, items)
    index = cache_states(model, tokenizer, items, files.REASON_ONLY_CACHE, READ_LAYERS)
    record = write_record("measurement", "reason_only_cache",
                          {"pairs": len(index), "saved_tokens": sum(v[2] for v in index.values()),
                           "format": FORMAT, "per_format": args.per_format,
                           "splits": {s: sum(i["split"] == s for i in items) for s in CACHED_SPLITS}})
    print(f"cached {len(index)} reason-only prompts; record {record}")
    return 0


def spot_check_job(args) -> int:
    """Rerun the saved batch of each of 32 drawn episodes and compare its states exactly; also
    report how far each differs from a pass over the episode alone."""
    model, tokenizer = load_original(), load_tokenizer()
    meta = read_meta(files.CONVERSATION_CACHE)
    episodes = {e["episode_id"]: e for e in _cached_episodes(args.per_format)}
    cached = sorted(set(episodes) & {i for b in meta["batches"] for i in b})
    drawn = random.Random(files.SMALL_DRAW_SEED).sample(cached, min(SPOT_CHECK_EPISODES, len(cached)))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch_of = {i: b for b in meta["batches"] for i in b}
    rows = []
    for eid in sorted(drawn):
        batch = batch_order(tokenize(tokenizer, [episodes[i] for i in batch_of[eid]]))
        fresh = dict(zip([it["episode_id"] for it in batch], saved_states(model, batch, meta["layers"], pad_id)))
        alone = saved_states(model, tokenize(tokenizer, [episodes[eid]]), meta["layers"], pad_id)[0]
        saved = episode_rows(files.CONVERSATION_CACHE, eid)
        rows.append({"episode_id": eid, "equal_in_its_batch": bool(torch.equal(fresh[eid], saved)),
                     "max_abs_difference_alone": float((alone.float() - saved.float()).abs().max())})
    equal = sum(r["equal_in_its_batch"] for r in rows)
    verdict = "green" if equal == len(rows) else "red"
    record = write_record("measurement", "reader_cache_spot_check",
                          {"episodes": rows, "equal": equal, "checked": len(rows)}, verdict=verdict)
    print(f"{equal} of {len(rows)} saved episodes equal a fresh pass of their batch; "
          f"largest difference from a pass alone {max(r['max_abs_difference_alone'] for r in rows)}")
    print(f"verdict: {verdict}; record {record}")
    return 0 if verdict == "green" else 1


JOBS = {"template": template_job, "first-turn": first_turn_job, "cache": cache_job,
        "reason-only-cache": reason_only_cache_job, "spot-check": spot_check_job}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--per-format", type=int, default=None)
    args = parser.parse_args(argv)
    return JOBS[args.job](args)


if __name__ == "__main__":
    sys.exit(main())
