"""The reason-only reader: how much of the reader's separation needs no question.

Its prompt holds only the written reason: the system turn, then one user turn with the reason
and nothing else, so there is one prompt per pair, with the same template tail and the same
relative read position as the conversations. The selected variant is fitted on `train` pairs
with the `heldout` loss deciding its epoch count, and cross-fitted in the same five folds with
that count. Its score is the AUROC of `user_right` on pairs of wrong-first questions.

Run as `python -m check2.reader.reason_only [--per-format N]` after the selection and the
reason-only cache; writes the cross-fitted and `heldout` outputs, the fitted reader, and a
measurement record.

Run as `python -m check2.reader.reason_only test [--per-format N]` after that, on a session
with the model: loads the fitted reason-only reader, computes its `p` for every `test` pair
that has a reason in one forward pass over the reason-only prompts, saving no state, and
writes `data/reader/p_reason_only_test.jsonl` in the form of the other saved outputs and a
measurement record.
"""
import argparse
import sys

import torch

from check2.data.episodes import SYSTEM
from check2.data.pairs import situation
from check2.data.pool import Question
from check2.data.splits import load_splits
from check2.reader import files
from check2.reader.cache import batch_order, right_align, saved_states, tokenize
from check2.reader.crossfit import cross_fit, save_reader
from check2.reader.inputs import load_inputs
from check2.reader.model import Reader
from check2.reader.scores import contrast_records, flatten, one_vs_rest, selection_score, user_right_scores
from check2.reader.train import BATCH_SIZE, fit, predict
from check2.reader.window import window_length
from check2.record import write_record

FORMAT = "reason_only"
FORMATS = (FORMAT,)
MAIN_WRITERS = (None, "opus")
P_TEST = files.READER_DATA / "p_reason_only_test.jsonl"


def pair_key(qid: str, target: str) -> str:
    return f"{qid}:{target}"


def pairs_on_test(pairs: list[dict], split_of: dict[str, str]) -> list[dict]:
    """The `test` pairs that have a reason of the main writer, in key order; one pair per key."""
    chosen = sorted((p for p in pairs if p.get("reason") is not None and p.get("writer") in MAIN_WRITERS
                     and split_of.get(p["qid"]) == "test"), key=lambda p: pair_key(p["qid"], p["target"]))
    keys = [pair_key(p["qid"], p["target"]) for p in chosen]
    if len(set(keys)) != len(keys):
        raise ValueError("two test pairs with a reason share a question and a target")
    return chosen


def write_test_p(model, tokenizer, reader: Reader, pairs: list[dict], tail_position: int,
                 output=P_TEST) -> dict[str, list[float]]:
    """The reason-only reader's `p` for the given pairs, written to `output` as the other saved outputs are."""
    p = reason_only_p(model, tokenizer, reader, pairs, tail_position)
    files.write_jsonl(output, files.p_rows(p))
    return p


def reason_only_prompt(tokenizer, reason: str) -> str:
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": reason}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


def reason_only_items(tokenizer, pairs: list[dict], pool: dict[str, Question],
                      split_of: dict[str, str], splits: tuple[str, ...]) -> list[dict]:
    """One record per pair with a reason whose question is in one of `splits`, in key order."""
    items = []
    for p in pairs:
        if p.get("reason") is None or p.get("writer") not in MAIN_WRITERS or split_of.get(p["qid"]) not in splits:
            continue
        q = pool[p["qid"]]
        items.append({"episode_id": pair_key(p["qid"], p["target"]), "qid": p["qid"],
                      "split": split_of[p["qid"]], "form": "reasoned", "format": FORMAT,
                      "target": p["target"], "a1": q.a1, "gold": q.gold,
                      "situation": situation("reasoned", q.a1, q.gold, p["target"]),
                      "prompt": reason_only_prompt(tokenizer, p["reason"])})
    return sorted(items, key=lambda e: e["episode_id"])


@torch.inference_mode()
def reason_only_p(model, tokenizer, reader: Reader, pairs: list[dict], tail_position: int) -> dict:
    """`p` of the reason-only reader for each pair, keyed `qid:target`, from one forward pass over
    the reason-only prompts in left-padded batches; no state is saved."""
    device = next(reader.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    items = [{"episode_id": pair_key(p["qid"], p["target"]), "prompt": reason_only_prompt(tokenizer, p["reason"])}
             for p in pairs]
    ordered = batch_order(tokenize(tokenizer, items))
    out: dict[str, list[float]] = {}
    for start in range(0, len(ordered), BATCH_SIZE):
        chunk = ordered[start:start + BATCH_SIZE]
        rows = saved_states(model, chunk, reader.layers, pad_id)
        h, mask = right_align([r[:window_length(r.shape[0], tail_position)] for r in rows])
        p = reader(h.to(device), mask.to(device)).cpu().tolist()
        out.update((it["episode_id"], v) for it, v in zip(chunk, p))
    return {k: out[k] for k in sorted(out)}


def job_on_test(args) -> int:
    from check2.data import paths
    from check2.reader.crossfit import load_reader
    from check2.reader.original import load_original, load_tokenizer

    position = files.read_json(files.SELECTION)["read_position"]
    pairs = pairs_on_test(files.read_jsonl(paths.PAIRS), load_splits().split_of())
    if args.per_format is not None:
        keys = [{"episode_id": pair_key(p["qid"], p["target"]), "format": FORMAT} for p in pairs]
        kept = {k["episode_id"] for k in files.small_draw(keys, args.per_format)}
        pairs = [p for p in pairs if pair_key(p["qid"], p["target"]) in kept]
    model, tokenizer = load_original(), load_tokenizer()
    reader = load_reader(args.reader_path).to(next(model.parameters()).device)
    p = write_test_p(model, tokenizer, reader, pairs, position)
    values = {"pairs": len(p), "read_position": position, "per_format": args.per_format,
              "output": P_TEST.relative_to(paths.ROOT).as_posix(), "reader_path": args.reader_path}
    record = write_record("measurement", "reason_only_reader_test", values)
    print(f"reason-only p for {len(p)} test pairs, written to {P_TEST}; record {record}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("split", nargs="?", choices=("test",),
                        help="test: compute the fitted reader's p on the test pairs")
    parser.add_argument("--cache-dir", default=str(files.REASON_ONLY_CACHE))
    parser.add_argument("--reader-path", default=str(files.REASON_ONLY_WEIGHTS))
    parser.add_argument("--per-format", type=int, default=None)
    args = parser.parse_args(argv)
    if args.split == "test":
        return job_on_test(args)
    selection = files.read_json(files.SELECTION)
    variant, position = selection["variant"], selection["read_position"]
    items = files.read_jsonl(files.REASON_ONLY_ITEMS)
    train = files.small_draw([i for i in items if i["split"] == "train"], args.per_format)
    heldout = files.small_draw([i for i in items if i["split"] == "heldout"], args.per_format)
    inputs = load_inputs(args.cache_dir, train, heldout)
    first = fit(variant, inputs.bank, inputs.bank_layers, inputs.train_ids, inputs.label_of, position,
                files.SEED, heldout_ids=inputs.heldout_ids)
    p_heldout = dict(zip(inputs.heldout_ids, predict(first.reader, inputs.bank, inputs.bank_layers,
                                                    inputs.heldout_ids, position).tolist()))
    p_train, fold_fits = cross_fit(inputs, load_splits().folds, variant, position, first.best_epoch)
    save_reader(first.reader, args.reader_path)
    files.write_jsonl(files.REASON_ONLY_P_TRAIN, files.p_rows(p_train))
    files.write_jsonl(files.REASON_ONLY_P_HELDOUT, files.p_rows(p_heldout))

    def scores(p, rows):
        by_qid = contrast_records(rows, user_right_scores(p))
        return {"contrast_score": selection_score(flatten(by_qid), FORMATS), "contrast_questions": len(by_qid),
                "one_vs_rest": one_vs_rest(p, inputs.label_of)}

    values = {"variant": variant, "read_position": position, "epochs": first.best_epoch,
              "heldout_loss": first.heldout_loss, "per_format": args.per_format,
              "train_cross_fitted": scores(p_train, train), "heldout": scores(p_heldout, heldout),
              "folds": [{k: v for k, v in f.items() if not k.endswith("_questions")} for f in fold_fits],
              "pairs": {"train": len(train), "heldout": len(heldout)}, "reader_path": args.reader_path}
    record = write_record("measurement", "reason_only_reader", values)
    print(f"reason-only reader, cross-fitted train: {values['train_cross_fitted']}")
    print(f"heldout: {values['heldout']}; epochs {first.best_epoch}; record {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
