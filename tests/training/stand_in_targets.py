"""Stand-in targets for the training checks' episodes, made on a session before the sampled targets exist.

    python -m tests.training.stand_in_targets <out file>

The training checks draw, per format, `random.Random(2035).sample(sorted(ids), 8)` from the
`named` and `reasoned` `train` episodes. For those 16 episodes only, this writes a targets
file: each snap episode gets its constructed target, the question's first answer with its
letter replaced; each short episode gets the first qualifying of 4 replies sampled from the
original model with the target sampler's settings, seeded with the configuration's seed.
A short episode with no qualifying reply takes its first reply that ended, marked
`sampled_unqualified`, so that every drawn episode keeps a reply of a real length.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from check2.data import sampling
from check2.data.episodes import END_OF_TURN
from check2.data.pool import load_pool
from check2.data.targets import construct_snap_target, select_target
from check2.training.bench import drawn
from tests.training.made_up_targets import training_candidates

PER_FORMAT = 8


def main(out: Path) -> None:
    candidates = training_candidates()
    snap = drawn([e for e in candidates if e["format"] == "snap"], PER_FORMAT)
    short = drawn([e for e in candidates if e["format"] == "short"], PER_FORMAT)
    pool = {q.qid: q for q in load_pool()}
    records = [construct_snap_target(e, pool[e["qid"]]) for e in snap]
    model, tokenizer = sampling.load_original_model(sampling.model_revision())
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    lines = sampling.sample_batch(model, tokenizer, short, sampling.TARGET_SAMPLES, sampling.RUN_SEED, 0, end_id)
    for e, line in zip(short, lines):
        target = select_target(line["samples"], e)
        if target is None:
            index = next((i for i, s in enumerate(line["samples"]) if s["ended"]), 0)
            target = {"episode_id": e["episode_id"], "reply": line["samples"][index]["text"], "a2": None,
                      "source": "sampled_unqualified", "sample_index": index}
        records.append(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    for r in records:
        print(r["episode_id"], r["source"], r["sample_index"], len(tokenizer(r["reply"])["input_ids"]), "tokens")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
