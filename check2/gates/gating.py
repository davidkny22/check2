"""Gating: no addition reaches a token at or before the read, and both switch on at the token after it.

With random non-zero writers and adapter factors, on 64 episodes per format: every layer's
output at every token up to and including `r` has maximum absolute difference 0 from the
same pipeline with no additions; and the output of layer 3 at token `r + 1` differs from
that pipeline's, once with the write alone and once with the adapter alone. The first half
is checked with both additions, and with each alone. The planted fault shifts the mask one
token earlier, so that it includes `r`. With `--setting`, every arm runs with that decode
setting on, its preallocated cache sized for the format's reply cap as a run's is.
"""

from __future__ import annotations

import time

import torch

from check2.gates import faults
from check2.config.schema import WRITER_INPUTS
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, copy_in
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline, one_hot_situations

NAME = "gating"
EPISODES_PER_FORMAT = 64
BATCH = 16
ARMS = {"both": (True, True), "write_alone": (True, False), "adapter_alone": (False, True)}


def _outputs(bench, control, writers, adapter, batch, p, setting=None):
    pipeline = Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter)
    try:
        with setup.setting_for(setting, pipeline, batch, REPLY_CAPS[batch[0]["format"]]):
            return setup.prompt_pass_outputs(pipeline, batch, p)
    finally:
        pipeline.close()


def check(bench: setup.Bench, episodes: dict[str, list[dict]], plant: bool = False,
          implementation: str = "hooks", setting: str | None = None, writer_input: str = WRITER_INPUTS[0]) -> dict:
    control = ControlState()
    measure = [e for fmt in episodes for e in episodes[fmt]]
    writers, adapter = setup.random_additions(bench, control, measure, setting=setting, writer_input=writer_input)
    adapter = copy_in(adapter, implementation, bench.model)
    values: dict = {"implementation": implementation, "formats": {}}
    if setting is not None:
        values["setting"] = setting
    with faults.planted("gating", plant):
        for fmt, chosen in episodes.items():
            result = {"episodes": 0, **{f"{arm}_max_abs_difference_up_to_r": 0.0 for arm in ARMS},
                      "write_alone_changes_layer_3_at_r_plus_1": 0, "adapter_alone_changes_layer_3_at_r_plus_1": 0}
            for batch in setup.batches(chosen, BATCH):
                p = one_hot_situations(batch)
                prefilled, plain = _outputs(bench, ControlState(), None, None, batch, p, setting)
                for arm, (write_on, adapter_on) in ARMS.items():
                    _, added = _outputs(bench, control, writers if write_on else None,
                                        adapter if adapter_on else None, batch, p, setting)
                    worst = max(setup.max_difference_up_to_read(plain, added, prefilled).values())
                    key = f"{arm}_max_abs_difference_up_to_r"
                    result[key] = worst if not worst <= result[key] else result[key]
                    if arm != "both":
                        # Column 0 of the call after the prefix is the token r + 1 in every row.
                        a, b = plain[setup.GATING_LAYER][1][:, 0], added[setup.GATING_LAYER][1][:, 0]
                        changed = (a != b).any(dim=-1)
                        result[f"{arm}_changes_layer_3_at_r_plus_1"] += int(changed.sum())
                result["episodes"] += len(batch)
            values["formats"][fmt] = result
    first_half = all(v[f"{arm}_max_abs_difference_up_to_r"] == 0.0 for v in values["formats"].values() for arm in ARMS)
    second_half = all(v["write_alone_changes_layer_3_at_r_plus_1"] == v["episodes"]
                      and v["adapter_alone_changes_layer_3_at_r_plus_1"] == v["episodes"]
                      for v in values["formats"].values())
    values.update(first_half_green=first_half, second_half_green=second_half, green=first_half and second_half)
    return values


def main(argv=None) -> int:
    parser = setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="hooks")
    setup.add_setting_switch(parser)
    args = parser.parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=False)
    episodes = setup.draw(setup.episode_file(args, "heldout"), EPISODES_PER_FORMAT, args.tail_position, args.limit)
    values = check(bench, episodes, plant=args.plant, implementation=args.implementation, setting=args.setting,
                   writer_input=args.writer_input)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
