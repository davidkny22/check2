"""Identity: the hooks change nothing while the writers and the adapter are at initialisation.

With writers and adapter factors at initialisation and every hook installed, on 64 `test`
episodes per format: every layer's output on the prompt pass has maximum absolute
difference 0 from the same pipeline with no hooks, and greedy generation is
token-identical. The planted fault starts every writer's last-layer bias at 1e-2. With
`--setting`, both arms run with that decode setting on.
"""

from __future__ import annotations

import time

import torch

from check2.config.schema import WRITER_INPUTS
from check2.gates import faults
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, build
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline, one_hot_situations

NAME = "identity"
EPISODES_PER_FORMAT = 64
BATCH = 16


def _arm(bench: setup.Bench, batch: list[dict], p: torch.Tensor, cap: int, additions: bool,
         implementation: str = "hooks", setting: str | None = None, writer_input: str = WRITER_INPUTS[0]):
    control = ControlState()
    writers = adapter = None
    if additions:
        device = next(bench.model.parameters()).device
        writers = setup.writers_for(bench, writer_input).to(device)
        adapter = build(implementation, bench.model, control, setup.RANK, setup.ALPHA_LORA,
                        bench.layout.adapter_first, bench.layout.adapter_last)
    pipeline = Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter)
    try:
        with setup.setting_for(setting, pipeline, batch, cap):
            prefilled, outputs = setup.prompt_pass_outputs(pipeline, batch, p)
            tokens = [r.token_ids for r in pipeline.generate(prefilled, p, cap)]
    finally:
        pipeline.close()
    return prefilled, outputs, tokens


def check(bench: setup.Bench, episodes: dict[str, list[dict]], plant: bool = False,
          implementation: str = "hooks", setting: str | None = None, writer_input: str = WRITER_INPUTS[0]) -> dict:
    values: dict = {"implementation": implementation, "formats": {}}
    if setting is not None:
        values["setting"] = setting
    for fmt, chosen in episodes.items():
        worst, layer_of_worst, rows, same_tokens = 0.0, None, 0, 0
        for batch in setup.batches(chosen, BATCH):
            p = one_hot_situations(batch)
            prefilled, plain, plain_tokens = _arm(bench, batch, p, REPLY_CAPS[fmt], additions=False, setting=setting)
            with faults.planted("identity", plant):
                _, hooked, hooked_tokens = _arm(bench, batch, p, REPLY_CAPS[fmt], additions=True,
                                                implementation=implementation, setting=setting,
                                                writer_input=writer_input)
            for layer in plain:
                for call in range(2):
                    a, b = plain[layer][call], hooked[layer][call]
                    mask = prefilled.prefix_attention.bool() if call == 0 else torch.ones(a.shape[:2], dtype=torch.bool,
                                                                                         device=a.device)
                    diff = float((a[mask] - b[mask]).abs().max())
                    if not diff <= worst:
                        worst, layer_of_worst = diff, layer
            rows += len(batch)
            same_tokens += sum(x == y for x, y in zip(plain_tokens, hooked_tokens))
        values["formats"][fmt] = {"episodes": rows, "max_abs_difference": worst, "layer_of_max": layer_of_worst,
                                  "generations_identical": same_tokens}
    values["green"] = all(v["max_abs_difference"] == 0.0 and v["generations_identical"] == v["episodes"]
                          for v in values["formats"].values())
    return values


def main(argv=None) -> int:
    parser = setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="hooks")
    setup.add_setting_switch(parser)
    args = parser.parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=False)
    episodes = setup.draw(setup.episode_file(args, "test"), EPISODES_PER_FORMAT, args.tail_position, args.limit)
    values = check(bench, episodes, plant=args.plant, implementation=args.implementation, setting=args.setting,
                   writer_input=args.writer_input)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
