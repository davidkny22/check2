"""Cached equals uncached: generation from the cache gives the tokens a full recomputation gives.

In float32 on the built-in path, with random non-zero writers and adapter factors, on 16
episodes: cached generation and full recomputation give the same next token at every step,
under teacher forcing along the cached path's tokens, up to and including the first
end-of-turn token. Full recomputation is one call over the whole sequence with no cache,
the form training uses. The arm with no additions runs first; if it disagrees at any step,
the library path itself differs and the gate is red on that. In the arm with additions a
disagreement counts only where the cached path's margin between its top two tokens is
more than ten times the arm without additions' largest top-logit difference, per format,
over the steps where its tokens agree; one inside that bound is a tie, recorded with its
margin and not counted, and every row is compared to its end. The planted fault skips the
additions in calls over one token. With `--setting`, both arms run with that decode setting
on; the full recomputation makes no cache, so under the preallocated cache it runs as it is.
"""

from __future__ import annotations

import time

from check2.gates import faults
from check2.config.schema import WRITER_INPUTS
from check2.gates import mechanism_setup as setup
from check2.gates import token_agreement
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, copy_in
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline, one_hot_situations

NAME = "cached_equals_uncached"
MEANING = "the library path itself differs between cached decoding and full recomputation"
EPISODES_PER_FORMAT = 16


def compare(bench: setup.Bench, control: ControlState, writers, adapter, batch: list[dict], cap: int,
            setting: str | None = None) -> dict:
    p = one_hot_situations(batch)
    pipeline = Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter)
    try:
        with setup.setting_for(setting, pipeline, batch, cap):
            prefilled = pipeline.prefill(batch)
            replies = pipeline.generate(prefilled, p, cap, keep_scores=True)
            cached = [reply.token_ids for reply in replies]
            recomputed = pipeline.teacher_forced_logits(prefilled, p, cached, cached=False)
    finally:
        pipeline.close()
    out = token_agreement.compare_rows(batch, cached, recomputed, [reply.scores for reply in replies])
    out["rows_ended_inside_the_cap"] = sum(bool(t) and t[-1] == pipeline.end_of_turn_id for t in cached)
    return out


def check(bench: setup.Bench, episodes: dict[str, list[dict]], plant: bool = False,
          implementation: str = "hooks", setting: str | None = None, writer_input: str = WRITER_INPUTS[0]) -> dict:
    additions: dict = {}

    def compare_arm(arm: str, fmt: str) -> dict:
        if arm == "control":
            return compare(bench, ControlState(), None, None, episodes[fmt], REPLY_CAPS[fmt], setting)
        if not additions:
            control = ControlState()
            measure = [e for f in episodes for e in episodes[f]]
            additions["control"] = control
            additions["writers"], additions["adapter"] = setup.random_additions(bench, control, measure,
                                                                                setting=setting,
                                                                                writer_input=writer_input)
            additions["adapter"] = copy_in(additions["adapter"], implementation, bench.model)
        with faults.planted(NAME, plant):
            return compare(bench, additions["control"], additions["writers"], additions["adapter"],
                           episodes[fmt], REPLY_CAPS[fmt], setting)

    named = {"implementation": implementation, **({"setting": setting} if setting is not None else {})}
    return token_agreement.verdict(named, compare_arm, list(episodes), MEANING)


def main(argv=None) -> int:
    parser = setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="hooks")
    setup.add_setting_switch(parser)
    args = parser.parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=True)
    episodes = setup.draw(setup.episode_file(args, "heldout"), EPISODES_PER_FORMAT, args.tail_position, args.limit)
    values = check(bench, episodes, plant=args.plant, implementation=args.implementation, setting=args.setting,
                   writer_input=args.writer_input)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
