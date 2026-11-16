"""Split training equals plain: both training forms give the step's consumer loss and gradients exactly.

In float32 on the built-in path, on 16 training episodes, 8 of each format, with writers and
adapter factors random at scale 0.01, as in gradient parity: the consumer loss and the
gradient of every trained parameter, computed one episode at a time on whole sequences with
no padding, are matched
with relative difference at most 1e-4 by two forms, each run as padded batches of at most
8 rows: the shared-prefix form and the whole-sequence form. An out-of-memory error halves
the batch and is not a result; any other exception is red for its form. The shared-prefix
form's result selects the form training uses. The planted fault sets the recurrent state
to zero at the start of each continuation, which the shared-prefix form must catch.

A parameter's relative difference is the norm of the difference of the two gradients over
the norm of the reference's; the loss's is the absolute difference over the reference's.

`--implementation hooks|wrappers|library` (default `hooks`) names the adapter
implementation the seeded factors are copied into; the reference and both forms run
through it, and the record names it.

`--answer-weight 0.5` compares the two forms on the answer-weighted consumer loss: the
reference and both forms give each short-format episode's answer token that share of its
cross-entropy, and the record names the weight.
"""

from __future__ import annotations

import gc
import time
from dataclasses import fields

import torch

from check2.config.schema import WRITER_INPUTS, Training
from check2.data import paths
from check2.gates import faults, training_setup
from check2.gates import mechanism_setup as setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.training import bench as training_bench
from check2.training.answer_token import weighting_for
from check2.training.batches import plan_batches
from check2.training.examples import read_targets
from check2.training.forms import SharedPrefix, WholeSequence
from check2.training.objectives import consumer_step

NAME = "split_training"
EPISODES_PER_FORMAT = 8
MAX_ROWS = 8
TOLERANCE = 1e-4
# The values the answer token's weight takes, as the configuration allows them.
ANSWER_WEIGHTS = {f.name: f for f in fields(Training)}["answer_token_weight"].metadata["allowed"]


def _relative(a: torch.Tensor, reference: torch.Tensor) -> float:
    a, reference = a.double(), reference.double()
    scale = float(reference.norm())
    difference = float((a - reference).norm())
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / scale


def _run_form(form_class, pipeline, examples, p_of, beta, step_size, parameters, answer=None) -> dict:
    """The step's consumer loss and gradients in one form, halving the rows on an out-of-memory error."""
    rows, halvings = MAX_ROWS, []
    while rows >= 1:
        training_setup.clear_gradients(parameters)
        form = form_class(pipeline)
        out_of_memory = False
        try:
            batches = plan_batches(examples, float("inf"), form.batch_tokens, max_rows=rows)
            loss = 0.0
            for batch in batches:
                losses = consumer_step(form, batch, p_of(batch), beta, step_size, answer)
                loss += float(losses.total.double().sum()) / step_size
            return {"loss": loss, "gradients": training_setup.gradients(parameters), "max_rows": rows,
                    "batch_rows": [len(b) for b in batches], "out_of_memory_at_rows": halvings, "error": None}
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True
        except Exception as e:  # noqa: BLE001 - any other exception is red for its form
            pipeline.control.clear()
            return {"error": f"{type(e).__name__}: {str(e)[:500]}", "max_rows": rows,
                    "out_of_memory_at_rows": halvings}
        # Outside the handler, so that the failed pass's tensors are no longer held by the exception.
        if out_of_memory:
            pipeline.control.clear()
            training_setup.clear_gradients(parameters)
            gc.collect()
            torch.cuda.empty_cache()
            halvings.append(rows)
            rows //= 2
    return {"error": "out of memory at one row", "max_rows": 0, "out_of_memory_at_rows": halvings}


def check(bench: setup.Bench, episodes: dict[str, list[dict]], targets_file=None, plant: bool = False,
          implementation: str = "hooks", writer_input: str = WRITER_INPUTS[0], answer_weight: float = 0.0) -> dict:
    control = ControlState()
    chosen = [e for fmt in episodes for e in episodes[fmt]]
    writers, adapter = training_setup.random_additions(bench, control, implementation, writer_input)
    pipeline = Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter)
    try:
        examples = training_bench.examples(chosen, bench.tokenizer, targets_file)
        signal = {e["episode_id"]: row for e, row in zip(chosen, one_hot_situations(chosen))}
        if writers.wide:
            # Each episode's S: its one-hot p beside its own state at the read token, from a pass of it alone.
            signal = {e["episode_id"]: pipeline.writer_signal(signal[e["episode_id"]][None], pipeline.prefill([e]))[0]
                      for e in chosen}

        def p_of(batch):
            return torch.stack([signal[e.episode_id] for e in batch]).to(pipeline.device)

        beta, step_size = training_bench.BETA, len(examples)
        replies = {i: target["reply"] for i, target in read_targets(targets_file or paths.TARGETS).items()}
        answer = weighting_for(answer_weight, examples, replies, bench.tokenizer)
        parameters = training_setup.trained_parameters(pipeline)
        training_setup.clear_gradients(parameters)
        reference_loss = training_setup.one_at_a_time(pipeline, examples, p_of(examples), beta, step_size, answer)
        reference = training_setup.gradients(parameters)
        values: dict = {"implementation": implementation,
                        "episodes": {fmt: len(v) for fmt, v in episodes.items()}, "step_size": step_size,
                        "beta": beta, "tolerance": TOLERANCE, "reference_loss": reference_loss,
                        "additions_scale": training_setup.ADDITIONS_SCALE,
                        "targets": training_setup.targets_fact(targets_file),
                        "episode_ids": [e.episode_id for e in examples],
                        "trained_parameters": len(parameters),
                        "reference_zero_gradients": sorted(n for n, g in reference.items() if float(g.norm()) == 0.0),
                        "forms": {}}
        if answer is not None:
            values["answer_token_weight"] = answer.weight
            values["answer_positions"] = [answer.positions.get(e.episode_id) for e in examples]
        with faults.planted(NAME, plant):
            for form_class in (SharedPrefix, WholeSequence):
                run = _run_form(form_class, pipeline, examples, p_of, beta, step_size, parameters, answer)
                result = {k: v for k, v in run.items() if k != "gradients"}
                if run["error"] is None:
                    differences = {n: _relative(run["gradients"][n], reference[n]) for n in reference}
                    worst = max(differences, key=differences.get)
                    result.update(loss_relative_difference=abs(run["loss"] - reference_loss) / abs(reference_loss),
                                  max_gradient_relative_difference=differences[worst], worst_parameter=worst)
                    result["green"] = (result["loss_relative_difference"] <= TOLERANCE
                                       and result["max_gradient_relative_difference"] <= TOLERANCE)
                else:
                    result["green"] = False
                values["forms"][form_class.name] = result
    finally:
        pipeline.close()
    forms = values["forms"]
    values["training_form"] = "shared_prefix" if forms["shared_prefix"]["green"] else "whole_sequence"
    values["green"] = forms["shared_prefix"]["green"] and forms["whole_sequence"]["green"]
    return values


def main(argv=None) -> int:
    parser = training_setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--answer-weight", type=float, choices=ANSWER_WEIGHTS, default=ANSWER_WEIGHTS[0],
                        help="the answer token's share of a short-format episode's cross-entropy")
    args = parser.parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=True)
    episodes = training_setup.draw(args, EPISODES_PER_FORMAT)
    values = check(bench, episodes, args.targets, plant=args.plant, implementation=args.implementation,
                   writer_input=args.writer_input, answer_weight=args.answer_weight)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
