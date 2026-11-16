"""What the gates on the training forms share: their episodes, their random additions, the trained
parameters, and the plain reference.

The reference computes the consumer loss of a step and its gradients one episode at a
time, on the whole sequence with no padding, written here from the objective's formulas
and not through the training forms or the step loss: the mask is 1 at the tokens after
the read token, the logits are taken at the positions that predict the reply and the
end-of-turn token, the cross-entropy is their mean, and the KL term is
`sum softmax(orig) * (log_softmax(orig) - log_softmax(off))` averaged over them. Each
episode's two backward passes are scaled by the step's episode count, as a step's are.
With an answer weighting, the cross-entropy of a short-format episode is
`w * l[a] + (1 - w) * mean(l[t] for t != a)` over the same positions, with `a` the
episode's answer position as the weighting gives it from the reply's text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from check2.config.schema import WRITER_INPUTS
from check2.data import paths
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.adapter.hooks import HookAdapter
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, copy_in
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline
from check2.mechanism.write import Writers
from check2.training import bench as training_bench
from check2.training.answer_token import SHORT_FORMAT, AnswerWeighting
from check2.training.examples import TrainingExample

# The scale of the random writers and adapter factors in both checks of the training forms.
ADDITIONS_SCALE = 0.01


def parser(description: str):
    out = setup.parser(description)
    out.add_argument("--targets", default=None, help="the targets file; the build's by default")
    out.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="hooks",
                     help="the adapter implementation the seeded factors are copied into")
    return out


def draw(args, count: int) -> dict[str, list[dict]]:
    """Per format, `named` and `reasoned` `train` episodes with a target, drawn as a gate draws from a split."""
    out = {}
    for fmt in ("snap", "short"):
        chosen = training_bench.drawn(training_bench.candidate_episodes(args, fmt), count)
        out[fmt] = chosen if args.limit is None else chosen[:args.limit]
    return out


def targets_fact(targets_file) -> dict:
    """The targets file a gate ran on: its path as given and its SHA-256."""
    path = Path(targets_file) if targets_file else paths.TARGETS
    return {"file": path.name if path == paths.TARGETS else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def random_additions(bench: setup.Bench, control: ControlState, implementation: str = "hooks",
                     writer_input: str = WRITER_INPUTS[0]) -> tuple[Writers, GatedAdapter]:
    """Writers and adapter factors at `ADDITIONS_SCALE * randn`, from one seeded CPU generator, in a fixed order.

    Every writer parameter, in the order of `named_parameters`, then `A` and `B` of each
    adapted projection, in the adapter's order. The values are the same in every process
    and for every implementation: they are drawn into the hook adapter and then copied into
    the named one. The adapter is not installed; the pipeline that takes it installs it.
    With the wide input the writers take `S` and have more first-layer weights to draw.
    """
    generator = torch.Generator().manual_seed(setup.ADDITIONS_SEED)
    device = next(bench.model.parameters()).device
    writers = setup.writers_for(bench, writer_input)
    adapter = HookAdapter(bench.model, control, training_bench.RANK, training_bench.ALPHA_LORA,
                          bench.layout.adapter_first, bench.layout.adapter_last)
    with torch.no_grad():
        for _, parameter in writers.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * ADDITIONS_SCALE)
        for name, _ in adapter.modules:
            for factor in (adapter.a[name], adapter.b[name]):
                factor.copy_((torch.randn(factor.shape, generator=generator) * ADDITIONS_SCALE).to(device))
    if implementation != "hooks":
        adapter = copy_in(adapter, implementation, bench.model)
    return writers.to(device), adapter


def trained_parameters(pipeline: Pipeline) -> dict[str, torch.nn.Parameter]:
    """Every trained parameter by name: `writer.<first|second>.<layer>.<weight|bias>`, `adapter.<module>.<A|B>`."""
    out = {}
    if pipeline.writers is not None:
        for name, parameter in pipeline.writers.named_parameters():
            out[f"writer.{name}"] = parameter
    if pipeline.adapter is not None:
        for name, _ in pipeline.adapter.modules:
            out[f"adapter.{name}.A"] = pipeline.adapter.a[name]
            out[f"adapter.{name}.B"] = pipeline.adapter.b[name]
    return out


def layer_of(name: str) -> int:
    """The decoder layer a trained parameter belongs to; a writer belongs to the layer it writes into."""
    return int(name.split(".")[2])


def clear_gradients(parameters: dict[str, torch.nn.Parameter]) -> None:
    for parameter in parameters.values():
        parameter.grad = None


def gradients(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    """A float32 copy on the CPU of each parameter's gradient, zero where none was computed."""
    return {name: (torch.zeros_like(p, device="cpu") if p.grad is None else p.grad.detach().float().cpu().clone())
            for name, p in parameters.items()}


def one_at_a_time(pipeline: Pipeline, examples: list[TrainingExample], p: torch.Tensor, beta: float,
                  step_size: int, answer: AnswerWeighting | None = None) -> float:
    """The consumer loss of the step, with its gradients accumulated, one unpadded episode at a time."""
    control, model = pipeline.control, pipeline.model
    device = pipeline.device
    total = 0.0
    for i, e in enumerate(examples):
        ids = torch.tensor([e.sequence_ids], dtype=torch.long, device=device)
        after = (torch.arange(ids.shape[1], device=device) > e.read_index).to(torch.float32).unsqueeze(0)
        first = len(e.prompt_ids) - 1
        targets = torch.tensor(e.target_ids, dtype=torch.long, device=device)

        def logits(vectors: dict, adapter_on: bool) -> torch.Tensor:
            control.set(after, vectors, 1.0, (), adapter_active=adapter_on)
            hidden = pipeline.body(input_ids=ids).last_hidden_state
            return model.lm_head(hidden[0, first:first + len(targets)]).to(torch.float32)

        try:
            with torch.no_grad():
                orig = logits({}, False)
        finally:
            control.clear()
        try:
            off = logits({}, True)
            reference = orig.softmax(-1)
            kl = (reference * (orig.log_softmax(-1) - off.log_softmax(-1))).sum(-1).mean()
            (beta * kl / step_size).backward()
        finally:
            control.clear()
        try:
            on = logits(pipeline.writers(p[i:i + 1].to(device)), True)
            ce = F.cross_entropy(on, targets)
            if answer is not None and e.format == SHORT_FORMAT:
                per_token = F.cross_entropy(on, targets, reduction="none")
                a = answer.positions[e.episode_id]
                others = torch.cat([per_token[:a], per_token[a + 1:]])
                ce = answer.weight * per_token[a] + (1 - answer.weight) * others.mean()
            (ce / step_size).backward()
        finally:
            control.clear()
        total += (float(ce.detach()) + beta * float(kl.detach())) / step_size
    return total
