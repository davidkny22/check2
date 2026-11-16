"""Adapter matches its formula: a candidate adapts exactly the listed projections, by exactly the update.

For one candidate implementation, in float32 on the built-in path, with the same random
non-zero `A` and `B` loaded: the set of adapted modules is exactly the list (every layer of
the adapter's range; `q_proj`, `k_proj`, `v_proj`, `o_proj` and the three MLP projections in
each full-attention layer, `in_proj_qkv`, `in_proj_z`, `out_proj` and the three MLP
projections in each Gated DeltaNet layer); and every adapted module's output, and the
gradient of a reply loss with respect to every `A` and `B`, match
`W x + mask * (alpha_lora / rank) * B A x` computed directly, with relative difference at
most 1e-4.

Each projection's output is checked in every forward call of the pipeline's split at the
read token (the call over `0..r`, the call over the later prompt tokens and every decoding
step, to the end of the format's reply cap), in one right-padded call over every row's whole
sequence, and in the calls over whole sequences, one row each, that the gradient comes from.
Every linear module of the model is watched, so a module adapted that is not listed is seen
as well as a listed module left out. The gradient of the mean cross-entropy
of the generated replies is computed through the candidate and through a direct
computation of the formula in this file, on the same whole sequences. The scale
`alpha_lora / rank` in both comparisons is the settings', computed here, never the
adapter's own.

The planted fault (`--plant`) leaves the last listed module out of the candidate. The
second (`--plant-scale`) halves the scale of every adapter built while the gate runs.
"""

from __future__ import annotations

import contextlib
import time

import torch
import torch.nn.functional as F

from check2.gates import mechanism_setup as setup
from check2.mechanism import masks
from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, copy_in
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline, Prefilled
from check2.mechanism.model_parts import decoder_layers, text_model

NAME = "adapter_formula"
EPISODES_PER_FORMAT = 16
TOLERANCE = 1e-4
# Rows per backward pass. Only memory sets it: the gradient is that of one loss over all rows,
# and one row at a time keeps a float32 backward pass of the full model on one card.
GRADIENT_ROWS = 1
SPECIFIED = {
    "full_attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    "deltanet": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"),
    "all": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}


def specified_modules(model: torch.nn.Module, first_layer: int, last_layer: int) -> list[str]:
    """The projections the adapter must adapt, as `layers.<i>.<path>`, from the list this file states."""
    out = []
    for index in range(first_layer, last_layer + 1):
        kind = "full_attention" if hasattr(decoder_layers(model)[index], "self_attn") else "deltanet"
        out += [f"layers.{index}.{path}" for path in SPECIFIED[kind] + SPECIFIED["all"]]
    return out


def _resolve(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    for part in dotted.split("."):
        root = getattr(root, part)
    return root


def _relative(difference: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(reference.norm())
    return float(difference.norm()) / denominator if denominator > 0 else float(difference.norm() > 0)


class OutputProbe:
    """Watches every linear module's output and compares it with the formula computed directly.

    The formula's base part `W x` is computed by the same operation the module uses; the
    update `mask * (alpha_lora / rank) * B (A x)` is computed in float64 from the factors
    the gate loaded, so it does not depend on the candidate's code.
    """

    def __init__(self, model: torch.nn.Module, control: ControlState, factors: dict, scale: float) -> None:
        self.body = text_model(model)
        self.control = control
        self.factors = {name: (a.detach().double(), b.detach().double()) for name, (a, b) in factors.items()}
        self.scale = scale
        self.linears = {name: m for name, m in self.body.named_modules() if isinstance(m, torch.nn.Linear)}
        self.real: torch.Tensor | None = None
        self.worst = {"all": (0.0, None), "real": (0.0, None)}
        self.adapted: set[str] = set()
        self.calls = 0
        self.handles: list = []

    @contextlib.contextmanager
    def watching(self):
        """Watch the modules now at the linear modules' places, which an installed candidate may have replaced."""
        for name, original in self.linears.items():
            self.handles.append(_resolve(self.body, name).register_forward_hook(self._hook(name, original)))
        try:
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles = []

    def _note(self, key: str, value: float, name: str) -> None:
        if not value <= self.worst[key][0]:
            self.worst[key] = (value, name)

    def _hook(self, name: str, original: torch.nn.Linear):
        def hook(module, inputs, output):
            x, y = inputs[0].detach(), output.detach()
            base = F.linear(x, original.weight, original.bias).double()
            expected = base
            applies = self.control.adapter_applies()
            if applies and name in self.factors:
                a, b = self.factors[name]
                mask = self.control.mask_for(y).double()
                expected = base + mask.unsqueeze(-1) * self.scale * ((x.double() @ a.t()) @ b.t())
            difference = y.double() - expected
            self._note("all", _relative(difference, expected), name)
            real = self.real.bool() if self.real is not None else torch.ones(y.shape[:2], dtype=torch.bool,
                                                                             device=y.device)
            self._note("real", _relative(difference[real], expected[real]), name)
            if applies and _relative(y.double() - base, base) > TOLERANCE:
                self.adapted.add(name)
            self.calls += 1
        return hook


class DirectFormula:
    """The formula computed in this file, by hooks that replace each listed projection's output."""

    def __init__(self, model: torch.nn.Module, control: ControlState, factors: dict, scale: float) -> None:
        self.body = text_model(model)
        self.control = control
        self.scale = scale
        self.factors = {name: (a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True))
                        for name, (a, b) in factors.items()}

    @contextlib.contextmanager
    def installed(self):
        handles = []
        for name, (a, b) in self.factors.items():
            module = _resolve(self.body, name)

            def hook(m, inputs, output, a=a, b=b):
                x = inputs[0]
                mask = self.control.mask_for(output) if self.control.adapter_applies() else \
                    torch.zeros(output.shape[:2], device=output.device)
                update = mask.unsqueeze(-1) * self.scale * ((x.float() @ a.t()) @ b.t())
                return F.linear(x, m.weight, m.bias) + update.to(output.dtype)
            handles.append(module.register_forward_hook(hook))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def gradients(self) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
        return {name: (a.grad, b.grad) for name, (a, b) in self.factors.items()}


def _whole_sequences(prefilled: Prefilled, replies: list[list[int]], pad_id: int, rows: slice):
    """The rows' prompts and replies as one left-aligned call: ids, attention, read columns, reply columns."""
    prefix, attention = prefilled.prefix_ids[rows], prefilled.prefix_attention[rows]
    rest = prefilled.rest_ids[rows]
    chosen = replies[rows]
    width = max(len(r) for r in chosen)
    device = prefix.device
    tail = torch.full((len(chosen), width), pad_id, dtype=torch.long, device=device)
    tail_attention = torch.zeros((len(chosen), width), dtype=torch.long, device=device)
    for i, r in enumerate(chosen):
        tail[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        tail_attention[i, :len(r)] = 1
    ids = torch.cat([prefix, rest, tail], 1)
    full_attention = torch.cat([attention, torch.ones_like(rest), tail_attention], 1)
    reply_columns = torch.cat([torch.zeros_like(attention), torch.zeros_like(rest), tail_attention], 1)
    return ids, full_attention, prefilled.read_column[rows], reply_columns


def _whole_call(model: torch.nn.Module, control: ControlState, ids, attention, read_column,
                probe: OutputProbe | None) -> torch.Tensor:
    """One call over whole sequences with the control state set for it, as training calls the model."""
    control.set(masks.after_read_mask(attention, read_column, ids.shape[1]), adapter_active=True)
    if probe is not None:
        probe.real = attention.bool()
    try:
        positions = (attention.long().cumsum(-1) - 1).clamp(min=0)
        return text_model(model)(input_ids=ids, attention_mask=attention, position_ids=positions,
                                 use_cache=False).last_hidden_state
    finally:
        control.clear()
        if probe is not None:
            probe.real = None


def whole_sequence_outputs(model: torch.nn.Module, control: ControlState, prefilled: Prefilled,
                           replies: list[list[int]], pad_id: int, probe: OutputProbe) -> None:
    """Every row in one right-padded call over whole sequences, for the output check, with no gradient."""
    with torch.inference_mode():
        ids, attention, read_column, _ = _whole_sequences(prefilled, replies, pad_id, slice(None))
        _whole_call(model, control, ids, attention, read_column, probe)


def reply_loss_backward(model: torch.nn.Module, control: ControlState, prefilled: Prefilled,
                        replies: list[list[int]], pad_id: int, probe: OutputProbe | None = None) -> float:
    """Back-propagate the mean cross-entropy of the reply tokens over all rows; returns the loss.

    The rows go through in groups of `GRADIENT_ROWS`, each group one call over whole
    sequences; each group's loss is its share of the mean over every reply token of the batch.
    """
    total_tokens = sum(len(r) for r in replies)
    # Logits only at the positions that predict a reply token: the last prompt token onwards.
    first_reply = prefilled.prefix_ids.shape[1] + prefilled.rest_ids.shape[1]
    loss_value = 0.0
    for start in range(0, len(replies), GRADIENT_ROWS):
        ids, attention, read_column, reply_columns = _whole_sequences(prefilled, replies, pad_id,
                                                                      slice(start, start + GRADIENT_ROWS))
        hidden = _whole_call(model, control, ids, attention, read_column, probe)
        logits = model.lm_head(hidden[:, first_reply - 1:-1]).float()
        weights = reply_columns[:, first_reply:].float()
        per_token = F.cross_entropy(logits.transpose(1, 2), ids[:, first_reply:], reduction="none")
        loss = (per_token * weights).sum() / total_tokens
        loss.backward()
        loss_value += float(loss.detach())
    return loss_value


def _gradient_differences(candidate: GatedAdapter, reference: DirectFormula) -> dict:
    worst, worst_factor, zero_reference = 0.0, None, 0
    for name, (ga_ref, gb_ref) in reference.gradients().items():
        for which, got, want in (("A", candidate.a.get(name), ga_ref), ("B", candidate.b.get(name), gb_ref)):
            want = torch.zeros(1) if want is None else want
            if not bool(want.abs().sum() > 0):
                zero_reference += 1
            got_grad = torch.zeros_like(want) if got is None or got.grad is None else got.grad
            value = _relative(got_grad.double() - want.double(), want.double())
            if not value <= worst:
                worst, worst_factor = value, f"{name}.{which}"
    return {"max_relative_gradient_difference": worst, "factor_of_max": worst_factor,
            "factors_with_zero_reference_gradient": zero_reference}


def settings_scale() -> float:
    """The update's scale, `alpha_lora / rank`, from the settings the gate builds its adapter with.

    It is never read from an adapter, so a scale wrong in the code every implementation
    shares makes the candidate differ from the formula.
    """
    return setup.ALPHA_LORA / setup.RANK


def check_format(bench: setup.Bench, reference_adapter: GatedAdapter, implementation: str, batch: list[dict],
                 cap: int, plant: bool) -> dict:
    control = reference_adapter.control
    scale = settings_scale()
    factors = {name: (a.detach().clone(), b.detach().clone()) for name, (a, b) in reference_adapter.factors().items()}
    candidate = copy_in(reference_adapter, implementation, bench.model)
    if plant:
        candidate.modules = candidate.modules[:-1]
    declared = [name for name, _ in candidate.modules]
    probe = OutputProbe(bench.model, control, factors, scale)
    pipeline = Pipeline(bench.model, bench.tokenizer, control, adapter=candidate)
    try:
        with probe.watching():
            with torch.inference_mode():
                prefilled = pipeline.prefill(batch)
                replies = [reply.token_ids for reply in pipeline.generate(prefilled, None, cap)]
            whole_sequence_outputs(bench.model, control, prefilled, replies, pipeline.pad_id, probe)
            for parameter in candidate.parameters():
                parameter.grad = None
            candidate_loss = reply_loss_backward(bench.model, control, prefilled, replies, pipeline.pad_id, probe)
    finally:
        pipeline.close()
    direct = DirectFormula(bench.model, control, factors, scale)
    with direct.installed():
        reference_loss = reply_loss_backward(bench.model, control, prefilled, replies, pipeline.pad_id)
    out = {"episodes": len(batch), "declared_modules": declared, "observed_adapted": sorted(probe.adapted),
           "calls_checked": probe.calls,
           "max_relative_output_difference": probe.worst["all"][0], "module_of_max": probe.worst["all"][1],
           "max_relative_output_difference_at_real_tokens": probe.worst["real"][0],
           "module_of_max_at_real_tokens": probe.worst["real"][1],
           "candidate_loss": candidate_loss, "reference_loss": reference_loss,
           **_gradient_differences(candidate, direct)}
    for parameter in candidate.parameters():
        parameter.grad = None
    return out


@contextlib.contextmanager
def scale_halved():
    """The second planted fault: every adapter built while it is open scales its update by `alpha_lora / (2 rank)`.

    The fault sits in the code every implementation shares, so the adapter the gate draws
    its random factors through carries it as well as the candidate.
    """
    original = GatedAdapter.__init__

    def halved(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.scale = self.alpha_lora / (2 * self.rank)

    GatedAdapter.__init__ = halved
    try:
        yield
    finally:
        GatedAdapter.__init__ = original


def check(bench: setup.Bench, episodes: dict[str, list[dict]], implementation: str = "hooks",
          plant: bool = False, plant_scale: bool = False) -> dict:
    """The gate's values. `plant` leaves the last listed module out; `plant_scale` halves every adapter's scale."""
    with scale_halved() if plant_scale else contextlib.nullcontext():
        values = _check(bench, episodes, implementation, plant)
    values["planted_faults"] = [name for name, on in (("module_left_out", plant), ("scale_halved", plant_scale)) if on]
    return values


def _check(bench: setup.Bench, episodes: dict[str, list[dict]], implementation: str, plant: bool) -> dict:
    control = ControlState()
    measure = [e for fmt in episodes for e in episodes[fmt]]
    _, reference = setup.random_additions(bench, control, measure, writers_on=False)
    expected = specified_modules(bench.model, bench.layout.adapter_first, bench.layout.adapter_last)
    layers = sorted({int(name.split(".")[1]) for name in expected})
    values: dict = {"implementation": implementation, "specified_modules": len(expected), "layers": len(layers),
                    "projections_per_layer": {kind: len(SPECIFIED[kind] + SPECIFIED["all"])
                                              for kind in ("full_attention", "deltanet")},
                    "formats": {}}
    for fmt, batch in episodes.items():
        result = check_format(bench, reference, implementation, batch, REPLY_CAPS[fmt], plant)
        declared, observed = result.pop("declared_modules"), result.pop("observed_adapted")
        result["declared_set_matches"] = declared == expected
        result["observed_set_matches"] = observed == sorted(expected)
        result["missing_from_observed"] = sorted(set(expected) - set(observed))
        result["extra_in_observed"] = sorted(set(observed) - set(expected))
        values["formats"][fmt] = result
    values["green"] = all(
        v["declared_set_matches"] and v["observed_set_matches"]
        and v["max_relative_output_difference"] <= TOLERANCE
        and v["max_relative_gradient_difference"] <= TOLERANCE
        and v["factors_with_zero_reference_gradient"] == 0
        for v in values["formats"].values())
    return values


def main(argv=None) -> int:
    parser = setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), required=True)
    parser.add_argument("--plant-scale", action="store_true",
                        help="apply this gate's second planted fault, every adapter's scale halved; the gate must go red")
    args = parser.parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=True)
    episodes = setup.draw(setup.episode_file(args, "heldout"), EPISODES_PER_FORMAT, args.tail_position, args.limit)
    values = check(bench, episodes, implementation=args.implementation, plant=args.plant,
                   plant_scale=args.plant_scale)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
