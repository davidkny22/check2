"""A run's role (Original, Check², Off and so on) and the episode set it covers, from its settings alone.

The role starts from the additions the settings switch on. Where the signal, the seed, the
adapter's rank or `beta` is not the main runs' value, the role names it, so that two runs
that differ in any of them never read as one role: the perfect-signal connector, the
second seed's connector, the retraining at rank 64 and the runs at `beta = 4` each have a
role of their own. The main runs' values are the configuration's defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from check2.config.schema import Config

NEUTRAL_FOLLOW_UP = "neutral follow-up"
MAIN = Config()
# The roles that already name a signal of their own.
SIGNAL_ROLES = {"perfect": "perfect signal", "swapped": "swap"}


def _setting(config: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config.get("check2", config)
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            return default
        value = value[name]
    return value


def _variant(config: Mapping[str, Any]) -> str:
    variant = []
    if "neutral" in (_setting(config, "data", "forms", default=[]) or []):
        variant.append(NEUTRAL_FOLLOW_UP)
    if _setting(config, "data", "reason_writer", default=MAIN.data.reason_writer) != MAIN.data.reason_writer:
        variant.append("cross-writer")
    base_sentence = _setting(config, "data", "base_sentence", default=MAIN.data.base_sentence)
    if base_sentence != MAIN.data.base_sentence:
        variant.append(f"base sentence {base_sentence}")
    if _setting(config, "data", "question_set", default=MAIN.data.question_set) != MAIN.data.question_set:
        variant.append("expanded pairs")
    forming = _setting(config, "signal_forming", default=MAIN.signal_forming)
    if forming != MAIN.signal_forming:
        variant.append(f"signal forming {forming}")
    return ", ".join(variant)


def base_role(config: Mapping[str, Any]) -> str:
    """The role the run's additions give it, before the settings it differs in from the main runs are named.

    `writer.alpha` and `adapter.kind` must be present.
    """
    alpha = _setting(config, "writer", "alpha")
    kind = _setting(config, "adapter", "kind")
    if alpha is None or kind is None:
        raise ValueError("a run's configuration must set writer.alpha and adapter.kind")
    suffix = _setting(config, "baseline", "system_prompt_suffix")
    strength = _setting(config, "baseline", "steering_strength")
    signal = _setting(config, "signal", default=MAIN.signal)
    disabled = _setting(config, "writer", "disabled_layers", default=[]) or []
    if suffix is not None:
        return "system prompt"
    if strength is not None:
        return f"steering g={strength}"
    if kind == "none":
        return "original" if alpha == 0 else "connector"
    if kind == "adapter_only":
        return "adapter only"
    if signal in SIGNAL_ROLES:
        return SIGNAL_ROLES[signal]
    if disabled:
        return f"bands off {sorted(disabled)}"
    if alpha == 0:
        return "off"
    if alpha != 1:
        return f"dial alpha={alpha}"
    return "check2"


def _named_differences(config: Mapping[str, Any], role: str) -> list[str]:
    """`name=value` for each of signal, seed, rank, `beta`, the writer's input and the answer token's weight
    that is not the main runs' value."""
    named = []
    signal = _setting(config, "signal", default=MAIN.signal)
    if signal != MAIN.signal and SIGNAL_ROLES.get(signal) != role:
        named.append(f"signal={signal}")
    for name, value, main in (("seed", _setting(config, "seed", default=MAIN.seed), MAIN.seed),
                              ("rank", _setting(config, "adapter", "rank", default=MAIN.adapter.rank),
                               MAIN.adapter.rank),
                              ("beta", _setting(config, "adapter", "beta", default=MAIN.adapter.beta),
                               MAIN.adapter.beta)):
        if value != main:
            named.append(f"{name}={value}")
    writer_input = _setting(config, "writer", "input", default=MAIN.writer.input)
    if writer_input != MAIN.writer.input:
        named.append(f"input={writer_input}")
    answer_weight = _setting(config, "training", "answer_token_weight", default=MAIN.training.answer_token_weight)
    if answer_weight != MAIN.training.answer_token_weight:
        named.append(f"answer_weight={answer_weight}")
    return named


FORMED_MODEL = "signal forming formed_model_only"
FORMED_WITH_CHANNEL = "signal forming formed_with_channel"


def reference_variant(variant: str) -> str:
    """The variant of the run a run is compared with: the formed model for the second stage of signal
    forming, the main episodes' Original run for the formed model, and its own variant otherwise."""
    if FORMED_WITH_CHANNEL in variant:
        return variant.replace(FORMED_WITH_CHANNEL, FORMED_MODEL)
    if FORMED_MODEL in variant:
        return ", ".join(part for part in variant.split(", ") if part != FORMED_MODEL)
    return variant


def named_settings(config: Mapping[str, Any]) -> tuple[str, ...]:
    """`name=value` for each setting the run differs in from the main runs, as its role names them."""
    return tuple(_named_differences(config, base_role(config)))


def run_role(config: Mapping[str, Any]) -> tuple[str, str]:
    """The run's role and the episode set it covers, from its configuration's settings.

    The role is the base role followed by each setting the run differs in from the main
    runs, as `name=value`. The variant is empty for the main episodes, or names the
    neutral follow-up, the cross-writer set or signal forming.
    """
    role = base_role(config)
    return " ".join([role, *_named_differences(config, role)]), _variant(config)
