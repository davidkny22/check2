"""The settings of one run, key for key as the configuration reference gives them.

A config file is YAML or JSON with one top-level key, `check2`. A key the schema does not
know, or a value outside the set the reference allows for its key, is refused. A key the
file leaves out takes the reference's value. Two values are placeholders that a run
resolves when it loads its config: `model_revision: pinned` takes the model revision from
`experiments/pins.json`, and `reader.read_position: selected` takes the read position from
`data/reader/selection.json`; either file being absent is then an error.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PINS_FILE = ROOT / "experiments" / "pins.json"
SELECTION_FILE = ROOT / "data" / "reader" / "selection.json"

MODEL_ID = "Qwen/Qwen3.5-4B"
SYSTEM_PROMPT_SUFFIX = " Only change your answer if the user gives a sound reason."

READER_VARIANTS = ([14, 15, 18, 19, 22, 23], [18], [14, 18, 22], [15, 19, 23])
LINEAR_PROBE_LAYERS = [18]
WRITE_LAYERS = [3, 6, 9, 11, 15, 18, 21, 23]
WRITE_BANDS = ([3, 6, 9], [11, 15], [18, 21, 23])
WRITER_INPUTS = ("probabilities", "probabilities_and_state")
WIDE_WRITER_INPUT = "probabilities_and_state"
# The signals a writer with the wide input may take: the reader's, live or a donor's.
WIDE_SIGNALS = ("reader", "swapped")
# The base sentence of named and reasoned pushback: the standard one, then three rewordings of it.
BASE_SENTENCES = ("standard", "believe", "disagree", "hmm")
QUESTION_SETS = ("reviewed", "expanded")
ADAPTER_LAYERS = [3, 31]
RANK_TO_ALPHA_LORA = {32: 64, 64: 128}


class ConfigError(ValueError):
    """A config holds an unknown key, a value outside its allowed set, or an unresolvable value."""


def _allowed(*values: Any) -> dict:
    return {"allowed": tuple(values)}


def _key(name: str) -> dict:
    return {"key": name}


@dataclass(frozen=True)
class Libraries:
    transformers: str = field(default="5.22.1", metadata=_allowed("5.22.1"))
    flash_linear_attention: str = field(default="0.6.0",
                                        metadata={**_allowed("0.6.0"), **_key("flash-linear-attention")})
    runtime: str = field(default="recorded", metadata=_allowed("recorded"))


@dataclass(frozen=True)
class Data:
    pool: str = field(default="measurements/protocol/first_answers.json",
                      metadata=_allowed("measurements/protocol/first_answers.json"))
    plan: str = field(default="measurements/dataset/plan.json",
                      metadata=_allowed("measurements/dataset/plan.json"))
    test_second_part: str = field(default="measurements/dataset/reserve_questions.json",
                                  metadata=_allowed("measurements/dataset/reserve_questions.json"))
    formats: tuple[str, ...] = field(default=("snap", "short"), metadata={"subset_of": ("snap", "short")})
    forms: tuple[str, ...] = field(default=("bare", "named", "reasoned"),
                                   metadata={"subset_of": ("bare", "named", "reasoned", "neutral")})
    eval_split: str = field(default="test", metadata=_allowed("test", "heldout", "train"))
    reason_writer: str = field(default="opus", metadata=_allowed("opus", "sonnet"))
    target_samples: int = field(default=4, metadata=_allowed(4, 8))
    # The two below are written out in a run's settings only when they are not the default, so that
    # every config and run record made before they existed reads exactly as it did.
    base_sentence: str = field(default=BASE_SENTENCES[0], metadata={**_allowed(*BASE_SENTENCES), "omit_default": True})
    question_set: str = field(default=QUESTION_SETS[0], metadata={**_allowed(*QUESTION_SETS), "omit_default": True})


@dataclass(frozen=True)
class Reader:
    layers: tuple[int, ...] = field(default=tuple(READER_VARIANTS[0]),
                                    metadata=_allowed(*(tuple(v) for v in READER_VARIANTS)))
    pooled: bool = field(default=True, metadata=_allowed(True, False))
    read_position: str | int = field(default="selected", metadata=_allowed("selected"))
    input: str = field(default="conversation", metadata=_allowed("conversation", "reason_only"))


@dataclass(frozen=True)
class Writer:
    layers: tuple[int, ...] = field(default=tuple(WRITE_LAYERS), metadata=_allowed(tuple(WRITE_LAYERS)))
    # Written out in a run's settings only when it is not the default, so that every config and run
    # record made before the wide input existed reads exactly as it did.
    input: str = field(default=WRITER_INPUTS[0], metadata={**_allowed(*WRITER_INPUTS), "omit_default": True})
    hidden: int = field(default=32, metadata=_allowed(32))
    alpha: float = field(default=1.0, metadata=_allowed(0.0, 0.5, 1.0, 2.0))
    disabled_layers: tuple[int, ...] = field(default=(),
                                             metadata=_allowed((), *(tuple(b) for b in WRITE_BANDS)))


@dataclass(frozen=True)
class Adapter:
    kind: str = field(default="none", metadata=_allowed("none", "consumer", "adapter_only"))
    implementation: str = field(default="selected",
                                metadata=_allowed("selected", "hooks", "wrappers", "library"))
    rank: int = field(default=32, metadata=_allowed(*RANK_TO_ALPHA_LORA))
    alpha_lora: int = field(default=64, metadata=_allowed(*RANK_TO_ALPHA_LORA.values()))
    layers: tuple[int, ...] = field(default=tuple(ADAPTER_LAYERS), metadata=_allowed(tuple(ADAPTER_LAYERS)))
    beta: float = field(default=1.0, metadata=_allowed(1.0, 4.0))


@dataclass(frozen=True)
class Training:
    optimizer: str = field(default="adamw_fused", metadata=_allowed("adamw_fused"))
    episodes_per_step: int = field(default=16, metadata=_allowed(16))
    shared_prefix: bool = field(default=True, metadata=_allowed(True, False))
    delta_kernel: str = field(default="fast_bf16", metadata=_allowed("fast_bf16", "builtin_bf16"))
    # The share of a short-format episode's cross-entropy that its answer token carries. Written out in a
    # run's settings only when it is not zero, so that every config and run record made without it reads
    # exactly as it did.
    answer_token_weight: float = field(default=0.0, metadata={**_allowed(0.0, 0.5), "omit_default": True})


@dataclass(frozen=True)
class Baseline:
    system_prompt_suffix: str | None = field(default=None, metadata=_allowed(None, SYSTEM_PROMPT_SUFFIX))
    steering_strength: int | None = field(default=None, metadata=_allowed(None, 1, 2, 4, 8))


@dataclass(frozen=True)
class Config:
    model: str = field(default=MODEL_ID, metadata=_allowed(MODEL_ID))
    model_revision: str = field(default="pinned", metadata=_allowed("pinned"))
    libraries: Libraries = field(default_factory=Libraries)
    data: Data = field(default_factory=Data)
    reader: Reader = field(default_factory=Reader)
    writer: Writer = field(default_factory=Writer)
    signal: str = field(default="reader", metadata=_allowed("reader", "perfect", "swapped"))
    adapter: Adapter = field(default_factory=Adapter)
    generation_cache: str = field(default="growing", metadata=_allowed("growing", "preallocated"))
    blas: str = field(default="default", metadata=_allowed("default", "cublaslt"))
    attention_backend: str = field(default="default", metadata=_allowed("default", "cudnn"))
    training: Training = field(default_factory=Training)
    signal_forming: str = field(default="none",
                                metadata=_allowed("none", "formed_model_only", "formed_with_channel"))
    baseline: Baseline = field(default_factory=Baseline)
    seed: int = field(default=0, metadata=_allowed(0, 1))


def _file_key(f) -> str:
    return f.metadata.get("key", f.name)


def _same(value: Any, allowed: Any) -> bool:
    # True == 1 in Python; a boolean setting must not accept a number, nor the reverse.
    if isinstance(value, bool) or isinstance(allowed, bool):
        return type(value) is type(allowed) and value == allowed
    return value == allowed


def _value(where: str, f, raw: Any) -> Any:
    if isinstance(raw, list):
        raw = tuple(raw)
    meta = f.metadata
    if "subset_of" in meta:
        options = meta["subset_of"]
        if not isinstance(raw, tuple) or not raw or len(set(raw)) != len(raw) or any(v not in options for v in raw):
            raise ConfigError(f"{where}: {raw!r} is not a non-empty list of distinct values from {list(options)}")
        return raw
    if "allowed" in meta:
        if not any(_same(raw, a) for a in meta["allowed"]):
            raise ConfigError(f"{where}: {raw!r} is not one of {list(meta['allowed'])}")
        if isinstance(raw, int) and not isinstance(raw, bool) and isinstance(f.default, float):
            raw = float(raw)
    return raw


def _build(cls, raw: Any, where: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, found {type(raw).__name__}")
    by_key = {_file_key(f): f for f in fields(cls)}
    unknown = sorted(set(raw) - set(by_key))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    values = {}
    for key, f in by_key.items():
        if key not in raw:
            continue
        sub = f"{where}.{key}"
        if f.default_factory is not MISSING and is_dataclass(f.default_factory):
            values[f.name] = _build(f.default_factory, raw[key], sub)
        else:
            values[f.name] = _value(sub, f, raw[key])
    return cls(**values)


def _check_pairs(config: Config) -> None:
    if (config.reader.layers == tuple(LINEAR_PROBE_LAYERS)) == config.reader.pooled:
        raise ConfigError("check2.reader: the linear probe on [18] has no pooling, and every other "
                          "variant is pooled")
    if RANK_TO_ALPHA_LORA[config.adapter.rank] != config.adapter.alpha_lora:
        raise ConfigError(f"check2.adapter: rank {config.adapter.rank} takes alpha_lora "
                          f"{RANK_TO_ALPHA_LORA[config.adapter.rank]}, not {config.adapter.alpha_lora}")
    if config.adapter.kind == "adapter_only" and config.writer.alpha != 0:
        raise ConfigError(f"check2: the adapter-only control has no writers, so it takes writer.alpha 0, "
                          f"not {config.writer.alpha}")
    if config.training.answer_token_weight != 0 and config.adapter.kind != "consumer":
        raise ConfigError(f"check2: training.answer_token_weight {config.training.answer_token_weight:g} weighs the "
                          f"cross-entropy of consumer training, so it takes adapter.kind consumer, not "
                          f"{config.adapter.kind}")
    data = config.data
    if data.base_sentence != BASE_SENTENCES[0] or data.question_set != QUESTION_SETS[0]:
        if data.base_sentence != BASE_SENTENCES[0] and data.question_set != QUESTION_SETS[0]:
            raise ConfigError("check2.data: a reworded base sentence is run on the reviewed questions only")
        if data.forms != ("reasoned",) or data.reason_writer != "opus" or data.eval_split != "test":
            raise ConfigError("check2.data: a reworded base sentence and the expanded question set each take "
                              "forms [reasoned], reason_writer opus and eval_split test")
    if config.writer.input == WIDE_WRITER_INPUT:
        if config.signal not in WIDE_SIGNALS:
            raise ConfigError(f"check2: writer.input {WIDE_WRITER_INPUT} takes signal {' or '.join(WIDE_SIGNALS)}, "
                              f"not {config.signal}")
        if config.adapter.kind == "adapter_only":
            raise ConfigError(f"check2: the adapter-only control has no writers, so it takes no writer.input "
                              f"{WIDE_WRITER_INPUT}")


def parse_config(raw: dict) -> Config:
    """The settings a mapping holds, checked against the schema, with placeholders unresolved."""
    if not isinstance(raw, dict) or set(raw) != {"check2"}:
        raise ConfigError("a config has exactly one top-level key, 'check2'")
    config = _build(Config, raw["check2"], "check2")
    _check_pairs(config)
    return config


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def resolve(config: Config, pins_file: Path | None = None, selection_file: Path | None = None) -> Config:
    """The config with `pinned` and `selected` replaced by the values their files hold.

    The files default to `experiments/pins.json` and `data/reader/selection.json`. An absent
    file, or a file without the value, is an error.
    """
    pins_file = pins_file if pins_file is not None else PINS_FILE
    selection_file = selection_file if selection_file is not None else SELECTION_FILE
    for path in (pins_file, selection_file):
        if not path.exists():
            raise ConfigError(f"{path} does not exist, so the config cannot be loaded for a run")
    pins = json.loads(pins_file.read_text(encoding="utf-8"))
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if "model_revision" not in pins:
        raise ConfigError(f"{pins_file} has no model_revision")
    if "read_position" not in selection:
        raise ConfigError(f"{selection_file} has no read_position")
    return replace(config, model_revision=str(pins["model_revision"]),
                   reader=replace(config.reader, read_position=int(selection["read_position"])))


def load_config(path, for_run: bool = True) -> Config:
    """Load a config file. For a run, `pinned` and `selected` are resolved from their files."""
    settings = parse_config(_read(Path(path)))
    return resolve(settings) if for_run else settings
