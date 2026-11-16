"""The consumer adapter computed by a library's position-gated adapter: Activated LoRA in `peft`.

`peft` places Activated LoRA layers on the named projections. Such a layer adds
`(lora_alpha / r) * B (A x)` only on the last `offset` tokens of each row of a forward call,
where `offset` is passed to the layer with the call. Here the offset of each row is taken
from the control state's mask when the call runs: the number of the call's tokens from the
row's first token after the read to the end of the call. In every call the pipeline makes
over a left-padded batch, the tokens after the read are exactly that last stretch of each
row, so the update is the one the mask gives. The factors are the adapter's own: the
library's layers hold the same `A` and `B` tensors. `remove` puts the original projections
back and leaves the model as it was.
"""

from __future__ import annotations

import re

import torch

from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.model_parts import decoder_layers

LIBRARY = "peft"
ADAPTER_NAME = "default"
OFFSETS_ARGUMENT = "alora_offsets"
# The library switches its gated variant on only when an invocation sequence is configured.
# Its value is never used here, because every call passes its offsets explicitly.
UNUSED_INVOCATION_TOKENS = [0]


class LibraryUnavailable(RuntimeError):
    """The library, or the position-gated adapter in it, cannot be found in this process."""


def _library():
    try:
        import peft
        from peft import LoraConfig, inject_adapter_in_model
        from peft.tuners.lora.variants import ALoraLinearVariant  # noqa: F401  (its presence is the check)
    except ImportError as error:
        raise LibraryUnavailable(f"{LIBRARY} with its Activated LoRA variant cannot be imported: {error}") from error
    return peft, LoraConfig, inject_adapter_in_model


def row_offsets(mask: torch.Tensor) -> list[int]:
    """Per row, the number of the call's tokens from the first one after the read to the end of the call."""
    after = mask.bool()
    length = after.shape[1]
    first = torch.where(after.any(dim=1), after.int().argmax(dim=1), torch.full_like(after[:, 0], length,
                                                                                    dtype=torch.long))
    return (length - first).tolist()


class LibraryAdapter(GatedAdapter):
    """The gated low-rank update, added by the library's Activated LoRA layers."""

    def __init__(self, model: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.model = model
        self.replaced: list[tuple[torch.nn.Module, str, torch.nn.Linear]] = []
        self.handles: list = []
        self._added_attributes: list[str] = []
        self._offsets_for: torch.Tensor | None = None
        self._offsets: list[int] = []

    def _parent(self, name: str) -> tuple[torch.nn.Module, str]:
        _, index, *path = name.split(".")
        parent = decoder_layers(self.model)[int(index)]
        for part in path[:-1]:
            parent = getattr(parent, part)
        return parent, path[-1]

    def _offsets_now(self, batch: int) -> list[int]:
        """The offsets for the call that is running, worked out once per call from the control state."""
        control = self.control
        if not control.adapter_applies():
            return [0] * batch
        if self._offsets_for is not control.mask:
            self._offsets_for = control.mask
            self._offsets = row_offsets(control.mask)
        return self._offsets

    def _pre_hook(self, module, args, kwargs):
        x = args[0]
        if self.control.adapter_applies():
            self.control.mask_for(x)  # refuses a mask set for another call
        kwargs[OFFSETS_ARGUMENT] = self._offsets_now(x.shape[0])
        return args, kwargs

    def install(self) -> None:
        if self.replaced:
            raise RuntimeError("the adapter's library layers are already installed")
        _, LoraConfig, inject_adapter_in_model = _library()
        full_names = {id(module): full for full, module in self.model.named_modules()}
        targets = [full_names[id(projection)] for _, projection in self.modules]
        pattern = "|".join(re.escape(t) for t in targets)
        config = LoraConfig(r=self.rank, lora_alpha=self.alpha_lora, lora_dropout=0.0, target_modules=pattern,
                            bias="none", use_rslora=False, task_type="CAUSAL_LM",
                            alora_invocation_tokens=UNUSED_INVOCATION_TOKENS)
        before = set(vars(self.model))
        for name, projection in self.modules:
            self.replaced.append((*self._parent(name), projection))
        inject_adapter_in_model(config, self.model, adapter_name=ADAPTER_NAME)
        self._added_attributes = sorted(set(vars(self.model)) - before)
        for name, projection in self.modules:
            parent, attribute = self._parent(name)
            layer = getattr(parent, attribute)
            if getattr(layer, "base_layer", None) is not projection:
                raise RuntimeError(f"the library did not place its layer on {name}")
            if layer.scaling[ADAPTER_NAME] != self.scale:
                raise RuntimeError(f"the library's scale on {name} is {layer.scaling[ADAPTER_NAME]}, not {self.scale}")
            layer.lora_A[ADAPTER_NAME].weight = self.a[name]
            layer.lora_B[ADAPTER_NAME].weight = self.b[name]
            self.handles.append(layer.register_forward_pre_hook(self._pre_hook, with_kwargs=True))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for parent, attribute, projection in self.replaced:
            setattr(parent, attribute, projection)
        self.replaced = []
        for attribute in self._added_attributes:
            delattr(self.model, attribute)
        self._added_attributes = []
        self._offsets_for, self._offsets = None, []
