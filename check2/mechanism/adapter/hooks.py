"""The consumer adapter computed in a forward hook on each adapted projection.

The hook sees the projection's input `x` and its output `W x`, and returns the output plus
`mask * (alpha_lora / rank) * B (A x)` in the output's dtype, reading the control state
when it runs.
"""

from __future__ import annotations

from check2.mechanism.adapter.base import GatedAdapter

ADAPTER_HOOK_TAG = "check2_adapter_module"


class HookAdapter(GatedAdapter):
    """The gated low-rank update, added by forward hooks on the named projections."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.handles: list = []

    def _hook(self, name: str):
        control = self.control

        def hook(module, inputs, output):
            if not control.adapter_applies():
                return None
            mask = control.mask_for(output)
            change = mask.unsqueeze(-1) * self.update(name, inputs[0])
            return output + change.to(output.dtype)

        setattr(hook, ADAPTER_HOOK_TAG, name)
        return hook

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("the adapter's hooks are already installed")
        self.handles = [module.register_forward_hook(self._hook(name)) for name, module in self.modules]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
