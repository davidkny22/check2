"""The reader: hidden states over the window to `p`, three probabilities in the order of `OUTPUTS`.

    x[t] = concat over l of Linear_l(LayerNorm_l(h[l][t]))          each d -> 64
    a    = softmax over the window of (x[t] . u / sqrt(64 * |layers|))
    z    = concat(sum_t a[t] * x[t], x[r])
    p    = softmax(Linear_2(Dropout_0.1(GELU(Linear_1(z)))))        2 * 64 * |layers| -> 128 -> 3

`h` is `[B, len(layers), W, d]`, right-aligned so the read token `r` is the last column, and
`mask` is `[B, W]`, true on window tokens. Without pooling, `z` is `x[r]` alone. The linear
probe is one linear map from the read token's states to the three logits.
"""
import math

import torch
from torch import nn

OUTPUTS = ("user_right", "hold", "recheck")
PROJECTED = 64
HIDDEN = 128
DROPOUT = 0.1
DEFAULT_VARIANT = "six_layers"
VARIANTS = {
    "linear_probe": {"layers": [18], "pooled": False, "linear_probe": True},
    "six_layers": {"layers": [14, 15, 18, 19, 22, 23], "pooled": True, "linear_probe": False},
    "deltanet_only": {"layers": [14, 18, 22], "pooled": True, "linear_probe": False},
    "attention_only": {"layers": [15, 19, 23], "pooled": True, "linear_probe": False},
}


class Reader(nn.Module):
    def __init__(self, layers: list[int], hidden_size: int, pooled: bool = True, linear_probe: bool = False):
        super().__init__()
        self.layers = list(layers)
        self.hidden_size = hidden_size
        self.pooled = pooled
        self.linear_probe = linear_probe
        n = len(self.layers)
        if linear_probe:
            self.probe = nn.Linear(n * hidden_size, len(OUTPUTS))
            return
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(n))
        self.projections = nn.ModuleList(nn.Linear(hidden_size, PROJECTED) for _ in range(n))
        width = PROJECTED * n
        if pooled:
            self.u = nn.Parameter(torch.zeros(width))
        self.linear_1 = nn.Linear(2 * width if pooled else width, HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear_2 = nn.Linear(HIDDEN, len(OUTPUTS))

    def settings(self) -> dict:
        return {"layers": self.layers, "hidden_size": self.hidden_size, "pooled": self.pooled,
                "linear_probe": self.linear_probe}

    def logits(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if h.shape[1] != len(self.layers):
            raise ValueError(f"h holds {h.shape[1]} layers; this reader reads {len(self.layers)}")
        h = h.float()
        mask = mask.bool()
        if self.linear_probe:
            return self.probe(h[:, :, -1].flatten(1))
        x = torch.cat([proj(norm(h[:, i])) for i, (norm, proj)
                       in enumerate(zip(self.norms, self.projections))], dim=-1)    # B, W, 64n
        x = torch.where(mask[:, :, None], x, torch.zeros((), dtype=x.dtype, device=x.device))
        x_read = x[:, -1]
        if self.pooled:
            score = (x @ self.u) / math.sqrt(x.shape[-1])
            score = score.masked_fill(~mask, float("-inf"))
            a = torch.softmax(score, dim=-1)
            z = torch.cat([(a[:, :, None] * x).sum(dim=1), x_read], dim=-1)
        else:
            z = x_read
        return self.linear_2(self.dropout(nn.functional.gelu(self.linear_1(z))))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h, mask), dim=-1)
