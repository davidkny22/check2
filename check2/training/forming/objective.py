"""The loss of the forming update, trained together with a copy of the reader.

    L = CE_reader(p, situation) + KL( softmax(logits_orig) || softmax(logits_formed) )

`p` is the reader copy's output on the formed model's states over the episode's window,
read at the read token; `CE_reader` is the reader's cross-entropy against the situation,
each episode's term weighted by its situation's class weight. The logits are the model's
next-token output at the last prompt token: the original model's with the update off and
without gradient, the formed model's with the update on. The write and the consumer adapter
are absent. The sequence is the whole prompt, left-padded, so that every row's last prompt
token sits in the last column and, since the read token is one index of a fixed template
tail, every row's read token sits in one column.

Each term is the mean over the step's episodes: a forward batch back-propagates the sum of
its episodes' terms divided by the step's episode count, so the step's gradient does not
depend on how the step is split into forward batches. The update's control state is set
for the formed pass and stays set until that pass's backward pass has finished.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from check2.mechanism.generate import USER_TURN_HEADER, _positions, window_start
from check2.mechanism.model_parts import decoder_layers, text_model
from check2.reader.model import OUTPUTS, Reader
from check2.training.forming.update import FormingUpdate
from check2.training.loss import batch_objective


class FormingStepError(ValueError):
    """An episode or a batch the forming step cannot lay out."""


@dataclass(frozen=True)
class FormingExample:
    """An episode's prompt as token ids, its read index, where its window starts, and its situation."""

    episode_id: str
    prompt_ids: tuple[int, ...]
    read_index: int
    window_start: int
    label: int


def make_example(episode: dict, tokenizer) -> FormingExample:
    """The forming example of one episode, with the read index the episode record carries."""
    ids = tuple(tokenizer([episode["prompt"]])["input_ids"][0])
    if len(ids) != episode["n_prompt_tokens"]:
        raise FormingStepError(f"{episode['episode_id']}: {len(ids)} prompt tokens, the record says "
                               f"{episode['n_prompt_tokens']}")
    r = episode["read_index"]
    if not 0 <= r < len(ids) - 1:
        raise FormingStepError(f"{episode['episode_id']}: read index {r} leaves no prompt token after it")
    header = tuple(tokenizer.encode(USER_TURN_HEADER, add_special_tokens=False))
    return FormingExample(episode_id=episode["episode_id"], prompt_ids=ids, read_index=r,
                          window_start=window_start(list(ids), r, header),
                          label=OUTPUTS.index(episode["situation"]))


@dataclass
class FormingBatch:
    """A forward batch of whole prompts, left-padded."""

    examples: list[FormingExample]
    ids: torch.Tensor            # [B, W]
    attention: torch.Tensor      # [B, W], 1 at real tokens
    read_column: torch.Tensor    # [B], one value repeated
    window_column: torch.Tensor  # [B], the column of each row's first window token
    labels: torch.Tensor         # [B]


def batch_tokens(batch: list[FormingExample]) -> int:
    """The tokens a forward batch carries gradient over: its rows times its padded width."""
    return len(batch) * max(len(e.prompt_ids) for e in batch)


def layout(examples: list[FormingExample], pad_id: int, device) -> FormingBatch:
    if not examples:
        raise FormingStepError("an empty batch")
    width = max(len(e.prompt_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.long)
    for b, e in enumerate(examples):
        ids[b, width - len(e.prompt_ids):] = torch.tensor(e.prompt_ids)
        attention[b, width - len(e.prompt_ids):] = 1
    first = torch.tensor([width - len(e.prompt_ids) for e in examples])
    read_column = first + torch.tensor([e.read_index for e in examples])
    if len(set(read_column.tolist())) != 1:
        raise FormingStepError("the rows' read tokens fall in different columns; their tails differ")
    window_column = first + torch.tensor([e.window_start for e in examples])
    labels = torch.tensor([e.label for e in examples])
    return FormingBatch(examples=list(examples), ids=ids.to(device), attention=attention.to(device),
                        read_column=read_column.to(device), window_column=window_column.to(device),
                        labels=labels.to(device))


def _last_logits(model: torch.nn.Module, batch: FormingBatch, read_layers=None):
    """The logits `[B, V]` in float32 at the last prompt token, and the read layers' outputs if asked."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    blocks = decoder_layers(model)
    for layer in read_layers or ():
        def keep(module, args, output, layer=layer):
            captured[layer] = output
        handles.append(blocks[layer].register_forward_hook(keep))
    try:
        width = batch.ids.shape[1]
        out = text_model(model)(input_ids=batch.ids, attention_mask=batch.attention,
                                position_ids=_positions(batch.attention, width), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    logits = model.lm_head(out.last_hidden_state[:, -1]).to(torch.float32)
    return logits, captured


def reader_window(captured: dict[int, torch.Tensor], layers: list[int],
                  batch: FormingBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """The reader's input: `h [B, layers, W, d]` ending at the read token, and `mask [B, W]`, true on window tokens."""
    read = int(batch.read_column[0])
    start = int(batch.window_column.min())
    h = torch.stack([captured[layer][:, start:read + 1] for layer in layers], 1)
    columns = torch.arange(start, read + 1, device=batch.ids.device)
    mask = columns.unsqueeze(0) >= batch.window_column.unsqueeze(1)
    return h, mask


@dataclass
class FormingLosses:
    """One forward batch's losses per episode, detached: the total, the reader's term and the KL term."""

    total: torch.Tensor
    cross_entropy: torch.Tensor
    kl: torch.Tensor


def episode_terms(reader_logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor,
                  reference_logits: torch.Tensor, formed_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each episode's weighted reader cross-entropy and its KL term, `[B]` each."""
    weights = class_weights.to(reader_logits.device)[labels]
    cross_entropy = weights * F.cross_entropy(reader_logits, labels, reduction="none")
    reference = F.log_softmax(reference_logits.detach(), dim=-1)
    kl = F.kl_div(F.log_softmax(formed_logits, dim=-1), reference, log_target=True, reduction="none").sum(-1)
    return cross_entropy, kl


def forming_step(model: torch.nn.Module, update: FormingUpdate, reader: Reader, examples: list[FormingExample],
                 class_weights: torch.Tensor, step_size: int, pad_id: int) -> FormingLosses:
    """One forward batch of a forming step: the reference pass, the formed pass and its backward pass.

    Gradients accumulate on the update's factors and the reader copy's weights.
    """
    batch = layout(examples, pad_id, update.device)
    update.clear()
    with torch.no_grad():
        reference, _ = _last_logits(model, batch)
    update.set_call(batch.attention, batch.read_column, batch.ids.shape[1])
    try:
        formed, captured = _last_logits(model, batch, reader.layers)
        h, mask = reader_window(captured, reader.layers, batch)
        cross_entropy, kl = episode_terms(reader.logits(h, mask), batch.labels, class_weights, reference, formed)
        del captured, h
        batch_objective(cross_entropy + kl, step_size).backward()
    finally:
        update.clear()
    cross_entropy, kl = cross_entropy.detach(), kl.detach()
    return FormingLosses(total=cross_entropy + kl, cross_entropy=cross_entropy, kl=kl)
