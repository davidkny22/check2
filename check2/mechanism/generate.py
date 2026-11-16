"""Generation split at the read token: the one object every run and gate drives.

A batch of episodes is run in one cached forward pass split at the read token `r`:

1. `prefill` tokenises the prompts, left-pads the tokens `0..r` so that every row's read
   token sits in the last column, and runs the model on them with the mask all 0. The read
   layers' outputs over the window are kept for the reader.
2. `signal` gives `p`: the reader's output, the one-hot situation, or a tensor the caller
   supplies. The reader's output is computed whenever a reader is present. With writers
   that take the wide signal, every call that writes hands them `S = concat(p, n)`, with
   `n` from the output of layer 18 at each row's own read token, kept by `prefill` from the
   reader's kept outputs, or by a hook of its own when the pipeline has no reader.
3. `generate` continues from a copy of the cache over the remaining prompt tokens, then
   decodes one token per call. At every such token the write and the adapter act as the
   control state says.

After the read token every row has the same number of prompt tokens left, because the read
position is one relative index in a fixed template tail, so left-padding the prefix aligns
the whole batch. A run with no additions goes through the same calls.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from check2.data.episodes import END_OF_TURN
from check2.evaluation.scoring import APPENDED_READ_SUFFIX, SITUATIONS
from check2.mechanism import masks, wide_signal
from check2.mechanism.control import ControlState
from check2.mechanism.model_parts import decoder_layers, text_model
from check2.mechanism.write import Writers, install_write_hooks

if TYPE_CHECKING:
    from check2.mechanism.adapter.base import GatedAdapter

LETTERS = "ABCD"
# The reply cap of each format, in new tokens.
REPLY_CAPS = {"snap": 16, "short": 256}
USER_TURN_HEADER = "<|im_start|>user\n"


class PipelineError(RuntimeError):
    """A batch the pipeline cannot run as the split at the read token requires."""


@dataclass
class StepScores:
    """The two highest logits at each decoding step, and their token ids, in float32."""

    top_ids: torch.Tensor     # [steps, 2]
    top_values: torch.Tensor  # [steps, 2]

    @property
    def tokens(self) -> list[int]:
        return self.top_ids[:, 0].tolist()

    def margin(self, step: int) -> float:
        return float(self.top_values[step, 0] - self.top_values[step, 1])


@dataclass
class Reply:
    text: str
    token_ids: list[int]
    ended: bool
    scores: StepScores | None = None


@dataclass
class Prefilled:
    """A batch after the call over tokens `0..r`, ready to continue from its cache."""

    episodes: list[dict]
    prompt_ids: list[list[int]]
    prefix_ids: torch.Tensor        # [B, P], left-padded
    prefix_attention: torch.Tensor  # [B, P], 1 at real tokens
    rest_ids: torch.Tensor          # [B, R], the prompt tokens after r
    read_column: torch.Tensor       # [B]
    cache: object
    window_states: torch.Tensor | None = None  # [B, read layers, W, d], the read token last
    window_mask: torch.Tensor | None = None    # [B, W]
    # [B, d], the output of layer 18 at each row's read token; kept when the writers take the wide signal.
    read_state: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.prefix_ids.shape[0]


def _positions(attention: torch.Tensor, length: int) -> torch.Tensor:
    """Each token's position counted from its own row's first token, for the call's last columns."""
    return (attention.long().cumsum(-1) - 1).clamp(min=0)[:, -length:]


def window_start(token_ids: list[int], read_index: int, header_ids: tuple[int, ...]) -> int:
    """The index of the first token of the pushback turn's content: after the last user header before r."""
    size = len(header_ids)
    for i in range(read_index - size, -1, -1):
        if tuple(token_ids[i:i + size]) == header_ids:
            return i + size
    raise PipelineError("no user turn header before the read token")


def _choose(logits: torch.Tensor, sampling: dict | None, generator: torch.Generator | None) -> torch.Tensor:
    if sampling is None:
        return logits.argmax(-1)
    scores = logits / float(sampling["temperature"])
    top_k = sampling.get("top_k")
    if top_k:
        kth = torch.topk(scores, int(top_k), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    top_p = sampling.get("top_p")
    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(scores, descending=True, dim=-1)
        cumulative = ordered.softmax(-1).cumsum(-1)
        drop = cumulative - ordered.softmax(-1) >= float(top_p)
        ordered = ordered.masked_fill(drop, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, order, ordered)
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)


class Pipeline:
    """The model, its additions and the split at the read token, for generation and its checks."""

    def __init__(self, model: torch.nn.Module, tokenizer, control: ControlState, writers: Writers | None = None,
                 adapter: "GatedAdapter | None" = None, reader=None, forming=None) -> None:
        self.model, self.tokenizer, self.control = model, tokenizer, control
        self.writers, self.adapter, self.reader = writers, adapter, reader
        # An installed update that acts at and before the read token, with a control state of its own.
        self.forming = forming
        self.body = text_model(model)
        self.device = next(model.parameters()).device
        self.end_of_turn_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.end_of_turn_id
        self.wide = writers is not None and writers.wide
        if self.wide and reader is not None and writers.state_layer not in reader.layers:
            raise PipelineError(f"writers that take the wide signal need layer {writers.state_layer} at the read "
                                f"token, and the reader reads layers {list(reader.layers)}")
        self.handles = install_write_hooks(model, control, writers.layers) if writers is not None else []
        if adapter is not None:
            adapter.install()

    def close(self) -> None:
        """Remove every hook this pipeline installed."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.adapter is not None:
            self.adapter.remove()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # The forward call every path goes through.

    def _run(self, ids: torch.Tensor, attention: torch.Tensor, read_column: torch.Tensor, cache,
             vectors: dict[int, torch.Tensor], alpha: float, disabled_layers,
             use_cache: bool = True) -> tuple[torch.Tensor, object]:
        """One forward call over `ids`, the last columns of `attention`, with the control state set for it."""
        length = ids.shape[1]
        mask = masks.after_read_mask(attention, read_column, length)
        self.control.set(mask.to(self.device), vectors, alpha, disabled_layers,
                         adapter_active=self.adapter is not None)
        try:
            if self.forming is not None:
                self.forming.set_call(attention, read_column, length)
            out = self.body(input_ids=ids, attention_mask=attention, position_ids=_positions(attention, length),
                            past_key_values=cache, use_cache=use_cache)
        finally:
            self.control.clear()
            if self.forming is not None:
                self.forming.clear()
        return out.last_hidden_state, out.past_key_values

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden).to(torch.float32)

    def _vectors(self, p: torch.Tensor | None, prefilled: Prefilled) -> dict[int, torch.Tensor]:
        if self.writers is None:
            return {}
        if p is None:
            raise PipelineError("a pipeline with writers needs p")
        return self.writers(self.writer_signal(p, prefilled))

    def writer_signal(self, p: torch.Tensor, prefilled: Prefilled) -> torch.Tensor:
        """What the writers receive: `p`, or for wide writers `S = concat(p, n)` from each row's own read state."""
        if not self.wide:
            return p.to(self.device)
        if prefilled.read_state is None:
            raise PipelineError("writers that take the wide signal need the read token's state, and the batch has none")
        return wide_signal.signal(p.to(self.device), prefilled.read_state.to(self.device))

    # The call over tokens 0..r.

    def prefill(self, episodes: list[dict]) -> Prefilled:
        if not episodes:
            raise PipelineError("an empty batch")
        prompt_ids = self.tokenizer([e["prompt"] for e in episodes])["input_ids"]
        prefixes, rests, read_index = [], [], []
        for e, ids in zip(episodes, prompt_ids):
            if len(ids) != e["n_prompt_tokens"]:
                raise PipelineError(f"{e['episode_id']}: {len(ids)} prompt tokens, the record says "
                                    f"{e['n_prompt_tokens']}")
            r = e["read_index"]
            if not 0 <= r < len(ids) - 1:
                raise PipelineError(f"{e['episode_id']}: read index {r} leaves no prompt token after it")
            prefixes.append(ids[:r + 1])
            rests.append(ids[r + 1:])
            read_index.append(r)
        if len({len(rest) for rest in rests}) != 1:
            raise PipelineError("the rows have different numbers of prompt tokens after the read token")
        width = max(len(x) for x in prefixes)
        size = len(episodes)
        prefix_ids = torch.full((size, width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((size, width), dtype=torch.long)
        for b, x in enumerate(prefixes):
            prefix_ids[b, width - len(x):] = torch.tensor(x)
            attention[b, width - len(x):] = 1
        first = torch.tensor([width - len(x) for x in prefixes])
        read_column = masks.read_columns(first, torch.tensor(read_index))
        prefix_ids, attention = prefix_ids.to(self.device), attention.to(self.device)
        rest_ids = torch.tensor(rests, dtype=torch.long, device=self.device)

        captured: dict[int, torch.Tensor] = {}
        read_state: dict[str, torch.Tensor] = {}
        handles, window, window_mask = [], 0, None
        if self.reader is not None:
            header = tuple(self.tokenizer.encode(USER_TURN_HEADER, add_special_tokens=False))
            starts = [window_start(ids, r, header) for ids, r in zip(prompt_ids, read_index)]
            lengths = [r - s + 1 for r, s in zip(read_index, starts)]
            window = max(lengths)
            window_mask = torch.tensor([[0] * (window - n) + [1] * n for n in lengths], dtype=torch.bool,
                                       device=self.device)
            blocks = decoder_layers(self.model)
            for layer in self.reader.layers:
                def keep(module, args, output, layer=layer):
                    captured[layer] = output[:, -window:].detach().clone()
                handles.append(blocks[layer].register_forward_hook(keep))
        elif self.wide:
            # With no reader, the pipeline keeps layer 18 at the read token itself: the last column of the call.
            def keep_read(module, args, output):
                read_state["state"] = output[:, -1].detach().clone()
            handles.append(decoder_layers(self.model)[self.writers.state_layer].register_forward_hook(keep_read))
        try:
            with torch.inference_mode():
                _, cache = self._run(prefix_ids, attention, read_column, None, {}, 1.0, ())
        finally:
            for handle in handles:
                handle.remove()
        states = torch.stack([captured[layer] for layer in self.reader.layers], 1) if self.reader is not None else None
        state = None
        if self.wide:
            # Every row's prefix ends at its read token, so the read token is the last column of the window.
            state = (states[:, wide_signal.state_position(self.reader.layers, self.writers.state_layer), -1].clone()
                     if states is not None else read_state["state"])
        return Prefilled(episodes=list(episodes), prompt_ids=[list(x) for x in prompt_ids], prefix_ids=prefix_ids,
                         prefix_attention=attention, rest_ids=rest_ids, read_column=read_column, cache=cache,
                         window_states=states, window_mask=window_mask, read_state=state)

    # The signal.

    def signal(self, prefilled: Prefilled, source) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`(p, p_reader)`: the `p` to use, and the reader's output whenever a reader is present."""
        p_reader = None
        if self.reader is not None:
            with torch.inference_mode():
                p_reader = self.reader(prefilled.window_states, prefilled.window_mask).to(torch.float32)
        if isinstance(source, torch.Tensor):
            if tuple(source.shape) != (prefilled.batch_size, len(SITUATIONS)):
                raise PipelineError(f"a supplied p must be [{prefilled.batch_size}, 3], not {tuple(source.shape)}")
            p = source.to(torch.float32)
        elif source == "reader":
            if p_reader is None:
                raise PipelineError("the signal source is the reader, but the pipeline has none")
            p = p_reader
        elif source == "perfect":
            p = one_hot_situations(prefilled.episodes)
        else:
            raise PipelineError(f"unknown signal source {source!r}")
        return p.to(self.device), p_reader

    # Continuing from the cache.

    def prompt_pass(self, prefilled: Prefilled, p: torch.Tensor | None, alpha: float = 1.0,
                    disabled_layers=()) -> torch.Tensor:
        """The call over the prompt tokens after `r`; returns the logits at the last prompt token."""
        with torch.inference_mode():
            cache = copy.deepcopy(prefilled.cache)
            attention = torch.cat([prefilled.prefix_attention, torch.ones_like(prefilled.rest_ids)], 1)
            hidden, _ = self._run(prefilled.rest_ids, attention, prefilled.read_column, cache,
                                  self._vectors(p, prefilled), alpha, disabled_layers)
            return self._logits(hidden[:, -1])

    def _decode(self, prefilled: Prefilled, p, max_new_tokens: int, alpha: float, disabled_layers, sampling,
                forced: list[list[int]] | None, keep_scores: bool) -> list[tuple[list[int], bool, StepScores | None]]:
        size = prefilled.batch_size
        steps = max_new_tokens if forced is None else max((len(f) for f in forced), default=0)
        tokens: list[list[int]] = [[] for _ in range(size)]
        top_ids: list[list[torch.Tensor]] = [[] for _ in range(size)]
        top_values: list[list[torch.Tensor]] = [[] for _ in range(size)]
        done = [False] * size if forced is None else [len(f) == 0 for f in forced]
        ended = [False] * size
        generator = None
        if sampling is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(sampling["seed"]))
        with torch.inference_mode():
            cache = copy.deepcopy(prefilled.cache)
            vectors = self._vectors(p, prefilled)
            attention = prefilled.prefix_attention
            step_ids = prefilled.rest_ids
            for step in range(steps):
                attention = torch.cat([attention, torch.ones_like(step_ids)], 1)
                hidden, cache = self._run(step_ids, attention, prefilled.read_column, cache, vectors, alpha,
                                          disabled_layers)
                logits = self._logits(hidden[:, -1])
                if forced is None:
                    chosen = _choose(logits, sampling, generator)
                else:
                    chosen = torch.tensor([f[step] if step < len(f) else self.pad_id for f in forced],
                                          device=self.device)
                if keep_scores:
                    best = torch.topk(logits, 2, dim=-1)
                    best_ids, best_values = best.indices.cpu(), best.values.cpu()
                chosen_list = chosen.tolist()
                for b in range(size):
                    if done[b]:
                        continue
                    token = chosen_list[b]
                    tokens[b].append(token)
                    if keep_scores:
                        top_ids[b].append(best_ids[b])
                        top_values[b].append(best_values[b])
                    if forced is not None:
                        done[b] = len(tokens[b]) == len(forced[b])
                    elif token == self.end_of_turn_id:
                        done[b] = ended[b] = True
                if all(done) or step == steps - 1:
                    break
                fed = [self.pad_id if done[b] else chosen_list[b] for b in range(size)]
                step_ids = torch.tensor(fed, dtype=torch.long, device=self.device).unsqueeze(1)
        out = []
        for b in range(size):
            scores = None
            if keep_scores:
                scores = StepScores(torch.stack(top_ids[b]) if top_ids[b] else torch.zeros((0, 2), dtype=torch.long),
                                    torch.stack(top_values[b]) if top_values[b] else torch.zeros((0, 2)))
            out.append((tokens[b], ended[b], scores))
        return out

    def generate(self, prefilled: Prefilled, p: torch.Tensor | None, max_new_tokens: int, alpha: float = 1.0,
                 disabled_layers=(), sampling: dict | None = None, keep_scores: bool = False) -> list[Reply]:
        """Continue from the cache and decode, greedy unless `sampling` gives temperature, top_p, top_k, seed."""
        replies = []
        for token_ids, ended, scores in self._decode(prefilled, p, max_new_tokens, alpha, disabled_layers, sampling,
                                                     None, keep_scores):
            body = token_ids[:-1] if ended else token_ids
            replies.append(Reply(text=self.tokenizer.decode(body, skip_special_tokens=True), token_ids=token_ids,
                                 ended=ended, scores=scores))
        return replies

    def teacher_forced_logits(self, prefilled: Prefilled, p: torch.Tensor | None, continuation_ids: list[list[int]],
                              cached: bool = True, alpha: float = 1.0, disabled_layers=()) -> list[StepScores]:
        """The scores predicting each given continuation token, one row's list per batch row.

        With `cached`, the continuation is fed one token per call from the cache, as in
        generation. Without it, the whole sequence is recomputed in one call with no cache.
        """
        if len(continuation_ids) != prefilled.batch_size:
            raise PipelineError("one continuation per batch row is needed")
        if cached:
            return [scores for _, _, scores in self._decode(prefilled, p, 0, alpha, disabled_layers, None,
                                                             continuation_ids, True)]
        return self._recompute(prefilled, p, continuation_ids, alpha, disabled_layers)

    def _recompute(self, prefilled: Prefilled, p, continuation_ids: list[list[int]], alpha: float,
                   disabled_layers) -> list[StepScores]:
        size = prefilled.batch_size
        fed = [list(c[:-1]) for c in continuation_ids]
        width = max((len(f) for f in fed), default=0)
        tail = torch.full((size, width), self.pad_id, dtype=torch.long)
        tail_attention = torch.zeros((size, width), dtype=torch.long)
        for b, f in enumerate(fed):
            tail[b, :len(f)] = torch.tensor(f, dtype=torch.long)
            tail_attention[b, :len(f)] = 1
        ids = torch.cat([prefilled.prefix_ids, prefilled.rest_ids, tail.to(self.device)], 1)
        attention = torch.cat([prefilled.prefix_attention, torch.ones_like(prefilled.rest_ids),
                               tail_attention.to(self.device)], 1)
        start = prefilled.prefix_ids.shape[1] + prefilled.rest_ids.shape[1] - 1
        out = []
        with torch.inference_mode():
            hidden, _ = self._run(ids, attention, prefilled.read_column, None, self._vectors(p, prefilled), alpha,
                                  disabled_layers, use_cache=False)
            for b, c in enumerate(continuation_ids):
                logits = self._logits(hidden[b, start:start + len(c)])
                best = torch.topk(logits, 2, dim=-1)
                out.append(StepScores(best.indices.cpu(), best.values.cpu()))
        return out

    # The appended read of a reply that states no answer.

    def appended_read(self, prefilled: Prefilled, p: torch.Tensor | None, replies: list[Reply], alpha: float = 1.0,
                      disabled_layers=()) -> list[str]:
        """The option letter with the highest logit after `Final answer: (` is appended to prompt plus reply.

        The read continues from the prefix cache under the same additions, `p` and `alpha`
        as the reply.
        """
        scores = self.appended_letter_logits(prefilled, p, replies, alpha, disabled_layers)
        return [LETTERS[i] for i in scores.argmax(-1).tolist()]

    def appended_letter_logits(self, prefilled: Prefilled, p: torch.Tensor | None, replies: list[Reply],
                               alpha: float = 1.0, disabled_layers=()) -> torch.Tensor:
        """The logits `[B, 4]` of the letters A to D after `Final answer: (`, as the appended read sees them."""
        if len(replies) != prefilled.batch_size:
            raise PipelineError("one reply per batch row is needed")
        texts = [e["prompt"] + r.text + APPENDED_READ_SUFFIX for e, r in zip(prefilled.episodes, replies)]
        encoded = self.tokenizer(texts)["input_ids"]
        rows = []
        for e, ids, prompt in zip(prefilled.episodes, encoded, prefilled.prompt_ids):
            r = e["read_index"]
            if ids[:r + 1] != prompt[:r + 1]:
                raise PipelineError(f"{e['episode_id']}: the appended text changes the tokens up to the read token")
            rows.append(ids[r + 1:])
        size, width = len(rows), max(len(x) for x in rows)
        tail = torch.full((size, width), self.pad_id, dtype=torch.long)
        tail_attention = torch.zeros((size, width), dtype=torch.long)
        for b, x in enumerate(rows):
            tail[b, :len(x)] = torch.tensor(x, dtype=torch.long)
            tail_attention[b, :len(x)] = 1
        attention = torch.cat([prefilled.prefix_attention, tail_attention.to(self.device)], 1)
        letter_ids = [self.tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
        with torch.inference_mode():
            cache = copy.deepcopy(prefilled.cache)
            hidden, _ = self._run(tail.to(self.device), attention, prefilled.read_column, cache,
                                  self._vectors(p, prefilled), alpha, disabled_layers)
            last = hidden[torch.arange(size, device=self.device), torch.tensor([len(x) - 1 for x in rows],
                                                                                device=self.device)]
            return self._logits(last)[:, letter_ids]


def one_hot_situations(episodes: list[dict]) -> torch.Tensor:
    """The one-hot `p` of each episode's situation, in the order (user_right, hold, recheck)."""
    p = torch.zeros((len(episodes), len(SITUATIONS)), dtype=torch.float32)
    for b, e in enumerate(episodes):
        if e.get("situation") not in SITUATIONS:
            raise PipelineError(f"{e['episode_id']}: no situation to make a one-hot p from")
        p[b, SITUATIONS.index(e["situation"])] = 1.0
    return p
