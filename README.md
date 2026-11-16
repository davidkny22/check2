# Check²

Check² ("check twice") is a mechanism for a language model to read its own judgment of a
user's pushback and act on it before it answers. It was built to test whether a model can
check itself from the inside. On Qwen3.5-4B, a small reader takes that judgment from the
model's hidden states as three numbers, small writers add it back into the model at every
token of the reply, and an adapter is trained to act on it. The signal and its write are
called the channel.

The judgment is there to read. On questions the model first got wrong, the reader tells a
right pushback from a wrong one, each with a written reason, at 0.87 on a score where 0.5
is chance (AUROC). The model's
confidence scores 0.50 to 0.67 on the same questions. On 783 test pairs with matched right and wrong
written reasons, Check² cut the adoption of wrong answers from 84.6% to 12.2% in one-letter
replies and from 34.6% to 26.0% in short replies, and the change runs through the signal.
The cost in accepted right corrections stayed within a limit set in advance only in short
replies. A plain adapter matched the channel's rates. What the channel adds is a signal
that can be read, switched off, scaled and overwritten.

## The terms

A model answers a multiple-choice question, a user pushes back once with a written reason,
and the model replies. A pushback is *right* when the option the user pushes toward is the
answer key's, and *wrong* otherwise. *Wrong adoption* is the share of wrong pushbacks after
which the model ends on the user's option, and *right adoption* is that share for right
pushbacks. The pushback asks for the reply in one of two formats: one letter, or under 100
words ending in a stated final answer.

## Results

The table gives adoptions under reasoned pushback on the test set, as counts out of 534
wrong and 249 right pushbacks in each format. † marks a condition defined after the test results were first
read.

| Condition | One-letter: wrong adopted | One-letter: right adopted | Short: wrong adopted | Short: right adopted |
|---|---:|---:|---:|---:|
| Original model | 452 | 246 | 185 | 241 |
| System sentence | 434 | 243 | 154 | 239 |
| Steering ×2 | 460 | 243 | 148 | 234 |
| Adapter only | 77 | 197 | 142 | 233 |
| Check² | 65 | 189 | 139 | 232 |
| Check², second seed | 46 | 157 | 131 | 234 |
| Check² at half strength | 141 | 226 | 143 | 237 |
| Check², answer weighted† | 67 | 199 | 117 | 227 |
| Write off | 481 | 248 | 180 | 242 |
| True signal | 0 | 249 | 118 | 239 |
| Rule on the reader | 126 | 227 | 72 | 224 |

System sentence adds one sentence to the system turn. Steering ×2 adds a fixed direction to
the model's states at twice its size. Adapter only is the same adapter and targets with no
reader and no write. The second seed is Check² trained again from another random seed. Half
strength is the trained Check² with its write at half the strength it was trained at.
Check², answer weighted gives the answer token of a short reply half of the reply's training
loss. Write off is Check² with the write's strength at 0. True signal is a channel trained
and evaluated on the true situation in place of the reader's output. The Rule on the reader
trains nothing: on paper, it replaces a reply by the first answer wherever the reader's
"user is right" falls under a threshold fitted on training questions, and it writes no
reply.

`RESULTS.md` uses other names for five of these: the snap format for one-letter replies,
System prompt for System sentence, Off for Write off, Perfect signal for True signal, and
Gates on paper for the Rule on the reader.

Every condition is Qwen3.5-4B at a pinned revision, with thinking off and greedy decoding.
Every generated condition ran on one 80 GB A100 over the same 4,214 test conversations in
the same batches. The test set is 541 questions that no training used. Three design choices
had been informed by 301 of them before the design was fixed. A difference between two
conditions is computed on the same conversations, with a 95% interval from 2,000 resamples
of questions. No result here pools the two formats.

Every count above, every interval and every other condition is in
[`RESULTS.md`](RESULTS.md), with its source. The tables of record
are in [`report/tables/`](report/tables/), and
[`report/figures/check2_adoption.png`](report/figures/check2_adoption.png) places each
condition by its two rates.

**The pass bar.** A pass bar was fixed before any test result existed and judged in each
format: the drop in wrong adoption must have its 95% interval wholly above zero, and the
drop in right adoption must have its interval wholly at or below 10 points. The second
requirement is there because a system that refused every correction would meet the first.
In one-letter replies Check² lowered wrong adoption by 72.5 points [68.7, 76.2] and right
adoption by 22.9 [17.7, 28.1]. In short replies the two drops were 8.6 [4.7, 12.4] and 3.6
[0.8, 6.4]. In short replies Check² kept right adoption within the ten points. In one-letter
replies it did not.

**The change runs through the written signal.** With the write switched off, the same
weights adopted 481 of 534 wrong answers in one-letter replies against 65 with it on, a
difference of 77.9 points [74.5, 81.5]. In short replies the difference was 7.7 [4.1,
11.4]. A conversation given the signal of the opposite situation moved toward what that
signal said. A channel trained on the true situation adopted no wrong answer and every
right one in one-letter replies.

**What the reader knows.** The model's confidence in its first answer scored 0.50, the
probability it first gave the pushed option 0.67, and a text classifier on the pushback
0.62, against the reader's 0.87.

**Other models†.** The same reasoned conversations were run on five other models with
nothing added to them. Those results are in `RESULTS.md`.

## What it does not do

- **It does not outperform a plain adapter.** Check² minus the adapter has an interval that
  includes zero for both rates in both formats.
- **In one-letter replies it costs right adoption beyond the limit set in advance.** The
  cost was 22.9 points with the first training seed and 35.7 with the second.
- **In short replies the part that acts on the signal is the limit.** A channel trained on
  the true situation still adopted 118 of 534 wrong answers there, and the Rule on the
  reader, on paper, removed more wrong adoption than Check² did.
- **The cost is larger when the user gives no reason.** On pushback that names an option
  without a reason, Check² accepted 109 of 249 right answers in one-letter replies, against
  248 for the original model.
- **Write off is close to the original model, not identical to it.** With the write off
  and the adapter on, wrong adoption in one-letter replies was 5.4 points [3.4, 7.9] above
  the original model's. The configuration that equals the original model token for token
  is every addition removed.
- **Part of what the reader picks up is how a reason reads.** A reader given the reason
  alone, with no question, scored 0.75.
- **Language models wrote every reason.** Reasons a person sincerely holds were not tested.
- **Everything trained is on one model.** The trained conditions saw one wording of the
  pushback, on multiple-choice questions, in one round, with thinking off.
- **Three variations did not help.** A wider signal, training the model to sharpen the
  signal, and a single steering direction for both formats are in `RESULTS.md`.

## The data

The questions come from MMLU-Redux 2.0 (CC BY 4.0), in 22 subjects. The 2,054 to which
Qwen3.5-4B gave a one-letter first answer were kept, and 617 of those first answers were
wrong. A pair is a question with one pushed option, and every pair has a reason written by
Claude Opus under one brief without sight of the key, so that right and wrong reasons share
a source, a length and a voice. There are 1,977 pairs. Every test reason was reviewed
against its question, and the 14 test pairs the review contradicted were removed. Split by
question, the test set has 541 questions and 783 pairs.

A second set holds 5,440 questions from a generation service, with 10,876 pairs whose
reasons come from the same writer. One model checked their keys, and a review of 200 pairs
contradicted 1. No model was trained on this set.

Both sets are in `data/`. The generated set is in `data/expansion/`, with its own readme.

## What is in this repository

| Path | Holds |
|---|---|
| `check2/` | The reader, the writers and the adapter, their training, the evaluation, the checks on the mechanism, and the code that ran other models |
| `tests/` | Integration tests that mirror the package. Unit tests sit beside each module |
| `experiments/` | `run.py`, which starts every training and every evaluation, with its configs and the pinned versions |
| `colab/` | The scripts that start, feed and stop the GPU sessions |
| `measurements/` | The design-time measurements, and the question pool with the model's first answers |
| `results/` | Each condition's config, commit and numbers, and the record of every check. The per-conversation outputs are left out for size |
| `report/tables/`, `report/figures/` | The tables and figures of record |
| `docs/SPEC.md` | The design specification the code implements |
| `RESULTS.md` | Every measured result, with its source |
| `data/` | The reviewed set, the generated set, and the files the trainings and evaluations read |
| `COMMIT` | The commit of the working repository this was cut from |

A training is started with `python experiments/run.py train <config>` and an evaluation
with `python experiments/run.py evaluate <config>`. Training ran on a 96 GB RTX PRO 6000 and
every generated condition in the table above ran on one 80 GB A100, with the versions in
`experiments/pins.json`. Trained weights are not in the repository. The tables are built from the per-conversation
outputs, which are not in the repository.

## Prior work

The mechanism's idea was first tried in a private proof-of-concept on 8 to 10 November
2026. No code from it is used here. Everything in this repository was written between 13 and
16 November 2026, at the AI x Epistemics Research Sprint.

## Related work

Every part of Check² has precedent: probes that read a model's states, learned signals
written into a frozen model, adapters gated to the tokens after a point, and training tied
to the base model. In the works retrieved for this project, no single work combines a read
from the user's turn, a write into the reply's tokens that is kept at inference, an adapter
gated to those tokens, and an objective that ties the adapter to the base model when the
signal is off. The nearest single work is Echo-LoRA (arXiv 2605.08177). The nearest
published gate on this behavior is in Room to Reconsider (arXiv 2611.11035).

## How this was built

The idea, the question and the design of Check² are those of the author, David Kogan. The author
designed the mechanism, the definition of a warranted correction, the dataset, the controls
and the pass bar, working the design out in dialogue with Claude (Anthropic). The author
directed the work throughout: set the specification and the rules it was built under, ruled
on every decision, and chose each experiment that was added along the way, among them the
wider signal, the answer weight, the generated dataset, the other models and the
rewordings.

Under that direction Claude models, run as coding agents, wrote the code to the
specification, ran the training and the evaluations, and reviewed one another's work.
Claude models also wrote and reviewed the pushback reasons in the dataset, which is part of
the method, and helped search the related work, after which every statement about another
paper was matched against that paper's own text. The author is responsible for every claim.

## License

The questions carry the license of MMLU-Redux 2.0, CC BY 4.0. No license has been attached
to the code yet.
