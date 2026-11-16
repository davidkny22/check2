# Expanded question set

Four-option multiple-choice questions written by a generation service from the `train`
questions of one subject per job, kept apart from the reviewed set. No training and no run
reads this folder. Each kept question has two written reasons, one arguing for its key and
one for another option.

## How a question was kept

- A generated row is usable when its prompt splits into one question and exactly four
  distinct options and its key is one of A to D. A row whose question text, lower-cased
  with all whitespace removed, matches a question of the reviewed pool, a seed question, or
  an earlier usable row is dropped. Subjects are compared in sorted order.
- A Sonnet curator answered every remaining question without its key. A question is kept
  only where the curator's answer equals the key and the curator records no problem with
  the question. The keys were therefore checked by one model.
- The four options of a kept question are in an order drawn with `random.Random(2040)`
  over the questions sorted by id, with the key moved with its option. A question with an
  option that names other options keeps its order, and its `options_reordered` is false.
- Every kept question has two pairs: its key, and one other option drawn with
  `random.Random(2041)` from the options that are neither the key nor the model's complying
  wrong first answer. A writer wrote one reason per pair without seeing the key: an Opus
  writer, except for the pairs the record below names as written by Sonnet.
- The reasons were checked by a sample: a soundness review of 200 pairs drawn with
  `random.Random(2042)` from the pairs with a reason, sorted by question id and then
  target. A sampled pair that an earlier review judged with the same reason text keeps
  that judgement; the others were reviewed for this sample. A pair is contradicted when a
  reason for the key is rated unsound or a reason for another option is rated sound;
  contradicted pairs are removed and listed in `removed_pairs.json`.

## Counts per subject

| subject | rows returned | do not split | pool or seed | repeat | usable | curated | curator agrees with key | kept by curator | repeat of an earlier subject | kept | options kept in order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abstract_algebra | 362 | 3 | 62 | 6 | 359 | 291 | 280 | 264 | 0 | 264 | 3 |
| astronomy | 365 | 0 | 65 | 4 | 365 | 296 | 277 | 228 | 0 | 228 | 0 |
| college_computer_science | 357 | 2 | 61 | 1 | 355 | 293 | 270 | 255 | 0 | 255 | 3 |
| college_mathematics | 298 | 1 | 0 | 0 | 297 | 297 | 234 | 211 | 0 | 211 | 1 |
| college_physics | 367 | 0 | 67 | 1 | 367 | 299 | 255 | 238 | 0 | 238 | 0 |
| conceptual_physics | 365 | 1 | 69 | 3 | 364 | 292 | 282 | 256 | 1 | 255 | 1 |
| econometrics | 365 | 17 | 48 | 0 | 348 | 300 | 283 | 281 | 0 | 281 | 6 |
| elementary_mathematics | 361 | 0 | 62 | 4 | 361 | 295 | 270 | 263 | 0 | 263 | 0 |
| formal_logic | 344 | 0 | 46 | 1 | 344 | 297 | 272 | 228 | 0 | 228 | 16 |
| high_school_computer_science | 373 | 10 | 72 | 5 | 363 | 286 | 275 | 271 | 0 | 271 | 1 |
| high_school_geography | 528 | 2 | 136 | 1 | 526 | 389 | 384 | 354 | 0 | 354 | 2 |
| high_school_macroeconomics | 367 | 150 | 67 | 2 | 217 | 148 | 145 | 136 | 0 | 136 | 0 |
| high_school_mathematics | 312 | 1 | 48 | 3 | 311 | 260 | 179 | 166 | 0 | 166 | 0 |
| high_school_microeconomics | 368 | 0 | 69 | 2 | 368 | 297 | 241 | 226 | 0 | 226 | 0 |
| high_school_physics | 356 | 2 | 59 | 0 | 354 | 295 | 283 | 266 | 0 | 266 | 1 |
| high_school_statistics | 363 | 0 | 66 | 2 | 363 | 295 | 288 | 280 | 0 | 280 | 3 |
| logical_fallacies | 355 | 0 | 56 | 1 | 355 | 298 | 281 | 256 | 0 | 256 | 0 |
| machine_learning | 360 | 0 | 60 | 0 | 360 | 300 | 291 | 285 | 0 | 285 | 0 |
| management | 353 | 1 | 55 | 2 | 352 | 295 | 294 | 292 | 0 | 292 | 1 |
| marketing | 360 | 1 | 65 | 0 | 359 | 294 | 286 | 273 | 0 | 273 | 1 |
| philosophy | 355 | 151 | 55 | 0 | 204 | 149 | 148 | 146 | 0 | 146 | 0 |
| professional_accounting | 350 | 1 | 53 | 5 | 349 | 291 | 278 | 266 | 0 | 266 | 0 |
| total | 7984 | 343 | 1341 | 43 | 7641 | 6257 | 5796 | 5441 | 1 | 5440 | 39 |

`usable` counts rows that split cleanly with a letter key, before the pool, seed and
repeat drops; `curated` counts the questions the curator answered.

## First answers, pairs and reasons

- Kept questions: 5440. The model's first answers were generated: 5412 comply, and 703 of those are wrong.
- Pairs: 10880, two per kept question.
- Pairs with a reason that counts: 10880 of 10880. A reason counts for its pair on its own terms, whether or not its writer's file is whole: its job id is one of its file's jobs and is written once there, and its text keeps the writer brief's rules that need no judgement (20 to 40 words, none of the phrases of confidence or doubt the brief names, one line with no heading or bullet point).
- 8 writer job files of 150 were stopped part-way by the model provider's safety classifier (`writer_012.json`, `writer_025.json`, `writer_026.json`, `writer_040.json`, `writer_041.json`, `writer_042.json`, `writer_047.json`, `writer_048.json`; the notices are in `stopped_writers.json`). The reasons their writers finished count; 9828 reasons in all come from the files of 150. A stopped file was written again in smaller files by fresh writers, each saving its output after every ten reasons so that a stop keeps what it finished:
  - 1052 unwritten jobs went to Opus writers at medium effort in 36 files of at most 30 (`writer30_NNN.json`), which added 872 reasons.
  - 180 unwritten jobs went to Opus writers at medium effort in 18 files of at most 10 (`writer10_NNN.json`), which added 170 reasons.
- 10 jobs still unwritten after the files of at most 10 were set aside, and are listed with their question, subject and target in `set_aside.json`. The table below gives them per subject.
- The owner agreed a second writer model for the jobs set aside. They went once to Sonnet writers at medium effort in 1 file of at most 10 (`sonnet10_NNN.json`), under the same brief, which wrote 10 reasons. Those pairs carry `"writer": "sonnet"` in `pairs.jsonl`; every other pair carries `"writer": "opus"`.
- Pairs without a reason: 0.
- Writer job files that are not whole, with why; the reasons they hold that count are kept:
  - `writer10_016.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer30_014.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer30_019.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer30_023.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer30_025.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer30_027.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer30_030.json`: it has no reasons file.
  - `writer_012.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer_025.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer_026.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer_040.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer_041.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer_042.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.
  - `writer_047.json`: its writer was stopped by the model provider's safety classifier; its reasons file fails its check (2 job ids missing).
  - `writer_048.json`: its writer was stopped by the model provider's safety classifier; it has no reasons file.

## Checks on the reasons

- Style check: the cross-validated AUROC of a text-only classifier at telling reasons for the key from the others, grouped by question, is 0.7479 over 10880 reasons. It is recorded, not used to remove anything.
- Soundness review of 200 sampled pairs: 1 contradicted, a share of 0.005. Of the reasons for the key, 102 were rated sound, 1 unsound and 0 unclear; of the reasons for another option, 93 were rated unsound, 0 sound and 4 unclear. 30 of the judgements were kept from earlier review samples, whose reviewer saw the same reason, and 170 were made for this sample.
- Earlier review samples of 200, drawn with the same seed before the last reasons were written, found 3 pairs contradicted, and they stay removed. Of these, the number this sample also drew, with the same reason and the same judgement, is 0.
- Pairs removed: 4. Pairs in `pairs.jsonl` after the contradicted pairs are removed: 10876.

## Pairs per subject

| subject | pairs | reason from the files of 150 | added by Opus files of 30 | added by Opus files of 10 | set aside | added by Sonnet files of 10 | without a reason | reviewed in the sample | contradicted in the sample | removed | in `pairs.jsonl` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abstract_algebra | 528 | 528 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 528 |
| astronomy | 456 | 456 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 456 |
| college_computer_science | 510 | 510 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 510 |
| college_mathematics | 422 | 272 | 150 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 422 |
| college_physics | 476 | 476 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 476 |
| conceptual_physics | 510 | 510 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 1 | 509 |
| econometrics | 562 | 562 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 562 |
| elementary_mathematics | 526 | 226 | 270 | 30 | 0 | 0 | 0 | 8 | 0 | 0 | 526 |
| formal_logic | 456 | 456 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 1 | 455 |
| high_school_computer_science | 542 | 542 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 542 |
| high_school_geography | 708 | 708 | 0 | 0 | 0 | 0 | 0 | 14 | 1 | 1 | 707 |
| high_school_macroeconomics | 272 | 154 | 118 | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 271 |
| high_school_mathematics | 332 | 0 | 212 | 120 | 0 | 0 | 0 | 2 | 0 | 0 | 332 |
| high_school_microeconomics | 452 | 452 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 452 |
| high_school_physics | 532 | 380 | 122 | 20 | 10 | 10 | 0 | 11 | 0 | 0 | 532 |
| high_school_statistics | 560 | 560 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 560 |
| logical_fallacies | 512 | 512 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 512 |
| machine_learning | 570 | 570 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 570 |
| management | 584 | 584 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 584 |
| marketing | 546 | 546 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 546 |
| philosophy | 292 | 292 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 292 |
| professional_accounting | 532 | 532 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 532 |
| total | 10880 | 9828 | 872 | 170 | 10 | 10 | 0 | 200 | 1 | 4 | 10876 |

`removed` counts every contradicted pair, from this sample and from earlier review samples.

