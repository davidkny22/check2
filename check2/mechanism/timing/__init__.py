"""How long short-format generation takes, where a decoding step's time goes, and what a decode setting changes.

Every command runs the pipeline's generation, split at the read token, under
`torch.inference_mode()`, on the first batch of 96 short-format `heldout` episodes in the
order of the runs (by prompt length, ties by episode id), generated to the end of the reply
cap, with no write and `p` absent. Each timed batch comes after one warm-up run of the same
batch, and the timer is synchronised with the card. The appended read is not timed: the
generation path does not include it, and the same work is timed in every arm.

    python -m check2.mechanism.timing generation --implementation none|hooks|wrappers|library
    python -m check2.mechanism.timing profile
    python -m check2.mechanism.timing setting --setting preallocated_cache|cublaslt|cudnn_attention

Each also takes `--model tiny|full` (full by default) and `--limit N`, the first N episodes of
the batch. `generation` times three repeats and records their spread. `profile` profiles the
decoding step in the middle of the reply cap with no adapter. `setting` times the batch with
the setting off and on, three repeats each, and counts the episodes whose final answer
differs from the default's. Each writes a measurement record.
"""
