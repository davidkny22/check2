"""Where the expanded set's inputs and outputs live, and the numbers its draws use."""
from check2.data.paths import DATA

ADAPTION = DATA / "adaption"
EXPANSION = DATA / "expansion"

GENERATED = EXPANSION / "generated"
DROPPED = EXPANSION / "dropped"
WITHOUT_KEYS = EXPANSION / "questions_without_keys"
CURATION = EXPANSION / "curation"
QUESTIONS = EXPANSION / "questions.jsonl"
QUESTIONS_WITHOUT_KEYS = EXPANSION / "questions_without_keys.jsonl"
FIRST_ANSWERS = EXPANSION / "first_answers.jsonl"
JOBS = EXPANSION / "jobs"
REASONS = EXPANSION / "reasons"
REVIEW_JOBS = EXPANSION / "review_jobs"
REVIEWS = EXPANSION / "reviews"
PAIRS = EXPANSION / "pairs.jsonl"
# Writer job files whose writer the model provider's safety classifier stopped, each with
# the notice the writer or its run reported.
STOPPED_WRITERS = EXPANSION / "stopped_writers.json"
# Jobs still unwritten after the writers of the smaller files, each with its question,
# subject and target.
SET_ASIDE = EXPANSION / "set_aside.json"
COUNTS = EXPANSION / "counts.json"
README = EXPANSION / "README.md"

# Rows generated before the per-subject jobs, as (subject, raw files in the order they are read).
EARLIER_RAW = {
    "college_mathematics": ("generated_raw_college_mathematics.jsonl",),
    "high_school_geography": ("generated_raw.jsonl", "generated_raw_300.jsonl"),
}
# The college mathematics questions already answered by a Sonnet curator, and the file that
# gives those answers' question ids their text.
EARLIER_CURATION = ADAPTION / "curation_college_mathematics_sonnet.json"
EARLIER_CURATED_QUESTIONS = ADAPTION / "generated_questions_college_mathematics.jsonl"

OPTION_ORDER_SEED = 2040
SECOND_PAIR_SEED = 2041
REVIEW_SEED = 2042
REVIEW_SAMPLE_SIZE = 200
JOB_FILE_SIZE = 150
REASON_WORDS = (20, 40)
