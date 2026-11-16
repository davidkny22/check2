"""Where the data files live, relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

POOL = ROOT / "measurements" / "protocol" / "first_answers.json"
PLAN = ROOT / "measurements" / "dataset" / "plan.json"
TEST_SECOND_PART = ROOT / "measurements" / "dataset" / "reserve_questions.json"
MEASURED_JOBS = ROOT / "measurements" / "dataset" / "jobs"

DATA = ROOT / "data"
SPLITS = DATA / "splits.json"
PAIRS = DATA / "pairs.jsonl"
EPISODES = DATA / "episodes"
JOBS = DATA / "jobs"
REASONS = DATA / "reasons"
REVIEWS = DATA / "reviews"
REMOVED_PAIRS = REVIEWS / "removed_pairs.json"
STYLE_SCORES = REVIEWS / "style_scores.json"
TARGETS = DATA / "targets" / "targets.jsonl"
CACHE_INDEX = DATA / "reader" / "cache_index.json"
CROSSWRITER_JOBS = JOBS / "crosswriter.json"
CROSSWRITER_REASONS = REASONS / "reasons_crosswriter.json"

LETTERS = "ABCD"
