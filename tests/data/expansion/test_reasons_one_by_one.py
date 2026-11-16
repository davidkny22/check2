"""Reasons counted one by one: a reason counts for its pair on its own terms, whether or
not its file is whole."""
from check2.data.expansion.reasons import counted_reasons, reason_verdicts, text_fault, unwritten_jobs

OK = " ".join(["word"] * 25)


def _jobs(*ids):
    return [{"job_id": j, "question": "Q?", "options": {}, "argue_for": "A"} for j in ids]


def test_a_reason_counts_from_a_file_that_is_not_whole():
    jobs = _jobs("e/s/1:A", "e/s/1:B", "e/s/2:C")
    written = {"writer": "opus", "reasons": [{"job_id": "e/s/1:A", "reason": OK}]}
    counted = counted_reasons(jobs, written)
    assert counted == {"writer": "opus", "reasons": [{"job_id": "e/s/1:A", "reason": OK}]}
    assert [j["job_id"] for j in unwritten_jobs(jobs, {"e/s/1:A"})] == ["e/s/1:B", "e/s/2:C"]


def test_a_reason_refused_for_its_word_count_while_its_neighbours_count():
    jobs = _jobs("e/s/1:A", "e/s/1:B", "e/s/2:C")
    short = " ".join(["word"] * 19)
    long = " ".join(["word"] * 41)
    written = {"writer": "opus", "reasons": [{"job_id": "e/s/1:A", "reason": OK},
                                             {"job_id": "e/s/1:B", "reason": short},
                                             {"job_id": "e/s/2:C", "reason": " ".join(["word"] * 40)}]}
    verdicts = [(r["job_id"], fault) for r, fault in reason_verdicts(jobs, written)]
    assert verdicts == [("e/s/1:A", None), ("e/s/1:B", "19 words"), ("e/s/2:C", None)]
    assert [r["job_id"] for r in counted_reasons(jobs, written)["reasons"]] == ["e/s/1:A", "e/s/2:C"]
    assert text_fault(long) == "41 words"


def test_a_job_id_that_belongs_to_another_file_is_refused():
    jobs = _jobs("e/s/1:A", "e/s/1:B")
    written = {"writer": "opus", "reasons": [{"job_id": "e/s/1:A", "reason": OK},
                                             {"job_id": "e/t/9:D", "reason": OK}]}
    verdicts = [(r["job_id"], fault) for r, fault in reason_verdicts(jobs, written)]
    assert verdicts == [("e/s/1:A", None), ("e/t/9:D", "job id not in this file")]
    assert [r["job_id"] for r in counted_reasons(jobs, written)["reasons"]] == ["e/s/1:A"]


def test_a_job_id_written_twice_in_one_file_counts_neither_time():
    jobs = _jobs("e/s/1:A", "e/s/1:B")
    written = {"writer": "opus", "reasons": [{"job_id": "e/s/1:A", "reason": OK},
                                             {"job_id": "e/s/1:A", "reason": OK + " more"},
                                             {"job_id": "e/s/1:B", "reason": OK}]}
    assert [r["job_id"] for r in counted_reasons(jobs, written)["reasons"]] == ["e/s/1:B"]


def test_the_brief_s_named_phrases_and_a_second_line_are_refused():
    assert text_fault("Obviously " + OK) == "uses 'Obviously'"
    assert text_fault(OK + " as I see it") == "uses 'as I see it'"
    assert text_fault(OK[:50] + "\n" + OK[50:]) == "has a heading, a bullet point or more than one line"
    assert text_fault("- " + OK) == "has a heading, a bullet point or more than one line"
    # A name in code or a formula with underscores or stars is not markdown.
    assert text_fault("Names like __init__ and x**2 " + OK) is None
    assert text_fault("Clearlyish " + OK) is None
