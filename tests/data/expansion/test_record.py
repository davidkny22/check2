"""The README written from the counts file."""
from check2.data.expansion.record import readme


def test_readme_states_counts_per_subject_and_the_totals():
    parse = {"a": {"rows": 10, "does_not_split": 1, "dropped_pool_or_seed": 2, "dropped_repeat": 1, "usable": 9},
             "b": {"rows": 20, "does_not_split": 0, "dropped_pool_or_seed": 3, "dropped_repeat": 0, "usable": 20}}
    keep = {s: {"curated": 6, "answer_equals_key": 5, "kept_by_curator": 4,
                "dropped_repeat_of_earlier_subject": 0, "kept": 4, "options_kept_in_order": 1} for s in parse}
    checks = {"review": {"reviewed": 200, "contradicted": 3, "contradicted_share": 0.015,
                         "key_target": {"sound": 90}, "other_target": {"unsound": 100}},
              "style": {"reasons": 400, "auroc": 0.61}, "pairs_kept": 397}
    text = readme({"parse": parse, "keep": keep, "checks": checks})
    assert "| a | 10 | 1 | 2 | 1 | 9 | 6 | 5 | 4 | 0 | 4 | 1 |" in text
    assert "| total | 30 | 1 | 5 | 1 | 29 | 12 | 10 | 8 | 0 | 8 | 2 |" in text
    assert "a share of 0.015" in text
    assert "checked by one model" in text
    assert "—" not in text


def test_readme_names_files_without_reasons_and_states_the_pairs_per_subject():
    pairs = {"first_answers": "used", "questions": 4, "complying": 4, "wrong_first": 1, "pairs": 8}
    reasons = {"pairs": 8, "reasons_written": 6, "sample_judgement_carried": 1, "sample_to_review": 3,
               "writer_files_without_reasons": {"writer_002.json": "its writer was stopped by the model "
                                                                   "provider's safety classifier"}}
    per_subject = {"a": {"pairs": 4, "with_reason": 4, "reviewed": 3, "contradicted": 1, "removed": 2, "kept": 2},
                   "b": {"pairs": 4, "with_reason": 2, "reviewed": 1, "contradicted": 0, "removed": 0, "kept": 2}}
    checks = {"review": {"reviewed": 4, "contradicted": 1, "contradicted_share": 0.25,
                         "key_target": {"sound": 2}, "other_target": {"unsound": 1, "sound": 1}},
              "style": {"reasons": 6, "auroc": 0.5}, "pairs_kept": 4, "removed": 2,
              "removed_by_first_sample": 2, "removed_by_first_sample_in_this_sample": 1,
              "per_subject": per_subject}
    text = readme({"pairs": pairs, "reasons": reasons, "checks": checks})
    assert "  - `writer_002.json`: its writer was stopped by the model provider's safety classifier." in text
    assert "- Pairs with a reason that counts: 6 of 8." in text
    assert "- Pairs without a reason: 2." in text
    assert "1 of the judgements were kept from earlier review samples" in text
    assert "3 were made for this sample" in text
    assert "Earlier review samples of 200, drawn with the same seed before the last reasons were written, found 2 "            "pairs contradicted, and they stay removed. Of these, the number this sample also drew" in text
    assert "with the same reason and the same judgement, is 1." in text
    assert "- Pairs removed: 2. Pairs in `pairs.jsonl`" in text
    assert "| a | 4 | 4 | 0 | 3 | 1 | 2 | 2 |" in text
    assert "| b | 4 | 2 | 2 | 1 | 0 | 0 | 2 |" in text
    assert "| total | 8 | 6 | 2 | 4 | 1 | 2 | 4 |" in text
    assert "—" not in text


def test_readme_states_what_each_set_of_smaller_files_added_and_the_jobs_set_aside():
    pairs = {"first_answers": "used", "questions": 6, "complying": 6, "wrong_first": 1, "pairs": 12}
    reasons = {"pairs": 12, "reasons_written": 11,
               "reasons_by_file_group": {"writer": 6, "writer30": 3, "writer10": 1, "sonnet10": 1}}
    smaller = {"writer30": {"sources": ["writer_002.json"], "size": 30, "unwritten": 6, "writer": "opus",
                            "files": {"writer30_001.json": 6}},
               "writer10": {"sources": ["writer_002.json"], "size": 10, "unwritten": 3, "writer": "opus",
                            "files": {"writer10_001.json": 3}},
               "sonnet10": {"sources": ["writer_002.json"], "size": 10, "unwritten": 2, "writer": "sonnet",
                            "files": {"sonnet10_001.json": 2}}}
    set_aside = {"jobs": 2, "per_subject": {"a": 1, "b": 1}}
    per_subject = {"a": {"pairs": 6, "with_reason": 6, "reasons_from_writer": 4, "reasons_from_writer30": 1,
                         "reasons_from_writer10": 0, "set_aside": 1, "reasons_from_sonnet10": 1, "reviewed": 2,
                         "contradicted": 0, "removed": 0, "kept": 6},
                   "b": {"pairs": 6, "with_reason": 5, "reasons_from_writer": 2, "reasons_from_writer30": 2,
                         "reasons_from_writer10": 1, "set_aside": 1, "reviewed": 1,
                         "contradicted": 0, "removed": 0, "kept": 5}}
    checks = {"review": {"reviewed": 3, "contradicted": 0, "contradicted_share": 0.0,
                         "key_target": {"sound": 2}, "other_target": {"unsound": 1}},
              "style": {"reasons": 11, "auroc": 0.5}, "pairs_kept": 11, "removed": 0, "per_subject": per_subject}
    text = readme({"pairs": pairs, "reasons": reasons, "smaller_files": smaller, "set_aside": set_aside,
                   "checks": checks})
    assert "- 1 writer job file of 150 was stopped part-way by the model provider's safety classifier " \
           "(`writer_002.json`; the notices are in `stopped_writers.json`)." in text
    assert "6 reasons in all come from the files of 150." in text
    assert "  - 6 unwritten jobs went to Opus writers at medium effort in 1 file of at most 30 " \
           "(`writer30_NNN.json`), which added 3 reasons." in text
    assert "  - 3 unwritten jobs went to Opus writers at medium effort in 1 file of at most 10 " \
           "(`writer10_NNN.json`), which added 1 reasons." in text
    assert "- 2 jobs still unwritten after the files of at most 10 were set aside" in text
    assert "in 1 file of at most 10 (`sonnet10_NNN.json`), under the same brief, which wrote 1 reasons." in text
    assert '`"writer": "sonnet"` in `pairs.jsonl`' in text
    assert "| subject | pairs | reason from the files of 150 | added by Opus files of 30 | added by Opus files of 10 " \
           "| set aside | added by Sonnet files of 10 | without a reason |" in text
    assert "| a | 6 | 4 | 1 | 0 | 1 | 1 | 0 | 2 | 0 | 0 | 6 |" in text
    assert "| b | 6 | 2 | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 5 |" in text
    assert "| total | 12 | 6 | 3 | 1 | 2 | 1 | 1 | 3 | 0 | 0 | 11 |" in text
    assert "—" not in text
