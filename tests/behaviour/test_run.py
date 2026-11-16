"""The runner on a stub backend: first answers, follow-ups built on them, resumption and the record."""

from check2.behaviour.backend import Reply
from check2.behaviour.replies import read_replies
from check2.behaviour.run import run


class StubBackend:
    """Answers every first question with "C" and every follow-up with "Final answer: (B)"."""

    name = "stub"
    thinking = "off"
    batch_size = 4

    def __init__(self):
        self.asked: list[str] = []

    def generate_batches(self, items):
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            self.asked += [i.item_id for i in batch]
            yield [Reply(i.item_id, "C" if i.kind == "first" else "Final answer: (B)", True, False) for i in batch]


def test_the_runner_answers_first_then_follow_ups_and_resumes(tmp_path):
    folder = tmp_path / "behaviour" / "stub-model"
    records = []

    def write_record(kind, name, values, config):
        records.append((kind, name, values, config))
        path = tmp_path / "record.json"
        path.write_text("{}", encoding="utf-8")
        return path

    backend = StubBackend()
    run(backend, folder, "stub/model", "abc", limit=3, log=lambda *_: None, write_record=write_record)
    rows = read_replies(folder)
    firsts = [r for r in rows if r["kind"] == "first"]
    assert [r["source"] for r in firsts] == ["pool"] * 3 + ["expanded"] * 3
    # The third pool question, abstract_algebra/10, is a test question with pairs on A and B:
    # with the first answer C all its pairs and its neutral follow-up are asked, after every
    # first answer.
    kinds = [r["kind"] for r in rows]
    assert kinds[:6] == ["first"] * 6 and set(kinds[6:]) == {"reasoned", "neutral"}
    assert {r["item_id"] for r in rows if r["qid"] == "abstract_algebra/10" and r["kind"] == "neutral"} == {
        "abstract_algebra/10:neutral:snap:-", "abstract_algebra/10:neutral:short:-"}
    kind, name, values, config = records[0]
    assert (kind, name) == ("run", "behaviour/stub-model")
    assert values["model"] == "stub/model" and values["revision"] == "abc" and values["thinking"] == "off"
    assert values["replies"] == len(rows) == values["first_items"] + values["follow_up_items"]
    assert (folder / "record.json").exists()
    again = StubBackend()
    run(again, folder, "stub/model", "abc", limit=3, log=lambda *_: None, write_record=write_record)
    assert again.asked == [] and len(read_replies(folder)) == len(rows)
