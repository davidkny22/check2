"""The template assumption under a given tokenizer: each episode's prompt has the token count, the
template tail and the read index its record holds, and its window holds exactly the last user
turn's content."""
from check2.data.episodes import TAIL_LENGTH, TemplateError
from check2.reader.window import last_user_content, spans

MAX_LISTED = 20


def check_episodes(tokenizer, episodes: list[dict], tail_ids: dict[str, list[int]],
                   tail_position: int) -> tuple[int, list[str]]:
    """The number of episodes that fail, and up to `MAX_LISTED` findings, one per failing episode."""
    encoded = tokenizer([e["prompt"] for e in episodes])["input_ids"]
    failing, findings = 0, []
    for e, ids in zip(episodes, encoded):
        problems = []
        if len(ids) != e["n_prompt_tokens"]:
            problems.append(f"{len(ids)} tokens, the record says {e['n_prompt_tokens']}")
        if list(ids[-TAIL_LENGTH:]) != list(tail_ids[e["format"]]):
            problems.append(f"tail {list(ids[-TAIL_LENGTH:])} is not the {e['format']} tail")
        if e["read_index"] != len(ids) - TAIL_LENGTH + tail_position:
            problems.append(f"read index {e['read_index']} is not tail[{tail_position}]")
        try:
            window_start, tail_start = spans(list(ids), tokenizer)
            if tokenizer.decode(ids[window_start:tail_start]) != last_user_content(e["prompt"]):
                problems.append("the window's tokens do not decode to the last user turn's content")
        except TemplateError as err:
            problems.append(str(err))
        if problems:
            failing += 1
            if len(findings) < MAX_LISTED:
                findings.append(f"{e['episode_id']}: " + "; ".join(problems))
    return failing, findings
