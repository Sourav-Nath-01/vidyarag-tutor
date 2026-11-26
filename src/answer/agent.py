"""
A bounded agent loop around grounded answers.

The first answer uses the top search results. Sometimes the model says those
passages don't answer the question (NOT_COVERED) even though the relevance
gate judged them on-topic. Then the right lecture was usually found but the
wrong moment in it. Search keeps one chunk per lecture, and the cross-encoder
favours chunks that *name* the topic over chunks that *explain* it. In the
quicksort-analysis lecture, for example, the intro ("we have seen quicksort...")
outranks "...the average case is n log n", which never says "quicksort".

So the tutor takes one more, different retrieval action and answers again:

    answer(top results) ── grounded or gated ──► done
         │ NOT_COVERED, although on-topic
         ▼
    hint terms (LLM) ─► re-rank every chunk of the top lectures ─► answer(new passages) ─► done

The hint terms only steer the ranking. The second answer must still come from
the passages, and the same checks run on it. There is at most one retry, so
cost and latency stay bounded. Every step is reported, so the page can show
how the answer was found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from .grounded import ChatModel, Source, answer_events

HINT_PROMPT = """\
List the technical terms, formulas and phrases a professor would most likely \
say while answering this question in a lecture. Reply with a comma-separated \
list of at most 12 terms and nothing else."""
MAX_HINT_CHARS = 200


@dataclass
class Deeper:
    """What looking deeper found: new sources and how they were chosen."""
    sources: list[Source]
    lectures: list[str] = field(default_factory=list)
    chunks_scored: int = 0


def hint_terms(question: str, llm: ChatModel) -> str:
    """Terms a lecture answering the question would contain ("" if none usable).
    LLM errors propagate; the caller falls back to the question alone."""
    reply = "".join(llm.stream([
        {"role": "system", "content": HINT_PROMPT},
        {"role": "user", "content": question},
    ]))
    for line in reply.strip().splitlines():
        line = " ".join(line.split()).strip(" .\"'`")
        if line and "NOT_COVERED" not in line:
            return line[:MAX_HINT_CHARS].rsplit(",", 1)[0] if len(line) > MAX_HINT_CHARS else line
    return ""


def tutor_events(question: str, sources: list[Source], llm: ChatModel, *,
                 relevance: float | None, min_relevance: float,
                 dig: Callable[[str], Deeper | None] | None = None,
                 hint_llm: ChatModel | None = None,
                 on_hint_error: Callable[[Exception], None] | None = None) -> Iterator[tuple[str, dict]]:
    """
    Stream one answer as (event, data) pairs, looking deeper once if needed.

    Events: `sources`, `token`, `done` as in answer_events, plus `step` while
    looking deeper. A second `sources` event replaces the first; citations in
    the final answer refer to it. `done` carries `steps` (what was tried) and
    `looked_deeper`.

    `dig(hints)` re-ranks the top lectures' chunks for the question plus hint
    terms and returns the new sources (None or empty: nothing better found).
    """
    first_done = None
    for event, data in answer_events(question, sources, llm, relevance=relevance, min_relevance=min_relevance):
        if event == "done":
            if dig is not None and data["not_covered"] and not data["gated"]:
                first_done = data
                break
            data = {**data, "looked_deeper": False,
                    "steps": [{"action": "answer", "sources": len(sources), "result": _outcome(data)}]}
        yield event, data
    if first_done is None:
        return

    steps = [{"action": "answer", "sources": len(sources), "result": "not_covered"}]
    yield "step", {"action": "look_deeper", "stage": "start"}
    hints = ""
    try:
        hints = hint_terms(question, hint_llm or llm)
    except Exception as e:                      # quota or provider trouble: dig with the question alone
        if on_hint_error:
            on_hint_error(e)
    deeper = dig(hints)
    if not deeper or not deeper.sources:
        steps.append({"action": "look_deeper", "hints": hints, "result": "nothing_found"})
        yield "done", {**first_done, "looked_deeper": True, "steps": steps}
        return

    steps.append({"action": "look_deeper", "hints": hints, "lectures": deeper.lectures,
                  "chunks_scored": deeper.chunks_scored, "result": f"{len(deeper.sources)} passages"})
    yield "step", {"action": "look_deeper", "stage": "found", "hints": hints,
                   "lectures": deeper.lectures, "chunks_scored": deeper.chunks_scored}
    for event, data in answer_events(question, deeper.sources, llm, relevance=None, min_relevance=min_relevance):
        if event == "done":
            steps.append({"action": "answer", "sources": len(deeper.sources), "result": _outcome(data)})
            data = {**data, "relevance": first_done["relevance"], "looked_deeper": True, "steps": steps}
        yield event, data


def _outcome(done: dict) -> str:
    if done["gated"]:
        return "gated"
    if done["not_covered"]:
        return "not_covered"
    return "grounded" if done["grounded"] else "ungrounded"
