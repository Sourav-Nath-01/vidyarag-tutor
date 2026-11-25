"""
Follow-up questions.

A follow-up such as "what is its worst case?" can't be searched on its own:
"its" means nothing to the retriever. Before retrieval, a small LLM rewrites it
into a standalone question ("what is the worst case of quicksort?") using the
last few turns of the conversation.

The conversation is used only to work out what the follow-up refers to. The
answer is still written only from the lecture passages retrieved for the
rewritten question, so citations always point at those passages and nothing
from an earlier answer is treated as evidence.

The server keeps no conversation state: the client sends the recent turns with
each follow-up, so any replica (or a fresh one after scale-to-zero) can serve it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

MAX_TURNS = 3            # older turns rarely matter for "it"/"that" and cost tokens
ANSWER_CHARS = 600       # enough of each earlier answer to know what it was about
MAX_QUESTION_CHARS = 300

REWRITE_SYSTEM_PROMPT = """\
You turn a student's follow-up question into a standalone search query for a \
library of computer-science lecture videos.

Rules:
- Use the conversation only to work out what the follow-up refers to ("it", \
"this", "that algorithm", "the second one", "why?", "what about ...").
- Keep the student's intent. Don't answer the question, and don't add facts or \
topics that aren't in the follow-up or the conversation.
- If the follow-up already makes sense on its own, or starts a new topic, \
return it unchanged.
- Reply with the standalone question only, on one line."""

_CITATION = re.compile(r"\s*\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")
_LABEL = re.compile(r"^(?:standalone(?:\s+(?:question|query))?|rewritten(?:\s+question)?|question|query)\s*[:\-–]\s*",
                    re.IGNORECASE)


@dataclass(frozen=True)
class Turn:
    """One earlier exchange: what was asked (ideally its standalone form) and the answer."""
    question: str
    answer: str = ""


class ChatModel(Protocol):
    model: str

    def stream(self, messages: list[dict]) -> Iterator[str]: ...


def _plain(text: str, limit: int) -> str:
    """Answer text without citation markers or bold, cut at a word boundary."""
    text = " ".join(_CITATION.sub("", text).replace("**", "").split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    return text


def build_rewrite_messages(question: str, history: Iterable[Turn]) -> list[dict]:
    lines = []
    for turn in list(history)[-MAX_TURNS:]:
        lines.append(f"Student: {_plain(turn.question, MAX_QUESTION_CHARS)}")
        if turn.answer.strip():
            lines.append(f"Tutor: {_plain(turn.answer, ANSWER_CHARS)}")
    conversation = "\n".join(lines)
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation:\n{conversation}\n\nFollow-up: {question}\n\nStandalone question:"},
    ]


def clean_rewrite(reply: str) -> str | None:
    """The model's reply as one clean question, or None if it isn't usable."""
    for line in reply.strip().splitlines():
        line = _LABEL.sub("", line.strip()).strip().strip("\"'`“”‘’").strip()
        if line:
            break
    else:
        return None
    line = " ".join(line.split())
    if len(line) < 3 or len(line) > MAX_QUESTION_CHARS or "NOT_COVERED" in line:
        return None
    return line


def same_question(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^\w]+", " ", s.lower()).strip()  # noqa: E731
    return norm(a) == norm(b)


def rewrite_question(question: str, history: Iterable[Turn], llm: ChatModel) -> str:
    """
    The follow-up as a standalone question, or the question unchanged when
    there's no history or the model's reply isn't usable. LLM errors propagate,
    so the caller can decide how to fall back.
    """
    question = " ".join(question.split())
    turns = [t for t in history if t.question.strip()]
    if not turns:
        return question
    reply = "".join(llm.stream(build_rewrite_messages(question, turns)))
    return clean_rewrite(reply) or question
