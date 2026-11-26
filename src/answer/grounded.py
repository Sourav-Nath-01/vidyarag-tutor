"""
Grounded answers: a tutor-style answer built only from retrieved lecture
passages, with every claim cited back to a playable moment in a lecture.

The flow, as a stream of (event, data) pairs the API turns into SSE:

    sources  the numbered passages the answer may use ([n] = result card n)
    token    answer text as the model writes it
    done     what the citation check found (or that the lectures don't cover it)

Safeguards, all enforced in code rather than trusted to the model:
  • Relevance gate: if even the best passage scores below a threshold with the
    cross-encoder, answer "not covered" without calling the LLM at all.
  • The model must reply NOT_COVERED when the passages don't answer the
    question; that sentinel is never shown as an answer.
  • Citation check: [n] must point at a real source; answers with no valid
    citation are flagged as ungrounded; per-sentence citation coverage is
    reported so it can be evaluated and shown.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterator, Protocol

NOT_COVERED = "NOT_COVERED"
PASSAGE_WORDS = 220          # per source: the chunk plus the start of the next one

SYSTEM_PROMPT = f"""You are VidyaRAG Tutor, a teaching assistant for NPTEL computer-science \
lectures by IIT professors.

Answer the student's question using ONLY the numbered lecture passages below. They are \
speech-to-text transcripts, so silently fix obvious transcription mistakes.

Rules:
- End every sentence that states a fact with citations like [1] or [2][3] pointing to the \
passages that support it. Never cite a passage number that isn't listed.
- Use no outside knowledge. If the passages don't answer the question, reply with exactly \
{NOT_COVERED} and nothing else.
- If they answer only part of it, answer that part and say briefly what these lectures don't cover.
- Teach well: open with a direct one- or two-sentence answer, then explain step by step in \
plain language, using short paragraphs or a short list. Stay under 220 words.
- Refer to "the lecture", not to "passages" or "transcripts"."""

_CITE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# Citations written after the full stop ("…the heap. [1]") belong to the
# sentence before them; move them inside it before splitting into sentences.
_CITE_AFTER_STOP = re.compile(r"([.!?])((?:\s*\[\d{1,2}(?:\s*,\s*\d{1,2})*\])+)")


class ChatModel(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


@dataclass
class Source:
    n: int
    segment_id: str
    video_id: str
    course: str
    lecture: str
    start_sec: float
    deep_link: str
    text: str                       # what the model reads (chunk + start of the next one)
    end_sec: float = 0.0
    instructor: str = ""
    transcript: str = ""            # the chunk alone, for display
    ocr_text: str = ""
    score: float | None = None      # set when chosen by looking deeper

    def public(self) -> dict:
        """Everything a client needs to show and play the source, minus the prompt text."""
        d = asdict(self)
        d.pop("text")
        d["timestamp"] = format_timestamp(self.start_sec)
        return d


def format_timestamp(sec: float) -> str:
    s = int(sec)
    h, m = divmod(s // 60, 60)
    return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m}:{s % 60:02d}"


def rerank_passage(seg: dict) -> str:
    """The exact passage text retriever._rerank scores, so gate thresholds match search scores."""
    transcript = seg.get("transcript", "").strip()
    ocr = "" if seg.get("ocr_failed", True) else \
        seg.get("ocr_text", "").replace("\n---\n", " ").replace("\n", " ").strip()
    return (f"{ocr} {transcript}" if ocr else transcript)[:512]


def build_sources(segment_ids: list[str], metadata: list[dict], index_of: dict[str, int],
                  *, exclude_next: set[str] | None = None) -> list[Source]:
    """Turn retrieved chunk IDs into numbered sources with a little following context.

    Chunks are cut at slide changes, so an explanation often continues into the
    next chunk; the start of that chunk is included when it's the same lecture,
    unless it is in `exclude_next` (already a source of its own).
    """
    exclude_next = exclude_next or set()
    sources = []
    for sid in segment_ids:
        i = index_of.get(sid)
        if i is None:
            continue
        seg = metadata[i]
        vid = seg.get("video_id") or seg.get("youtube_url", "").split("v=")[-1][:11]
        text = seg.get("transcript", "")
        if (i + 1 < len(metadata) and metadata[i + 1].get("youtube_url") == seg.get("youtube_url")
                and metadata[i + 1].get("segment_id") not in exclude_next):
            text += " " + metadata[i + 1].get("transcript", "")
        ocr = seg.get("ocr_text", "") if not seg.get("ocr_failed", True) else ""
        body = " ".join(text.split()[:PASSAGE_WORDS])
        if ocr.strip():
            body = f"[slide] {' '.join(ocr.replace('---', ' ').split()[:40])}\n{body}"
        sources.append(Source(
            n=len(sources) + 1, segment_id=sid, video_id=vid,
            course=seg.get("course_name", ""), lecture=seg.get("lecture_title", ""),
            start_sec=float(seg.get("start_sec", 0)), deep_link=seg.get("youtube_deep_link", ""),
            text=body, end_sec=float(seg.get("end_sec", 0) or 0), instructor=seg.get("instructor", ""),
            transcript=" ".join(seg.get("transcript", "").split()), ocr_text=ocr.strip()))
    return sources


def build_messages(question: str, sources: list[Source]) -> list[dict]:
    passages = "\n\n".join(
        f"[{s.n}] {s.course} — {s.lecture} (at {format_timestamp(s.start_sec)})\n{s.text}"
        for s in sources)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Lecture passages:\n\n{passages}\n\nStudent's question: {question}"},
    ]


@dataclass
class AnswerCheck:
    not_covered: bool
    cited: list[int]
    invalid_citations: list[int]
    sentences: int
    cited_sentences: int

    @property
    def grounded(self) -> bool:
        return not self.not_covered and bool(self.cited) and not self.invalid_citations

    def as_dict(self) -> dict:
        d = asdict(self)
        d["grounded"] = self.grounded
        d["citation_coverage"] = round(self.cited_sentences / self.sentences, 3) if self.sentences else None
        return d


def check_answer(text: str, n_sources: int) -> AnswerCheck:
    """Verify an answer's citations against the sources it was given."""
    stripped = text.strip()
    if stripped.startswith(NOT_COVERED):
        return AnswerCheck(True, [], [], 0, 0)
    cited, invalid = set(), set()
    for m in _CITE.finditer(stripped):
        for part in m.group(1).split(","):
            n = int(part)
            (cited if 1 <= n <= n_sources else invalid).add(n)
    # Count sentences that state something (skip headings / very short fragments).
    attached = _CITE_AFTER_STOP.sub(lambda m: f"{m.group(2)}{m.group(1)}", stripped)
    sentences = [s for s in _SENTENCE_END.split(attached) if len(s.split()) >= 5]
    return AnswerCheck(False, sorted(cited), sorted(invalid), len(sentences),
                       sum(1 for s in sentences if _CITE.search(s)))


def answer_events(question: str, sources: list[Source], llm: ChatModel, *,
                  relevance: float | None, min_relevance: float) -> Iterator[tuple[str, dict]]:
    """Run one grounded answer as a stream of (event, data) pairs.

    LLM errors propagate to the caller, after `sources` has been sent, so the
    client still has the search results to show.
    """
    yield "sources", {"sources": [s.public() for s in sources]}
    if not sources or (relevance is not None and relevance < min_relevance):
        yield "done", {**AnswerCheck(True, [], [], 0, 0).as_dict(), "gated": True,
                       "relevance": relevance, "answer": ""}
        return

    parts: list[str] = []
    held = ""            # hold back the opening until we know it isn't NOT_COVERED
    releasing = False
    for token in llm.stream(build_messages(question, sources)):
        parts.append(token)
        if releasing:
            yield "token", {"t": token}
            continue
        held += token
        head = held.lstrip()
        if NOT_COVERED.startswith(head):
            continue                           # could still become the sentinel
        if head.startswith(NOT_COVERED):
            break                              # the model declined; don't spend more tokens
        releasing = True
        yield "token", {"t": held}
    if not releasing and held.strip() and not held.lstrip().startswith(NOT_COVERED):
        yield "token", {"t": held}             # a very short reply that was still on hold

    answer = "".join(parts)
    check = check_answer(answer, len(sources))
    yield "done", {**check.as_dict(), "gated": False, "relevance": relevance,
                   "answer": "" if check.not_covered else answer}
