"""
Tests for grounded answers: src/answer/grounded.py and POST /ask.

The LLM, the reranker and the retriever are all fakes, so this runs in CI with
no API key, no models and no network.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from answer.grounded import (  # noqa: E402
    NOT_COVERED, answer_events, build_sources, check_answer, format_timestamp)
from answer.llm import RateLimited  # noqa: E402


def _seg(sid, vid, start, text, lecture="Socket Programming – II", course="Computer Networks"):
    return {"segment_id": sid, "video_id": vid, "youtube_url": f"https://www.youtube.com/watch?v={vid}",
            "youtube_deep_link": f"https://www.youtube.com/watch?v={vid}&t={int(start)}s",
            "start_sec": start, "end_sec": start + 60, "transcript": text, "course_name": course,
            "lecture_title": lecture, "ocr_text": "", "ocr_failed": True}


METADATA = [
    _seg("cn-lec026-c3-001", "JExfKvUgrtI", 0.0, "A server socket calls bind then listen."),
    _seg("cn-lec026-c3-002", "JExfKvUgrtI", 60.0, "Then accept returns a new connected socket."),
    _seg("cn-lec049-c3-001", "GmrbL1-7qyU", 950.0, "Ethernet uses CSMA/CD.", lecture="Ethernet (Contd.)"),
]
INDEX_OF = {s["segment_id"]: i for i, s in enumerate(METADATA)}


class FakeLLM:
    model = "fake-model"

    def __init__(self, tokens=None, error=None):
        self.tokens = tokens if tokens is not None else [
            "A TCP server binds and listens [1]. ", "Accept then returns a socket [1]."]
        self.error = error
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        yield from self.tokens


# ── grounded.py ──────────────────────────────────────────────────────────────

def test_format_timestamp():
    assert format_timestamp(950.4) == "15:50"
    assert format_timestamp(3725) == "1:02:05"


def test_sources_include_next_chunk_of_the_same_lecture_only():
    s1, s2 = build_sources(["cn-lec026-c3-001", "cn-lec026-c3-002"], METADATA, INDEX_OF)
    assert "accept returns" in s1.text           # continued into the next chunk
    assert "CSMA" not in s2.text                 # but never into another lecture
    assert (s1.n, s2.n) == (1, 2)
    assert s1.public()["timestamp"] == "0:00" and "text" not in s1.public()


def test_check_answer_counts_valid_and_invalid_citations():
    c = check_answer("Bind comes first [1]. Then listen [1, 2]. Also UDP exists [7].", n_sources=2)
    assert c.cited == [1, 2] and c.invalid_citations == [7] and not c.grounded


def test_check_answer_coverage_and_grounding():
    c = check_answer("The server binds to a port first [1]. It then waits for clients to connect.", 2)
    assert c.grounded and c.sentences == 2 and c.cited_sentences == 1
    assert c.as_dict()["citation_coverage"] == 0.5


def test_citation_after_the_full_stop_counts_for_that_sentence():
    # The model often cites after the punctuation ("…memory space. [1]"); seen
    # live as "0% of sentences cited" on a fully cited list.
    text = ("A process owns its code, heap and stack. [1]\n"
            "1. **Memory** – A thread shares the heap of its process. [1, 2]  \n"
            "2. **Lifetime** – Threads end when their process ends. [2] Each thread has its own stack. [1]\n"
            "These points cover the main differences between them.")
    c = check_answer(text, 2)
    assert (c.sentences, c.cited_sentences) == (5, 4)
    assert c.cited == [1, 2] and c.grounded


def test_answer_without_citations_is_not_grounded():
    assert not check_answer("Sockets are endpoints for communication between two programs.", 3).grounded


def test_not_covered_sentinel_is_never_streamed():
    llm = FakeLLM(tokens=["NOT_", "COVERED"])
    events = list(answer_events("q", build_sources(["cn-lec026-c3-001"], METADATA, INDEX_OF), llm,
                                relevance=2.0, min_relevance=-4))
    assert [e for e, _ in events] == ["sources", "done"]
    assert events[-1][1]["not_covered"] and events[-1][1]["answer"] == ""


def test_gate_skips_the_llm_when_nothing_is_relevant():
    llm = FakeLLM()
    events = list(answer_events("q", build_sources(["cn-lec026-c3-001"], METADATA, INDEX_OF), llm,
                                relevance=-9.5, min_relevance=-4))
    assert events[-1][1]["not_covered"] and events[-1][1]["gated"]
    assert llm.calls == []


def test_short_reply_on_hold_is_still_streamed():
    llm = FakeLLM(tokens=["No [1]."])
    events = list(answer_events("q", build_sources(["cn-lec026-c3-001"], METADATA, INDEX_OF), llm,
                                relevance=2.0, min_relevance=-4))
    assert ("token", {"t": "No [1]."}) in events


# ── llm.py ───────────────────────────────────────────────────────────────────
class _FakeStreamResponse:
    """Mimics requests: decode_unicode=True decodes with .encoding, which
    requests sets to ISO-8859-1 for a text/* response without a charset."""
    status_code, headers, encoding = 200, {}, "ISO-8859-1"

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode=False):
        for line in self.body.split(b"\n"):
            yield line.decode(self.encoding) if decode_unicode else line


def test_stream_decodes_utf8_even_without_a_charset(monkeypatch):
    from answer.llm import LLMClient
    chunk = {"choices": [{"delta": {"content": "the receiver’s window — रिसीवर"}}]}
    body = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n".encode()
    fake = types.ModuleType("requests")
    fake.RequestException = Exception
    fake.post = lambda *a, **kw: _FakeStreamResponse(body)
    monkeypatch.setitem(sys.modules, "requests", fake)
    assert list(LLMClient(api_key="k").stream([])) == ["the receiver’s window — रिसीवर"]


# ── POST /ask ────────────────────────────────────────────────────────────────

def _fake_retriever(index_dir: Path, scores: list[float]) -> types.ModuleType:
    fake = types.ModuleType("retriever")
    fake.INDEXES, fake.EMBEDDING_MODEL, fake.RERANKER_MODEL = index_dir, "fake-embedder", "fake-reranker"
    fake._index_cache, fake.calls = {}, []
    fake._get_embed_model = lambda: object()
    fake.reranker = types.SimpleNamespace(predict=lambda pairs: scores[:len(pairs)])
    fake._get_rerank_model = lambda: fake.reranker

    def _load_index(strategy):
        fake._index_cache[strategy] = {"metadata": METADATA}
        return fake._index_cache[strategy]

    def search(query, strategy, top_k, **kwargs):
        fake.calls.append(query)
        return [{"segment_id": s["segment_id"]} for s in METADATA[:top_k]]

    fake._load_index, fake.search = _load_index, search
    return fake


@pytest.fixture(scope="module")
def api():
    index_dir = Path(tempfile.mkdtemp())
    for name in ("faiss_c3.index", "metadata_c3.json"):
        (index_dir / name).write_text("stub")
    mp = pytest.MonkeyPatch()
    mp.setenv("DEFAULT_STRATEGY", "c3")
    mp.setenv("SEARCH_QUEUE_TIMEOUT_SEC", "0.1")
    mp.delenv("LLM_API_KEY", raising=False)
    mp.syspath_prepend(str(_ROOT))
    mp.setitem(sys.modules, "retriever", _fake_retriever(index_dir, [3.0, 1.0, -2.0]))
    sys.modules.pop("api.app", None)
    try:
        yield importlib.import_module("api.app")
    finally:
        sys.modules.pop("api.app", None)
        mp.undo()


@pytest.fixture(scope="module")
def client(api):
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def llm(api, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(api, "_llm", fake)
    api._answer_cache.clear()
    return fake


def _events(resp) -> list[tuple[str, dict]]:
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_ask_is_disabled_without_a_key(client):
    assert client.get("/info").json()["answers_enabled"] is False
    assert client.post("/ask", json={"question": "what is bind"}).status_code == 503


def test_ask_streams_sources_tokens_and_a_checked_answer(client, llm):
    r = client.post("/ask", json={"question": "how does a TCP server accept clients",
                                  "segment_ids": ["cn-lec026-c3-001", "cn-lec049-c3-001"]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    sources = events[0][1]["sources"]
    assert [s["n"] for s in sources] == [1, 2]                     # [n] = the order given
    assert sources[1]["deep_link"].endswith("t=950s") and sources[1]["timestamp"] == "15:50"
    done = events[-1][1]
    assert done["grounded"] and done["cited"] == [1] and done["model"] == "fake-model"
    prompt = llm.calls[0][1]["content"]
    assert "accept returns a new connected socket" in prompt       # neighbour chunk included


def test_ask_rejects_unknown_segment_ids(client, llm):
    r = client.post("/ask", json={"question": "anything here", "segment_ids": ["nope-c3-001"]})
    assert r.status_code == 422


def test_ask_searches_when_no_segment_ids_are_given(client, llm, api):
    before = len(api._ret.calls)
    r = client.post("/ask", json={"question": "what does listen do", "top_k": 2})
    assert len(api._ret.calls) == before + 1 and len(_events(r)[0][1]["sources"]) == 2


def test_irrelevant_passages_are_not_sent_to_the_llm(client, llm, api, monkeypatch):
    monkeypatch.setattr(api._ret.reranker, "predict", lambda pairs: [-9.0] * len(pairs))
    done = _events(client.post("/ask", json={"question": "how does bitcoin mining work",
                                             "segment_ids": ["cn-lec026-c3-001"]}))[-1][1]
    assert done["not_covered"] and done["gated"] and llm.calls == []


def test_rate_limit_becomes_an_error_event_after_sources(client, llm):
    llm.error = RateLimited("429", retry_after=30)
    events = _events(client.post("/ask", json={"question": "what is bind",
                                               "segment_ids": ["cn-lec026-c3-001"]}))
    assert [e for e, _ in events] == ["sources", "error"]
    assert events[1][1]["code"] == "rate_limited" and events[1][1]["retry_after"] == 30


def test_repeat_question_is_served_from_cache(client, llm):
    body = {"question": "What is  BIND", "segment_ids": ["cn-lec026-c3-001"]}
    first = _events(client.post("/ask", json=body))[-1][1]
    second = _events(client.post("/ask", json={**body, "question": "what is bind"}))[-1][1]
    assert first["cached"] is False and second["cached"] is True and len(llm.calls) == 1


def test_not_covered_reply(client, llm):
    llm.tokens = [NOT_COVERED]
    events = _events(client.post("/ask", json={"question": "explain quantum tunnelling",
                                               "segment_ids": ["cn-lec026-c3-001"]}))
    assert "token" not in [e for e, _ in events] and events[-1][1]["not_covered"]


# ── follow-ups ───────────────────────────────────────────────────────────────
from answer.followup import (  # noqa: E402
    MAX_TURNS, Turn, build_rewrite_messages, clean_rewrite, rewrite_question)

QUICKSORT = [Turn("What is quicksort?",
                  "Quicksort picks a **pivot** and partitions the array around it [1]. "
                  "It then sorts both parts recursively [1, 2].")]


def test_rewrite_prompt_uses_recent_turns_without_citations():
    history = [Turn(f"question {i}", f"answer {i} [1]") for i in range(5)] + QUICKSORT
    prompt = build_rewrite_messages("what is its worst case?", history)[1]["content"]
    assert "question 2" not in prompt and "question 3" in prompt   # last MAX_TURNS only
    assert MAX_TURNS == 3
    assert "[1]" not in prompt and "**" not in prompt
    assert prompt.rstrip().endswith("Follow-up: what is its worst case?\n\nStandalone question:")


@pytest.mark.parametrize("reply, expected", [
    ("What is the worst case of quicksort?", "What is the worst case of quicksort?"),
    ('Standalone question: "What is the worst case of quicksort?"', "What is the worst case of quicksort?"),
    ("Standalone question:\nWhy is quicksort O(n^2) in the worst case?\nExtra line",
     "Why is quicksort O(n^2) in the worst case?"),
    ("   ", None),
    ("NOT_COVERED", None),
    ("x" * 400, None),
])
def test_clean_rewrite(reply, expected):
    assert clean_rewrite(reply) == expected


def test_rewrite_resolves_a_follow_up():
    llm = FakeLLM(tokens=["What is the worst case ", "of quicksort?"])
    assert rewrite_question("what is its  worst case?", QUICKSORT, llm) == "What is the worst case of quicksort?"


def test_rewrite_without_history_or_usable_reply_keeps_the_question():
    llm = FakeLLM(tokens=[""])
    assert rewrite_question("what is a socket?", [], llm) == "what is a socket?"
    assert llm.calls == []                                          # nothing to resolve, no LLM call
    assert rewrite_question("what about its cost?", QUICKSORT, llm) == "what about its cost?"


@pytest.fixture
def rewriter(api, monkeypatch):
    fake = FakeLLM(tokens=["What is the worst case of quicksort?"])
    monkeypatch.setattr(api, "_rewriter", fake)
    return fake


def _turns(history):
    return [{"question": t.question, "answer": t.answer} for t in history]


def test_rewrite_endpoint_is_disabled_without_a_key(client):
    assert client.get("/info").json()["followups_enabled"] is False
    assert client.post("/rewrite", json={"question": "and its worst case?"}).status_code == 503


def test_rewrite_endpoint_returns_a_standalone_question(client, rewriter):
    r = client.post("/rewrite", json={"question": "what is its worst case?", "history": _turns(QUICKSORT)})
    body = r.json()
    assert r.status_code == 200 and body["rewritten"] is True and body["fallback"] is None
    assert body["standalone"] == "What is the worst case of quicksort?" and body["model"] == "fake-model"
    assert "Quicksort picks a pivot" in rewriter.calls[0][1]["content"]


def test_first_question_is_not_rewritten(client, rewriter):
    body = client.post("/rewrite", json={"question": "What is quicksort?"}).json()
    assert body["standalone"] == "What is quicksort?" and body["rewritten"] is False and rewriter.calls == []


def test_rewrite_falls_back_to_the_question_when_rate_limited(client, rewriter):
    rewriter.error = RateLimited("429", retry_after=20)
    body = client.post("/rewrite", json={"question": "what is its worst case?", "history": _turns(QUICKSORT)}).json()
    assert body["standalone"] == "what is its worst case?" and body["rewritten"] is False
    assert body["fallback"] == "rate_limited"


def test_rewrite_rejects_oversized_history(client, rewriter):
    history = [{"question": "q", "answer": "a"}] * 9
    assert client.post("/rewrite", json={"question": "and then?", "history": history}).status_code == 422
