"""
Label pooled passages with an LLM relevance judge → benchmark/qrels.jsonl.

    JUDGE_API_KEY=... python benchmark/judge.py            # resumes where it stopped

Works with any OpenAI-compatible chat API (default: Groq's free tier). The key
is read from the environment only and never written anywhere. Judgments are
appended to qrels.jsonl as they arrive, so when the provider's daily limit is
hit the script stops cleanly and the next run continues from there.

Grades (TREC-style):
  2  the passage itself explains or answers the question
  1  on the topic and partly useful, but doesn't answer it
  0  not relevant
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "v1"

SYSTEM = """You are a strict relevance assessor for a search engine over university \
lecture transcripts (speech-to-text, so expect transcription errors).

For each numbered passage, decide how well THAT PASSAGE helps a student with the question:
  2 = the passage itself explains or answers the question (the core of what was asked)
  1 = on the topic, and would partly help, but does not answer the question
  0 = not relevant, or only shares words with the question

Judge only the passage text, not the lecture title. Do not reward a passage
for being from the right course. Be consistent across passages.

Reply with JSON only: {"judgments": [{"id": <n>, "grade": 0|1|2, "why": "<= 12 words"}]}"""


class DailyLimit(Exception):
    pass


_limits_shown = False


def _show_limits(headers) -> None:
    """Print the account's real quota once (providers send it as headers)."""
    global _limits_shown
    if not _limits_shown:
        keys = ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens")
        found = {k: headers.get(k) for k in keys if headers.get(k)}
        if found:
            print("provider limits:", found, flush=True)
        _limits_shown = True


def chat(base_url: str, key: str, model: str, messages: list, max_retries: int = 6) -> str:
    body = {"model": model, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"}}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "vidyarag-benchmark"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                _show_limits(resp.headers)
                return json.load(resp)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 429:
                wait = float(e.headers.get("retry-after") or 30)
                if wait > 300 or "per day" in detail.lower():
                    raise DailyLimit(detail) from None
                time.sleep(wait + 1)
                continue
            if e.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            raise SystemExit(f"judge API error {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    raise SystemExit("judge API kept failing")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=ROOT / "benchmark" / "pool.jsonl")
    ap.add_argument("--qrels", type=Path, default=ROOT / "benchmark" / "qrels.jsonl")
    # `or`: CI passes unset repository variables as empty strings
    ap.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL") or "https://api.groq.com/openai/v1")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL") or "openai/gpt-oss-120b")
    ap.add_argument("--batch", type=int, default=10, help="passages per request")
    ap.add_argument("--max-requests", type=int, default=10_000)
    args = ap.parse_args()

    key = os.getenv("JUDGE_API_KEY")
    if not key:
        raise SystemExit("set JUDGE_API_KEY (never commit it)")

    pool = [json.loads(l) for l in args.pool.read_text().splitlines() if l.strip()]
    done = set()
    if args.qrels.exists():
        done = {(r["qid"], r["segment_id"]) for r in map(json.loads, args.qrels.read_text().splitlines())}
    todo = defaultdict(list)
    for p in pool:
        if (p["qid"], p["segment_id"]) not in done:
            todo[p["qid"]].append(p)
    remaining = sum(map(len, todo.values()))
    print(f"{len(pool)} pooled passages, {len(done)} already judged, {remaining} to go "
          f"(model {args.model})", flush=True)

    requests_made = judged = 0
    with open(args.qrels, "a", encoding="utf-8") as out:
        try:
            for qid, passages in todo.items():
                for i in range(0, len(passages), args.batch):
                    if requests_made >= args.max_requests:
                        raise DailyLimit("--max-requests reached")
                    batch = passages[i:i + args.batch]
                    numbered = "\n\n".join(
                        f"[{n}] ({p['course']} — {p['lecture']})\n{p['passage']}"
                        for n, p in enumerate(batch, 1))
                    reply = chat(args.base_url, key, args.model, [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Question: {batch[0]['query']}\n\n{numbered}"},
                    ])
                    requests_made += 1
                    grades = {j["id"]: j for j in json.loads(reply).get("judgments", [])}
                    for n, p in enumerate(batch, 1):
                        j = grades.get(n)
                        if j is None or j.get("grade") not in (0, 1, 2):
                            continue   # left unjudged; picked up on the next run
                        out.write(json.dumps({
                            "qid": qid, "segment_id": p["segment_id"], "video_id": p["video_id"],
                            "grade": j["grade"], "why": j.get("why", ""),
                            "judge": args.model, "prompt": PROMPT_VERSION,
                        }, ensure_ascii=False) + "\n")
                        judged += 1
                    out.flush()
        except DailyLimit as e:
            print(f"stopping: provider limit reached ({str(e)[:120]}) — rerun later to continue")
    print(f"judged {judged} passages in {requests_made} requests; "
          f"{remaining - judged} left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
