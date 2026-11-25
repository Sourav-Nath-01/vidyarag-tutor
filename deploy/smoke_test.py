#!/usr/bin/env python3
"""
smoke_test.py — end-to-end check of a running VidyaRAG API (stdlib only)
=======================================================================
Waits for /ready, runs a real search, and checks input validation.
Used in three places:
  • CI, inside the freshly built container:  docker exec -i api python - <URL> < deploy/smoke_test.py
  • CD, against the live Azure URL right after a deploy
  • locally:  python deploy/smoke_test.py http://localhost:8000

Exit code 0 = healthy, 1 = failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(method: str, url: str, body: dict | None = None, timeout: float = 120):
    """Returns (status, json_body, headers); headers lookups are case-insensitive."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}"), resp.headers
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except ValueError:
            payload = {}
        return e.code, payload, e.headers


def ask_events(url: str, body: dict, timeout: float = 120) -> list[tuple[str, dict]]:
    """POST to an SSE endpoint and return its (event, data) pairs."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode()
    events = []
    for block in filter(None, text.split("\n\n")):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields and "data" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base_url", help="e.g. http://127.0.0.1:7860 or https://<app>.azurecontainerapps.io")
    p.add_argument("--timeout", type=int, default=300, help="max seconds to wait for /ready")
    p.add_argument("--query", default="how does binary search tree insertion work")
    p.add_argument("--expect-version", default=None,
                   help="keep waiting until /ready reports this version (e.g. the git SHA just "
                        "deployed), so we test the NEW revision rather than the old one")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    # 1) Wait for readiness (covers scale-from-zero cold starts on Azure)
    t0 = time.time()
    last = "no response yet"
    while True:
        try:
            status, body, _ = call("GET", f"{base}/ready", timeout=60)
            version_ok = args.expect_version is None or body.get("version") == args.expect_version
            if status == 200 and version_ok:
                print(f"✓ ready after {time.time() - t0:.0f}s  {body}")
                break
            last = f"HTTP {status} {body}" + ("" if version_ok else f" (waiting for version {args.expect_version})")
        except Exception as e:  # connection refused, DNS, timeout while waking up...
            last = f"{type(e).__name__}: {e}"
        if time.time() - t0 > args.timeout:
            print(f"✗ not ready after {args.timeout}s — last: {last}")
            return 1
        time.sleep(5)

    # 2) Service info + the search web page
    status, info, _ = call("GET", f"{base}/info")
    print(f"✓ GET /info → {status}  version={info.get('version')}  strategies={info.get('available_strategies')}")
    with urllib.request.urlopen(f"{base}/", timeout=60) as resp:
        page = resp.read().decode("utf-8", "replace")
        if resp.status != 200 or "VidyaRAG" not in page:
            print(f"✗ search page not served correctly (HTTP {resp.status})")
            return 1
    print("✓ GET / → search page (HTML) served")

    # 3) Real search through the whole pipeline
    status, res, headers = call("POST", f"{base}/search", {"query": args.query, "top_k": 3})
    if status != 200 or not res.get("results"):
        print(f"✗ search failed: HTTP {status} {res}")
        return 1
    top = res["results"][0]
    print(f"✓ POST /search → {res['n_results']} results in {res['latency_sec']}s "
          f"(server {headers.get('X-Response-Time-ms', '?')} ms)")
    print(f"  top hit: {top.get('course_name')} — {top.get('lecture_title')}  "
          f"score={top.get('retrieval_score'):.4f}")

    # 4) Validation still works
    status, _, _ = call("POST", f"{base}/search", {"query": "x"})
    if status != 422:
        print(f"✗ expected 422 for a too-short query, got {status}")
        return 1
    print("✓ invalid input rejected with 422")

    # 5) Grounded answers: disabled without an LLM key (CI), real call in production
    if not info.get("answers_enabled"):
        status, _, _ = call("POST", f"{base}/ask", {"question": args.query})
        if status != 503:
            print(f"✗ /ask should be 503 when answers are disabled, got {status}")
            return 1
        print("✓ POST /ask → 503 (answers disabled: no LLM key on this instance)")
    else:
        ids = [r["segment_id"] for r in res["results"] if r.get("segment_id")]
        events = ask_events(f"{base}/ask", {"question": args.query, "segment_ids": ids})
        kinds = [e for e, _ in events]
        done = next((d for e, d in events if e == "done"), None)
        error = next((d for e, d in events if e == "error"), None)
        if error and error.get("code") == "rate_limited":
            print("⚠ POST /ask → rate-limited by the LLM provider's free tier (not a deploy failure)")
        elif kinds[:1] != ["sources"] or done is None:
            print(f"✗ /ask stream malformed or failed: events={kinds} error={error}")
            return 1
        else:
            print(f"✓ POST /ask → {kinds.count('token')} tokens, grounded={done['grounded']} "
                  f"cited={done['cited']} not_covered={done['not_covered']} "
                  f"first token {done.get('first_token_ms')} ms · {done.get('model')}")

    # 6) Follow-ups: a question that only makes sense after the previous one
    history = [{"question": "What is quicksort?",
                "answer": "Quicksort picks a pivot, partitions the array around it and sorts both parts recursively."}]
    status, body, _ = call("POST", f"{base}/rewrite", {"question": "What is its worst case?", "history": history})
    if not info.get("followups_enabled"):
        if status != 503:
            print(f"✗ /rewrite should be 503 when follow-ups are disabled, got {status}")
            return 1
        print("✓ POST /rewrite → 503 (follow-ups disabled: no LLM key on this instance)")
    elif status != 200 or not body.get("standalone"):
        print(f"✗ /rewrite failed: HTTP {status} {body}")
        return 1
    elif body.get("fallback") == "llm_error":
        # Not a quota blip: the provider rejected the call (e.g. a retired model ID).
        print("✗ /rewrite → the LLM call failed (llm_error); check REWRITE_MODEL and the app logs")
        return 1
    elif body.get("fallback"):
        print(f"⚠ POST /rewrite → used the question as typed ({body['fallback']}); not a deploy failure")
    else:
        resolved = "quicksort" in body["standalone"].lower()
        print(f"{'✓' if resolved else '⚠'} POST /rewrite → “{body['standalone']}” "
              f"in {body['ms']} ms · {body.get('model')}")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
