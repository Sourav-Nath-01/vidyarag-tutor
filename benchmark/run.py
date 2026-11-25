"""
Run every benchmark query through several retrieval configurations.

    python benchmark/run.py                  # all configs → benchmark/runs/
    python benchmark/run.py --pool           # also write benchmark/pool.jsonl

Meant to run inside the production Docker image (CI: benchmark.yml), so the
numbers describe exactly the models and index that are deployed. Each config
returns its top-10 unique lectures; --pool collects every distinct passage
any config returned, with enough text for a relevance judge to read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))

# name → retriever.search() switches. "full" is what production serves.
CONFIGS = {
    "dense":        {"use_bm25": False, "use_rerank": False},
    "hybrid":       {"use_bm25": True,  "use_rerank": False},
    "dense+rerank": {"use_bm25": False, "use_rerank": True},
    "full":         {"use_bm25": True,  "use_rerank": True},
}
TOP_K = 10


def video_id(url: str) -> str:
    return url.split("v=", 1)[1].split("&", 1)[0]


def passage(seg: dict, words: int = 160) -> str:
    text = " ".join(seg.get("transcript", "").split()[:words])
    ocr = " ".join(seg.get("ocr_text", "").replace("\n---\n", " ").split()[:40])
    return text + (f"\n[slide text] {ocr}" if ocr and not seg.get("ocr_failed") else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, default=ROOT / "benchmark" / "queries.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "benchmark" / "runs")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--strategy", default="c3", help="index to search (demo = tiny local smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    args = ap.parse_args()

    import retriever

    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()][:args.limit]
    metadata = retriever._load_index(args.strategy)["metadata"]
    by_id = {s["segment_id"]: s for s in metadata}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool: dict[tuple[str, str], dict] = {}
    for name in args.configs.split(","):
        t0, rows = time.time(), []
        for q in queries:
            results = retriever.search(q["query"], strategy=args.strategy, top_k=TOP_K, **CONFIGS[name])
            ranked = [{"rank": r["rank"], "segment_id": r["segment_id"],
                       "video_id": video_id(r["youtube_url"]), "start_sec": r["start_sec"],
                       "score": r["retrieval_score"]} for r in results]
            rows.append({"qid": q["qid"], "results": ranked})
            for r in ranked:
                key = (q["qid"], r["segment_id"])
                if key not in pool:
                    seg = by_id[r["segment_id"]]
                    pool[key] = {"qid": q["qid"], "query": q["query"], "segment_id": r["segment_id"],
                                 "video_id": r["video_id"], "course": seg["course_name"],
                                 "lecture": seg["lecture_title"], "start_sec": seg["start_sec"],
                                 "passage": passage(seg), "found_by": []}
                pool[key]["found_by"].append(name)
        (args.out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name:13} {len(queries)} queries in {time.time() - t0:.0f}s", flush=True)

    if args.pool:
        out = ROOT / "benchmark" / "pool.jsonl"
        out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pool.values()))
        per_q = len(pool) / len(queries)
        print(f"pool: {len(pool)} passages to judge ({per_q:.1f} per query) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
