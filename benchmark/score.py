"""
Score benchmark runs against relevance labels.

    python benchmark/score.py [--runs benchmark/runs] [--qrels benchmark/qrels.jsonl]

Labels are per passage (0/1/2, see judge.py). A lecture's grade for a query
is the best grade of any judged passage from it. A query is *answerable* if
at least one passage got grade 2; ranking metrics use answerable queries only,
and the rest are reported separately (they test refusal, not ranking).

Metrics, all @10 over a config's top-10 unique lectures:
  MRR        1 / rank of the first lecture graded 2
  nDCG       graded lecture gains (2^g − 1), ideal = best judged lectures
  Recall     share of the query's grade-2 lectures that appear in the top 10
  P1-answer  the top-1 *passage* itself is graded 2 ("the first thing you read answers it")
Each with a 95% bootstrap confidence interval over queries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = 10


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def per_query_metrics(results: list[dict], seg_grade: dict, lec_grade: dict) -> dict:
    top = results[:K]
    gains = [lec_grade.get(r["video_id"], 0) for r in top]
    first = next((i for i, g in enumerate(gains, 1) if g == 2), None)
    dcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains, 1))
    ideal = sorted(lec_grade.values(), reverse=True)[:K]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    relevant = {v for v, g in lec_grade.items() if g == 2}
    return {
        "MRR": 1 / first if first else 0.0,
        "nDCG": dcg / idcg if idcg else 0.0,
        "Recall": len(relevant & {r["video_id"] for r in top}) / len(relevant),
        "P1-answer": float(bool(top) and seg_grade.get(top[0]["segment_id"], 0) == 2),
    }


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "benchmark" / "runs")
    ap.add_argument("--qrels", type=Path, default=ROOT / "benchmark" / "qrels.jsonl")
    ap.add_argument("--queries", type=Path, default=ROOT / "benchmark" / "queries.jsonl")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    queries = {q["qid"]: q for q in load_jsonl(args.queries)}
    seg_grade: dict[str, dict] = defaultdict(dict)
    lec_grade: dict[str, dict] = defaultdict(dict)
    for r in load_jsonl(args.qrels):
        seg_grade[r["qid"]][r["segment_id"]] = r["grade"]
        lec_grade[r["qid"]][r["video_id"]] = max(r["grade"], lec_grade[r["qid"]].get(r["video_id"], 0))

    answerable = sorted(q for q in queries if 2 in lec_grade.get(q, {}).values())
    unanswerable = sorted(q for q in queries if q in lec_grade and q not in answerable)
    unjudged = sorted(q for q in queries if q not in lec_grade)

    report, holes = {}, 0
    for run_file in sorted(args.runs.glob("*.jsonl")):
        runs = {r["qid"]: r["results"] for r in load_jsonl(run_file)}
        holes += sum(1 for q in answerable for r in runs.get(q, [])[:K]
                     if r["segment_id"] not in seg_grade[q])
        per_q = [per_query_metrics(runs.get(q, []), seg_grade[q], lec_grade[q]) for q in answerable]
        report[run_file.stem] = {
            m: {"mean": sum(x[m] for x in per_q) / len(per_q), "ci95": bootstrap_ci([x[m] for x in per_q])}
            for m in ("MRR", "nDCG", "Recall", "P1-answer")
        } if per_q else {}

    print(f"Queries: {len(queries)} total · {len(answerable)} answerable (≥1 passage graded 2) · "
          f"{len(unanswerable)} with no answering passage · {len(unjudged)} not judged yet")
    if holes:
        print(f"warning: {holes} retrieved passages have no label (counted as 0)")
    metrics = ("MRR", "nDCG", "Recall", "P1-answer")
    print("\n| config | " + " | ".join(f"{m}@{K}" if m != "P1-answer" else m for m in metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1].get("MRR", {}).get("mean", 0)):
        cells = [f"{m[x]['mean']:.3f} ({m[x]['ci95'][0]:.2f}–{m[x]['ci95'][1]:.2f})" for x in metrics] if m else ["—"] * 4
        print(f"| {name} | " + " | ".join(cells) + " |")

    intents = defaultdict(lambda: [0, 0])
    for q in answerable + unanswerable:
        intents[queries[q]["intent"]][q in answerable] += 1
    print("\nAnswerable by intent: " + ", ".join(
        f"{k} {v[1]}/{sum(v)}" for k, v in sorted(intents.items())))

    if args.json_out:
        args.json_out.write_text(json.dumps({"answerable": len(answerable), "unanswerable": unanswerable,
                                             "configs": report}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
