# Retrieval benchmark (v2)

A lecture-level benchmark for the deployed search, replacing v1's grader. That
grader counted a hit when the course matched and the timestamp was within 90 s,
without checking the lecture.

## Method

Pooled relevance judgments, the standard way IR test collections (e.g. TREC)
are built when nobody can label all 16,050 passages for every query:

1. **Queries** ([`queries.jsonl`](queries.jsonl), 168). v1's 93 queries, 63 new
   student-style doubts spread over all 9 courses (comparisons, "why"
   questions, mechanisms from later lectures), and 12 questions the courses
   most likely don't cover, kept to test refusals later.
2. **Pool** ([`run.py`](run.py) → [`pool.jsonl`](pool.jsonl), 3,264 passages).
   Every query runs through 4 configurations (dense only, hybrid BM25+dense,
   dense+rerank, full = production). Every distinct passage in any
   configuration's top-10 lectures is pooled. 54% of pooled passages came from
   only one configuration.
3. **Labels** ([`judge.py`](judge.py) → `qrels.jsonl`). An LLM judge grades
   each passage 0 (not relevant), 1 (on topic, partly useful) or 2 (answers
   the question), reading only the passage text. Runs in CI
   ([`judge.yml`](../.github/workflows/judge.yml)) with the key in a repository
   secret, and resumes across the free tier's daily limits.
4. **Human check.** A sample of labels is re-judged by a person, and the
   agreement rate is reported next to the results.
5. **Scores** ([`score.py`](score.py)). Per configuration: lecture-level
   MRR / nDCG / Recall@10, and *P1-answer* (the first passage itself answers
   the question), each with a 95% bootstrap confidence interval. Ranking
   metrics use queries with at least one grade-2 passage; the rest are
   reported separately.

## Reproduce

```bash
python benchmark/run.py --pool              # CI: benchmark.yml (needs the C3 index + models)
JUDGE_API_KEY=... python benchmark/judge.py # CI: judge.yml
python benchmark/score.py
```

## Known limits

- Labels come from one LLM judge, so the human agreement check is what makes
  them trustworthy.
- Pooling only finds relevant passages that at least one configuration
  retrieved, so recall is relative to the pool, as in any pooled collection.
