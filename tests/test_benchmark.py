"""Unit tests for the benchmark metric maths (benchmark/score.py)."""

import importlib.util
import math
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "bench_score", Path(__file__).resolve().parents[1] / "benchmark" / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def _results(*videos):
    return [{"rank": i, "video_id": v, "segment_id": f"{v}-s"} for i, v in enumerate(videos, 1)]


def test_perfect_ranking():
    m = score.per_query_metrics(_results("a", "b"), {"a-s": 2}, {"a": 2, "b": 0})
    assert m == {"MRR": 1.0, "nDCG": 1.0, "Recall": 1.0, "P1-answer": 1.0}


def test_mrr_counts_only_grade_2_lectures():
    m = score.per_query_metrics(_results("x", "y", "a"), {}, {"x": 1, "a": 2})
    assert m["MRR"] == pytest.approx(1 / 3)
    assert m["P1-answer"] == 0.0   # top passage was only partly relevant


def test_lecture_found_but_passage_not_the_answer():
    """Right lecture at rank 1, but the retrieved passage itself isn't graded 2."""
    m = score.per_query_metrics(_results("a"), {"a-s": 1, "a-other": 2}, {"a": 2})
    assert m["MRR"] == 1.0 and m["P1-answer"] == 0.0


def test_ndcg_and_recall_partial():
    lec = {"a": 2, "b": 2, "c": 1}
    m = score.per_query_metrics(_results("c", "a", "z"), {}, lec)
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert m["nDCG"] == pytest.approx(dcg / idcg)
    assert m["Recall"] == 0.5


def test_only_top_10_counts():
    m = score.per_query_metrics(_results(*[f"n{i}" for i in range(10)], "a"), {}, {"a": 2})
    assert m["MRR"] == 0.0 and m["Recall"] == 0.0


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = score.bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi and 0.35 < lo and hi < 0.65
    assert score.bootstrap_ci([0.7] * 20) == pytest.approx((0.7, 0.7))
