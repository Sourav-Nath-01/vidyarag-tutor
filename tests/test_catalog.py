"""
Guards for data/catalog/lectures.json, the folder → YouTube video mapping.

v1 linked 71 of 522 lectures to the wrong video because its title matcher
ignored numbers ("Socket Programming – I" == "Socket Programming – II").
These tests pin the invariants that bug broke, plus unit tests of the matcher.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_collection"))

from lecture_catalog import (  # noqa: E402
    CatalogError,
    folder_title_keys,
    match_course,
    normalize_title,
    video_id_in_name,
)

CATALOG = json.loads((ROOT / "data" / "catalog" / "lectures.json").read_text())
LECTURES = [r for r in CATALOG if r["duplicate_of"] is None]


# ── the committed catalogue ──────────────────────────────────────────────────

def test_catalog_covers_every_course():
    assert {r["course_id"] for r in CATALOG} == {
        "cn", "coa", "cv", "daa", "dbms", "dl", "dsa", "ml", "os"}
    assert len(LECTURES) == 522


def test_every_lecture_has_a_video_and_title():
    for r in LECTURES:
        assert len(r["video_id"]) == 11, r["folder"]
        assert r["title"] and r["playlist_index"], r["folder"]


def test_no_two_lectures_share_a_video():
    dupes = [v for v, n in Counter(r["video_id"] for r in LECTURES).items() if n > 1]
    assert not dupes, f"videos linked to several lectures: {dupes}"


def test_no_two_lectures_share_a_playlist_position():
    keys = Counter((r["course_id"], r["playlist_index"]) for r in LECTURES)
    assert not [k for k, n in keys.items() if n > 1]


def test_folder_title_equals_video_title():
    """The regression test: numbers and part markers must agree exactly."""
    for r in LECTURES:
        if r["match"] == "id-in-folder-name":
            assert video_id_in_name(r["folder"]) == r["video_id"]
            continue
        assert normalize_title(r["title"]) in folder_title_keys(r["folder"]), (
            f"{r['folder']!r} is linked to {r['title']!r}")


def test_duplicates_point_at_a_real_lecture_with_the_same_video():
    by_folder = {r["folder"]: r for r in LECTURES}
    for r in CATALOG:
        if r["duplicate_of"] is not None:
            assert r["video_id"] == by_folder[r["duplicate_of"]]["video_id"]


def test_deployed_index_is_built_from_this_catalogue():
    """deploy/index.lock.json must pin the release built from the current inputs.

    rebuild-index.yml tags each release with a hash of these files, so editing
    the catalogue without rebuilding and re-pinning fails here, not in prod.
    """
    import hashlib

    inputs = ["data/catalog/lectures.json", "data/catalog/c3_segment_map.json",
              "scripts/data_fix/rebuild_c3_index.py",
              "src/retrieval/embedder.py", "src/retrieval/bm25_builder.py"]
    digest = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in inputs)).hexdigest()[:12]
    lock = json.loads((ROOT / "deploy" / "index.lock.json").read_text())
    assert lock["release"].endswith(f"/index-c3-{digest}"), (
        f"index.lock.json is stale: inputs hash to index-c3-{digest}; "
        "push to run rebuild-index.yml, then pin its release")
    assert set(lock["files"]) == {"faiss_c3.index", "metadata_c3.json", "bm25_c3.pkl"}


# ── the matcher ──────────────────────────────────────────────────────────────

def _pl(*titles):
    return [{"video_id": f"vid{i:08d}", "playlist_index": i, "title": t}
            for i, t in enumerate(titles, 1)]


def _folders(*names):
    return [{"folder": n, "content_hash": n} for n in names]


def test_part_numbers_are_not_ignored():
    pl = _pl("Lecture 24: Socket Programming – I", "Lecture 25: Socket Programming – II")
    rows = match_course(_folders("Lecture 25_ Socket Programming – II_480p",
                                 "Lecture 24 _ Socket Programming – I_480p"), pl)
    assert {r["folder"]: r["playlist_index"] for r in rows} == {
        "Lecture 25_ Socket Programming – II_480p": 2,
        "Lecture 24 _ Socket Programming – I_480p": 1,
    }


def test_sanitised_punctuation_and_prefixes_match():
    pl = _pl("Lecture 01: What is an Algorithm?", "#2 Database Architecture | Intro",
             "Basic Pipelining Concepts")
    rows = match_course(_folders("Lecture 01_ What is an Algorithm _1080p",
                                 "#2 Database Architecture _ Intro_1080p",
                                 "40_Basic Pipelining Concepts_480p"), pl)
    assert [r["playlist_index"] for r in rows] == [1, 2, 3]
    assert all(r["match"] == "exact" for r in rows)


def test_identical_downloads_are_marked_duplicate():
    pl = _pl("Topic A")
    folders = [{"folder": "Topic A_480p", "content_hash": "same"},
               {"folder": "Topic A_480p (1)", "content_hash": "same"}]
    rows = {r["folder"]: r for r in match_course(folders, pl)}
    assert rows["Topic A_480p"]["duplicate_of"] is None
    assert rows["Topic A_480p (1)"]["duplicate_of"] == "Topic A_480p"


def test_repeated_titles_are_assigned_in_playlist_order():
    pl = _pl("Amdahl's Law", "Amdahl's Law")
    rows = match_course(_folders("16_Amdahl's Law_480p", "15_Amdahl's Law_480p"), pl)
    assert {r["folder"]: r["playlist_index"] for r in rows} == {
        "15_Amdahl's Law_480p": 1, "16_Amdahl's Law_480p": 2}


def test_video_id_in_folder_name_is_used():
    pl = _pl("Lec 8.8 Adding Noise")
    pl[0]["video_id"] = "TtU9GhLeOk4"
    (row,) = match_course(_folders("youtube video #TtU9GhLeOk4_720p"), pl)
    assert (row["video_id"], row["title"]) == ("TtU9GhLeOk4", "Lec 8.8 Adding Noise")


def test_unmatched_folder_fails_instead_of_guessing():
    with pytest.raises(CatalogError, match="no exact title match"):
        match_course(_folders("Socket Programming – III_480p"),
                     _pl("Socket Programming – I", "Socket Programming – II"))


def test_two_lectures_claiming_one_video_fails():
    pl = _pl("Intro")
    with pytest.raises(CatalogError, match="both matched"):
        match_course(_folders("Intro_480p", "Intro_720p"), pl)
