"""
lecture_catalog.py — map lecture transcript folders to their YouTube videos.

Each folder under data/raw/<COURSE>/ is named after the video file it was
transcribed from, which is the YouTube title with ':' '|' '?' sanitised and a
resolution suffix added:

    YouTube title:  "Lecture 25: Socket Programming – II"
    folder:         "Lecture 25_ Socket Programming – II_480p"

The original matcher (1_metadata_builder.py) compared titles by word overlap
and dropped every number and every word shorter than three letters, so
"Socket Programming – I" and "Socket Programming – II" looked identical and
72 of 527 lectures were linked to the wrong video.

This module matches on the *whole* normalised title, numbers and roman
numerals included, and treats anything it cannot match exactly as an error
instead of guessing.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

_RESOLUTION_SUFFIX = re.compile(r"[ _]*\d{3,4}p(?: \(\d+\))?$")
_COPY_SUFFIX = re.compile(r" \(\d+\)$")
_NUMBER_PREFIX = re.compile(r"^(?:lecture\s*)?(\d+)\s*_\s*", re.IGNORECASE)
_VIDEO_ID_IN_NAME = re.compile(r"youtube video #([A-Za-z0-9_-]{11})")


def normalize_title(text: str) -> str:
    """Lowercase, keep only letters and digits, collapse whitespace.

    Numbers and short tokens ("ii", "4") are kept on purpose: they are what
    tells "Part 1" from "Part 2".
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text).split())


def strip_download_suffix(folder: str) -> str:
    """'Lecture 3_Foo_480p (1)' -> 'Lecture 3_Foo'."""
    return _COPY_SUFFIX.sub("", _RESOLUTION_SUFFIX.sub("", folder))


def is_copy(folder: str) -> bool:
    """True for a second download of the same file, e.g. '..._480p (1)'."""
    return bool(_COPY_SUFFIX.search(folder))


def folder_prefix_number(folder: str) -> int | None:
    """The download-order prefix some courses use ('40_Basic Pipelining')."""
    m = _NUMBER_PREFIX.match(folder)
    return int(m.group(1)) if m else None


def folder_title_keys(folder: str) -> list[str]:
    """Normalised titles to look up, most specific first."""
    base = strip_download_suffix(folder)
    keys = [normalize_title(base)]
    m = _NUMBER_PREFIX.match(base)
    if m:
        keys.append(normalize_title(base[m.end():]))
    return keys


def video_id_in_name(folder: str) -> str | None:
    """Folders that were saved as 'youtube video #<id>' carry the ID itself."""
    m = _VIDEO_ID_IN_NAME.search(folder)
    return m.group(1) if m else None


class CatalogError(ValueError):
    """Raised when folders cannot be matched one-to-one to videos."""


def match_course(folders: list[dict], playlist: list[dict]) -> list[dict]:
    """Match one course's folders to its playlist.

    folders:  [{"folder": str, "content_hash": str}, ...]
    playlist: [{"video_id": str, "playlist_index": int, "title": str}, ...]

    Returns one dict per folder with video_id, title, playlist_index, match
    and duplicate_of (the folder holding an identical transcript, or None).
    Raises CatalogError if any folder is unmatched or two distinct lectures
    claim the same video.
    """
    by_id = {v["video_id"]: v for v in playlist}
    by_title: dict[str, list[dict]] = defaultdict(list)
    for v in playlist:
        by_title[normalize_title(v["title"])].append(v)

    # 1. Identical transcripts are the same lecture downloaded twice; keep one.
    canonical_of: dict[str, str] = {}
    by_hash: dict[str, list[str]] = defaultdict(list)
    for f in folders:
        by_hash[f["content_hash"]].append(f["folder"])
    for names in by_hash.values():
        names.sort(key=lambda n: (is_copy(n), folder_prefix_number(n) or 0, n))
        for n in names:
            canonical_of[n] = names[0]

    # 2. Match each folder by exact normalised title (or the ID in its name).
    out, unmatched = {}, []
    same_title_groups: dict[tuple, list[str]] = defaultdict(list)
    for f in folders:
        name = f["folder"]
        vid = video_id_in_name(name)
        if vid:
            v = by_id.get(vid)
            out[name] = {"video_id": vid, "match": "id-in-folder-name",
                         "title": v["title"] if v else None,
                         "playlist_index": v["playlist_index"] if v else None}
            continue
        hits = next((by_title[k] for k in folder_title_keys(name) if by_title.get(k)), [])
        if not hits:
            unmatched.append(name)
        elif len(hits) == 1:
            out[name] = {**_pick(hits[0]), "match": "exact"}
        else:
            # Several videos share this exact title (e.g. untitled parts).
            same_title_groups[tuple(h["video_id"] for h in hits)].append(name)
    if unmatched:
        raise CatalogError(f"no exact title match for: {unmatched}")

    # 3. Same-title videos: pair distinct lectures with videos in playlist order.
    for vids, names in same_title_groups.items():
        lectures = sorted({canonical_of[n] for n in names},
                          key=lambda n: (folder_prefix_number(n) or 0, n))
        videos = sorted((by_id[v] for v in vids), key=lambda v: v["playlist_index"])
        if len(lectures) > len(videos):
            raise CatalogError(f"{len(lectures)} lectures but {len(videos)} videos titled "
                               f"{videos[0]['title']!r}")
        pick = {lec: videos[i] for i, lec in enumerate(lectures)}
        for n in names:
            out[n] = {**_pick(pick[canonical_of[n]]), "match": "exact+order"}

    # 4. Duplicates follow their canonical copy; distinct lectures need distinct videos.
    claimed: dict[str, str] = {}
    rows = []
    for f in folders:
        name = f["folder"]
        canon = canonical_of[name]
        row = {"folder": name, **out[canon], "duplicate_of": canon if canon != name else None}
        if row["duplicate_of"] is None:
            other = claimed.setdefault(row["video_id"], name)
            if other != name:
                raise CatalogError(f"{other!r} and {name!r} both matched video {row['video_id']}")
        rows.append(row)
    return rows


def _pick(v: dict) -> dict:
    return {"video_id": v["video_id"], "title": v["title"], "playlist_index": v["playlist_index"]}
