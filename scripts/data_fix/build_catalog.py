"""
Build data/catalog/lectures.json: which YouTube video each lecture folder is.

    python scripts/data_fix/build_catalog.py --raw-dir ../vidyarag/data/raw

Fetches each course playlist with yt-dlp (cached in --playlists-dir), matches
folders to videos with src/data_collection/lecture_catalog.py, and fails
loudly if any folder cannot be matched exactly. The output is committed, so
every link correction is reviewable in git and checked by tests/test_catalog.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_collection"))
from lecture_catalog import CatalogError, match_course  # noqa: E402


def fetch_playlist(url: str, cache: Path) -> list[dict]:
    if not cache.exists():
        out = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--flat-playlist",
             "--print", "%(id)s\t%(playlist_index)s\t%(title)s", url],
            check=True, capture_output=True, text=True,
        ).stdout
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out)
    rows = []
    for line in cache.read_text().splitlines():
        vid, idx, title = line.split("\t", 2)
        rows.append({"video_id": vid, "playlist_index": int(idx), "title": title})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--playlists-dir", type=Path, default=Path(".cache/playlists"))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "catalog" / "lectures.json")
    args = ap.parse_args()

    catalog, errors = [], []
    for course_dir in sorted(p for p in args.raw_dir.iterdir() if p.is_dir()):
        metas = sorted(course_dir.glob("*/metadata.json"))
        if not metas:
            continue
        folders, old = [], {}
        for m in metas:
            meta = json.loads(m.read_text())
            mm = m.parent / "multimodal.json"
            folders.append({"folder": m.parent.name,
                            "content_hash": hashlib.sha1(mm.read_bytes()).hexdigest()})
            old[m.parent.name] = meta
        first = old[folders[0]["folder"]]
        course_id = first["course_id"]
        playlist = fetch_playlist(first["playlist_url"], args.playlists_dir / f"{course_id}.tsv")
        try:
            rows = match_course(folders, playlist)
        except CatalogError as e:
            errors.append(f"[{course_id}] {e}")
            continue
        for r in rows:
            o = old[r["folder"]]
            catalog.append({
                "course_id": course_id,
                **r,
                "old_video_id": o.get("video_id") or None,
                "old_lecture_number": o.get("lecture_number"),
            })

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    catalog.sort(key=lambda r: (r["course_id"], r["playlist_index"] or 0, r["folder"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n")

    lectures = [r for r in catalog if r["duplicate_of"] is None]
    wrong = [r for r in lectures if r["video_id"] != r["old_video_id"]]
    print(f"{len(catalog)} folders → {len(lectures)} distinct lectures "
          f"({len(catalog) - len(lectures)} duplicate downloads)")
    print(f"{len(wrong)} lectures were linked to the wrong video (or none)")
    for course, n in sorted(Counter(r["course_id"] for r in wrong).items()):
        print(f"  {course:5} {n}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
