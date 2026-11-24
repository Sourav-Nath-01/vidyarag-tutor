"""
Work out which lecture every chunk of an existing index really came from.

    python scripts/data_fix/assign_segments.py \
        --metadata ../vidyarag/data/indexes/metadata_c3.json \
        --raw-dir  ../vidyarag/data/raw

In the v1 index, chunks are labelled with (course, lecture_number), but the
broken matcher gave several folders the same lecture_number, so e.g. chunks of
"Socket Programming – I" and "– II" share one label, one title and one video.
For every chunk this script finds the folder whose transcript contains the
chunk's text at the chunk's start time, then writes the folder's correct
video_id (from data/catalog/lectures.json) to data/catalog/c3_segment_map.json.

Chunks that come from a duplicate download are mapped to null (dropped).
The map is small and committed, so the index rebuild in CI needs only public
inputs: the v1 index files and this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def squash(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog" / "lectures.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "catalog" / "c3_segment_map.json")
    args = ap.parse_args()

    raw_bytes = args.metadata.read_bytes()
    segments = json.loads(raw_bytes)
    catalog = json.loads(args.catalog.read_text())
    folder_dirs = {p.name: p for p in args.raw_dir.glob("*/*") if p.is_dir()}

    # Folders that carried each old (course, lecture_number) label.
    by_old_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in catalog:
        by_old_key[(row["course_id"], row["old_lecture_number"])].append(row)

    transcript_cache: dict[str, tuple[str, set[float]]] = {}

    def folder_text(folder: str) -> tuple[str, set[float]]:
        if folder not in transcript_cache:
            utts = json.loads((folder_dirs[folder] / "multimodal.json").read_text())
            transcript_cache[folder] = (
                squash(" ".join(u["text"] for u in utts)),
                {round(float(u["start"]), 2) for u in utts},
            )
        return transcript_cache[folder]

    video_of, ambiguous, orphan = [], [], []
    seen: set[tuple] = set()
    for i, seg in enumerate(segments):
        cands = by_old_key.get((seg["course_id"], seg["lecture_number"]), [])
        if len(cands) > 1:
            text, start = squash(seg.get("transcript", "")), round(float(seg["start_sec"]), 2)
            cands = [c for c in cands
                     if text in folder_text(c["folder"])[0] and start in folder_text(c["folder"])[1]]
        lectures = {c["duplicate_of"] or c["folder"] for c in cands}
        if not lectures:
            orphan.append(seg["segment_id"])
            video_of.append(None)
            continue
        if len(lectures) > 1:
            ambiguous.append((seg["segment_id"], sorted(lectures)))
            video_of.append(None)
            continue
        vid = next(c["video_id"] for c in cands)
        # The same chunk indexed twice (from a duplicate download): keep the first.
        key = (vid, seg["start_sec"], seg["end_sec"], hashlib.sha1(seg.get("transcript", "").encode()).hexdigest())
        video_of.append(None if key in seen else vid)
        seen.add(key)

    if orphan or ambiguous:
        print(f"{len(orphan)} chunks matched no folder: {orphan[:5]}", file=sys.stderr)
        print(f"{len(ambiguous)} chunks matched several lectures: {ambiguous[:5]}", file=sys.stderr)
        return 1

    old_video = {r["video_id"]: r for r in catalog if r["duplicate_of"] is None}
    moved = sum(1 for s, v in zip(segments, video_of)
                if v and not s["youtube_url"].endswith(v))
    out = {
        "about": "video_id of the lecture each chunk of the source index came from; "
                 "null = duplicate chunk, drop it. Built by scripts/data_fix/assign_segments.py.",
        "source": {"index_repo": "SouravNath/vidyarag-indexes", "file": args.metadata.name,
                   "sha256": hashlib.sha256(raw_bytes).hexdigest(), "count": len(segments)},
        "video_id": video_of,
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")

    kept = [v for v in video_of if v]
    print(f"{len(segments)} chunks → {len(kept)} kept, {len(segments) - len(kept)} duplicate chunks dropped")
    print(f"{moved} chunks now point at a different video than before")
    print(f"{len(set(kept))} lectures have chunks "
          f"(catalogue has {len(old_video)}; missing: "
          f"{sorted(set(old_video) - set(kept))[:5]})")
    per_course = Counter(next(r['course_id'] for r in catalog if r['video_id'] == v) for v in set(kept))
    print("lectures per course:", dict(sorted(per_course.items())))
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
