"""
Rebuild the C3 index with corrected lecture links. Runs in GitHub Actions.

    python scripts/data_fix/rebuild_c3_index.py --out-dir build/index

Inputs, all public or committed:
  • the v1 C3 index (faiss_c3.index, metadata_c3.json) from the HF dataset
  • data/catalog/lectures.json     — the correct video for every lecture
  • data/catalog/c3_segment_map.json — which lecture every v1 chunk came from

What it does
  1. Relabels every chunk with its real lecture: title, number, video URL,
     deep link, a unique segment_id, plus a new explicit video_id field.
     Chunks from duplicate downloads are dropped.
  2. Each chunk's embedding text includes its lecture title, so chunks whose
     title changed are re-embedded. All other vectors are reused, after
     re-embedding a sample of unchanged chunks to prove the stored vectors
     came from the same model and text recipe (otherwise it aborts).
  3. Rebuilds BM25 over the deduplicated corpus.
  4. Writes the three index files plus manifest.json (sha256 of every file,
     counts, inputs) so the Docker build can verify what it downloads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "data_collection"))

from bm25_builder import build_bm25_text, tokenise  # noqa: E402
from embedder import build_passage_text  # noqa: E402
from segment_utils import build_deep_link  # noqa: E402

STRATEGY = "c3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relabel(segments: list[dict], video_of: list, catalog: dict) -> tuple[list[dict], list[int]]:
    """Return corrected chunks and, for each, its row in the old index."""
    out, rows = [], []
    for i, (seg, vid) in enumerate(zip(segments, video_of)):
        if vid is None:
            continue
        lec = catalog[vid]
        chunk_no = int(re.search(r"-(\d+)$", seg["segment_id"]).group(1))
        new = copy.deepcopy(seg)
        new.update(
            video_id=vid,
            lecture_title=lec["title"],
            lecture_number=lec["playlist_index"],
            youtube_url=f"https://www.youtube.com/watch?v={vid}",
            youtube_deep_link=build_deep_link(vid, seg["start_sec"]),
            segment_id=f"{seg['course_id']}-lec{lec['playlist_index']:03d}-{STRATEGY}-{chunk_no:03d}",
        )
        out.append(new)
        rows.append(i)
    ids = [s["segment_id"] for s in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("segment_id collision after relabelling")
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", type=Path, default=None,
                    help="folder with the v1 faiss_c3.index + metadata_c3.json (default: download)")
    ap.add_argument("--src-repo", default="SouravNath/vidyarag-indexes")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--verify-sample", type=int, default=48)
    ap.add_argument("--min-cosine", type=float, default=0.995)
    args = ap.parse_args()

    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    t_start = time.time()
    catalog_rows = json.loads((ROOT / "data/catalog/lectures.json").read_text())
    catalog = {r["video_id"]: r for r in catalog_rows if r["duplicate_of"] is None}
    seg_map = json.loads((ROOT / "data/catalog/c3_segment_map.json").read_text())

    # ── inputs ────────────────────────────────────────────────────────────────
    if args.src_dir is None:
        from huggingface_hub import hf_hub_download
        args.src_dir = Path(hf_hub_download(args.src_repo, f"metadata_{STRATEGY}.json",
                                            repo_type="dataset")).parent
        hf_hub_download(args.src_repo, f"faiss_{STRATEGY}.index", repo_type="dataset")
    meta_path = args.src_dir / f"metadata_{STRATEGY}.json"
    if sha256(meta_path) != seg_map["source"]["sha256"]:
        raise SystemExit(f"{meta_path} is not the file c3_segment_map.json was built from")
    old_segments = json.loads(meta_path.read_text())
    old_index = faiss.read_index(str(args.src_dir / f"faiss_{STRATEGY}.index"))
    old_vectors = old_index.reconstruct_n(0, old_index.ntotal)
    assert len(old_segments) == old_index.ntotal == seg_map["source"]["count"]
    print(f"v1 index: {old_index.ntotal:,} chunks, dim {old_index.d}")

    # ── 1. relabel ───────────────────────────────────────────────────────────
    segments, rows = relabel(old_segments, seg_map["video_id"], catalog)
    old_texts = [build_passage_text(old_segments[r]) for r in rows]
    new_texts = [build_passage_text(s) for s in segments]
    changed = [i for i, (a, b) in enumerate(zip(old_texts, new_texts)) if a != b]
    print(f"relabelled: {len(segments):,} chunks kept, {len(old_segments) - len(segments)} "
          f"duplicates dropped, {len(changed):,} need re-embedding")

    # ── 2. embeddings ────────────────────────────────────────────────────────
    model = SentenceTransformer(args.model, device="cpu")
    if model.get_sentence_embedding_dimension() != old_index.d:
        raise SystemExit("model dimension differs from the v1 index")
    encode = lambda texts: model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=len(texts) > 100)

    unchanged = sorted(set(range(len(segments))) - set(changed))
    sample = random.Random(0).sample(unchanged, min(args.verify_sample, len(unchanged)))
    cos = (encode([new_texts[i] for i in sample]) * old_vectors[[rows[i] for i in sample]]).sum(axis=1)
    print(f"reuse check on {len(sample)} unchanged chunks: cosine min {cos.min():.5f}, mean {cos.mean():.5f}")
    if cos.min() < args.min_cosine:
        raise SystemExit("stored vectors don't match this model/text recipe — re-embed everything instead")

    vectors = old_vectors[rows].copy()
    t0 = time.time()
    if changed:
        vectors[changed] = encode([new_texts[i] for i in changed])
    embed_sec = time.time() - t0
    print(f"re-embedded {len(changed):,} chunks in {embed_sec:.0f}s")

    # ── 3. write index files ─────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(old_index.d)
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, str(args.out_dir / f"faiss_{STRATEGY}.index"))
    (args.out_dir / f"metadata_{STRATEGY}.json").write_text(json.dumps(segments, ensure_ascii=False))
    corpus = [tokenise(build_bm25_text(s)) for s in segments]
    with open(args.out_dir / f"bm25_{STRATEGY}.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi(corpus), "corpus": corpus}, f, protocol=pickle.HIGHEST_PROTOCOL)

    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(args.out_dir.iterdir()) if p.name != "manifest.json"}
    hours = sum(s["duration_sec"] for s in segments) / 3600
    manifest = {
        "strategy": STRATEGY,
        "chunks": len(segments),
        "lectures": len({s["video_id"] for s in segments}),
        "hours": round(hours, 1),
        "embedding_model": args.model,
        "reembedded_chunks": len(changed),
        "duplicate_chunks_dropped": len(old_segments) - len(segments),
        "reuse_check": {"sample": len(sample), "min_cosine": round(float(cos.min()), 6)},
        "source": seg_map["source"],
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "files": files,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    print(f"done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
