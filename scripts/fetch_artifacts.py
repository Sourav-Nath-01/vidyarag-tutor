#!/usr/bin/env python3
"""
fetch_artifacts.py — download model weights + search indexes ahead of time
==========================================================================
Used by deploy/Dockerfile at *build* time so the production image is
self-contained: the container never has to reach the Hugging Face Hub at
runtime, cold starts don't depend on a third-party download, and every
image tag pins an exact set of models + indexes.

    python scripts/fetch_artifacts.py                       # c3 indexes + both models
    python scripts/fetch_artifacts.py --strategies c1,c2,c3 # more indexes
    python scripts/fetch_artifacts.py --skip-models         # indexes only

Index files listed in --index-lock (deploy/index.lock.json) are downloaded
from the GitHub release it names and must match the sha256 pinned there;
anything else comes from the public HF dataset (default
SouravNath/vidyarag-indexes). Models are cached in HF_HOME (the Dockerfile
points this inside the image).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download_pinned(url: str, target: Path, expected_sha256: str) -> None:
    """Download url → target, refusing to keep it unless the hash matches."""
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    actual = _sha256(tmp)
    if actual != expected_sha256:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    tmp.rename(target)


def _mb(path: Path) -> float:
    # Skip symlinks: the HF cache stores each file once under blobs/ and links
    # to it from snapshots/, so following links would count it twice.
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*")
               if p.is_file() and not p.is_symlink()) / 1e6


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", default=os.getenv("INDEX_STRATEGIES", "c3"),
                   help="comma-separated strategy keys to fetch (default: c3)")
    p.add_argument("--index-repo", default=os.getenv("INDEX_REPO_ID", "SouravNath/vidyarag-indexes"))
    p.add_argument("--index-dir", default=None,
                   help="where to put index files (default: $PROJECT_ROOT/data/indexes)")
    p.add_argument("--index-lock", default=None,
                   help="JSON naming a release URL and the sha256 of each index file in it")
    p.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    p.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    p.add_argument("--skip-models", action="store_true", help="only fetch indexes")
    args = p.parse_args()

    root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
    index_dir = Path(args.index_dir) if args.index_dir else root / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    lock = json.loads(Path(args.index_lock).read_text()) if args.index_lock else {"files": {}}
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"[fetch] indexes {strategies} → {index_dir}", flush=True)
    if lock["files"]:
        print(f"[fetch] pinned by {args.index_lock}: {lock['release']}", flush=True)
    for strategy in strategies:
        for fname in (f"faiss_{strategy}.index", f"metadata_{strategy}.json", f"bm25_{strategy}.pkl"):
            target = index_dir / fname
            pinned = lock["files"].get(fname)
            if target.exists() and (pinned is None or _sha256(target) == pinned):
                print(f"  = {fname} already present ({_mb(target):.0f} MB)", flush=True)
                continue
            t0 = time.time()
            if pinned:
                _download_pinned(f"{lock['release']}/{fname}", target, pinned)
                source = "release, sha256 verified"
            else:
                hf_hub_download(repo_id=args.index_repo, repo_type="dataset",
                                filename=fname, local_dir=str(index_dir))
                source = args.index_repo
            print(f"  ↓ {fname} ({_mb(target):.0f} MB, {time.time() - t0:.0f}s, {source})", flush=True)

    if not args.skip_models:
        # Instantiating the models downloads exactly the files the runtime will
        # load (not every format in the repo, e.g. ONNX/OpenVINO duplicates).
        from sentence_transformers import CrossEncoder, SentenceTransformer

        for label, load in (
            (args.embedding_model, lambda: SentenceTransformer(args.embedding_model, device="cpu")),
            (args.reranker_model,  lambda: CrossEncoder(args.reranker_model, device="cpu")),
        ):
            t0 = time.time()
            load()
            print(f"[fetch] model {label} ready ({time.time() - t0:.0f}s)", flush=True)

    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if hf_home.exists():
        print(f"[fetch] model cache {hf_home}: {_mb(hf_home):.0f} MB", flush=True)
        for child in sorted(p for p in hf_home.rglob("*") if p.is_dir() and p.parent.name == "hub"):
            print(f"         {child.name}: {_mb(child):.0f} MB", flush=True)

    print(f"[fetch] index dir total: {_mb(index_dir):.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
