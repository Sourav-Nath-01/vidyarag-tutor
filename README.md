# VidyaRAG Tutor — an AI doubt-solver for NPTEL lectures

> **v2 of [VidyaRAG](https://github.com/Sourav-Nath-01/vidyarag)** | *Hybrid multimodal retrieval (BGE-large + BM25 + RRF + cross-encoder) over 228 hours of IIT lectures*

**v1** ([Sourav-Nath-01/vidyarag](https://github.com/Sourav-Nath-01/vidyarag)) is the finished retrieval system: slide-boundary chunking, a 9-experiment ablation and a production deployment. **v2 (this repo)** builds on it to answer students' doubts directly from their professors' lectures:

- [x] Correct every result's video link: a data audit found 71 of 522 lectures (14%) linked to the wrong YouTube video ([details](#data-integrity-fix))
- [ ] Lecture-level retrieval benchmark to replace v1's lenient grader ([why](#a-note-on-the-v1-metrics)): tooling done, labels in progress ([benchmark/](benchmark/))
- [x] Grounded answers with clickable video citations, and refusal when the lectures don't cover a question ([how](#grounded-answers))
- [ ] Bounded agentic loop: plan → search → check evidence → re-search → answer; Hindi/Hinglish questions
- [ ] Answer-quality evaluation (faithfulness, citation accuracy, refusals) alongside the retrieval metrics

[![GitHub](https://img.shields.io/badge/GitHub-Sourav--Nath--01%2Fvidyarag--tutor-181717?logo=github)](https://github.com/Sourav-Nath-01/vidyarag-tutor)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-SouravNath%2Fvidyarag-ffd21e)](https://huggingface.co/spaces/SouravNath/vidyarag)
[![Tests](https://github.com/Sourav-Nath-01/vidyarag-tutor/actions/workflows/tests.yml/badge.svg)](https://github.com/Sourav-Nath-01/vidyarag-tutor/actions/workflows/tests.yml)
[![Build & Deploy](https://github.com/Sourav-Nath-01/vidyarag-tutor/actions/workflows/build-and-deploy.yml/badge.svg)](https://github.com/Sourav-Nath-01/vidyarag-tutor/actions/workflows/build-and-deploy.yml)
[![Live on Azure](https://img.shields.io/badge/Live-Azure%20Container%20Apps-0078D4?logo=microsoftazure)](https://vidyarag-tutor.yellowwave-9a72b46f.uaenorth.azurecontainerapps.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[🚀 Try the live demo](https://vidyarag-tutor.yellowwave-9a72b46f.uaenorth.azurecontainerapps.io/)** · **[⚡ REST API docs](https://vidyarag-tutor.yellowwave-9a72b46f.uaenorth.azurecontainerapps.io/docs)** · **[🤗 Streamlit demo on HF Spaces](https://huggingface.co/spaces/SouravNath/vidyarag)** · **[📄 Read the full report](report/vidyarag_report.pdf)**

> The Azure app scales to zero when idle, so the first search after a quiet period takes a few extra seconds while the container starts.

An AI-powered semantic search system over NPTEL lecture transcripts and OCR slide content: 16,050 indexed chunks from 522 lectures across 9 CS courses.

---

## Grounded answers

Ask a doubt and the tutor answers **only from what the professors say**, citing the moments it used. Each **[n]** in the answer is a button that plays result *n* from the cited timestamp.

```
question ─► hybrid search ─► top-5 lecture passages ─► relevance gate ─► LLM (streamed) ─► citation check ─► answer
                                  (+ next passage                │                                │
                                   for context)            "not covered"                 grounded? which [n]?
                                                          (no LLM call)               % of statements cited
```

- **Relevance gate.** The cross-encoder scores the passages before any LLM call. On the benchmark queries, every out-of-scope question's best passage scored at most −3.4, while 155 of the 156 real questions scored above that. The exception is the HOG question, which this CV course may not teach. Below −4.0 the tutor answers "not covered" without spending LLM quota. The threshold will be re-calibrated on the benchmark labels.
- **Grounding rules in the prompt, checks in code.** The model may use only the numbered passages, must cite them, and must reply `NOT_COVERED` when they don't answer the question. That sentinel is never shown to users. The server checks every citation against the passages it sent, and reports `grounded`, `cited` and `citation_coverage` with each answer.
- **Built for a free LLM tier.** Groq `openai/gpt-oss-20b` streams over Server-Sent Events, so the first words show in about a second. Repeated questions are served from a cache. A rate limit degrades to "search results still work" instead of an error page. The key lives only in an Azure Container Apps secret.
- **API.** `POST /ask` streams `sources` → `token`… → `done` (or `error`). Pass the `segment_ids` from a `/search` to answer from exactly those results, or omit them to search first.

---

## System Architecture

<p align="center">
  <img src="img/Architecture.png" alt="System Architecture" width="850">
</p>

---

## Key Results

| Configuration | MRR | Recall@5 | Recall@10 |
|---|---|---|---|
| C1 transcript only (baseline) | 0.4478 | 0.5952 | 0.6786 |
| C2 + OCR + BM25 | 0.6148 | 0.7619 | 0.8214 |
| **C3 + OCR + BM25 (Full System)** | **0.8259** | **0.9524** | **0.9643** |

Evaluated on **84 human-annotated queries** across 9 NPTEL courses (DSA, DAA, Deep Learning, OS, DBMS, CV, COA, ML, CN).

<p align="center">
  <img src="img/results_e1_e9.png" alt="System Experiment Results E1–E9" width="900">
</p>

### A note on the v1 metrics

These are v1's numbers, and v2 found two reasons to treat them as upper bounds rather than final results:

- **The grader doesn't check the lecture.** A result counts as correct if it is from the right *course* and starts within 90 s of the reference timestamp (`_is_correct` in `src/retrieval/evaluator.py`). A chunk from a different lecture of that course, at a similar timestamp, also passes.
- **Some reference answers don't line up with the data.** Mapped onto the corrected catalogue, 65 of the 84 timestamped queries resolve to exactly one lecture. Of the other 19, 8 point past the end of the named lecture, 4 name a lecture number that doesn't exist, and 7 name a number that covered several merged lectures.

The comparison *between* configurations (C1 vs C2 vs C3, with or without OCR/BM25) still reflects real differences, because every configuration was graded the same way. The absolute values are optimistic. Replacing them with a lecture-level benchmark is next on the roadmap.

---

## Data integrity fix

A v2 audit compared every lecture folder with the real YouTube playlists. **71 of 522 lectures (14%) were linked to the wrong video** (35 of 90 in Computer Vision), so a search result could play "Socket Programming – I" while quoting a transcript from "Socket Programming – II".

- **Root cause:** v1's matcher compared titles by word overlap after dropping every number and every word shorter than three letters, so "– I" vs "– II" and "Part 1" vs "Part 2" looked identical and the first title won. Folders that matched the same video also shared a lecture number, which merged their chunks under one title. The same mistake inflated v1's lecture count to 455 and double-counted 5 lectures that had been downloaded twice.
- **Fix:** [`lecture_catalog.py`](src/data_collection/lecture_catalog.py) matches on the whole normalised title, numbers included, and treats anything it can't match exactly as an error. All 527 folders matched exactly. The result is committed as a reviewable [catalogue](data/catalog/lectures.json): 522 lectures plus 5 duplicate downloads. Each of the 16,200 v1 chunks was then traced to its real lecture by matching its text and start time against each candidate transcript, with no ambiguous cases. 2,450 chunks changed video and 150 duplicate chunks were dropped.
- **Rebuild:** the lecture title is part of each chunk's embedding text, so [`rebuild_c3_index.py`](scripts/data_fix/rebuild_c3_index.py) re-embeds the 2,450 changed chunks on GitHub Actions. It reuses the other vectors only after re-embedding a sample of unchanged chunks and confirming they reproduce the stored vectors. The index is published as an immutable release, and the Docker build verifies every file against the sha256 pinned in [`deploy/index.lock.json`](deploy/index.lock.json).
- **Guard:** [`tests/test_catalog.py`](tests/test_catalog.py) fails CI if a folder's title and its video's title ever disagree, or if two lectures share a video.

---

## Features

- **Semantic lecture retrieval** using BGE-large-en-v1.5 dense embeddings (FAISS)
- **OCR-enhanced multimodal search** — Whisper ASR transcripts fused with Tesseract slide text
- **Novel C3 slide-boundary chunking** via OCR Jaccard similarity (83% MRR improvement over fixed-window baseline)
- **Hybrid retrieval** — FAISS dense + BM25 sparse, fused via Reciprocal Rank Fusion
- **Cross-encoder reranking** (ms-marco-MiniLM-L-6-v2)
- **LLM query analysis** — intent classification + query expansion via local Ollama
- **LLM-as-judge evaluation** — automated relevance scoring via Llama 3.2
- **REST API** — FastAPI `/search` endpoint for production use
- **Streamlit UI** — course filter, YouTube deep-links, query history
- **Full ablation study** — 9 system experiments + chunk parameter sensitivity

---

## Tech Stack

| Layer | Tools |
|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5`, Sentence Transformers |
| Vector Search | FAISS (IndexFlatIP, cosine similarity) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| ASR | Whisper (OpenAI) |
| OCR | Tesseract + pytesseract |
| LLM | Ollama (Llama 3.2:3b) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Evaluation | MRR, Recall@K, LLM-judge |

---

## Project Structure

```bash
vidyarag/
│
├── app.py                      # Streamlit UI — main entry point
├── DEPLOYMENT.md               # Production deployment write-up (Azure + CI/CD)
├── requirements.txt            # Full dev / pipeline dependency set
├── env.example                 # Copy to .env and edit
├── pytest.ini
│
├── src/
│   ├── data_collection/        # ASR transcription, OCR, chunking
│   │   ├── 1_metadata_builder.py
│   │   ├── 2_chunker_c1.py     # Fixed 30s window chunking
│   │   ├── 2_chunker_c2.py     # Utterance / word-count chunking
│   │   ├── 2_chunker_c3.py     # Slide-boundary chunking (novel, OCR Jaccard)
│   │   └── segment_utils.py
│   └── retrieval/
│       ├── embedder.py         # BGE-large FAISS index builder
│       ├── bm25_builder.py     # BM25 index builder
│       ├── retriever.py        # Full retrieval pipeline
│       └── evaluator.py        # Ablation study + metrics
│
├── api/
│   ├── app.py                  # FastAPI REST server (+ serves the web app at /)
│   └── static/                 # Search web app: index.html, app.js, app.css
│
├── scripts/                    # Developer tooling & offline pipeline
│   ├── build_indexes.py        # Build FAISS + BM25 indexes
│   ├── quick_demo.py           # CPU end-to-end demo (no GPU)
│   ├── eval_app.py             # Annotation / evaluation Streamlit UI
│   ├── generate_charts.py      # README result charts
│   ├── kaggle_build_indexes.ipynb
│   ├── kaggle_upload_to_hf.py  # Push built indexes to an HF Dataset
│   ├── fetch_artifacts.py      # Download models + indexes at image build time
│   └── eval/                   # Metric summarisation helpers
│       ├── eval_summary.py
│       ├── sensitivity_summary.py
│       └── add_types.py
│
├── deploy/                     # Deployment (Azure + Hugging Face)
│   ├── Dockerfile              # Production image for the FastAPI service
│   ├── smoke_test.py           # End-to-end check used by CI and after deploys
│   ├── azure/setup.sh          # One-time Azure Container Apps provisioning
│   ├── streamlit_space.py      # Deploy the Streamlit demo Space
│   ├── api_space.py            # Deploy the Docker FastAPI Space
│   ├── zerogpu_space.py        # Deploy the Gradio ZeroGPU Space
│   ├── app_zerogpu.py          # Gradio entry point for ZeroGPU
│   ├── requirements/           # Per-target runtime deps
│   │   ├── streamlit-space.txt
│   │   ├── api.txt
│   │   └── zerogpu.txt
│   └── space_cards/            # HF Space README front-matter cards
│       ├── streamlit.md
│       ├── api.md
│       └── zerogpu.md
│
├── data/
│   ├── raw/                    # Whisper transcripts + OCR output
│   ├── processed/              # Chunked segments (segments_c1/c2/c3.jsonl)
│   ├── indexes/                # FAISS + BM25 index files
│   └── eval/                   # Annotations, experiment results, metrics
│
├── report/
│   ├── vidyarag_report.tex     # Full technical report (LaTeX)
│   ├── vidyarag_report.pdf     # Compiled report
│   └── figures/                # Report figures (PDF + PNG)
│
├── configs/courses.json        # Course list / OCR flags
├── img/                        # README diagrams and charts
├── .github/workflows/          # CI (tests.yml) + CI/CD (build-and-deploy.yml)
└── tests/
    ├── test_core.py            # Retrieval unit tests (pytest)
    └── test_api.py             # API tests: probes, validation, back-pressure
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sourav-Nath-01/vidyarag-tutor.git
cd vidyarag
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy and configure environment variables:

```bash
cp env.example .env
# Edit .env — set PROJECT_ROOT, EMBEDDING_DEVICE, etc.
```

---

## Usage

### Streamlit UI (primary demo)

```bash
streamlit run app.py
# Custom port:
streamlit run app.py --server.port 8502
```

### FastAPI REST Server

```bash
uvicorn api.app:app --reload --port 8000
# Then POST to: http://localhost:8000/search
```

Example request:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does binary search tree insertion work", "strategy": "c3", "top_k": 5}'
```

### CLI retrieval

```bash
python src/retrieval/retriever.py --query "BST insertion" --strategy c3
python src/retrieval/retriever.py --query "backpropagation" --llm --verbose
```

### Quick Demo (No GPU required)

Runs the **full pipeline** (dense + BM25 + RRF + reranker) on 300 real segments using the lightweight `all-MiniLM-L6-v2` model. Completes in ~2 minutes on any CPU.

```bash
python scripts/quick_demo.py
```

Demo indexes are pre-built in `data/indexes/` — no rebuild needed.
To build the full indexes yourself: `python scripts/build_indexes.py --strategy c3`.

### Annotation / Evaluation UI

Interactive Streamlit tool used to create and validate the 84-query human-annotated evaluation set with timestamp grounding.

```bash
streamlit run scripts/eval_app.py
```

---

## Retrieval Pipeline

```
Query
  ↓
[Optional] LLM query analysis (intent + expansion)   ← Ollama
  ↓
BGE-large query embedding                            ← dense
  ↓
FAISS top-100 candidates                             ← dense
  ↓
BM25 top-100 candidates                              ← sparse
  ↓
Reciprocal Rank Fusion (RRF)                         ← fusion
  ↓
Content-type score boost (code/theory/conceptual)
  ↓
Cross-encoder reranking (ms-marco-MiniLM)            ← rerank
  ↓
Lecture-level deduplication
  ↓
Top-K results with YouTube deep-links
```

---

## Chunking Strategies

| Strategy | Description | Novel? |
|---|---|---|
| C1 | Fixed 30-second chunks — simple baseline | No |
| C2 | Word-count boundaries (≈200 words), respects sentence structure | No |
| **C3** | **Slide-boundary detection via OCR Jaccard similarity** — boundaries placed where consecutive slide OCR overlap drops below threshold | **Yes** |

**C3 algorithm detail:**
```
For consecutive segments (N, N+1):
  sim = |words(OCR_N) ∩ words(OCR_N+1)| / |words(OCR_N) ∪ words(OCR_N+1)|
  if sim < threshold → new chunk boundary (slide changed)
Fallback: word-count boundary when OCR confidence is low
Min/max duration rules applied after initial segmentation
```

---

## Evaluation

### GROUP 1 — System Experiments (9 configs)

| Exp | Strategy | OCR | BM25 | MRR | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| E1 | C1 | ❌ | ❌ | 0.4478 | 0.5952 | 0.6786 |
| E2 | C1 | ✅ | ❌ | 0.6536 | 0.7857 | 0.8452 |
| E3 | C1 | ✅ | ✅ | 0.6564 | 0.7976 | 0.8571 |
| E4 | C2 | ❌ | ❌ | 0.4059 | 0.5595 | 0.7024 |
| E5 | C2 | ✅ | ❌ | 0.6066 | 0.7262 | 0.7857 |
| E6 | C2 | ✅ | ✅ | 0.6148 | 0.7619 | 0.8214 |
| E7 | C3 | ❌ | ❌ | 0.4487 | 0.6429 | 0.6786 |
| E8 | C3 | ✅ | ❌ | 0.8200 | 0.9643 | 0.9643 |
| **E9** | **C3** | **✅** | **✅** | **0.8259** | **0.9524** | **0.9643** |

Evaluation set: **84 human-annotated queries**, timestamp-grounded (90s tolerance window).

### GROUP 2 — Chunk Parameter Sensitivity

| Variant | MRR | Recall@5 | Recall@10 |
|---|---|---|---|
| C2 (150w) | 0.4527 | 0.6310 | 0.7381 |
| C2 (200w) | 0.4059 | 0.5595 | 0.7024 |
| C2 (250w) | 0.3762 | 0.5238 | 0.6429 |
| C3 (t=0.25) | 0.4494 | 0.6429 | 0.6905 |
| C3 (t=0.30) | 0.4487 | 0.6429 | 0.6786 |
| C3 (t=0.40) | 0.4467 | 0.6548 | 0.6905 |

<p align="center">
  <img src="img/sensitivity_chart.png" alt="Chunk Parameter Sensitivity" width="700">
</p>

### Running the evaluation

```bash
python src/retrieval/evaluator.py               # all experiments
python src/retrieval/evaluator.py --experiment E9
python src/retrieval/evaluator.py --group1-only
python src/retrieval/evaluator.py --no-llm-judge
```

### Running tests

```bash
pytest tests/ -v
```

---

## Dataset Statistics

| Variant | Segments | Avg Duration | Avg Words | OCR Fail% |
|---|---|---|---|---|
| C1 (30s) | 24,804 | 33.2s | 80.6 | 3.4% |
| C2 (200w) | 9,841 | 85.3s | 203.1 | 1.9% |
| C3 (t=0.30) | ~15,000 | ~54s | ~131 | ~1.8% |

---

## Future Improvements

- Fine-tune bi-encoder on query-segment pairs (contrastive learning)
- ColBERT late-interaction reranking
- Multilingual retrieval (Hindi NPTEL lectures)
- Slide image embeddings (CLIP-based)
- Voice-based query support
- Adaptive chunk merging

---

## Deployment

The search web app and REST API run as one Docker container on **Azure
Container Apps** ([live](https://vidyarag-tutor.yellowwave-9a72b46f.uaenorth.azurecontainerapps.io/)).
A GitHub Actions CI/CD pipeline tests, builds, smoke-tests, ships and verifies
every change. Full write-up: **[DEPLOYMENT.md](DEPLOYMENT.md)**.

```
PR / push → unit tests → docker build → smoke test (offline, 2 CPU / 4 GB)
          → push to GHCR → deploy to Azure (OIDC, no secrets) → verify the new revision
```

- **Image**: CPU-only PyTorch, models and C3 index baked in at build time, non-root, `HEALTHCHECK`
- **Runtime**: 2 vCPU / 4 GiB, scale-to-zero, HTTPS, startup/readiness/liveness probes on `/health` and `/ready`
- **API hardening**: warm-up before accepting traffic, concurrency limit with `503 + Retry-After`, structured access logs
- **Web app**: dependency-free HTML/JS at `/`, inline video playback from the matching timestamp, shareable `?q=` URLs, strict Content Security Policy

```bash
docker build -f deploy/Dockerfile -t vidyarag-tutor .
docker run --rm -p 8000:7860 --memory 4g vidyarag-tutor
python deploy/smoke_test.py http://localhost:8000
```

The interactive Streamlit demo is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/SouravNath/vidyarag)
(`deploy/streamlit_space.py`). `deploy/` also holds HF Docker and ZeroGPU
targets (`api_space.py`, `zerogpu_space.py`).

---

## Contributing

Contributions are welcome through pull requests and issue discussions.

---

## License

This project is licensed under the MIT License.
