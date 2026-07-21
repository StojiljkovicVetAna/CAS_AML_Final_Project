# Canine Research RAG Backend

RAG system developed as a final project for the Certificate of Advanced Studies
(CAS) in Advanced Machine Learning 2024/2025 at the University of Bern. Matteo
Boi and Ana Stojiljkovic from the Data Science Lab contributed equally to the
project.

This repository contains the backend pipeline for a scientific question
answering system over dog behaviour literature. It extracts text from papers,
creates retrieval-ready chunks, uploads embeddings to ChromaDB, and serves a
FastAPI backend for retrieval-augmented answers.

## Architecture

```text
Literature input
  -> extraction
  -> section-aware chunking
  -> embedding upload to ChromaDB
  -> FastAPI retrieval backend
  -> optional reranking
  -> grounded LLM answer
```

The deployed backend uses the selected final retrieval profile:

```text
retrieve 20 candidate chunks
rerank with Jina
pass the top 8 chunks to the LLM
```

The profile is defined in
`src/final_project/backend/deployment.py`. Request-level retrieval overrides
are disabled by default so the deployed app keeps the evaluated methodology.

## Input Options

The extraction module supports two literature routes:

- local PDFs in `input_literature/PDFs/`;
- online articles listed as DOI/PMID/PMCID values in
  `input_literature/list_DOIs.txt`.

For PDFs, GROBID is the default extractor. If GROBID is unavailable or returns
empty text, the pipeline falls back to a cleaned PyMuPDF extraction. For online
articles, the pipeline resolves identifiers through PubMed/PMC and extracts
available full text from PMC XML.

Raw PDFs, extracted full text, and generated chunks are intentionally ignored by
Git because they can contain copyrighted paper content.

## Repository Structure

```text
src/final_project/extraction/   text extraction from PDFs and PMC articles
src/final_project/chunking/     section-aware academic chunking
src/final_project/backend/      FastAPI app, retrieval, embeddings, reranking
src/final_project/evaluation/   retrieval and answer-quality evaluation tools
tests/                          offline parser and pipeline tests
data/evaluation/                final aggregated evaluation summaries
```

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Create a private environment file from the template:

```bash
cp .env.example .env
```

Put real credentials only in `.env`. The file is ignored by Git.

## Main Configuration

Minimum settings for the deployed RAG backend:

```text
BACKEND_API_KEY=
ALLOWED_ORIGINS=https://your-frontend.onrender.com

CHROMA_MODE=cloud
CHROMA_API_KEY=
CHROMA_TENANT=
CHROMA_DATABASE=
CHROMA_COLLECTION=dog_behavior_papers

EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY_EMBED=

RAG_PIPELINE=reranked
ALLOW_RETRIEVAL_OVERRIDES=false
RETRIEVE_TOP_K=20
CONTEXT_TOP_N=8
RERANKER_PROVIDER=jina
JINA_RERANKER_API_KEY=

LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.0-flash
GOOGLE_API_KEY_LLM=
```

`BACKEND_API_KEY` is a lightweight shared token between the frontend and
backend. The frontend sends the same value as `VITE_BACKEND_API_KEY`.

## Extraction And Chunking

Run extraction over the configured PDF folder and DOI list:

```bash
rag-extract \
  --pdf-dir input_literature/PDFs \
  --doi-file input_literature/list_DOIs.txt \
  --output-dir data/extracted
```

Create chunks:

```bash
rag-chunk \
  --input-dir data/extracted/documents \
  --output-dir data/chunks
```

The chunker keeps section metadata, filters common academic layout noise, and
targets chunks of approximately 400 tokens with overlap for context continuity.

## Upload To ChromaDB

After `data/chunks/chunks.jsonl` exists, upload embeddings:

```bash
rag-upload-chunks \
  --chunks data/chunks/chunks.jsonl \
  --manifest data/chunks/upload_manifest.json \
  --batch-size 32 \
  --skip-existing
```

Use `--reset-collection` only when intentionally replacing the whole ChromaDB
collection.

## Run The Backend

Local development:

```bash
uvicorn final_project.backend.app:app --reload --host 0.0.0.0 --port 8000
```

Production start command, for example on Render:

```bash
uvicorn final_project.backend.app:app --host 0.0.0.0 --port $PORT
```

Render settings:

```text
Build command: pip install -r requirements.txt
Start command: uvicorn final_project.backend.app:app --host 0.0.0.0 --port $PORT
Python version: controlled by runtime.txt
```

Main endpoints:

```text
GET  /
GET  /api/health
POST /api/search
POST /api/chat
```

`/api/search` returns retrieved chunks and source metadata. `/api/chat` returns
a grounded answer plus the supporting source list.

## Evaluation

The repository includes scripts for comparing retrieval strategies and judging
answer quality. The final aggregated summaries are kept in `data/evaluation/`.
Detailed row-level files containing retrieved contexts are excluded from Git.

Run tests with:

```bash
pytest
```
