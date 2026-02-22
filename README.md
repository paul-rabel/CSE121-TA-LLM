# CSE 121 TA LLM

A local question-answering service for University of Washington CSE 121 course content.

The project combines retrieval, deterministic answer rules, and optional LLM rewriting to produce concise answers with source attribution.

## Static Demo (No Backend)

A backend-free demo is published on GitHub Pages:

- [CSE 121 TA LLM Demo](https://paul-rabel.github.io/CSE121-TA-LLM/)

## Overview

The service is designed for high-precision course Q&A:

- Retrieval-first architecture over indexed course pages
- Deterministic handling for structured course facts (for example, labels, quiz dates, instructor info)
- Optional LLM rewrite layer for natural phrasing
- Source-aware responses with ranking, snippets, and optional debug reasoning
- Session memory and clarification stitching for follow-up questions

## Core Components

- `answer_service.py`: HTTP API (`/api/chat`), retrieval pipeline, deterministic logic, optional LLM rewrite
- `embeddings.py`: chunking, embeddings, and reverse token index generation
- `firecrawl_crawler.py`: source crawl utility with retry/backoff and failure reports
- `benchmark_runner.py`: benchmark harness for regression checks
- `benchmarks/cse121_accuracy_cases.json`: benchmark cases
- `tests/`: unit and API integration tests
- `static/chat.html`: local web chat UI

## Runtime Modes

### Deterministic + Extractive (simple)

Use this mode for maximum stability and zero LLM dependency.

```bash
ENABLE_LLM_RESPONSE=0 .venv/bin/python answer_service.py
```

### LLM Rewrite Mode (recommended)

This mode attempts an LLM rewrite on top of retrieval evidence. If citations or grounding checks fail, the service falls back to deterministic/extractive output.

```bash
export OLLAMA_MODEL=llama3.2:latest
ENABLE_LLM_RESPONSE=1 .venv/bin/python answer_service.py
```

## Quick Start

### 1. Verify index artifacts

Required:

- `data/vectors/metadata.json`
- `data/vectors/cse121_faiss.index`

Optional (recommended, auto-generated if missing):

- `data/vectors/token_index.json`

### 2. Start the service

```bash
ENABLE_LLM_RESPONSE=0 .venv/bin/python answer_service.py
```
or
```bash
ENABLE_LLM_RESPONSE=1 .venv/bin/python answer_service.py
```

### 3. Open the UI

- `http://127.0.0.1:8000`

### 4. Smoke test

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"When is R4 due and what assignments are eligible?"}'
```

## API

### `POST /api/chat`

Request:

```json
{
  "message": "When is R4 due and what assignments are eligible?",
  "session_id": "browser-session-123",
  "debug": false
}
```

Response shape:

```json
{
  "answer": "...",
  "answer_mode": "deterministic",
  "confidence": 8.123,
  "sources": [
    {
      "rank": 1,
      "title": "CSE 121",
      "url": "https://courses.cs.washington.edu/courses/cse121/26wi/",
      "chunk_index": 31,
      "section": "Course Staff",
      "distance": 0.1234,
      "score": 12.3456,
      "snippet": "..."
    }
  ],
  "memory_applied": ["Used recent context: R4."],
  "query_used": "What about it? R4 due deadline"
}
```

`answer_mode` values:

- `deterministic`
- `llm`
- `extractive`
- `no_answer`
- `clarification`

Optional response fields:

- `memory_applied`
- `query_used`
- `needs_clarification` (when `answer_mode` is `clarification`)

## Configuration

Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ENABLE_LLM_RESPONSE` | `1` | Enable/disable LLM rewrite mode |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model name |
| `ENABLE_DENSE_RETRIEVAL` | `0` | Enable dense retrieval in hybrid ranking |
| `LLM_REQUIRE_VALID_CITATIONS` | `1` | Reject LLM answers with invalid/missing `[source N]` citations |
| `ENABLE_SESSION_MEMORY` | `1` | Enable multi-turn context memory |
| `ENABLE_CLARIFICATION_STITCH` | `1` | Stitch follow-up clarification replies into unresolved prior query |
| `DEBUG_SOURCE_DETAILS` | `0` | Include `why`/`evidence` debug fields in source payload |

## Rebuild Data

### 1. Crawl source pages

Set `FIRECRAWL_API_KEY` in `.env`, then run:

```bash
.venv/bin/python firecrawl_crawler.py --retries 3 --initial-backoff 1.5 --timeout-secs 45
```

Outputs:

- `data/raw/cse121_dump_*.json`
- optional `*_failures.json` if some pages fail

### 2. Build vectors and token index

```bash
.venv/bin/python embeddings.py --input data/raw/cse121_dump_YYYYMMDD_HHMMSS.json
```

If `--input` is omitted, the newest `data/raw/cse121_dump_*.json` is selected.

Artifacts produced:

- `data/vectors/metadata.json` (chunk metadata, including section labels)
- `data/vectors/cse121_faiss.index` (dense vector index)
- `data/vectors/token_index.json` (reverse token postings index)

## Testing

Run targeted unit tests:

```bash
python3 -m unittest tests.test_answer_logic
```

Run API integration tests:

```bash
python3 -m unittest tests.test_api_integration
```

Run full suite:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Note: in restricted sandboxes, socket binding may be blocked; integration tests are skipped in that case.

## Benchmark

```bash
python3 benchmark_runner.py
```

Benchmark cases are defined in `benchmarks/cse121_accuracy_cases.json`.

## Troubleshooting

- `Missing metadata file`: regenerate `data/vectors/metadata.json` via `embeddings.py`.
- `ENABLE_LLM_RESPONSE=1` but no Ollama: run Ollama or set `ENABLE_LLM_RESPONSE=0`.
- Low-quality answers: confirm index artifacts come from the same crawl snapshot.
- Missing `token_index.json`: start the service once (it auto-builds from metadata) or rebuild with `embeddings.py`.
