# CSE 121 TA LLM

Local question-answering service for CSE 121 course content.

It includes:
- A crawler for course pages (`firecrawl_crawler.py`)
- A chunking/embedding pipeline (`embeddings.py`)
- A local HTTP chat API + web UI (`answer_service.py`)

## Static Demo on GitHub Pages (No Backend)

This repo includes a backend-free demo:
- https://paul-rabel.github.io/CSE121-TA-LLM/

What it does:
- Runs fully in the browser.
- Uses a built-in demo Q&A set for common course questions.
- Does not call `/api/chat`.

### Run demo locally
```bash
cd <repo-root>
python3 -m http.server 8080
```
Open:
- `http://127.0.0.1:8080/docs/`

## Quick Start (Recommended, No Ollama Required)

This path uses deterministic + extractive answers only.

### 1. Verify indexed data exists
Required files:
- `data/vectors/metadata.json`
- `data/vectors/cse121_faiss.index`

### 2. Start the server
```bash
ENABLE_LLM_RESPONSE=0 .venv/bin/python answer_service.py
```

### 3. Open the app
- `http://127.0.0.1:8000`

### 4. Optional smoke test
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"When is R4 due and what assignments are eligible?"}'
```

## Optional Modes

### Use local Ollama for natural-language answers
```bash
export OLLAMA_MODEL=llama3.2:latest
ENABLE_LLM_RESPONSE=1 .venv/bin/python answer_service.py
```

If Ollama is unavailable, the service falls back to extractive answers.

### Enable dense retrieval (hybrid lexical + dense)
Only enable if `all-MiniLM-L6-v2` is already available locally.
```bash
ENABLE_DENSE_RETRIEVAL=1 ENABLE_LLM_RESPONSE=0 .venv/bin/python answer_service.py
```

## API

### `POST /api/chat`

Request:
```json
{
  "message": "When is R4 due and what assignments are eligible?"
}
```

Response (shape):
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
      "distance": 0.1234,
      "score": 12.3456,
      "snippet": "..."
    }
  ]
}
```

`answer_mode` can be:
- `deterministic`
- `llm`
- `extractive`
- `no_answer`

## Rebuild Data (Optional)

### 1. Crawl source pages
Set `FIRECRAWL_API_KEY` in `.env`, then run:
```bash
.venv/bin/python firecrawl_crawler.py
```

### 2. Build vectors
`embeddings.py` currently points to a specific raw JSON file via `RAW_JSON_PATH`.
Update that path if you generated a new crawl file, then run:
```bash
.venv/bin/python embeddings.py
```

## Troubleshooting

- `Missing metadata file`: generate/recover `data/vectors/metadata.json`.
- `ENABLE_LLM_RESPONSE=1` but no Ollama: either install/run Ollama or set `ENABLE_LLM_RESPONSE=0`.
- Empty or weak answers: confirm vector files exist and are from the same crawl snapshot.
