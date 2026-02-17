# CSE 121 TA LLM

This repo contains:
- A crawler for CSE 121 course pages
- An embedding/index pipeline using FAISS
- A local webchat + answer service over the indexed content

## Run the webchat

1. Ensure artifacts exist:
- `data/vectors/cse121_faiss.index`
- `data/vectors/metadata.json`

2. Start the service:

```bash
.venv/bin/python answer_service.py
```

3. Open:
- `http://127.0.0.1:8000`

## API

### `POST /api/chat`

Request JSON:

```json
{
  "message": "When is Quiz 1?"
}
```

Response JSON:

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "CSE 121",
      "url": "https://courses.cs.washington.edu/courses/cse121/26wi/",
      "chunk_index": 30,
      "distance": 12.3456,
      "snippet": "..."
    }
  ]
}
```
