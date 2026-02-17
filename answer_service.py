import json
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).parent
VECTOR_DIR = ROOT / "data" / "vectors"
INDEX_PATH = VECTOR_DIR / "cse121_faiss.index"
METADATA_PATH = VECTOR_DIR / "metadata.json"
CHAT_HTML_PATH = ROOT / "static" / "chat.html"

TOP_K = 5
MAX_CONTEXT_CHARS = 1100


@dataclass
class SearchResult:
    text: str
    url: str
    title: str
    chunk_index: int
    distance: float


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self) -> None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

        with METADATA_PATH.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.index = None
        self.model = None

        # Run offline-first: if embedding model is unavailable, use keyword search.
        try:
            if INDEX_PATH.exists():
                self.index = faiss.read_index(str(INDEX_PATH))
                self.model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    local_files_only=True,
                )
        except Exception as exc:
            print(
                "Embedding model unavailable; using keyword retrieval only.\n"
                f"Reason: {exc}",
                file=sys.stderr,
            )

    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        query_tokens = set(tokenize(query))
        ranked = []
        for row in self.metadata:
            text = row.get("text", "")
            text_tokens = set(tokenize(text))
            overlap = len(query_tokens & text_tokens)
            if query.lower() in text.lower():
                overlap += 5
            if overlap <= 0:
                continue
            meta = row.get("metadata", {})
            ranked.append(
                (
                    overlap,
                    SearchResult(
                        text=text,
                        url=meta.get("url", ""),
                        title=meta.get("title", "Untitled"),
                        chunk_index=int(meta.get("chunk_index", -1)),
                        distance=float(1 / (overlap + 1)),
                    ),
                )
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in ranked[:top_k]]

    def search(self, query: str, top_k: int = TOP_K) -> List[SearchResult]:
        if self.index is None or self.model is None:
            return self._keyword_search(query, top_k)

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_vec, top_k)
        results: List[SearchResult] = []

        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata[idx]
            meta = row.get("metadata", {})
            results.append(
                SearchResult(
                    text=row.get("text", ""),
                    url=meta.get("url", ""),
                    title=meta.get("title", "Untitled"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=float(distance),
                )
            )
        if results:
            return results
        return self._keyword_search(query, top_k)


def build_answer(query: str, results: List[SearchResult]) -> str:
    if not results:
        return "I could not find any relevant course information."

    query_tokens = set(tokenize(query))
    best_lines = []
    seen = set()

    for result in results:
        for line in result.text.splitlines():
            line = line.strip()
            if len(line) < 20 or line in seen:
                continue
            seen.add(line)
            tokens = set(tokenize(line))
            overlap = len(query_tokens & tokens)
            if overlap == 0 and len(best_lines) < 2:
                best_lines.append((overlap, line))
            elif overlap > 0:
                best_lines.append((overlap, line))

    best_lines.sort(key=lambda x: x[0], reverse=True)
    selected = [line for _, line in best_lines[:3]]
    if not selected:
        selected = [results[0].text[:300].strip()]

    body = " ".join(selected)
    if len(body) > MAX_CONTEXT_CHARS:
        body = body[: MAX_CONTEXT_CHARS - 3] + "..."
    return body


class ChatHandler(BaseHTTPRequestHandler):
    retriever: Retriever = None  # set at startup

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, body: str, status: int = HTTPStatus.OK) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_chat_page(self) -> None:
        if not CHAT_HTML_PATH.exists():
            self._send_text("chat.html is missing", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        body = CHAT_HTML_PATH.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_chat_page()
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_text("Not Found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/chat":
            self._send_text("Not Found", HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "Invalid JSON payload"}, HTTPStatus.BAD_REQUEST)
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
            return

        results = self.retriever.search(message, TOP_K)
        answer = build_answer(message, results)
        sources = []
        for r in results:
            sources.append(
                {
                    "title": r.title,
                    "url": r.url,
                    "chunk_index": r.chunk_index,
                    "distance": round(r.distance, 4),
                    "snippet": r.text[:220].strip(),
                }
            )

        self._send_json({"answer": answer, "sources": sources})

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    retriever = Retriever()
    ChatHandler.retriever = retriever

    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), ChatHandler)
    print(f"Webchat running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
