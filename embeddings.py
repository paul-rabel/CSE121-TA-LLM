import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_VECTOR_DIR = Path("data/vectors")
DEFAULT_RAW_GLOB = "cse121_dump_*.json"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOKEN_INDEX_NAME = "token_index.json"

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

TOKEN_NORMALIZATION_MAP = {
    "assignments": "assignment",
    "projects": "project",
    "resubs": "resub",
    "resubmission": "resub",
    "resubmissions": "resub",
    "professor": "instructor",
    "professors": "instructor",
    "instructors": "instructor",
    "tas": "ta",
    "classes": "class",
    "lectures": "lecture",
    "sections": "section",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS vectors from crawled course pages.")
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to crawled JSON file. If omitted, the newest data/raw/cse121_dump_*.json is used.",
    )
    parser.add_argument(
        "--vector-dir",
        default=str(DEFAULT_VECTOR_DIR),
        help="Output directory for vectors and metadata (default: data/vectors).",
    )
    parser.add_argument(
        "--index-out",
        help="Optional explicit output path for the FAISS index file.",
    )
    parser.add_argument(
        "--metadata-out",
        help="Optional explicit output path for metadata JSON.",
    )
    parser.add_argument(
        "--token-index-out",
        help="Optional explicit output path for token postings index JSON.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Chunk size for text splitting (default: 700).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for text splitting (default: 100).",
    )
    return parser.parse_args()


def resolve_input_path(input_path: Optional[str]) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    if not DEFAULT_RAW_DIR.exists():
        raise FileNotFoundError(f"Raw crawl directory not found: {DEFAULT_RAW_DIR}")

    candidates = list(DEFAULT_RAW_DIR.glob(DEFAULT_RAW_GLOB))
    if not candidates:
        raise FileNotFoundError(
            "No crawl files found. Expected something like data/raw/cse121_dump_*.json"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_pages(raw_json_path: Path) -> List[Dict[str, Any]]:
    with raw_json_path.open("r", encoding="utf-8") as f:
        pages = json.load(f)
    if not isinstance(pages, list):
        raise ValueError(f"Expected a JSON list in {raw_json_path}")
    return pages


def build_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    all_chunks_with_data: List[Dict[str, Any]] = []

    for page in pages:
        content_obj = page.get("content") or {}
        if not isinstance(content_obj, dict):
            continue

        content_md = content_obj.get("markdown", "")
        metadata_dict = content_obj.get("metadata", {})

        if not content_md or not content_md.strip():
            continue

        chunks = text_splitter.split_text(content_md)

        for i, chunk in enumerate(chunks):
            section = infer_chunk_section(chunk, fallback_title=metadata_dict.get("title", "Untitled"))
            all_chunks_with_data.append(
                {
                    "text": chunk,
                    "metadata": {
                        "url": page.get("url", ""),
                        "title": metadata_dict.get("title", "Untitled"),
                        "chunk_index": i,
                        "section": section,
                    },
                }
            )

    return all_chunks_with_data


def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)


def tokenize_for_index(text: str) -> List[str]:
    tokens: List[str] = []
    for raw_token in re.findall(r"[a-z0-9]+", text.lower()):
        token = TOKEN_NORMALIZATION_MAP.get(raw_token, raw_token)
        if token in COMMON_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def infer_chunk_section(chunk: str, fallback_title: str) -> str:
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading_match:
            heading = heading_match.group(1).strip()
            if heading:
                return heading
    return fallback_title


def build_token_postings(chunks: List[Dict[str, Any]]) -> Dict[str, List[List[int]]]:
    postings: Dict[str, List[List[int]]] = defaultdict(list)
    for doc_id, row in enumerate(chunks):
        meta = row.get("metadata", {})
        combined = " ".join(
            [
                str(meta.get("title", "")),
                str(meta.get("section", "")),
                str(row.get("text", "")),
            ]
        )
        tokens = tokenize_for_index(combined)
        if not tokens:
            continue
        tf = Counter(tokens)
        for token, count in tf.items():
            postings[token].append([doc_id, int(count)])

    # Deterministic ordering for reproducible builds.
    ordered: Dict[str, List[List[int]]] = {}
    for token in sorted(postings.keys()):
        token_postings = postings[token]
        token_postings.sort(key=lambda item: (item[0], item[1]))
        ordered[token] = token_postings
    return ordered


def main() -> int:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")

    raw_json_path = resolve_input_path(args.input_path)

    vector_dir = Path(args.vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)

    faiss_index_path = Path(args.index_out) if args.index_out else vector_dir / "cse121_faiss.index"
    metadata_path = Path(args.metadata_out) if args.metadata_out else vector_dir / "metadata.json"
    token_index_path = (
        Path(args.token_index_out)
        if args.token_index_out
        else vector_dir / DEFAULT_TOKEN_INDEX_NAME
    )
    ensure_parent_dir(faiss_index_path)
    ensure_parent_dir(metadata_path)
    ensure_parent_dir(token_index_path)

    print(f"Loading crawl data from: {raw_json_path}")
    pages = load_pages(raw_json_path)

    all_chunks_with_data = build_chunks(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    texts_to_embed = [item["text"] for item in all_chunks_with_data]

    print(f"Total chunks created: {len(texts_to_embed)}")
    if not texts_to_embed:
        raise ValueError("No text chunks were produced. Check crawl input content.")

    print(f"Loading local embedding model ({args.model})...")
    embeddings_model = HuggingFaceEmbeddings(model_name=args.model)

    print("Generating embeddings (this may take a minute)...")
    embeddings = embeddings_model.embed_documents(texts_to_embed)
    embeddings_np = np.array(embeddings, dtype="float32")

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    faiss.write_index(index, str(faiss_index_path))
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks_with_data, f, indent=2, ensure_ascii=False)
    token_index_payload = {
        "version": 1,
        "doc_count": len(all_chunks_with_data),
        "token_postings": build_token_postings(all_chunks_with_data),
    }
    with token_index_path.open("w", encoding="utf-8") as f:
        json.dump(token_index_payload, f, indent=2, ensure_ascii=False)

    print(f"Success! FAISS index saved to {faiss_index_path} with {index.ntotal} vectors.")
    print(f"Metadata saved to {metadata_path}")
    print(f"Token postings index saved to {token_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
