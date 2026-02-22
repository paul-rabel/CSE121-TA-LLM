import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib.request import Request, urlopen
from urllib.parse import urlparse

import faiss
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency for local env loading
    load_dotenv = None


ROOT = Path(__file__).parent
VECTOR_DIR = ROOT / "data" / "vectors"
INDEX_PATH = VECTOR_DIR / "cse121_faiss.index"
METADATA_PATH = VECTOR_DIR / "metadata.json"
CHAT_HTML_PATH = ROOT / "static" / "chat.html"

if load_dotenv:
    load_dotenv()

TOP_K = 5
MAX_CONTEXT_CHARS = 1100
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
MIN_SENTENCE_SCORE = 2.0
MIN_QUERY_TOKEN_COVERAGE = 0.45
ENABLE_DENSE_RETRIEVAL = os.getenv("ENABLE_DENSE_RETRIEVAL", "0") == "1"
ENABLE_LLM_RESPONSE = os.getenv("ENABLE_LLM_RESPONSE", "1") == "1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT_SECS = int(os.getenv("OLLAMA_TIMEOUT_SECS", "45"))

NO_ANSWER_TEXT = (
    "I couldn't find a reliable answer in the indexed course content for that question."
)
NO_ANSWER_SENTINEL = "NO_ANSWER_FOUND"

DATE_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b|\b\d{1,2}/\d{1,2}\b",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", re.IGNORECASE)

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
    "me",
    "my",
    "of",
    "our",
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
    "your",
    "you",
}

LOCATION_HINTS = ("room", "hall", "building", "cse2", "gug", "zoom", "online", "campus")
TIME_HINTS = ("due", "quiz", "exam", "deadline", "released")
STAFF_HINTS = ("instructor", "teaching assistant", "course staff", "quiz section", "office hours", "ta")
STAFF_STRONG_HINTS = ("instructor", "teaching assistant", "course staff", "quiz section", "office hours")
SCHEDULE_HINTS = (
    "lecture @",
    "class sessions",
    "class session",
    "quiz sections meet",
    "tuesdays/thursdays",
    "schedule",
    "in cse2",
    "in gug",
)
ASSIGNMENT_HINTS = ("assignment", "project", "resub", "eligible", "released", "submission")
LABEL_IN_LINE_PATTERN = re.compile(r"\b(?:quiz|[cpr])\s*0?\d+\b", re.IGNORECASE)
LINE_LABEL_PATTERN = re.compile(
    r"\b(?:quiz\s*0?(\d+)|([cpr])\s*0?(\d+)|resub(?:mission)?\s*0?(\d+))\b",
    re.IGNORECASE,
)

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

QUERY_TOKEN_CORRECTIONS = {
    "elligivle": "eligible",
    "elligible": "eligible",
    "eligable": "eligible",
    "eligibile": "eligible",
    "prof": "professor",
    "proffesor": "professor",
    "profeser": "professor",
    "assisgnment": "assignment",
    "assigment": "assignment",
}


@dataclass
class SearchResult:
    doc_id: int
    text: str
    url: str
    title: str
    chunk_index: int
    distance: float
    score: float


@dataclass
class EvidenceSelection:
    lines: List[Tuple[str, int]]
    top_score: float
    confident: bool


def tokenize(text: str) -> List[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens: List[str] = []
    for token in raw_tokens:
        split_suffix = re.match(r"^([cpr]\d+)(due|released)$", token)
        if split_suffix:
            tokens.append(split_suffix.group(1))
            tokens.append(split_suffix.group(2))
            continue
        tokens.append(token)
    return tokens


def normalize_token(token: str, for_query: bool = False) -> str:
    normalized = token.lower()
    if for_query:
        normalized = QUERY_TOKEN_CORRECTIONS.get(normalized, normalized)
    normalized = TOKEN_NORMALIZATION_MAP.get(normalized, normalized)
    return normalized


def query_terms(text: str, for_query: bool = False) -> List[str]:
    normalized_tokens = (normalize_token(token, for_query=for_query) for token in tokenize(text))
    return [t for t in normalized_tokens if t not in COMMON_STOPWORDS]


def looks_like_name_line(line: str) -> bool:
    candidate = line.strip()
    if len(candidate) < 6 or len(candidate) > 90:
        return False
    return bool(
        re.match(
            r"^[A-Z][A-Za-z'`.-]+(?: [A-Z][A-Za-z'`.-]+){1,3}(?:\s+(?:he/him/his|she/her/hers|they/them/theirs))?$",
            candidate,
        )
    )


def infer_query_intents(query: str) -> Dict[str, bool]:
    q = query.lower()
    query_tokens = set(query_terms(query, for_query=True))
    label_hints = extract_query_labels(query)

    wants_time = any(word in q for word in ("when", "time", "date", "deadline", "due", "schedule"))
    wants_location = any(word in q for word in ("where", "location", "room", "building"))
    wants_policy = any(word in q for word in ("policy", "late", "resub", "grading", "eligible", "credit"))
    wants_staff = (
        any(phrase in q for phrase in ("professor", "instructor", "teaching assistant", "course staff", "staff"))
        or bool(query_tokens & {"instructor", "ta", "staff"})
    )
    wants_schedule = (
        any(phrase in q for phrase in ("my class", "class schedule", "class time", "quiz section", "class session"))
        or bool(query_tokens & {"class", "lecture", "section", "session", "schedule"})
    )
    wants_assignments = (
        bool(label_hints)
        or bool(query_tokens & {"assignment", "project", "resub", "quiz", "eligible"})
        or any(word in q for word in ("assignment", "project"))
    )
    wants_tas = any(phrase in q for phrase in ("teaching assistant", "tas")) or bool(query_tokens & {"ta"})
    wants_instructor = (
        any(phrase in q for phrase in ("professor", "instructor", "teacher"))
        or bool(query_tokens & {"instructor"})
    ) and not wants_tas

    return {
        "wants_time": wants_time,
        "wants_location": wants_location,
        "wants_policy": wants_policy,
        "wants_staff": wants_staff,
        "wants_schedule": wants_schedule,
        "wants_assignments": wants_assignments,
        "wants_tas": wants_tas,
        "wants_instructor": wants_instructor,
    }


def extract_query_labels(query: str) -> List[str]:
    labels = []
    q = query.lower()

    for prefix in ("quiz", "p", "c", "r"):
        for match in re.finditer(rf"\b{prefix}\s*0?(\d+)\b", q):
            number = int(match.group(1))
            labels.append(f"{prefix}{number}")
            labels.append(f"{prefix}{number:02d}")
            if prefix == "quiz":
                labels.append(f"quiz {number}")
                labels.append(f"quiz {number:02d}")

    for match in re.finditer(r"\bresub(?:mission)?\s*0?(\d+)\b", q):
        number = int(match.group(1))
        labels.append(f"r{number}")
        labels.append(f"r{number:02d}")
        labels.append(f"resub {number}")

    # Handle natural phrasing like "project 0" / "checkpoint 2".
    for match in re.finditer(r"\bproject\s*0?(\d+)\b", q):
        number = int(match.group(1))
        labels.append(f"p{number}")
        labels.append(f"p{number:02d}")

    for match in re.finditer(r"\bcheckpoint\s*0?(\d+)\b", q):
        number = int(match.group(1))
        labels.append(f"c{number}")
        labels.append(f"c{number:02d}")

    # Preserve order while deduplicating.
    return list(dict.fromkeys(labels))


def canonicalize_label_token(token: str) -> Optional[str]:
    value = token.lower().strip()
    match = re.fullmatch(r"(quiz|[cpr])\s*0?(\d+)", value)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return f"{prefix}{number}"
    match = re.fullmatch(r"resub(?:mission)?\s*0?(\d+)", value)
    if match:
        return f"r{int(match.group(1))}"
    return None


def query_label_targets(label_hints: List[str]) -> Dict[str, set]:
    targets: Dict[str, set] = defaultdict(set)
    for hint in label_hints:
        canonical = canonicalize_label_token(hint)
        if not canonical:
            continue
        prefix = "quiz" if canonical.startswith("quiz") else canonical[0]
        targets[prefix].add(canonical)
    return dict(targets)


def extract_line_labels_by_prefix(text: str) -> Dict[str, set]:
    labels: Dict[str, set] = defaultdict(set)
    for match in LINE_LABEL_PATTERN.finditer(text.lower()):
        quiz_num = match.group(1)
        prefix = match.group(2)
        pref_num = match.group(3)
        resub_num = match.group(4)
        if quiz_num is not None:
            labels["quiz"].add(f"quiz{int(quiz_num)}")
        elif prefix is not None and pref_num is not None:
            labels[prefix].add(f"{prefix}{int(pref_num)}")
        elif resub_num is not None:
            labels["r"].add(f"r{int(resub_num)}")
    return dict(labels)


def line_label_alignment(line: str, label_targets: Dict[str, set]) -> Tuple[bool, bool]:
    if not label_targets:
        return (False, False)
    line_labels = extract_line_labels_by_prefix(line)
    has_exact = False
    has_conflict = False
    for prefix, targets in label_targets.items():
        found = line_labels.get(prefix, set())
        if not found:
            continue
        if found & targets:
            has_exact = True
        if any(label not in targets for label in found):
            has_conflict = True
    return (has_exact, has_conflict)


def strict_single_label_prefixes(label_targets: Dict[str, set]) -> set:
    strict_prefixes = set()
    for prefix in ("c", "p", "quiz"):
        targets = label_targets.get(prefix, set())
        if len(targets) == 1:
            strict_prefixes.add(prefix)
    return strict_prefixes


def line_has_prefix_conflict(line: str, label_targets: Dict[str, set], strict_prefixes: set) -> bool:
    if not strict_prefixes:
        return False
    line_labels = extract_line_labels_by_prefix(line)
    for prefix in strict_prefixes:
        found = line_labels.get(prefix, set())
        targets = label_targets.get(prefix, set())
        if found and not found.issubset(targets):
            return True
    return False


def has_exact_label_evidence(lines: List[Tuple[str, int]], label_targets: Dict[str, set]) -> bool:
    if not lines or not label_targets:
        return False
    strict_prefixes = strict_single_label_prefixes(label_targets)
    for line, _ in lines:
        has_exact_label, _ = line_label_alignment(line, label_targets)
        if not has_exact_label:
            continue
        if line_has_prefix_conflict(line, label_targets, strict_prefixes):
            continue
        return True
    return False


def expanded_query_terms(
    query: str,
    intents: Optional[Dict[str, bool]] = None,
    label_hints: Optional[List[str]] = None,
) -> List[str]:
    if intents is None:
        intents = infer_query_intents(query)
    if label_hints is None:
        label_hints = extract_query_labels(query)

    expanded = list(query_terms(query, for_query=True))
    seen = set(expanded)

    def add_terms(*terms: str) -> None:
        for term in terms:
            if term not in seen:
                expanded.append(term)
                seen.add(term)

    if intents["wants_staff"]:
        add_terms("instructor", "staff", "ta", "teaching", "assistant")
    if intents["wants_tas"]:
        add_terms("teaching", "assistant", "quiz", "section")
    if intents["wants_instructor"]:
        add_terms("instructor")
    if intents["wants_schedule"]:
        add_terms("class", "lecture", "section", "session", "schedule")
    if intents["wants_assignments"] or intents["wants_policy"]:
        add_terms("assignment", "project", "resub", "eligible", "released")
    if intents["wants_time"]:
        add_terms("time", "date", "due", "deadline")

    for label in label_hints:
        compact = label.replace(" ", "")
        if compact:
            add_terms(compact)

    return expanded


def normalize_line(line: str) -> str:
    # Keep text readable while stripping markdown/table noise.
    cleaned = line.replace("|", " ").replace("`", " ").replace("<br>", " ")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\b([cpr]\d+)(due|released)\b", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def score_line(
    line: str,
    query_tokens: set,
    intents: Dict[str, bool],
    rank_bias: float,
    label_targets: Dict[str, set],
    strict_prefixes: set,
) -> float:
    if len(line) < 18:
        return 0.0
    line_lower = line.lower()
    has_email = bool(re.search(r"\b[a-z0-9._%+-]+@(?:uw|cs)\b", line_lower))
    has_name = looks_like_name_line(line)
    has_staff_hint = any(word in line_lower for word in STAFF_HINTS)

    tokens = set(query_terms(line))
    if not tokens:
        return 0.0

    overlap = len(tokens & query_tokens)
    staff_override = intents["wants_staff"] and (has_staff_hint or has_email or has_name)
    if overlap == 0 and not staff_override:
        return 0.0
    score = float(overlap) + rank_bias
    if overlap == 0 and staff_override:
        score = 0.6 + rank_bias

    if intents["wants_time"] and (DATE_PATTERN.search(line) or TIME_PATTERN.search(line)):
        score += 2.5
    if intents["wants_time"] and any(word in line_lower for word in TIME_HINTS):
        score += 1.0
    if intents["wants_location"] and any(word in line_lower for word in LOCATION_HINTS):
        score += 1.5
    if intents["wants_location"] and re.search(r"\b(?:cse2|gug)\s*[a-z]?\d+\b", line_lower):
        score += 2.0
    if intents["wants_policy"] and any(word in line_lower for word in ("policy", "resub", "late", "grading", "credit")):
        score += 1.0
    if intents["wants_staff"] and any(word in line_lower for word in STAFF_HINTS):
        score += 2.2
    if intents["wants_staff"] and has_email:
        score += 1.1
    if intents["wants_staff"] and has_name:
        score += 1.0
    if intents["wants_schedule"] and (
        any(word in line_lower for word in SCHEDULE_HINTS)
        or re.search(r"\b(?:mon|tue|wed|thu|fri)(?:day|s)?\b", line_lower)
    ):
        score += 1.8
    if intents["wants_assignments"] and (
        any(word in line_lower for word in ASSIGNMENT_HINTS)
        or LABEL_IN_LINE_PATTERN.search(line_lower)
    ):
        score += 1.3
    if intents["wants_tas"] and ("teaching assistants" in line_lower or "quiz section" in line_lower):
        score += 2.0
    if intents["wants_tas"] and "instructor" in line_lower and "teaching assistant" not in line_lower:
        score -= 1.2
    if intents["wants_tas"] and has_name and "quiz section" in line_lower:
        score += 1.2
    if intents["wants_tas"] and has_name and "quiz section" not in line_lower:
        score += 2.0
    if intents["wants_tas"] and line_lower.startswith("quiz section"):
        score -= 1.2
    if intents["wants_instructor"] and "instructor" in line_lower:
        score += 2.8
    if intents["wants_instructor"] and "instructor" not in line_lower and "teaching assistants" in line_lower:
        score -= 1.4
    if intents["wants_instructor"] and has_name and "quiz section" not in line_lower:
        score += 1.6
    if intents["wants_instructor"] and "office hours" in line_lower and "instructor:" not in line_lower and "quiz section" not in line_lower:
        score -= 1.8
    if intents["wants_assignments"] and not intents["wants_policy"] and "resub" in line_lower:
        score -= 0.9
    if intents["wants_schedule"] and not intents["wants_assignments"] and any(word in line_lower for word in ("due", "released", "resub")):
        score -= 1.2

    if label_targets:
        has_exact_label, has_conflicting_label = line_label_alignment(line_lower, label_targets)
        if has_exact_label:
            score += 4.0
        if has_conflicting_label and not has_exact_label:
            score -= 7.0
        elif has_conflicting_label and has_exact_label:
            score -= 1.5
        if line_has_prefix_conflict(line_lower, label_targets, strict_prefixes):
            score -= 8.0

    if intents["wants_time"] and not (DATE_PATTERN.search(line) or TIME_PATTERN.search(line) or any(w in line_lower for w in TIME_HINTS)):
        score -= 1.0
    if intents["wants_location"] and not any(word in line_lower for word in LOCATION_HINTS):
        score -= 2.0
    if intents["wants_staff"] and not any(word in line_lower for word in STAFF_HINTS):
        score -= 0.8
    if intents["wants_schedule"] and not (
        DATE_PATTERN.search(line)
        or TIME_PATTERN.search(line)
        or any(word in line_lower for word in SCHEDULE_HINTS)
    ):
        score -= 0.7
    if intents["wants_assignments"] and not (
        any(word in line_lower for word in ASSIGNMENT_HINTS)
        or LABEL_IN_LINE_PATTERN.search(line_lower)
    ):
        score -= 0.5

    # Penalize boilerplate-ish rows.
    if line_lower in {"week", "topic"} or line_lower.startswith("---"):
        score -= 1.0
    return score


class Retriever:
    def __init__(self) -> None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

        with METADATA_PATH.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # BM25 lexical index (works fully offline).
        self.doc_term_freq: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.avg_doc_len = 1.0
        self._build_bm25_index()

        self.index = None
        self.model = None

        # Optional dense retrieval (disabled by default for deterministic offline behavior).
        if ENABLE_DENSE_RETRIEVAL:
            try:
                if INDEX_PATH.exists():
                    self.index = faiss.read_index(str(INDEX_PATH))
                    self.model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        local_files_only=True,
                    )
            except Exception as exc:
                print(
                    "Embedding model unavailable; using lexical retrieval only.\n"
                    f"Reason: {exc}",
                    file=sys.stderr,
                )

    def _build_bm25_index(self) -> None:
        for row in self.metadata:
            meta = row.get("metadata", {})
            text = row.get("text", "")
            title = meta.get("title", "")
            joined = f"{title} {text}"
            terms = query_terms(joined)
            tf = Counter(terms)
            self.doc_term_freq.append(tf)
            doc_len = sum(tf.values()) or 1
            self.doc_lengths.append(doc_len)
            for term in tf:
                self.doc_freq[term] += 1

        if self.doc_lengths:
            self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)

    def _result_from_doc(self, doc_id: int, score: float, distance: float) -> SearchResult:
        row = self.metadata[doc_id]
        meta = row.get("metadata", {})
        return SearchResult(
            doc_id=doc_id,
            text=row.get("text", ""),
            url=meta.get("url", ""),
            title=meta.get("title", "Untitled"),
            chunk_index=int(meta.get("chunk_index", -1)),
            distance=float(distance),
            score=float(score),
        )

    def _bm25_score_doc(self, query_tokens: List[str], doc_id: int) -> float:
        tf = self.doc_term_freq[doc_id]
        dl = self.doc_lengths[doc_id]
        n_docs = len(self.metadata)
        score = 0.0
        for token in query_tokens:
            freq = tf.get(token, 0)
            if freq <= 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_doc_len)
            score += idf * (freq * (BM25_K1 + 1)) / denom
        return score

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        intents = infer_query_intents(query)
        label_hints = extract_query_labels(query)
        label_targets = query_label_targets(label_hints)
        strict_prefixes = strict_single_label_prefixes(label_targets)
        q_tokens = expanded_query_terms(query, intents=intents, label_hints=label_hints)
        if not q_tokens:
            return []

        q_phrase = query.lower().strip()
        ranked: List[Tuple[int, float]] = []

        for doc_id, row in enumerate(self.metadata):
            score = self._bm25_score_doc(q_tokens, doc_id)

            text_lower = row.get("text", "").lower()
            title_lower = row.get("metadata", {}).get("title", "").lower()
            url_lower = row.get("metadata", {}).get("url", "").lower()
            combined_lower = f"{title_lower}\n{text_lower}"

            if q_phrase and q_phrase in text_lower:
                score += 4.0
            if q_phrase and q_phrase in title_lower:
                score += 2.0
            if DATE_PATTERN.search(q_phrase) and DATE_PATTERN.search(text_lower):
                score += 1.0
            if label_targets:
                has_exact_label, has_conflicting_label = line_label_alignment(combined_lower, label_targets)
                if has_exact_label:
                    score += 6.0
                else:
                    score -= 1.8
                if has_conflicting_label and not has_exact_label:
                    score -= 7.0
                elif has_conflicting_label and has_exact_label:
                    score -= 1.5
                if line_has_prefix_conflict(combined_lower, label_targets, strict_prefixes):
                    score -= 8.0
            if intents["wants_staff"]:
                if "/staff/" in url_lower:
                    score += 5.0
                if "course staff" in title_lower:
                    score += 2.5
                if "teaching assistants" in text_lower or "instructor:" in text_lower:
                    score += 2.0
                if "/getting_help/" in url_lower:
                    score -= 0.6
            if intents["wants_instructor"]:
                if "instructor" in title_lower or "instructor" in text_lower:
                    score += 4.0
                else:
                    score -= 2.4
                if "teaching assistants" in text_lower:
                    score -= 1.3
            if intents["wants_tas"]:
                if "teaching assistants" in text_lower or "quiz section" in text_lower:
                    score += 3.5
                if "instructor" in text_lower and "teaching assistants" not in text_lower:
                    score -= 1.0
            if intents["wants_schedule"]:
                if "/staff/" in url_lower:
                    score -= 1.5
                if "lecture @" in text_lower or "class sessions" in text_lower or "quiz sections meet" in text_lower:
                    score += 3.0
                if "/syllabus/" in url_lower and ("class sessions" in text_lower or "quiz sections" in text_lower):
                    score += 2.0
                if DATE_PATTERN.search(text_lower) or TIME_PATTERN.search(text_lower):
                    score += 1.0
            if intents["wants_assignments"]:
                if intents["wants_policy"] and "/resubs/" in url_lower:
                    score += 2.0
                if not intents["wants_policy"] and "/resubs/" in url_lower:
                    score -= 1.0
                if "/assignments/" in url_lower:
                    score += 1.7
                if any(word in text_lower for word in ("assignment", "project", "resub", "released", "eligible")):
                    score += 1.5
                if not intents["wants_policy"] and "resub" in text_lower:
                    score -= 0.8
            if score <= 0:
                continue

            ranked.append((doc_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        limit_multiplier = 6 if label_targets else 4
        return [
            self._result_from_doc(doc_id, score, distance=1.0 / (score + 1.0))
            for doc_id, score in ranked[: max(top_k, TOP_K) * limit_multiplier]
        ]

    def _dense_search(self, query: str, top_k: int) -> List[SearchResult]:
        if self.index is None or self.model is None:
            return []

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_vec, max(top_k, TOP_K) * 4)
        results: List[SearchResult] = []
        for distance, doc_id in zip(distances[0], indices[0]):
            if doc_id < 0 or doc_id >= len(self.metadata):
                continue
            dense_score = 1.0 / (1.0 + float(distance))
            results.append(self._result_from_doc(doc_id, dense_score, float(distance)))
        return results

    def search(self, query: str, top_k: int = TOP_K) -> List[SearchResult]:
        lexical_results = self._bm25_search(query, top_k)
        if self.index is None or self.model is None:
            return lexical_results[:top_k]

        dense_results = self._dense_search(query, top_k)
        if not dense_results:
            return lexical_results[:top_k]

        fused_scores: Dict[int, float] = defaultdict(float)
        best_distance: Dict[int, float] = {}

        for rank, result in enumerate(lexical_results):
            fused_scores[result.doc_id] += 1.0 / (RRF_K + rank + 1)
            best_distance[result.doc_id] = result.distance

        for rank, result in enumerate(dense_results):
            fused_scores[result.doc_id] += 1.0 / (RRF_K + rank + 1)
            if result.doc_id not in best_distance:
                best_distance[result.doc_id] = result.distance
            else:
                best_distance[result.doc_id] = min(best_distance[result.doc_id], result.distance)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            self._result_from_doc(doc_id, score=fused_score, distance=best_distance.get(doc_id, 1.0))
            for doc_id, fused_score in ranked[:top_k]
        ]


def collect_evidence(query: str, results: List[SearchResult]) -> EvidenceSelection:
    if not results:
        return EvidenceSelection(lines=[], top_score=0.0, confident=False)

    intents = infer_query_intents(query)
    label_hints = extract_query_labels(query)
    label_targets = query_label_targets(label_hints)
    strict_prefixes = strict_single_label_prefixes(label_targets)
    query_token_set = set(expanded_query_terms(query, intents=intents, label_hints=label_hints))
    if not query_token_set:
        query_token_set = set(tokenize(query))

    candidates: List[Tuple[float, int, str, int]] = []
    seen_lines = set()

    for rank, result in enumerate(results):
        rank_bias = max(0.0, 0.8 - rank * 0.15)
        for raw_line in result.text.splitlines():
            line = normalize_line(raw_line).strip("# ").strip()
            if line in seen_lines:
                continue
            seen_lines.add(line)
            line_score = score_line(
                line,
                query_token_set,
                intents,
                rank_bias,
                label_targets,
                strict_prefixes,
            )
            if line_score <= 0:
                continue
            candidates.append((line_score, rank, line, rank + 1))

    if not candidates:
        return EvidenceSelection(lines=[], top_score=0.0, confident=False)

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    top_score = candidates[0][0]
    cutoff = max(MIN_SENTENCE_SCORE, top_score * 0.58)
    selected = [candidate for candidate in candidates if candidate[0] >= cutoff][:3]
    if not selected:
        selected = [candidates[0]]

    if label_targets:
        filtered = []
        for candidate in selected:
            has_exact_label, has_conflicting_label = line_label_alignment(candidate[2], label_targets)
            if has_conflicting_label and not has_exact_label:
                continue
            if line_has_prefix_conflict(candidate[2], label_targets, strict_prefixes):
                continue
            filtered.append(candidate)
        if filtered:
            selected = filtered
        else:
            exact_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if line_label_alignment(candidate[2], label_targets)[0]
                    and not line_has_prefix_conflict(candidate[2], label_targets, strict_prefixes)
                ),
                None,
            )
            if exact_candidate is not None:
                selected = [exact_candidate]
            else:
                selected = [candidates[0]]

    if intents["wants_tas"] and selected and not any(looks_like_name_line(candidate[2]) for candidate in selected):
        name_candidate = next((candidate for candidate in candidates if looks_like_name_line(candidate[2])), None)
        if name_candidate is not None:
            if len(selected) >= 3:
                selected[-1] = name_candidate
            else:
                selected.append(name_candidate)

    if intents["wants_schedule"] and intents["wants_time"]:
        has_clock_time = any(re.search(r"\b\d{1,2}:\d{2}\b", candidate[2]) for candidate in selected)
        if not has_clock_time:
            timed_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if re.search(r"\b\d{1,2}:\d{2}\b", candidate[2])
                    and not any(word in candidate[2].lower() for word in ("due", "resub", "released"))
                    and not (
                        label_targets
                        and (
                            line_label_alignment(candidate[2], label_targets)[1]
                            and not line_label_alignment(candidate[2], label_targets)[0]
                        )
                    )
                ),
                None,
            )
            if timed_candidate is not None:
                if len(selected) >= 3:
                    selected[-1] = timed_candidate
                else:
                    selected.append(timed_candidate)

    lines = [(line, source_num) for _, _, line, source_num in selected]
    confident = top_score >= MIN_SENTENCE_SCORE

    query_core_tokens = {t for t in query_terms(query, for_query=True) if len(t) >= 3}
    selected_tokens = set()
    for line, _ in lines:
        selected_tokens.update(query_terms(line))
    skip_coverage_gate = bool(label_hints and intents["wants_assignments"])
    if query_core_tokens and not skip_coverage_gate:
        token_coverage = len(query_core_tokens & selected_tokens) / len(query_core_tokens)
        confident = confident and token_coverage >= MIN_QUERY_TOKEN_COVERAGE

    # If user asked about a specific label (Quiz 1, P2, C3, etc),
    # require at least one selected line to match the same canonical label.
    if label_targets:
        confident = confident and has_exact_label_evidence(lines, label_targets)

    if intents["wants_time"]:
        has_time_evidence = any(
            DATE_PATTERN.search(line)
            or TIME_PATTERN.search(line)
            or any(word in line.lower() for word in TIME_HINTS)
            for line, _ in lines
        )
        confident = confident and has_time_evidence

    if intents["wants_location"]:
        has_location_evidence = any(
            any(word in line.lower() for word in LOCATION_HINTS)
            or re.search(r"\b(?:cse2|gug)\s*[a-z]?\d+\b", line.lower())
            for line, _ in lines
        )
        confident = confident and has_location_evidence

    if intents["wants_staff"]:
        has_staff_evidence = any(
            any(word in line.lower() for word in STAFF_STRONG_HINTS)
            or re.search(r"\b[a-z0-9._%+-]+@(?:uw|cs)\b", line.lower())
            for line, _ in lines
        )
        confident = confident and has_staff_evidence

    if intents["wants_tas"]:
        has_ta_evidence = any(
            "teaching assistants" in line.lower() or "quiz section" in line.lower()
            for line, _ in lines
        )
        confident = confident and has_ta_evidence

    if intents["wants_instructor"]:
        has_instructor_evidence = any("instructor" in line.lower() for line, _ in lines)
        confident = confident and has_instructor_evidence

    return EvidenceSelection(lines=lines, top_score=top_score, confident=confident)


def build_extractive_answer(selection: EvidenceSelection) -> str:
    if not selection.lines:
        return NO_ANSWER_TEXT

    lines = []
    for line, source_num in selection.lines:
        text = line if len(line) <= 240 else line[:237] + "..."
        lines.append(f"- {text} [source {source_num}]")

    answer = "Best matching details:\n" + "\n".join(lines)
    if len(answer) > MAX_CONTEXT_CHARS:
        answer = answer[: MAX_CONTEXT_CHARS - 3] + "..."
    return answer


def _format_canonical_label(label: str) -> str:
    if label.startswith("quiz"):
        return f"Quiz {label[4:]}"
    return f"{label[0].upper()}{label[1:]}"


def _label_sort_key(label: str) -> Tuple[int, int]:
    if label.startswith("c"):
        return (0, int(label[1:]))
    if label.startswith("p"):
        return (1, int(label[1:]))
    if label.startswith("r"):
        return (2, int(label[1:]))
    if label.startswith("quiz"):
        return (3, int(label[4:]))
    return (4, 0)


def build_deterministic_label_answer(query: str, selection: EvidenceSelection) -> Optional[str]:
    if not selection.lines:
        return None

    query_lower = query.lower()
    wants_due = any(word in query_lower for word in ("when", "due", "deadline", "time"))
    wants_eligible = any(
        word in query_lower for word in ("eligible", "submit", "submission", "assignments", "assignment")
    )
    wants_name = any(
        phrase in query_lower
        for phrase in ("what is", "what's", "name", "called", "assignment", "project", "title")
    )
    if not (wants_due or wants_eligible or wants_name):
        return None

    targets = query_label_targets(extract_query_labels(query))
    r_targets = sorted(targets.get("r", set()))
    c_targets = sorted(targets.get("c", set()))
    p_targets = sorted(targets.get("p", set()))
    strict_prefixes = strict_single_label_prefixes(targets)

    target_label = None
    target_prefix = None
    if len(r_targets) == 1:
        target_label = r_targets[0]
        target_prefix = "r"
    elif len(c_targets) == 1 and not p_targets and not r_targets:
        target_label = c_targets[0]
        target_prefix = "c"
    elif len(p_targets) == 1 and not c_targets and not r_targets:
        target_label = p_targets[0]
        target_prefix = "p"
    else:
        return None

    target_display = _format_canonical_label(target_label)
    target_only = {target_prefix: {target_label}}

    due_time = None
    assignment_labels: List[str] = []
    assignment_seen = set()
    assignment_name = None
    fallback_label_line = None

    for line, _ in selection.lines:
        has_exact, _ = line_label_alignment(line, target_only)
        if not has_exact:
            continue
        if line_has_prefix_conflict(line, target_only, strict_prefixes):
            continue
        if fallback_label_line is None:
            fallback_label_line = line
        line_lower = line.lower()
        if target_prefix in {"c", "p"} and "resub" in line_lower and "i.s." not in line_lower:
            # Avoid treating resub-cycle lines as the canonical assignment definition.
            continue
        if due_time is None:
            due_match = re.search(
                r"\bdue\b[^0-9]*(\d{1,2}:\d{2}\s*(?:am|pm)\s*(?:pt|pst|pdt)?)",
                line,
                flags=re.IGNORECASE,
            )
            if due_match:
                due_time = due_match.group(1)
            else:
                is_match = re.search(r"\bi\.s\.\s*by\s*(\d{1,2}:\d{2}\s*(?:am|pm)\s*(?:pt|pst|pdt)?)", line, flags=re.IGNORECASE)
                if is_match:
                    due_time = is_match.group(1)
                elif due_time is None:
                    generic_time = re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", line, flags=re.IGNORECASE)
                    if generic_time:
                        due_time = generic_time.group(0)
        if assignment_name is None and target_prefix in {"c", "p"}:
            label_upper = target_label.upper()
            name_match = re.search(
                rf"\b{re.escape(label_upper)}\b\s*(?:-\s*)?(.+?)\s+i\.s\.",
                line,
                flags=re.IGNORECASE,
            )
            if name_match:
                candidate_name = name_match.group(1).strip(" -:")
                invalid_name = (
                    not candidate_name
                    or "resub" in candidate_name.lower()
                    or candidate_name.startswith(",")
                    or re.search(r"\b[cp]\s*0?\d+\b", candidate_name, flags=re.IGNORECASE) is not None
                )
                if not invalid_name:
                    assignment_name = candidate_name
            else:
                list_match = re.search(rf"\b{re.escape(label_upper)}\b\s*-\s*(.+)$", line, flags=re.IGNORECASE)
                if list_match:
                    candidate_name = list_match.group(1).strip(" -:")
                    if candidate_name and "resub" not in candidate_name.lower():
                        assignment_name = candidate_name
        for label_match in re.finditer(r"\b([cp])\s*0?(\d+)\b", line, flags=re.IGNORECASE):
            prefix = label_match.group(1).lower()
            number = int(label_match.group(2))
            canonical = f"{prefix}{number}"
            if canonical not in assignment_seen:
                assignment_seen.add(canonical)
                assignment_labels.append(canonical)

    if not due_time and not assignment_labels and not assignment_name:
        return None

    parts = []
    if target_prefix in {"c", "p"} and wants_name and assignment_name:
        parts.append(f"{target_display} is {assignment_name}.")
    elif target_prefix in {"c", "p"} and wants_name and fallback_label_line:
        clipped = fallback_label_line if len(fallback_label_line) <= 200 else fallback_label_line[:197] + "..."
        parts.append(f"{target_display}: {clipped}")
    if wants_due and due_time:
        parts.append(f"{target_display} is due by {due_time}.")
    if target_prefix == "r" and wants_eligible and assignment_labels:
        assignment_text = ", ".join(_format_canonical_label(label) for label in assignment_labels)
        parts.append(f"Eligible assignments for {target_display}: {assignment_text}.")

    if not parts:
        return None
    return " ".join(parts)


def _extract_ollama_output_text(payload: dict) -> str:
    text = payload.get("response")
    if isinstance(text, str):
        return text.strip()
    return ""


def build_llm_answer(
    query: str,
    selection: EvidenceSelection,
    results: List[SearchResult],
) -> Optional[str]:
    if not ENABLE_LLM_RESPONSE:
        return None
    if not selection.confident or not selection.lines:
        return NO_ANSWER_TEXT

    evidence_block = "\n".join(
        f"[source {source_num}] {line}" for line, source_num in selection.lines
    )
    source_meta = "\n".join(
        f"[source {i}] {r.title} - {r.url}" for i, r in enumerate(results, start=1)
    )

    system_prompt = (
        "You are a helpful CSE 121 course assistant.\n"
        "Use only the provided evidence lines.\n"
        f"If the evidence is insufficient or ambiguous, reply exactly: {NO_ANSWER_SENTINEL}\n"
        "If sufficient, answer naturally in 2-4 sentences and cite evidence as [source N]."
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Evidence lines:\n{evidence_block}\n\n"
        f"Sources:\n{source_meta}"
    )

    prompt = f"{system_prompt}\n\n{user_prompt}"
    request_body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    req = Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
        data=json.dumps(request_body).encode("utf-8"),
    )

    try:
        with urlopen(req, timeout=OLLAMA_TIMEOUT_SECS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = _extract_ollama_output_text(payload)
        if not text:
            return None
        if text.strip() == NO_ANSWER_SENTINEL:
            return NO_ANSWER_TEXT
        return text
    except (urlerror.URLError, TimeoutError, ValueError) as exc:
        print(f"Ollama call failed; falling back to extractive answer. Reason: {exc}", file=sys.stderr)
        return None


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
        selection = collect_evidence(message, results)
        label_targets = query_label_targets(extract_query_labels(message))
        label_exact_evidence = has_exact_label_evidence(selection.lines, label_targets)
        answer_mode = "no_answer"
        answer = NO_ANSWER_TEXT

        deterministic_answer = build_deterministic_label_answer(message, selection)
        if deterministic_answer is not None:
            answer = deterministic_answer
            answer_mode = "deterministic"
        elif label_targets:
            # For explicit label questions, avoid hallucinations by requiring exact
            # label evidence and returning extractive text only from that evidence.
            if label_exact_evidence and selection.lines:
                answer = build_extractive_answer(selection)
                answer_mode = "extractive"
        elif selection.confident:
            llm_answer = build_llm_answer(message, selection, results)
            if llm_answer is not None:
                answer = llm_answer
                answer_mode = "llm" if llm_answer != NO_ANSWER_TEXT else "no_answer"
            else:
                answer = build_extractive_answer(selection)
                answer_mode = "extractive"

        sources = []
        for rank, r in enumerate(results, start=1):
            sources.append(
                {
                    "rank": rank,
                    "title": r.title,
                    "url": r.url,
                    "chunk_index": r.chunk_index,
                    "distance": round(r.distance, 4),
                    "score": round(r.score, 4),
                    "snippet": r.text[:220].strip(),
                }
            )

        self._send_json(
            {
                "answer": answer,
                "sources": sources,
                "answer_mode": answer_mode,
                "confidence": round(selection.top_score, 3),
            }
        )

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
