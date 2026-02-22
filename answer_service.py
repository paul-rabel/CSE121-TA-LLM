import json
import math
import os
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
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
TOKEN_INDEX_PATH = VECTOR_DIR / "token_index.json"
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
RERANK_CANDIDATE_MULTIPLIER = max(2, int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "4")))
POSTINGS_CANDIDATE_MULTIPLIER = max(2, int(os.getenv("POSTINGS_CANDIDATE_MULTIPLIER", "6")))
RERANK_SEMANTIC_WEIGHT = float(os.getenv("RERANK_SEMANTIC_WEIGHT", "3.5"))
SESSION_MEMORY_TURNS = max(1, int(os.getenv("SESSION_MEMORY_TURNS", "6")))
SESSION_MEMORY_MAX_SESSIONS = max(4, int(os.getenv("SESSION_MEMORY_MAX_SESSIONS", "256")))
ENABLE_SESSION_MEMORY = os.getenv("ENABLE_SESSION_MEMORY", "1") == "1"
DEBUG_SOURCE_DETAILS_DEFAULT = os.getenv("DEBUG_SOURCE_DETAILS", "0") == "1"
ENABLE_DENSE_RETRIEVAL = os.getenv("ENABLE_DENSE_RETRIEVAL", "0") == "1"
ENABLE_LLM_RESPONSE = os.getenv("ENABLE_LLM_RESPONSE", "1") == "1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()
OLLAMA_TIMEOUT_SECS = int(os.getenv("OLLAMA_TIMEOUT_SECS", "45"))
OLLAMA_TAGS_TIMEOUT_SECS = int(os.getenv("OLLAMA_TAGS_TIMEOUT_SECS", "3"))
LLM_CONTEXT_SOURCE_LIMIT = max(1, int(os.getenv("LLM_CONTEXT_SOURCE_LIMIT", "6")))
LLM_CONTEXT_LINES_PER_SOURCE = max(1, int(os.getenv("LLM_CONTEXT_LINES_PER_SOURCE", "3")))
LLM_CONTEXT_SNIPPET_CHARS = max(120, int(os.getenv("LLM_CONTEXT_SNIPPET_CHARS", "420")))
LLM_REQUIRE_VALID_CITATIONS = os.getenv("LLM_REQUIRE_VALID_CITATIONS", "1") == "1"
SECTION_WINDOW_CHUNKS = max(0, int(os.getenv("SECTION_WINDOW_CHUNKS", "1")))
SECTION_CONTEXT_MAX_CHUNKS = max(1, int(os.getenv("SECTION_CONTEXT_MAX_CHUNKS", "4")))
ENABLE_CLARIFICATION_STITCH = os.getenv("ENABLE_CLARIFICATION_STITCH", "1") == "1"
ALWAYS_ATTEMPT_LLM = os.getenv("ALWAYS_ATTEMPT_LLM", "1") == "1"

_OLLAMA_MODEL_CACHE = ""
_OLLAMA_MODEL_LOOKUP_ATTEMPTED = False
_OLLAMA_MODEL_LOCK = Lock()

NO_ANSWER_TEXT = (
    "I couldn't find a reliable answer in the indexed course content for that question."
)
NO_ANSWER_SENTINEL = "NO_ANSWER_FOUND"

DATE_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b|\b\d{1,2}/\d{1,2}\b",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", re.IGNORECASE)
INSTRUCTOR_LINE_PATTERN = re.compile(
    r"\*\*Instructor:\*\*\s*([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})"
)
EMAIL_PATTERN = re.compile(r"\b[a-z0-9._%+-]+@(?:uw|cs)\.washington\.edu\b", re.IGNORECASE)
LECTURE_DAY_PATTERN = re.compile(r"\|\s*(Mon|Tue|Wed|Thu|Fri)\s+\d{2}/\d{2}\s*\|\s*LES", re.IGNORECASE)
LECTURE_LOCATION_LINE_PATTERN = re.compile(r"lecture\s*@[^|]+", re.IGNORECASE)
QUIZ_LINE_PATTERN = re.compile(
    r"\|\s*(Mon|Tue|Wed|Thu|Fri)\s+(\d{2}/\d{2})\s*\|\s*QUIZ\s*0?(\d+)\s*([^|]*)",
    re.IGNORECASE,
)
SOURCE_CITATION_PATTERN = re.compile(r"\[source\s+(\d+)\]", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$")
CLARIFICATION_HINT_PATTERN = re.compile(
    r"(could you clarify|can you clarify|do you mean|which one|which .* are you asking|please clarify|please specify)",
    re.IGNORECASE,
)
WEEKDAY_PATTERN = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b",
    re.IGNORECASE,
)
ROOM_PATTERN = re.compile(r"\b(?:cse2|gug)\s*[a-z]?\d+\b", re.IGNORECASE)

ORDINAL_WORD_TO_INT = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

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
RELEASED_ASSIGNMENT_PATTERN = re.compile(
    r"Released\s+([CP]\d+)\s+(.+?)\s+I\.S\.\s+by\s+(.+?)(?:\s*\||$)",
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

FOLLOWUP_PREFIXES = (
    "what about",
    "how about",
    "and ",
    "also ",
    "same for",
    "what if",
)
REFERENTIAL_PATTERN = re.compile(r"\b(it|that|this|they|them|those|these)\b", re.IGNORECASE)


@dataclass
class SearchResult:
    doc_id: int
    text: str
    url: str
    title: str
    chunk_index: int
    distance: float
    score: float
    section: str = ""


@dataclass
class EvidenceSelection:
    lines: List[Tuple[str, int]]
    top_score: float
    confident: bool


@dataclass
class ConversationTurn:
    query: str
    answer: str
    labels: List[str]


class SessionMemoryManager:
    def __init__(self, max_sessions: int = SESSION_MEMORY_MAX_SESSIONS, turn_limit: int = SESSION_MEMORY_TURNS) -> None:
        self.max_sessions = max_sessions
        self.turn_limit = turn_limit
        self._sessions: Dict[str, deque] = {}
        self._pending_clarifications: Dict[str, str] = {}
        self._lock = Lock()

    def _ensure_session_unlocked(self, session_id: str) -> deque:
        if session_id not in self._sessions:
            if len(self._sessions) >= self.max_sessions:
                oldest_session = next(iter(self._sessions))
                self._sessions.pop(oldest_session, None)
                self._pending_clarifications.pop(oldest_session, None)
            self._sessions[session_id] = deque(maxlen=self.turn_limit)
        return self._sessions[session_id]

    def get_turns(self, session_id: Optional[str]) -> List[ConversationTurn]:
        if not session_id:
            return []
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return []
            return list(turns)

    def add_turn(self, session_id: Optional[str], turn: ConversationTurn) -> None:
        if not session_id:
            return
        with self._lock:
            turns = self._ensure_session_unlocked(session_id)
            turns.append(turn)

    def get_pending_clarification(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        with self._lock:
            return self._pending_clarifications.get(session_id)

    def set_pending_clarification(self, session_id: Optional[str], original_query: str) -> None:
        if not session_id or not original_query.strip():
            return
        with self._lock:
            self._ensure_session_unlocked(session_id)
            self._pending_clarifications[session_id] = original_query.strip()

    def clear_pending_clarification(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        with self._lock:
            self._pending_clarifications.pop(session_id, None)


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


def canonical_labels_from_query(query: str) -> List[str]:
    labels: List[str] = []
    for hint in extract_query_labels(query):
        canonical = canonicalize_label_token(hint)
        if canonical and canonical not in labels:
            labels.append(canonical)
    return labels


def rewrite_query_with_memory(query: str, history: List[ConversationTurn]) -> Tuple[str, List[str], List[str]]:
    if not history:
        return query, [], canonical_labels_from_query(query)

    rewritten = query
    notes: List[str] = []
    query_labels = canonical_labels_from_query(query)
    query_lower = query.lower().strip()
    current_intents = infer_query_intents(query)
    previous_turn = history[-1]
    previous_intents = infer_query_intents(previous_turn.query)
    previous_labels = previous_turn.labels
    query_terms_lower = query_terms(query, for_query=True)
    has_explicit_name_request = any(
        phrase in query_lower for phrase in ("what is", "what's", "name", "called", "title")
    )
    has_explicit_due_request = any(word in query_lower for word in ("when", "due", "deadline", "time"))
    has_followup_prefix = any(query_lower.startswith(prefix) for prefix in FOLLOWUP_PREFIXES)
    has_referential_term = bool(REFERENTIAL_PATTERN.search(query_lower))

    appended_terms: List[str] = []
    seen_terms = set()

    def append_term(term: str) -> None:
        if term not in seen_terms:
            appended_terms.append(term)
            seen_terms.add(term)

    # Resolve referential follow-ups like "When is it due?" to the last label.
    if not query_labels and previous_labels:
        assignment_followup = (
            current_intents["wants_assignments"]
            and previous_intents["wants_assignments"]
            and len(query_terms_lower) <= 6
        )
        if has_referential_term or has_followup_prefix or assignment_followup:
            context_label = previous_labels[0]
            append_term(_format_canonical_label(context_label))
            query_labels.append(context_label)
            notes.append(f"Used recent context: {_format_canonical_label(context_label)}.")

    # For short follow-ups ("what about C4?"), inherit prior intent if user omitted it.
    inherits_intent = (has_followup_prefix or has_referential_term) and not has_explicit_name_request
    if inherits_intent and query_labels:
        if previous_intents["wants_time"] and not current_intents["wants_time"] and not has_explicit_due_request:
            append_term("due")
            append_term("deadline")
            notes.append("Inherited due-time intent from previous turn.")
        if previous_intents["wants_location"] and not current_intents["wants_location"]:
            append_term("location")
            append_term("where")
            notes.append("Inherited location intent from previous turn.")
        if previous_intents["wants_staff"] and not current_intents["wants_staff"]:
            append_term("staff")
            append_term("instructor")
            notes.append("Inherited staff intent from previous turn.")
        if previous_intents["wants_policy"] and not current_intents["wants_policy"] and not current_intents["wants_assignments"]:
            append_term("policy")
            notes.append("Inherited policy intent from previous turn.")

    if appended_terms:
        rewritten = f"{query} {' '.join(appended_terms)}"

    return rewritten, notes, query_labels


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def normalize_line(line: str) -> str:
    # Keep text readable while stripping markdown/table noise.
    cleaned = line.replace("|", " ").replace("`", " ").replace("<br>", " ")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\b([cpr]\d+)(due|released)\b", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def infer_section_name(text: str, fallback: str = "") -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading_match = SECTION_HEADING_PATTERN.match(line)
        if heading_match:
            heading = normalize_line(heading_match.group(1)).strip()
            if heading:
                return heading
    return fallback


def is_clarification_request(text: str) -> bool:
    if not text:
        return False
    if "?" not in text:
        return False
    if CLARIFICATION_HINT_PATTERN.search(text):
        return True
    short_text = text.strip()
    if len(short_text) <= 180 and short_text.lower().startswith(("which ", "do you mean", "can you clarify")):
        return True
    return False


def should_stitch_pending_query(message: str) -> bool:
    lower = message.lower().strip()
    if not lower:
        return False
    if any(phrase in lower for phrase in ("new question", "different question", "never mind", "ignore that")):
        return False
    if len(query_terms(lower, for_query=True)) <= 12:
        return True
    # Long, clearly standalone prompts should start a fresh query.
    return not any(word in lower for word in (" when ", " where ", " who ", " what ", " how ", "?"))


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
        if not isinstance(self.metadata, list):
            raise ValueError(f"Expected metadata list in {METADATA_PATH}")
        for row in self.metadata:
            if not isinstance(row, dict):
                continue
            meta = row.setdefault("metadata", {})
            title = str(meta.get("title", "Untitled"))
            section = str(meta.get("section", "")).strip()
            if not section:
                section = infer_section_name(str(row.get("text", "")), fallback=title)
                meta["section"] = section

        # BM25 lexical index (works fully offline).
        self.doc_term_freq: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.avg_doc_len = 1.0
        self._build_bm25_index()
        self.docs_by_url: Dict[str, List[int]] = defaultdict(list)
        self.doc_position_by_id: Dict[int, int] = {}
        self.docs_by_url_section: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self._build_doc_navigation_index()
        self.token_postings: Dict[str, List[Tuple[int, int]]] = {}
        self.token_doc_freq: Dict[str, int] = {}
        self._load_token_postings_index()

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

    def _build_doc_navigation_index(self) -> None:
        self.docs_by_url.clear()
        self.doc_position_by_id.clear()
        self.docs_by_url_section.clear()

        for doc_id, row in enumerate(self.metadata):
            meta = row.get("metadata", {})
            url = str(meta.get("url", ""))
            self.docs_by_url[url].append(doc_id)

        for url, doc_ids in self.docs_by_url.items():
            doc_ids.sort(
                key=lambda doc_id: int(self.metadata[doc_id].get("metadata", {}).get("chunk_index", doc_id))
            )
            for position, doc_id in enumerate(doc_ids):
                self.doc_position_by_id[doc_id] = position
                meta = self.metadata[doc_id].get("metadata", {})
                section = str(meta.get("section", "")).strip() or str(meta.get("title", "Untitled"))
                self.docs_by_url_section[(url, section)].append(doc_id)

    def _build_token_postings_from_metadata(self) -> Dict[str, List[Tuple[int, int]]]:
        postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for doc_id, row in enumerate(self.metadata):
            meta = row.get("metadata", {})
            combined = " ".join(
                [
                    str(meta.get("title", "")),
                    str(meta.get("section", "")),
                    str(row.get("text", "")),
                ]
            )
            tf = Counter(query_terms(combined))
            for token, count in tf.items():
                postings[token][doc_id] = int(count)

        normalized: Dict[str, List[Tuple[int, int]]] = {}
        for token, doc_map in postings.items():
            pairs = sorted(doc_map.items(), key=lambda item: item[0])
            normalized[token] = pairs
        return normalized

    def _load_token_postings_index(self) -> None:
        loaded = False
        if TOKEN_INDEX_PATH.exists():
            try:
                with TOKEN_INDEX_PATH.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                raw_postings = payload.get("token_postings", {}) if isinstance(payload, dict) else {}
                if isinstance(raw_postings, dict):
                    parsed: Dict[str, List[Tuple[int, int]]] = {}
                    for token, rows in raw_postings.items():
                        if not isinstance(token, str) or not isinstance(rows, list):
                            continue
                        pairs: List[Tuple[int, int]] = []
                        for row in rows:
                            if not isinstance(row, list) or len(row) != 2:
                                continue
                            doc_id = int(row[0])
                            freq = int(row[1])
                            if doc_id < 0 or doc_id >= len(self.metadata) or freq <= 0:
                                continue
                            pairs.append((doc_id, freq))
                        if pairs:
                            parsed[token] = pairs
                    if parsed:
                        self.token_postings = parsed
                        loaded = True
            except Exception as exc:
                print(f"Token index load failed; rebuilding in memory. Reason: {exc}", file=sys.stderr)

        if not loaded:
            self.token_postings = self._build_token_postings_from_metadata()
            try:
                payload = {
                    "version": 1,
                    "doc_count": len(self.metadata),
                    "token_postings": {
                        token: [[doc_id, freq] for doc_id, freq in postings]
                        for token, postings in self.token_postings.items()
                    },
                }
                with TOKEN_INDEX_PATH.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"Token index save skipped. Reason: {exc}", file=sys.stderr)

        self.token_doc_freq = {token: len(postings) for token, postings in self.token_postings.items()}

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
            section=str(meta.get("section", "")),
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

    def _postings_search(self, query: str, top_k: int) -> List[SearchResult]:
        q_tokens = expanded_query_terms(query, intents=infer_query_intents(query), label_hints=extract_query_labels(query))
        if not q_tokens:
            return []
        q_token_set = set(q_tokens)
        n_docs = max(1, len(self.metadata))
        doc_scores: Dict[int, float] = defaultdict(float)
        q_phrase = query.lower().strip()

        for token in q_tokens:
            postings = self.token_postings.get(token)
            if not postings:
                continue
            df = max(1, self.token_doc_freq.get(token, len(postings)))
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id, freq in postings:
                doc_scores[doc_id] += idf * (1.0 + math.log1p(freq))

        if not doc_scores:
            return []

        ranked: List[Tuple[int, float]] = []
        for doc_id, score in doc_scores.items():
            row = self.metadata[doc_id]
            meta = row.get("metadata", {})
            text_lower = str(row.get("text", "")).lower()
            title_lower = str(meta.get("title", "")).lower()
            section_lower = str(meta.get("section", "")).lower()
            title_tokens = set(query_terms(title_lower))
            section_tokens = set(query_terms(section_lower))
            if q_phrase and q_phrase in text_lower:
                score += 3.5
            if q_phrase and q_phrase in title_lower:
                score += 1.8
            title_overlap = len(q_token_set & title_tokens)
            section_overlap = len(q_token_set & section_tokens)
            score += title_overlap * 0.9
            score += section_overlap * 1.2
            if score > 0:
                ranked.append((doc_id, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        limit = max(top_k, TOP_K) * POSTINGS_CANDIDATE_MULTIPLIER
        return [
            self._result_from_doc(doc_id, score, distance=1.0 / (score + 1.0))
            for doc_id, score in ranked[:limit]
        ]

    def _merge_lexical_results(
        self,
        bm25_results: List[SearchResult],
        postings_results: List[SearchResult],
        candidate_limit: int,
    ) -> List[SearchResult]:
        if not postings_results:
            return bm25_results[:candidate_limit]
        merged_scores: Dict[int, float] = defaultdict(float)
        best_distance: Dict[int, float] = {}

        for rank, result in enumerate(bm25_results):
            merged_scores[result.doc_id] += result.score + (1.0 / (rank + 1))
            best_distance[result.doc_id] = result.distance

        for rank, result in enumerate(postings_results):
            merged_scores[result.doc_id] += (result.score * 1.05) + (1.2 / (rank + 1))
            if result.doc_id not in best_distance:
                best_distance[result.doc_id] = result.distance
            else:
                best_distance[result.doc_id] = min(best_distance[result.doc_id], result.distance)

        ranked = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            self._result_from_doc(doc_id, score=score, distance=best_distance.get(doc_id, 1.0))
            for doc_id, score in ranked[:candidate_limit]
        ]

    def _page_text(self, url: str, max_chunks: int = 12) -> str:
        doc_ids = self.docs_by_url.get(url, [])
        if not doc_ids:
            return ""
        chunks = [str(self.metadata[doc_id].get("text", "")) for doc_id in doc_ids[:max_chunks]]
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    def _context_doc_ids_for_result(self, result: SearchResult) -> List[int]:
        if result.doc_id < 0 or result.doc_id >= len(self.metadata):
            return [result.doc_id]
        meta = self.metadata[result.doc_id].get("metadata", {})
        url = str(meta.get("url", ""))
        section = str(meta.get("section", "")).strip() or str(meta.get("title", "Untitled"))
        ordered_ids = self.docs_by_url.get(url, [result.doc_id])
        if not ordered_ids:
            return [result.doc_id]
        position = self.doc_position_by_id.get(result.doc_id, 0)
        start = max(0, position - SECTION_WINDOW_CHUNKS)
        end = min(len(ordered_ids), position + SECTION_WINDOW_CHUNKS + 1)
        selected: List[int] = list(ordered_ids[start:end])

        section_ids = self.docs_by_url_section.get((url, section), [])
        if section_ids:
            ranked_section_ids = sorted(
                section_ids,
                key=lambda doc_id: abs(self.doc_position_by_id.get(doc_id, 0) - position),
            )
            for doc_id in ranked_section_ids:
                if doc_id not in selected:
                    selected.append(doc_id)
                if len(selected) >= SECTION_CONTEXT_MAX_CHUNKS:
                    break

        selected.sort(key=lambda doc_id: self.doc_position_by_id.get(doc_id, 0))
        return selected[:SECTION_CONTEXT_MAX_CHUNKS]

    def expand_result_for_context(self, query: str, result: SearchResult) -> SearchResult:
        query_token_set = set(query_terms(query, for_query=True))
        doc_ids = self._context_doc_ids_for_result(result)
        if not doc_ids or any(doc_id < 0 or doc_id >= len(self.metadata) for doc_id in doc_ids):
            return result
        sections: List[str] = []
        token_hits = 0

        for doc_id in doc_ids:
            chunk_text = str(self.metadata[doc_id].get("text", "")).strip()
            if not chunk_text:
                continue
            sections.append(chunk_text)
            if query_token_set:
                token_hits += len(query_token_set & set(query_terms(chunk_text)))

        expanded_text = "\n".join(sections).strip()
        if not expanded_text:
            expanded_text = result.text

        # If the local section window has weak lexical overlap, include a larger
        # page slice so LLM has enough surrounding context to answer safely.
        if query_token_set and token_hits < max(1, len(query_token_set) // 3):
            page_text = self._page_text(result.url)
            if page_text:
                expanded_text = page_text

        return SearchResult(
            doc_id=result.doc_id,
            text=expanded_text,
            url=result.url,
            title=result.title,
            chunk_index=result.chunk_index,
            distance=result.distance,
            score=result.score,
            section=result.section,
        )

    def find_supporting_results(self, query: str, answer: str, limit: int = TOP_K) -> List[SearchResult]:
        if not answer.strip():
            return []
        query_tokens = set(query_terms(query, for_query=True))
        answer_tokens = set(query_terms(answer))
        answer_anchors = _extract_fact_anchors(answer)
        ranked: List[Tuple[int, float]] = []

        for doc_id, row in enumerate(self.metadata):
            meta = row.get("metadata", {})
            combined = " ".join(
                [
                    str(meta.get("title", "")),
                    str(meta.get("section", "")),
                    str(row.get("text", "")),
                ]
            )
            combined_tokens = set(query_terms(combined))
            if not combined_tokens:
                continue

            score = 0.0
            score += len(answer_tokens & combined_tokens) * 1.8
            score += len(query_tokens & combined_tokens) * 1.1

            combined_anchors = _extract_fact_anchors(combined)
            for key in ("labels", "dates", "times", "weekdays", "emails", "rooms"):
                score += len(answer_anchors[key] & combined_anchors[key]) * 2.2

            if score <= 0.0:
                continue
            ranked.append((doc_id, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            self._result_from_doc(doc_id, score=score, distance=1.0 / (score + 1.0))
            for doc_id, score in ranked[: max(1, limit)]
        ]

    def _dense_search(self, query: str, top_k: int) -> List[SearchResult]:
        if self.index is None or self.model is None:
            return []

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_vec, max(top_k, TOP_K))
        results: List[SearchResult] = []
        for distance, doc_id in zip(distances[0], indices[0]):
            if doc_id < 0 or doc_id >= len(self.metadata):
                continue
            dense_score = 1.0 / (1.0 + float(distance))
            results.append(self._result_from_doc(doc_id, dense_score, float(distance)))
        return results

    def _fuse_results(
        self,
        lexical_results: List[SearchResult],
        dense_results: List[SearchResult],
        candidate_limit: int,
    ) -> List[SearchResult]:
        if not dense_results:
            return lexical_results[:candidate_limit]
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
            for doc_id, fused_score in ranked[:candidate_limit]
        ]

    def _semantic_rerank_scores(self, query: str, candidates: List[SearchResult]) -> Dict[int, float]:
        if self.model is None or not candidates:
            return {}
        try:
            candidate_texts = [
                f"{result.title}\n{result.text[:900]}".strip()
                for result in candidates
            ]
            vectors = self.model.encode([query] + candidate_texts, convert_to_numpy=False)
            if not vectors or len(vectors) != len(candidate_texts) + 1:
                return {}
            query_vector = list(vectors[0])
            scores: Dict[int, float] = {}
            for result, doc_vector in zip(candidates, vectors[1:]):
                scores[result.doc_id] = cosine_similarity(query_vector, list(doc_vector))
            return scores
        except Exception:
            return {}

    def _rerank_results(self, query: str, candidates: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not candidates:
            return []

        intents = infer_query_intents(query)
        label_hints = extract_query_labels(query)
        label_targets = query_label_targets(label_hints)
        strict_prefixes = strict_single_label_prefixes(label_targets)
        query_token_set = set(expanded_query_terms(query, intents=intents, label_hints=label_hints))
        if not query_token_set:
            query_token_set = set(query_terms(query, for_query=True))
        semantic_scores = self._semantic_rerank_scores(query, candidates)
        query_lower = query.lower().strip()

        reranked: List[Tuple[float, SearchResult]] = []
        for rank, result in enumerate(candidates):
            combined_text = f"{result.title}\n{result.text}"
            combined_lower = combined_text.lower()
            doc_token_set = set(query_terms(combined_text))

            overlap = len(query_token_set & doc_token_set)
            coverage = overlap / max(1, len(query_token_set))
            phrase_hit = 1.0 if query_lower and len(query_lower) >= 6 and query_lower in combined_lower else 0.0

            score = (
                result.score * 0.35
                + overlap * 1.25
                + coverage * 4.5
                + phrase_hit * 2.2
            )
            score += semantic_scores.get(result.doc_id, 0.0) * RERANK_SEMANTIC_WEIGHT

            if label_targets:
                exact_line_hits = 0
                conflict_only_hits = 0
                for raw_line in result.text.splitlines():
                    line = normalize_line(raw_line).strip().lower()
                    if not line:
                        continue
                    has_exact_label, has_conflicting_label = line_label_alignment(line, label_targets)
                    has_prefix_conflict = line_has_prefix_conflict(line, label_targets, strict_prefixes)
                    if has_exact_label and not has_prefix_conflict:
                        exact_line_hits += 1
                    elif has_conflicting_label or has_prefix_conflict:
                        conflict_only_hits += 1

                if exact_line_hits > 0:
                    score += 6.0 + min(2.0, exact_line_hits * 0.4)
                elif conflict_only_hits > 0:
                    score -= 3.5

            if intents["wants_time"] and (DATE_PATTERN.search(combined_lower) or TIME_PATTERN.search(combined_lower)):
                score += 1.1
            if intents["wants_location"] and any(word in combined_lower for word in LOCATION_HINTS):
                score += 1.0
            if intents["wants_staff"] and any(word in combined_lower for word in STAFF_HINTS):
                score += 1.0
            if intents["wants_assignments"] and any(word in combined_lower for word in ASSIGNMENT_HINTS):
                score += 0.8

            score += max(0.0, 0.8 - rank * 0.06)
            reranked.append((score, result))

        reranked.sort(key=lambda item: item[0], reverse=True)
        return [
            self._result_from_doc(result.doc_id, score=score, distance=result.distance)
            for score, result in reranked[: max(top_k, TOP_K)]
        ]

    def search(self, query: str, top_k: int = TOP_K) -> List[SearchResult]:
        candidate_limit = max(top_k, TOP_K) * RERANK_CANDIDATE_MULTIPLIER
        bm25_results = self._bm25_search(query, candidate_limit)
        postings_results = self._postings_search(query, candidate_limit)
        lexical_results = self._merge_lexical_results(bm25_results, postings_results, candidate_limit)
        lexical_candidates = lexical_results[:candidate_limit]
        if self.index is None or self.model is None:
            return self._rerank_results(query, lexical_candidates, top_k)

        dense_results = self._dense_search(query, candidate_limit)
        fused_candidates = self._fuse_results(lexical_candidates, dense_results, candidate_limit)
        return self._rerank_results(query, fused_candidates, top_k)


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


def build_deterministic_label_answer(
    query: str,
    selection: EvidenceSelection,
    results: Optional[List[SearchResult]] = None,
) -> Optional[str]:
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

    if results and target_prefix in {"c", "p"} and (assignment_name is None or fallback_label_line is None):
        for result in results[: max(TOP_K * 2, 10)]:
            for raw_line in result.text.splitlines():
                line = normalize_line(raw_line).strip("# ").strip()
                if not line:
                    continue
                has_exact, _ = line_label_alignment(line, target_only)
                if not has_exact:
                    continue
                if line_has_prefix_conflict(line, target_only, strict_prefixes):
                    continue
                line_lower = line.lower()
                if target_prefix in {"c", "p"} and "resub" in line_lower and "i.s." not in line_lower:
                    continue
                if fallback_label_line is None:
                    fallback_label_line = line
                if assignment_name is None:
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
                if assignment_name is not None and fallback_label_line is not None:
                    break
            if assignment_name is not None and fallback_label_line is not None:
                break

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


def _iter_metadata_lines(metadata: List[dict]) -> List[str]:
    lines: List[str] = []
    for row in metadata:
        text = row.get("text", "")
        if not isinstance(text, str):
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line:
                lines.append(line)
    return lines


def _format_day_list(days: List[str]) -> str:
    if not days:
        return ""
    if len(days) == 1:
        return days[0]
    if len(days) == 2:
        return f"{days[0]} and {days[1]}"
    return ", ".join(days[:-1]) + f", and {days[-1]}"


def _extract_quiz_number(query: str) -> Optional[int]:
    q = query.lower()
    match = re.search(r"\bquiz\s*0?(\d+)\b", q)
    if match:
        return int(match.group(1))

    match = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+quiz\b", q)
    if match:
        return int(match.group(1))
    match = re.search(r"\bquiz\s+(\d+)(?:st|nd|rd|th)\b", q)
    if match:
        return int(match.group(1))

    for word, number in ORDINAL_WORD_TO_INT.items():
        if f"{word} quiz" in q or f"quiz {word}" in q:
            return number
    return None


def _answer_named_item_from_lines(target: str, lines: List[str]) -> Optional[str]:
    for raw_line in lines:
        if target not in raw_line.lower():
            continue
        line = normalize_line(raw_line).strip("# ").strip()
        if not line:
            continue
        release_match = RELEASED_ASSIGNMENT_PATTERN.search(line)
        if release_match:
            label = release_match.group(1).lower()
            title = release_match.group(2).strip(" -:")
            due = release_match.group(3).strip(" -:.")
            return f"{title} is {_format_canonical_label(label)}. It is due by {due}."
        return line
    return None


def build_deterministic_fact_answer(
    query: str,
    results: List[SearchResult],
    metadata: List[dict],
) -> Optional[str]:
    q = query.lower().strip()
    metadata_lines = _iter_metadata_lines(metadata)
    day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    day_full = {
        "Mon": "Monday",
        "Tue": "Tuesday",
        "Wed": "Wednesday",
        "Thu": "Thursday",
        "Fri": "Friday",
    }

    asks_instructor = (
        ("instructor" in q or "professor" in q or "who teaches" in q or "teacher" in q)
        and "office hour" not in q
        and "ta" not in q
    )
    if asks_instructor:
        name = None
        email = None
        for line in metadata_lines:
            if name is None:
                match = INSTRUCTOR_LINE_PATTERN.search(line)
                if match:
                    name = match.group(1).strip()
            if email is None and "instructor" in line.lower():
                match = EMAIL_PATTERN.search(line)
                if match:
                    email = match.group(0)
            if name and email:
                break
        if name:
            answer = f"The instructor is {name}."
            if email:
                answer += f" Instructor email: {email}."
            return answer

    asks_lecture_schedule = (
        "lecture" in q
        and any(word in q for word in ("when", "where", "time", "schedule", "day"))
    )
    if asks_lecture_schedule:
        day_counts: Counter = Counter()
        location_line = None
        for line in metadata_lines:
            for day_match in LECTURE_DAY_PATTERN.finditer(line):
                day_counts[day_match.group(1).title()] += 1
            if location_line is None:
                location_match = LECTURE_LOCATION_LINE_PATTERN.search(line)
                if location_match:
                    location_line = normalize_line(location_match.group(0)).strip(" _")

        if day_counts or location_line:
            ordered_by_freq = sorted(day_counts.keys(), key=lambda day: (-day_counts[day], day_order.get(day, 99)))
            ordered_days = sorted(ordered_by_freq[:2], key=lambda day: day_order.get(day, 99))
            full_days = [day_full.get(day, day) for day in ordered_days]
            parts = []
            if full_days:
                parts.append(f"Lectures meet on {_format_day_list(full_days)}.")
            if location_line:
                parts.append(f"Lecture times/locations: {location_line}.")
            return " ".join(parts)

    asks_quiz_time = "quiz" in q and any(word in q for word in ("when", "date", "time", "scheduled"))
    if asks_quiz_time:
        quiz_number = _extract_quiz_number(query)
        if quiz_number is not None:
            for line in metadata_lines:
                match = QUIZ_LINE_PATTERN.search(line)
                if not match:
                    continue
                current_quiz = int(match.group(3))
                if current_quiz != quiz_number:
                    continue
                weekday = match.group(1).title()
                date = match.group(2)
                return f"Quiz {quiz_number} is on {weekday} {date}."

    asks_esn = "esn" in q or "e/s/n" in q
    if asks_esn:
        has_explicit_definitions = any(
            "e(xcellent)" in line.lower() and "s(atisfactory)" in line.lower() and "n(ot yet)" in line.lower()
            for line in metadata_lines
        )
        if has_explicit_definitions:
            return "ESN stands for Excellent, Satisfactory, and Not yet."

        has_definition_heading = any("e/s/n grading definitions" in line.lower() for line in metadata_lines)
        has_esn_usage = any("esn grade" in line.lower() or "total esn" in line.lower() for line in metadata_lines)
        if has_definition_heading:
            return (
                "In this course, ESN refers to the E/S/N grading system. "
                "See the \"E/S/N Grading Definitions\" section on the Grading Rubrics page."
            )
        if has_esn_usage:
            return "ESN is a grading label used in this course for assignment and quiz assessment."

    asks_named_item = q.startswith("what is ") or q.startswith("what's ")
    if asks_named_item and not extract_query_labels(query):
        target = re.sub(r"^what(?:'s|\s+is)\s+", "", q).strip(" ?.!")
        if target and len(target.split()) >= 2 and not any(
            blocked in target for blocked in ("instructor", "lecture", "quiz", "policy", "esn")
        ):
            metadata_match = _answer_named_item_from_lines(target, metadata_lines)
            if metadata_match:
                return metadata_match

            result_lines: List[str] = []
            for result in results:
                result_lines.extend(result.text.splitlines())
            result_match = _answer_named_item_from_lines(target, result_lines)
            if result_match:
                return result_match

    return None


def _extract_ollama_output_text(payload: dict) -> str:
    text = payload.get("response")
    if isinstance(text, str):
        return text.strip()
    return ""


def ollama_model_from_tags_payload(payload: dict) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    for model_row in models:
        if not isinstance(model_row, dict):
            continue
        model_name = model_row.get("name") or model_row.get("model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return None


def discover_ollama_model(force_refresh: bool = False) -> Optional[str]:
    global _OLLAMA_MODEL_CACHE, _OLLAMA_MODEL_LOOKUP_ATTEMPTED
    with _OLLAMA_MODEL_LOCK:
        if _OLLAMA_MODEL_LOOKUP_ATTEMPTED and not force_refresh:
            return _OLLAMA_MODEL_CACHE or None

        model_name = ""
        req = Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            method="GET",
        )
        try:
            with urlopen(req, timeout=OLLAMA_TAGS_TIMEOUT_SECS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            model_name = ollama_model_from_tags_payload(payload) or ""
        except (urlerror.URLError, TimeoutError, ValueError):
            model_name = ""

        _OLLAMA_MODEL_CACHE = model_name
        _OLLAMA_MODEL_LOOKUP_ATTEMPTED = True
        return model_name or None


def preferred_ollama_model() -> Optional[str]:
    if OLLAMA_MODEL:
        return OLLAMA_MODEL
    return discover_ollama_model()


def request_ollama_generate(model: str, prompt: str) -> dict:
    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
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

    with urlopen(req, timeout=OLLAMA_TIMEOUT_SECS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_source_citations(text: str) -> List[int]:
    citations: List[int] = []
    seen = set()
    for match in SOURCE_CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            citations.append(number)
            seen.add(number)
    return citations


def llm_citations_are_valid(text: str, max_source_num: int) -> bool:
    citations = extract_source_citations(text)
    if not citations:
        return False
    return all(1 <= number <= max_source_num for number in citations)


def _strip_source_citations(text: str) -> str:
    return SOURCE_CITATION_PATTERN.sub(" ", text)


def _normalize_anchor(value: str) -> str:
    return normalize_line(value).lower()


def _extract_fact_anchors(text: str) -> Dict[str, set]:
    sanitized = _strip_source_citations(text)
    anchors: Dict[str, set] = {
        "labels": set(),
        "dates": set(),
        "times": set(),
        "weekdays": set(),
        "emails": set(),
        "rooms": set(),
    }
    for label in extract_query_labels(sanitized):
        canonical = canonicalize_label_token(label)
        if canonical:
            anchors["labels"].add(canonical)
    for match in DATE_PATTERN.finditer(sanitized):
        anchors["dates"].add(_normalize_anchor(match.group(0)))
    for match in TIME_PATTERN.finditer(sanitized):
        anchors["times"].add(_normalize_anchor(match.group(0)))
    for match in WEEKDAY_PATTERN.finditer(sanitized):
        anchors["weekdays"].add(_normalize_anchor(match.group(0)))
    for match in EMAIL_PATTERN.finditer(sanitized):
        anchors["emails"].add(_normalize_anchor(match.group(0)))
    for match in ROOM_PATTERN.finditer(sanitized):
        anchors["rooms"].add(_normalize_anchor(match.group(0)))
    return anchors


def llm_answer_is_grounded(answer: str, context_block: str) -> bool:
    answer_anchors = _extract_fact_anchors(answer)
    context_anchors = _extract_fact_anchors(context_block)
    for key in ("labels", "dates", "times", "weekdays", "emails", "rooms"):
        if answer_anchors[key] and not answer_anchors[key].issubset(context_anchors[key]):
            return False
    return True


def llm_answer_conflicts_with_preferred(answer: str, preferred_answer: str) -> bool:
    preferred_anchors = _extract_fact_anchors(preferred_answer)
    llm_anchors = _extract_fact_anchors(answer)
    for key in ("labels", "dates", "times", "emails", "rooms"):
        required = preferred_anchors[key]
        offered = llm_anchors[key]
        if required and offered and not offered.issubset(required):
            return True
    return False


def llm_source_limit(results: List[SearchResult]) -> int:
    return min(len(results), LLM_CONTEXT_SOURCE_LIMIT, TOP_K)


def _select_llm_context_lines(query: str, result: SearchResult, line_limit: int) -> List[str]:
    query_tokens = set(query_terms(query, for_query=True))
    intents = infer_query_intents(query)
    label_targets = query_label_targets(extract_query_labels(query))
    strict_prefixes = strict_single_label_prefixes(label_targets)
    query_lower = query.lower().strip()

    candidates: List[Tuple[float, str]] = []
    seen = set()
    for raw_line in result.text.splitlines():
        line = normalize_line(raw_line).strip("# ").strip()
        if not line or line in seen:
            continue
        seen.add(line)

        score = 0.0
        line_lower = line.lower()
        line_tokens = set(query_terms(line))
        overlap = len(query_tokens & line_tokens)
        score += overlap * 2.2

        if label_targets:
            has_exact_label, has_conflicting_label = line_label_alignment(line_lower, label_targets)
            if has_exact_label:
                score += 8.0
            if has_conflicting_label and not has_exact_label:
                score -= 4.0
            if line_has_prefix_conflict(line_lower, label_targets, strict_prefixes):
                score -= 4.5

        if intents["wants_time"] and (DATE_PATTERN.search(line) or TIME_PATTERN.search(line)):
            score += 1.8
        if intents["wants_location"] and any(word in line_lower for word in LOCATION_HINTS):
            score += 1.3
        if intents["wants_staff"] and any(word in line_lower for word in STAFF_HINTS):
            score += 1.3
        if intents["wants_assignments"] and any(word in line_lower for word in ASSIGNMENT_HINTS):
            score += 1.2
        if query_lower and len(query_lower) >= 8 and query_lower in line_lower:
            score += 2.6

        if score > 0:
            candidates.append((score, line))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [line for _, line in candidates[:line_limit]]
    if selected:
        return selected

    fallback = normalize_line(result.text)[:LLM_CONTEXT_SNIPPET_CHARS].strip()
    return [fallback] if fallback else []


def _context_results_for_llm(
    query: str,
    results: List[SearchResult],
    retriever: Optional[Retriever] = None,
) -> List[SearchResult]:
    limited_results = results[: llm_source_limit(results)]
    if not retriever:
        return limited_results
    expanded: List[SearchResult] = []
    for result in limited_results:
        try:
            expanded.append(retriever.expand_result_for_context(query, result))
        except Exception:
            expanded.append(result)
    return expanded


def build_llm_context(
    query: str,
    results: List[SearchResult],
    retriever: Optional[Retriever] = None,
) -> str:
    sections: List[str] = []
    for source_num, result in enumerate(_context_results_for_llm(query, results, retriever), start=1):
        selected_lines = _select_llm_context_lines(query, result, LLM_CONTEXT_LINES_PER_SOURCE)
        if not selected_lines:
            continue
        bullet_lines = "\n".join(f"- {line}" for line in selected_lines)
        source_title = result.title
        if result.section and result.section != result.title:
            source_title = f"{source_title} / {result.section}"
        sections.append(
            f"[source {source_num}] {source_title} - {result.url}\n{bullet_lines}"
        )
    return "\n\n".join(sections)


def build_llm_answer(
    query: str,
    selection: EvidenceSelection,
    results: List[SearchResult],
    retriever: Optional[Retriever] = None,
    preferred_answer: Optional[str] = None,
) -> Optional[str]:
    if not ENABLE_LLM_RESPONSE:
        return None
    if not results:
        return NO_ANSWER_TEXT

    context_block = build_llm_context(query, results, retriever)
    if not context_block.strip():
        return NO_ANSWER_TEXT

    model_name = preferred_ollama_model()
    if not model_name:
        print(
            "Ollama model unavailable; set OLLAMA_MODEL or install/pull any model. "
            "Falling back to extractive/deterministic answer.",
            file=sys.stderr,
        )
        return None

    system_prompt = (
        "You are a helpful CSE 121 course assistant.\n"
        "Use only the provided source snippets.\n"
        f"If the evidence is insufficient or ambiguous, reply exactly: {NO_ANSWER_SENTINEL}\n"
        "If the user question is ambiguous, ask one concise clarification question and stop.\n"
        "If sufficient, answer naturally in 2-4 sentences and cite evidence as [source N] on the course website.\n"
        "Do not infer dates, times, names, or policy details unless explicitly present in the snippets."
    )

    user_parts = [
        f"Question:\n{query}\n\n"
        f"Snippets from the course website:\n{context_block}"
    ]
    if preferred_answer and preferred_answer != NO_ANSWER_TEXT:
        user_parts.append(
            "Candidate factual answer from deterministic retrieval rules:\n"
            f"{preferred_answer}\n\n"
            "If supported by the snippets, preserve these facts and rewrite clearly."
        )
    user_prompt = "\n\n".join(user_parts)

    prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        payload = request_ollama_generate(model_name, prompt)
        model_error = payload.get("error")
        if isinstance(model_error, str) and model_error:
            # If a configured model name is stale/not installed, try a discovered
            # local model once before giving up.
            if OLLAMA_MODEL and "not found" in model_error.lower():
                discovered = discover_ollama_model(force_refresh=True)
                if discovered and discovered != model_name:
                    payload = request_ollama_generate(discovered, prompt)
                    model_error = payload.get("error")
            if isinstance(model_error, str) and model_error:
                raise ValueError(model_error)

        text = _extract_ollama_output_text(payload)
        if not text:
            return None
        if text.strip() == NO_ANSWER_SENTINEL:
            return NO_ANSWER_TEXT
        if is_clarification_request(text):
            return text
        if LLM_REQUIRE_VALID_CITATIONS:
            max_source_num = llm_source_limit(results)
            if not llm_citations_are_valid(text, max_source_num):
                return None
        if not llm_answer_is_grounded(text, context_block):
            return None
        if preferred_answer and preferred_answer != NO_ANSWER_TEXT:
            if llm_answer_conflicts_with_preferred(text, preferred_answer):
                return None
        return text
    except (urlerror.URLError, TimeoutError, ValueError) as exc:
        print(f"Ollama call failed; falling back to extractive answer. Reason: {exc}", file=sys.stderr)
        return None


def source_reasons_for_result(query: str, result: SearchResult, evidence_lines: List[str]) -> List[str]:
    reasons: List[str] = []
    intents = infer_query_intents(query)
    label_targets = query_label_targets(extract_query_labels(query))
    combined = f"{result.title}\n{result.text}"
    combined_lower = combined.lower()

    if evidence_lines:
        reasons.append("Contains a supporting line used directly in the answer.")

    if label_targets:
        has_exact_label, _ = line_label_alignment(combined_lower, label_targets)
        if has_exact_label:
            for prefix in ("r", "c", "p", "quiz"):
                labels = sorted(label_targets.get(prefix, set()))
                if labels:
                    reasons.append(f"Matches requested label {_format_canonical_label(labels[0])}.")
                    break

    if intents["wants_time"] and (DATE_PATTERN.search(combined_lower) or TIME_PATTERN.search(combined_lower)):
        reasons.append("Includes date/time details relevant to your question.")
    if intents["wants_location"] and any(word in combined_lower for word in LOCATION_HINTS):
        reasons.append("Includes location details relevant to your question.")
    if intents["wants_staff"] and any(word in combined_lower for word in STAFF_HINTS):
        reasons.append("Includes staff-related details relevant to your question.")
    if intents["wants_assignments"] and any(word in combined_lower for word in ASSIGNMENT_HINTS):
        reasons.append("Includes assignment policy/details relevant to your question.")

    if not reasons:
        reasons.append("Top reranked relevance match for your question.")
    return reasons[:3]


def best_source_snippet(query: str, result: SearchResult, max_chars: int = 220) -> str:
    lines = _select_llm_context_lines(query, result, line_limit=1)
    if lines:
        snippet = lines[0]
    else:
        snippet = normalize_line(result.text)
    snippet = snippet.strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def source_support_score(query: str, answer: str, result: SearchResult) -> float:
    combined = f"{result.title}\n{result.section}\n{result.text}"
    combined_tokens = set(query_terms(combined))
    query_tokens = set(query_terms(query, for_query=True))
    answer_tokens = set(query_terms(answer))
    score = 0.0
    score += len(query_tokens & combined_tokens) * 1.2
    score += len(answer_tokens & combined_tokens) * 1.6

    answer_anchors = _extract_fact_anchors(answer)
    result_anchors = _extract_fact_anchors(combined)
    for key in ("labels", "dates", "times", "weekdays", "emails", "rooms"):
        score += len(answer_anchors[key] & result_anchors[key]) * 2.4

    query_lower = query.lower()
    combined_lower = combined.lower()
    if "instructor" in query_lower and "/staff/" in result.url.lower():
        score += 2.0
    if "lecture" in query_lower and "lecture @" in combined_lower:
        score += 2.0
    if "quiz" in query_lower and "quiz" in combined_lower:
        score += 1.6
    if "esn" in query_lower and "e/s/n" in combined_lower:
        score += 1.6
    return score


def rank_sources_for_answer(query: str, answer: str, results: List[SearchResult]) -> List[SearchResult]:
    indexed = list(enumerate(results))
    indexed.sort(
        key=lambda item: (source_support_score(query, answer, item[1]), -item[0]),
        reverse=True,
    )
    return [result for _, result in indexed]


def build_chat_payload(
    message: str,
    retriever: Retriever,
    memory_manager: Optional[SessionMemoryManager] = None,
    session_id: Optional[str] = None,
    include_source_details: bool = DEBUG_SOURCE_DETAILS_DEFAULT,
) -> Dict[str, Any]:
    original_message = message.strip()
    if not original_message:
        raise ValueError("message is required")

    history = memory_manager.get_turns(session_id) if (memory_manager and ENABLE_SESSION_MEMORY) else []
    memory_notes: List[str] = []
    pending_clarification_query = (
        memory_manager.get_pending_clarification(session_id)
        if (memory_manager and ENABLE_SESSION_MEMORY and ENABLE_CLARIFICATION_STITCH)
        else None
    )

    stitched_query = original_message
    used_clarification_stitch = False
    if pending_clarification_query and should_stitch_pending_query(original_message):
        stitched_query = (
            f"{pending_clarification_query}\n"
            f"User clarification:\n{original_message}"
        )
        used_clarification_stitch = True
        memory_notes.append("Applied pending clarification context from previous turn.")

    query_for_search, rewrite_notes, _ = rewrite_query_with_memory(stitched_query, history)
    memory_notes.extend(rewrite_notes)
    query_lower = query_for_search.lower()
    label_targets = query_label_targets(extract_query_labels(query_for_search))
    wants_name_query = any(
        phrase in query_lower for phrase in ("what is", "what's", "name", "called", "title")
    )
    retrieval_k = TOP_K
    if wants_name_query:
        retrieval_k = max(TOP_K, 12)

    results = retriever.search(query_for_search, retrieval_k)
    selection = collect_evidence(query_for_search, results)
    label_exact_evidence = has_exact_label_evidence(selection.lines, label_targets)

    deterministic_answer = build_deterministic_label_answer(query_for_search, selection, results)
    retriever_metadata = getattr(retriever, "metadata", [])
    deterministic_fact_answer = build_deterministic_fact_answer(
        query_for_search,
        results,
        retriever_metadata,
    )
    fallback_mode = "no_answer"
    fallback_answer = NO_ANSWER_TEXT
    if deterministic_answer is not None:
        fallback_answer = deterministic_answer
        fallback_mode = "deterministic"
    elif deterministic_fact_answer is not None:
        fallback_answer = deterministic_fact_answer
        fallback_mode = "deterministic"
    elif label_targets:
        # For explicit label questions, avoid hallucinations by requiring exact
        # label evidence and returning extractive text only from that evidence.
        if label_exact_evidence and selection.lines:
            fallback_answer = build_extractive_answer(selection)
            fallback_mode = "extractive"
    elif selection.confident:
        fallback_answer = build_extractive_answer(selection)
        fallback_mode = "extractive"

    answer_mode = fallback_mode
    answer = fallback_answer

    should_attempt_llm = ALWAYS_ATTEMPT_LLM or fallback_mode == "no_answer"
    if should_attempt_llm:
        llm_answer = build_llm_answer(
            query_for_search,
            selection,
            results,
            retriever=retriever,
            preferred_answer=fallback_answer if fallback_mode != "no_answer" else None,
        )
        if llm_answer is not None:
            if llm_answer == NO_ANSWER_TEXT and fallback_mode != "no_answer":
                answer = fallback_answer
                answer_mode = fallback_mode
            elif is_clarification_request(llm_answer):
                if fallback_mode != "no_answer":
                    answer = fallback_answer
                    answer_mode = fallback_mode
                else:
                    answer = llm_answer
                    answer_mode = "clarification"
            else:
                answer = llm_answer
                answer_mode = "llm" if llm_answer != NO_ANSWER_TEXT else "no_answer"

    evidence_by_doc_id: Dict[int, List[str]] = defaultdict(list)
    for line, source_num in selection.lines:
        source_index = source_num - 1
        if 0 <= source_index < len(results):
            evidence_by_doc_id[results[source_index].doc_id].append(line)

    sources = []
    source_results = results[:TOP_K]
    if answer_mode == "deterministic":
        support_results: List[SearchResult] = []
        if hasattr(retriever, "find_supporting_results"):
            try:
                support_results = retriever.find_supporting_results(
                    query_for_search,
                    answer,
                    limit=max(TOP_K * 2, 8),
                )
            except Exception:
                support_results = []
        merged_by_doc_id: Dict[int, SearchResult] = {}
        for result in support_results + results:
            if result.doc_id not in merged_by_doc_id:
                merged_by_doc_id[result.doc_id] = result
        source_results = rank_sources_for_answer(
            query_for_search,
            answer,
            list(merged_by_doc_id.values()),
        )[:TOP_K]
    for rank, result in enumerate(source_results, start=1):
        evidence_lines = evidence_by_doc_id.get(result.doc_id, [])
        source_payload = {
            "rank": rank,
            "title": result.title,
            "url": result.url,
            "chunk_index": result.chunk_index,
            "section": result.section,
            "distance": round(result.distance, 4),
            "score": round(result.score, 4),
            "snippet": best_source_snippet(query_for_search, result),
        }
        if include_source_details:
            source_payload["why"] = source_reasons_for_result(query_for_search, result, evidence_lines)
            source_payload["evidence"] = evidence_lines[:2]
        sources.append(source_payload)

    if memory_manager and ENABLE_SESSION_MEMORY:
        if answer_mode == "clarification":
            pending_query = pending_clarification_query or original_message
            memory_manager.set_pending_clarification(session_id, pending_query)
        else:
            memory_manager.clear_pending_clarification(session_id)
        memory_manager.add_turn(
            session_id,
            ConversationTurn(
                query=original_message,
                answer=answer,
                labels=canonical_labels_from_query(query_for_search),
            ),
        )

    payload: Dict[str, Any] = {
        "answer": answer,
        "sources": sources,
        "answer_mode": answer_mode,
        "confidence": round(selection.top_score, 3),
    }
    if memory_notes:
        payload["memory_applied"] = memory_notes
    if used_clarification_stitch or query_for_search != original_message:
        payload["query_used"] = query_for_search
    if answer_mode == "clarification":
        payload["needs_clarification"] = True
    return payload


class ChatHandler(BaseHTTPRequestHandler):
    retriever: Retriever = None  # set at startup
    memory: SessionMemoryManager = SessionMemoryManager()

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
        raw_session_id = str(payload.get("session_id", "")).strip()
        session_id = raw_session_id[:80] if raw_session_id else None
        include_source_details = DEBUG_SOURCE_DETAILS_DEFAULT or bool(payload.get("debug", False))

        try:
            response = build_chat_payload(
                message=message,
                retriever=self.retriever,
                memory_manager=self.memory,
                session_id=session_id,
                include_source_details=include_source_details,
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(response)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    retriever = Retriever()
    ChatHandler.retriever = retriever
    ChatHandler.memory = SessionMemoryManager()

    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), ChatHandler)
    print(f"Webchat running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
