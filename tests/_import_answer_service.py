import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_faiss_module() -> None:
    try:
        import faiss  # noqa: F401
        return
    except Exception:
        pass

    faiss_stub = types.ModuleType("faiss")

    class _FakeFaissIndex:
        def search(self, _vectors, _top_k):
            return [[0.0]], [[-1]]

    def read_index(_path):
        return _FakeFaissIndex()

    faiss_stub.read_index = read_index
    sys.modules["faiss"] = faiss_stub


def _ensure_sentence_transformers_module() -> None:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return
    except Exception:
        pass

    sentence_transformers_stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            return [[0.0]]

    sentence_transformers_stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub


def import_answer_service():
    _ensure_faiss_module()
    _ensure_sentence_transformers_module()
    return importlib.import_module("answer_service")
