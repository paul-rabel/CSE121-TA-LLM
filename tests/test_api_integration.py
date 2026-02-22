import json
import threading
import unittest
from http.client import HTTPConnection

from tests._import_answer_service import import_answer_service


answer_service = import_answer_service()


class _FakeRetriever:
    def search(self, _query, _top_k):
        return [
            answer_service.SearchResult(
                doc_id=0,
                text=(
                    "C3 - Linked Lists i.s. due by 11:59 pm PT.\n"
                    "C3 - Linked Lists"
                ),
                url="https://courses.cs.washington.edu/courses/cse121/26wi/assignments/",
                title="Assignments",
                chunk_index=0,
                distance=0.1,
                score=12.0,
            )
        ]


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_llm_flag = answer_service.ENABLE_LLM_RESPONSE
        answer_service.ENABLE_LLM_RESPONSE = False
        answer_service.ChatHandler.retriever = _FakeRetriever()

        try:
            cls.server = answer_service.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                answer_service.ChatHandler,
            )
        except PermissionError as exc:
            # Some sandboxed environments block localhost binds. In those
            # cases skip integration tests and keep unit/benchmark coverage.
            raise unittest.SkipTest(f"Socket bind blocked in sandbox: {exc}") from exc
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        answer_service.ENABLE_LLM_RESPONSE = cls._previous_llm_flag

    def _request(self, method, path, payload=None):
        conn = HTTPConnection(self.host, self.port, timeout=5)
        headers = {}
        body = None

        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload)

        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw_body = response.read().decode("utf-8")
        conn.close()

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            parsed = raw_body
        return response.status, parsed

    def test_health_endpoint(self):
        status, body = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_chat_requires_message(self):
        status, body = self._request("POST", "/api/chat", payload={"message": "   "})
        self.assertEqual(status, 400)
        self.assertEqual(body.get("error"), "message is required")

    def test_chat_returns_deterministic_answer_with_sources(self):
        status, body = self._request(
            "POST",
            "/api/chat",
            payload={"message": "When is C3 due?", "debug": True},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body.get("answer_mode"), "deterministic")
        self.assertIn("C3 is due by 11:59 pm PT.", body.get("answer", ""))

        sources = body.get("sources", [])
        self.assertGreaterEqual(len(sources), 1)
        self.assertEqual(sources[0].get("title"), "Assignments")
        self.assertIsInstance(sources[0].get("why"), list)
        self.assertTrue(sources[0].get("why"))
        self.assertIsInstance(sources[0].get("evidence"), list)

    def test_chat_hides_source_reasoning_without_debug(self):
        status, body = self._request("POST", "/api/chat", payload={"message": "When is C3 due?"})
        self.assertEqual(status, 200)
        sources = body.get("sources", [])
        self.assertGreaterEqual(len(sources), 1)
        self.assertNotIn("why", sources[0])
        self.assertNotIn("evidence", sources[0])

    def test_chat_followup_uses_session_memory(self):
        session_id = "memory-test"
        first_status, first_body = self._request(
            "POST",
            "/api/chat",
            payload={"message": "When is C3 due?", "session_id": session_id},
        )
        self.assertEqual(first_status, 200)
        self.assertEqual(first_body.get("answer_mode"), "deterministic")

        second_status, second_body = self._request(
            "POST",
            "/api/chat",
            payload={"message": "What about it?", "session_id": session_id},
        )
        self.assertEqual(second_status, 200)
        self.assertEqual(second_body.get("answer_mode"), "deterministic")
        self.assertIn("C3 is due by 11:59 pm PT.", second_body.get("answer", ""))
        self.assertIn("memory_applied", second_body)


if __name__ == "__main__":
    unittest.main()
