import unittest
from types import SimpleNamespace

from tests._import_answer_service import import_answer_service


answer_service = import_answer_service()


class AnswerLogicTests(unittest.TestCase):
    def test_extract_query_labels_supports_natural_phrasing(self):
        labels = answer_service.extract_query_labels(
            "When is project 3 due, and what about resubmission 4 after checkpoint 2?"
        )
        self.assertIn("p3", labels)
        self.assertIn("p03", labels)
        self.assertIn("r4", labels)
        self.assertIn("r04", labels)
        self.assertIn("c2", labels)
        self.assertIn("c02", labels)

    def test_query_label_targets_canonicalizes_variants(self):
        targets = answer_service.query_label_targets(
            ["c03", "checkpoint 3", "resubmission 4", "quiz 2"]
        )
        self.assertEqual(targets["c"], {"c3"})
        self.assertEqual(targets["r"], {"r4"})
        self.assertEqual(targets["quiz"], {"quiz2"})

    def test_line_label_alignment_reports_exact_and_conflict(self):
        exact, conflict = answer_service.line_label_alignment(
            "C3 due by 11:59 pm PT, C2 released earlier.",
            {"c": {"c3"}},
        )
        self.assertTrue(exact)
        self.assertTrue(conflict)

    def test_score_line_prefers_exact_label_match(self):
        query = "When is C3 due?"
        intents = answer_service.infer_query_intents(query)
        label_hints = answer_service.extract_query_labels(query)
        label_targets = answer_service.query_label_targets(label_hints)
        strict_prefixes = answer_service.strict_single_label_prefixes(label_targets)
        query_tokens = set(
            answer_service.expanded_query_terms(query, intents=intents, label_hints=label_hints)
        )

        exact_line = "C3 - Linked Lists i.s. due by 11:59 pm PT."
        wrong_line = "C2 - Arrays i.s. due by 11:59 pm PT."

        exact_score = answer_service.score_line(
            exact_line,
            query_tokens,
            intents,
            rank_bias=0.8,
            label_targets=label_targets,
            strict_prefixes=strict_prefixes,
        )
        wrong_score = answer_service.score_line(
            wrong_line,
            query_tokens,
            intents,
            rank_bias=0.8,
            label_targets=label_targets,
            strict_prefixes=strict_prefixes,
        )

        self.assertGreater(exact_score, wrong_score)
        self.assertGreater(exact_score, answer_service.MIN_SENTENCE_SCORE)

    def test_memory_rewrite_does_not_override_explicit_name_query(self):
        history = [
            answer_service.ConversationTurn(
                query="When is R4 due?",
                answer="R4 is due by 11:59 pm PT.",
                labels=["r4"],
            )
        ]
        rewritten, notes, labels = answer_service.rewrite_query_with_memory("What is C2?", history)
        self.assertEqual(rewritten, "What is C2?")
        self.assertFalse(notes)
        self.assertEqual(labels, ["c2"])

    def test_deterministic_fact_answer_instructor(self):
        metadata = [
            {
                "text": (
                    "**Instructor:** Miya Natsuhara\n"
                    "**Instructor Email:** [mnats@cs.washington.edu](mailto:mnats@cs.washington.edu)"
                )
            }
        ]
        answer = answer_service.build_deterministic_fact_answer(
            "Who is the instructor?",
            results=[],
            metadata=metadata,
        )
        self.assertIsNotNone(answer)
        self.assertIn("Miya Natsuhara", answer)

    def test_deterministic_fact_answer_lecture_schedule(self):
        metadata = [
            {"text": "| Wed 01/21 | LES 04for Loops |"},
            {"text": "| Fri 01/23 | LES 05Methods |"},
            {"text": "_A lecture @ 11:30 in CSE2 G20; B lecture @ 2:30 in GUG 220_"},
        ]
        answer = answer_service.build_deterministic_fact_answer(
            "When is lecture?",
            results=[],
            metadata=metadata,
        )
        self.assertIsNotNone(answer)
        self.assertIn("Wednesday", answer)
        self.assertIn("Friday", answer)
        self.assertIn("11:30", answer)

    def test_deterministic_fact_answer_quiz_number(self):
        metadata = [
            {"text": "| Thu 03/05 | QUIZ 02Quiz 2: Conditionals, while Loops |"},
        ]
        answer = answer_service.build_deterministic_fact_answer(
            "When is the 2nd quiz?",
            results=[],
            metadata=metadata,
        )
        self.assertEqual(answer, "Quiz 2 is on Thu 03/05.")

    def test_deterministic_fact_answer_esn_definition(self):
        metadata = [
            {
                "text": (
                    "## E/S/N Grading Definitions\n"
                    "**E(xcellent)** ... **S(atisfactory)** ... **N(ot yet)** ..."
                )
            }
        ]
        answer = answer_service.build_deterministic_fact_answer(
            "What is ESN?",
            results=[],
            metadata=metadata,
        )
        self.assertEqual(answer, "ESN stands for Excellent, Satisfactory, and Not yet.")

    def test_deterministic_fact_answer_named_item_from_metadata(self):
        metadata = [
            {"text": "|  | Released<br>C3<br>Dance Dance Arrayvolution<br>I.S. by 11:59pm PT |"}
        ]
        answer = answer_service.build_deterministic_fact_answer(
            "What is Dance Dance Arrayvolution?",
            results=[],
            metadata=metadata,
        )
        self.assertIsNotNone(answer)
        self.assertIn("Dance Dance Arrayvolution is C3.", answer)

    def test_llm_citation_validation(self):
        self.assertTrue(
            answer_service.llm_citations_are_valid(
                "The instructor is Miya Natsuhara [source 1].",
                max_source_num=3,
            )
        )
        self.assertFalse(
            answer_service.llm_citations_are_valid(
                "The discussion board exists [source 4].",
                max_source_num=3,
            )
        )
        self.assertFalse(
            answer_service.llm_citations_are_valid(
                "The answer is in the syllabus.",
                max_source_num=3,
            )
        )

    def test_ollama_model_from_tags_payload(self):
        payload = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "mistral:7b"},
            ]
        }
        self.assertEqual(answer_service.ollama_model_from_tags_payload(payload), "llama3.2:3b")
        self.assertIsNone(answer_service.ollama_model_from_tags_payload({"models": []}))
        self.assertIsNone(answer_service.ollama_model_from_tags_payload({}))

    def test_llm_grounding_and_preferred_conflict_guards(self):
        context = (
            "[source 1] CSE 121 - https://courses.cs.washington.edu/courses/cse121/26wi/\n"
            "- Quiz 2 is on Thu 03/05.\n"
            "- C2 - Password Protector\n"
        )
        grounded = "Quiz 2 is on Thu 03/05. [source 1]"
        ungrounded = "Quiz 2 is on Thu 03/06. [source 1]"
        self.assertTrue(answer_service.llm_answer_is_grounded(grounded, context))
        self.assertFalse(answer_service.llm_answer_is_grounded(ungrounded, context))

        preferred = "Quiz 2 is on Thu 03/05."
        conflicting = "Quiz 2 is on Thu 03/06. [source 1]"
        compatible = "It is on Thu 03/05. [source 1]"
        self.assertTrue(answer_service.llm_answer_conflicts_with_preferred(conflicting, preferred))
        self.assertFalse(answer_service.llm_answer_conflicts_with_preferred(compatible, preferred))

    def test_best_source_snippet_uses_relevant_line(self):
        result = answer_service.SearchResult(
            doc_id=0,
            text=(
                "General announcements.\n"
                "Quiz 2 is on Thu 03/05 in your section.\n"
                "More details later."
            ),
            url="https://courses.cs.washington.edu/courses/cse121/26wi/",
            title="CSE 121",
            chunk_index=0,
            distance=0.2,
            score=7.5,
            section="Calendar",
        )
        snippet = answer_service.best_source_snippet("When is quiz 2?", result)
        self.assertIn("Quiz 2 is on Thu 03/05", snippet)

    def test_rank_sources_for_answer_prefers_anchor_matches(self):
        generic = answer_service.SearchResult(
            doc_id=1,
            text="General course information.",
            url="https://courses.cs.washington.edu/courses/cse121/26wi/",
            title="CSE 121",
            chunk_index=0,
            distance=0.3,
            score=5.0,
            section="Overview",
        )
        specific = answer_service.SearchResult(
            doc_id=2,
            text="| Thu 03/05 | QUIZ 02Quiz 2: Conditionals |",
            url="https://courses.cs.washington.edu/courses/cse121/26wi/",
            title="CSE 121",
            chunk_index=1,
            distance=0.2,
            score=5.1,
            section="Calendar",
        )
        ranked = answer_service.rank_sources_for_answer(
            "When is the 2nd quiz?",
            "Quiz 2 is on Thu 03/05.",
            [generic, specific],
        )
        self.assertEqual(ranked[0].doc_id, specific.doc_id)

    def test_preferred_ollama_model_prefers_env_value(self):
        original_model = answer_service.OLLAMA_MODEL
        answer_service.OLLAMA_MODEL = "custom-model:latest"
        try:
            self.assertEqual(answer_service.preferred_ollama_model(), "custom-model:latest")
        finally:
            answer_service.OLLAMA_MODEL = original_model

    def test_clarification_detection_and_stitch_heuristic(self):
        self.assertTrue(
            answer_service.is_clarification_request("Could you clarify whether you mean Quiz 1 or Quiz 2?")
        )
        self.assertFalse(answer_service.is_clarification_request("Quiz 2 is on Thu 03/05. [source 1]"))
        self.assertTrue(answer_service.should_stitch_pending_query("Quiz 2"))
        self.assertFalse(answer_service.should_stitch_pending_query("Never mind, new question: who are the TAs?"))

    def test_session_memory_pending_clarification(self):
        memory = answer_service.SessionMemoryManager(max_sessions=8, turn_limit=4)
        session_id = "clarify-test"
        self.assertIsNone(memory.get_pending_clarification(session_id))

        memory.set_pending_clarification(session_id, "When is quiz?")
        self.assertEqual(memory.get_pending_clarification(session_id), "When is quiz?")

        memory.clear_pending_clarification(session_id)
        self.assertIsNone(memory.get_pending_clarification(session_id))

    def test_clarification_flow_stitches_follow_up(self):
        class FakeRetriever:
            def __init__(self):
                self.metadata = []

            def search(self, _query, _top_k):
                return [
                    answer_service.SearchResult(
                        doc_id=0,
                        text="Course logistics and overview information.",
                        url="https://courses.cs.washington.edu/courses/cse121/26wi/",
                        title="CSE 121",
                        chunk_index=0,
                        distance=0.1,
                        score=10.0,
                        section="Calendar",
                    )
                ]

        memory = answer_service.SessionMemoryManager()
        retriever = FakeRetriever()
        session_id = "clarification-flow"

        original_llm = answer_service.build_llm_answer
        calls = SimpleNamespace(count=0)

        def fake_llm(query, _selection, _results, retriever=None, preferred_answer=None):
            _ = (retriever, preferred_answer)
            calls.count += 1
            if calls.count == 1:
                return "Could you clarify whether you mean Quiz 1 or Quiz 2?"
            self.assertIn("When is quiz?", query)
            self.assertIn("the second one", query)
            return "Quiz 2 is on Thu 03/05. [source 1]"

        answer_service.build_llm_answer = fake_llm
        try:
            first = answer_service.build_chat_payload(
                message="When is quiz?",
                retriever=retriever,
                memory_manager=memory,
                session_id=session_id,
            )
            self.assertEqual(first.get("answer_mode"), "clarification")
            self.assertTrue(first.get("needs_clarification"))

            second = answer_service.build_chat_payload(
                message="the second one",
                retriever=retriever,
                memory_manager=memory,
                session_id=session_id,
            )
            self.assertEqual(second.get("answer_mode"), "llm")
            self.assertIn("Quiz 2 is on Thu 03/05.", second.get("answer", ""))
            self.assertIn("query_used", second)
        finally:
            answer_service.build_llm_answer = original_llm


if __name__ == "__main__":
    unittest.main()
