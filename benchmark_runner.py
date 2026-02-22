import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_answer_service_module():
    try:
        from tests._import_answer_service import import_answer_service

        return import_answer_service()
    except Exception:
        import answer_service as answer_service_module

        return answer_service_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an accuracy benchmark against the local QA pipeline.")
    parser.add_argument(
        "--cases",
        default="benchmarks/cse121_accuracy_cases.json",
        help="Path to benchmark case JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every benchmark turn, including passing turns.",
    )
    return parser.parse_args()


def evaluate_turn(payload: Dict[str, Any], turn: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    expected_mode = turn.get("expect_mode")
    if expected_mode and payload.get("answer_mode") != expected_mode:
        failures.append(f"mode expected {expected_mode!r}, got {payload.get('answer_mode')!r}")

    answer_lower = str(payload.get("answer", "")).lower()
    for needle in turn.get("must_contain", []):
        if needle.lower() not in answer_lower:
            failures.append(f"answer missing substring {needle!r}")

    if turn.get("expect_memory") and not payload.get("memory_applied"):
        failures.append("expected memory_applied but it was missing")

    sources = payload.get("sources", [])
    if payload.get("answer_mode") != "no_answer" and not sources:
        failures.append("expected at least one source")
    if sources:
        first_source = sources[0]
        why = first_source.get("why")
        if not isinstance(why, list) or not why:
            failures.append("first source missing non-empty 'why' list")

    return (len(failures) == 0, failures)


def main() -> int:
    args = parse_args()
    case_path = Path(args.cases)
    if not case_path.exists():
        raise FileNotFoundError(f"Benchmark case file not found: {case_path}")

    answer_service = load_answer_service_module()

    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("Benchmark case file must contain a JSON list")

    retriever = answer_service.Retriever()
    memory = answer_service.SessionMemoryManager()

    total_turns = 0
    passed_turns = 0
    failures_report: List[str] = []

    for case in cases:
        case_id = str(case.get("id", f"case-{total_turns}"))
        turns = case.get("turns", [])
        if not isinstance(turns, list) or not turns:
            failures_report.append(f"[{case_id}] invalid or empty turns list")
            continue

        session_id = f"bench-{case_id}"
        for turn_index, turn in enumerate(turns, start=1):
            total_turns += 1
            message = str(turn.get("message", "")).strip()
            if not message:
                failures_report.append(f"[{case_id} turn {turn_index}] missing message")
                continue

            payload = answer_service.build_chat_payload(
                message=message,
                retriever=retriever,
                memory_manager=memory,
                session_id=session_id,
                include_source_details=True,
            )

            passed, turn_failures = evaluate_turn(payload, turn)
            if passed:
                passed_turns += 1
                if args.verbose:
                    print(f"PASS [{case_id} turn {turn_index}] {message}")
                continue

            summary = [f"FAIL [{case_id} turn {turn_index}] {message}"]
            summary.extend(f"  - {failure}" for failure in turn_failures)
            summary.append(f"  answer_mode={payload.get('answer_mode')!r}")
            summary.append(f"  answer={payload.get('answer')!r}")
            if payload.get("memory_applied"):
                summary.append(f"  memory_applied={payload.get('memory_applied')!r}")
            failures_report.append("\n".join(summary))

    print(f"Benchmark turns passed: {passed_turns}/{total_turns}")
    accuracy = (passed_turns / total_turns * 100.0) if total_turns else 0.0
    print(f"Turn accuracy: {accuracy:.1f}%")

    if failures_report:
        print("\nFailures:")
        for failure in failures_report:
            print(failure)
        return 1

    print("All benchmark checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
