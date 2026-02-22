import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from firecrawl import FirecrawlApp


DEFAULT_BASE_URL = "https://courses.cs.washington.edu/courses/cse121/26wi/"
DEFAULT_PAGES = [
    "",
    "syllabus/",
    "assignments/",
    "resubs/",
    "exams/",
    "getting_help/",
    "staff/",
    "rubrics/",
    "resources/",
    "course_tools/",
]
DEFAULT_OUTPUT_DIR = Path("data/raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl CSE 121 course pages via Firecrawl.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL to crawl (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        default=DEFAULT_PAGES,
        help="Page suffixes to crawl relative to --base-url.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for raw crawl output JSON (default: data/raw).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per page after the first failed attempt (default: 3).",
    )
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=1.5,
        help="Initial retry backoff in seconds (default: 1.5).",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=45,
        help="Scrape timeout hint in seconds when supported by Firecrawl SDK (default: 45).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with a non-zero code if any page fails after retries.",
    )
    return parser.parse_args()


def _scrape_kwargs(timeout_secs: int) -> dict:
    kwargs = {"formats": {"markdown": True, "metadata": True}}
    try:
        params = inspect.signature(FirecrawlApp.scrape).parameters
    except (TypeError, ValueError):
        return kwargs

    for timeout_name in ("timeout", "timeout_seconds", "request_timeout", "timeout_secs"):
        if timeout_name in params:
            kwargs[timeout_name] = timeout_secs
            break
    return kwargs


def scrape_with_retries(
    app: FirecrawlApp,
    url: str,
    retries: int,
    initial_backoff: float,
    timeout_secs: int,
):
    attempts = retries + 1
    backoff = max(0.1, initial_backoff)
    scrape_kwargs = _scrape_kwargs(timeout_secs)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return app.scrape(url, **scrape_kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            wait_seconds = backoff * (2 ** (attempt - 1))
            print(
                f"Scrape attempt {attempt}/{attempts} failed for {url}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to scrape {url} after {attempts} attempts") from last_error


def model_to_dict(result) -> dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"raw": str(result)}


def main() -> int:
    args = parse_args()

    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.initial_backoff <= 0:
        raise ValueError("--initial-backoff must be positive")
    if args.timeout_secs <= 0:
        raise ValueError("--timeout-secs must be positive")

    if load_dotenv:
        load_dotenv()

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("Missing FIRECRAWL_API_KEY in environment or .env")

    pages = args.pages if args.pages else [""]
    base_url = args.base_url

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cse121_dump_{timestamp}.json"

    firecrawl = FirecrawlApp(api_key=api_key)

    all_results = []
    failures = []

    for page in pages:
        url = urljoin(base_url, page)
        print(f"Scraping {url}")
        try:
            result = scrape_with_retries(
                firecrawl,
                url,
                retries=args.retries,
                initial_backoff=args.initial_backoff,
                timeout_secs=args.timeout_secs,
            )
            all_results.append(
                {
                    "url": url,
                    "content": model_to_dict(result),
                }
            )
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"Skipping {url}: {exc}", file=sys.stderr)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved {len(all_results)} pages to {output_file}")

    if failures:
        failure_file = output_file.with_name(output_file.stem + "_failures.json")
        with failure_file.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"{len(failures)} pages failed. Details: {failure_file}", file=sys.stderr)

    if not all_results:
        return 1
    if failures and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
