"""Analyze dialogs and evaluate CX-Ray support service quality.

Usage:
    python analyze.py [--input data/chats.json] [--output results/analysis.json] [--concurrency 5]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import ValidationError
from tqdm import tqdm

from config import (
    ANALYSIS_MODEL,
    ANALYSIS_TEMPERATURE,
    CHECKPOINT_ANALYSIS_PATH,
    CHECKPOINT_INTERVAL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RESULTS_PATH,
    MAX_RETRIES,
    OPENAI_API_KEY,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE,
    RETRY_BACKOFF_MAX,
    SEED,
)
from models import AnalysisResult
from prompts.analysis import SYSTEM_PROMPT, build_analysis_prompt
from validation import validate_analysis_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def save_analysis_checkpoint(results: list[dict[str, Any]], failed_count: int) -> None:
    """Save analysis checkpoint."""
    checkpoint_data: dict[str, Any] = {
        "results": results,
        "failed_count": failed_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_dir = os.path.dirname(CHECKPOINT_ANALYSIS_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Atomic write: write to temp file, then rename to avoid corruption on crash
    fd, tmp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)
        os.replace(tmp_path, CHECKPOINT_ANALYSIS_PATH)
    except BaseException:
        os.unlink(tmp_path)
        raise
    logger.info(f"Analysis checkpoint saved: {len(results)} results")


def load_analysis_checkpoint() -> tuple[list[dict[str, Any]], int] | None:
    """Load checkpoint to resume analysis."""
    if not os.path.exists(CHECKPOINT_ANALYSIS_PATH):
        return None
    try:
        with open(CHECKPOINT_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        logger.info(f"Analysis checkpoint loaded: {len(data.get('results', []))} results")
        return data.get("results", []), data.get("failed_count", 0)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def clear_analysis_checkpoint() -> None:
    """Remove checkpoint file after successful completion."""
    if os.path.exists(CHECKPOINT_ANALYSIS_PATH):
        os.remove(CHECKPOINT_ANALYSIS_PATH)
        logger.info("Analysis checkpoint removed")


def load_dataset(input_path: str) -> dict[str, Any]:
    """Load dataset from JSON file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if "chats" not in data:
        raise ValueError("File does not contain 'chats' field")

    return data


def _build_analysis_request(chat: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI API request parameters for dialog analysis."""
    user_prompt = build_analysis_prompt(chat["messages"])
    return {
        "model": ANALYSIS_MODEL,
        "temperature": ANALYSIS_TEMPERATURE,
        "seed": SEED,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def parse_analysis_response(raw_response: str, chat_id: str) -> dict[str, Any]:
    """Parse, validate, and apply rule-based corrections to API response.

    Args:
        raw_response: JSON string from API
        chat_id: chat ID for binding

    Returns:
        Dictionary with analysis results (post-validated)
    """
    data: dict[str, Any] = json.loads(raw_response)

    # Check required fields
    required = ["intent", "satisfaction", "quality_score", "agent_mistakes", "summary"]
    for field in required:
        if field not in data:
            raise ValueError(f"Response missing required field: '{field}'")

    result_data: dict[str, Any] = {
        "chat_id": chat_id,
        "intent": data["intent"],
        "satisfaction": data["satisfaction"],
        "quality_score": data["quality_score"],
        "agent_mistakes": data["agent_mistakes"],
        "summary": data["summary"],
        "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.8)))),
    }

    # Rule-based post-processing validation
    corrected_data, warnings = validate_analysis_result(result_data)
    if warnings:
        logger.info(f"Validation corrections for {chat_id}: {warnings}")
    corrected_data["validation_warnings"] = [str(w) for w in warnings]

    # Pydantic validation after corrections
    AnalysisResult(**corrected_data)

    return corrected_data


def _embed_ground_truth(result: dict[str, Any], chat: dict[str, Any]) -> dict[str, Any]:
    """Embed scenario ground truth into analysis result for evaluation."""
    scenario = chat.get("scenario")
    if scenario is None:
        return result
    result["ground_truth"] = {
        "expected_intent": scenario.get("category"),
        "has_hidden_dissatisfaction": scenario.get("has_hidden_dissatisfaction", False),
        "intended_agent_mistakes": scenario.get("intended_agent_mistakes", []),
        "case_type": scenario.get("case_type"),
        "mixed_intent": scenario.get("mixed_intent"),
    }
    return result


def analyze_single_chat(
    client: OpenAI,
    chat: dict[str, Any],
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Analyze a single dialog via OpenAI API (synchronous)."""
    chat_id: str = chat["id"]
    request_params = _build_analysis_request(chat)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**request_params)
            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("API returned empty response")
            result = parse_analysis_response(raw, chat_id)
            return _embed_ground_truth(result, chat)
        except RateLimitError:
            wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
            logger.warning(f"Rate limit, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logger.warning(f"Parse error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.warning(f"API error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to analyze chat {chat_id} after {max_retries} attempts")


async def async_analyze_single_chat(
    client: AsyncOpenAI,
    chat: dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Analyze a single dialog via OpenAI API (asynchronous)."""
    chat_id: str = chat["id"]
    request_params = _build_analysis_request(chat)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(**request_params)
            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("API returned empty response")
            result = parse_analysis_response(raw, chat_id)
            return _embed_ground_truth(result, chat)
        except RateLimitError:
            wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
            logger.warning(f"Rate limit for {chat_id}, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logger.warning(f"Parse error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.warning(f"API error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to analyze chat {chat_id} after {max_retries} attempts")


def save_results(
    results: list[dict[str, Any]],
    output_path: str,
    model: str = ANALYSIS_MODEL,
) -> None:
    """Save analysis results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output: dict[str, Any] = {
        "metadata": {
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "total_analyzed": len(results),
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved: {output_path} ({len(results)} analyses)")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics of analysis results."""

    total = len(results)
    if total == 0:
        logger.info("No results to analyze")
        return

    # Satisfaction statistics
    satisfaction_counts: dict[str, int] = {}
    for r in results:
        s: str = r["satisfaction"]
        satisfaction_counts[s] = satisfaction_counts.get(s, 0) + 1

    # Intent statistics
    intent_counts: dict[str, int] = {}
    for r in results:
        intent: str = r["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # Average quality
    avg_quality: float = sum(r["quality_score"] for r in results) / total

    # Agent mistakes
    all_mistakes: list[str] = []
    for r in results:
        all_mistakes.extend(r["agent_mistakes"])
    mistake_counts: dict[str, int] = {}
    for m in all_mistakes:
        mistake_counts[m] = mistake_counts.get(m, 0) + 1

    chats_with_mistakes: int = sum(1 for r in results if r["agent_mistakes"])

    # Validation corrections
    chats_with_corrections: int = sum(1 for r in results if r.get("validation_warnings"))

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal analyzed: {total} dialogs")
    print(f"Average quality score: {avg_quality:.2f} / 5.00")

    print("\nClient satisfaction:")
    for level in ["satisfied", "neutral", "unsatisfied"]:
        count = satisfaction_counts.get(level, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {level:12s}: {count:3d} ({pct:5.1f}%) {bar}")

    print("\nRequest categories:")
    for cat, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {cat:20s}: {count:3d} ({pct:5.1f}%)")

    print("\nAgent mistakes:")
    print(f"  Dialogs with mistakes: {chats_with_mistakes} / {total} ({chats_with_mistakes / total * 100:.1f}%)")
    if mistake_counts:
        for mistake, count in sorted(mistake_counts.items(), key=lambda x: -x[1]):
            print(f"  {mistake:25s}: {count}")

    print(f"\nValidation corrections applied: {chats_with_corrections} / {total}")

    print("=" * 60 + "\n")


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point for concurrent analysis."""
    dataset = load_dataset(args.input)
    chats: list[dict[str, Any]] = dataset["chats"]

    logger.info(f"Loaded {len(chats)} dialogs from {args.input}")
    logger.info(
        f"Starting async analysis "
        f"(model: {ANALYSIS_MODEL}, seed: {SEED}, concurrency: {args.concurrency})"
    )

    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    # Try to load checkpoint
    results: list[dict[str, Any]]
    checkpoint = load_analysis_checkpoint()
    if checkpoint is not None:
        results, failed = checkpoint
        analyzed_ids = {r["chat_id"] for r in results}
        chats_to_analyze = [c for c in chats if c["id"] not in analyzed_ids]
        logger.info(f"Resuming from checkpoint: {len(results)} results already available")
    else:
        results = []
        failed = 0
        chats_to_analyze = chats

    pbar = tqdm(total=len(chats_to_analyze), desc="Analyzing dialogs")

    async def _analyze_and_track(chat: dict[str, Any]) -> dict[str, Any] | BaseException:
        try:
            result = await async_analyze_single_chat(client, chat, semaphore)
            return result
        except BaseException as exc:
            return exc
        finally:
            pbar.update(1)

    # Launch all tasks at once — semaphore controls concurrency
    tasks = [
        asyncio.ensure_future(_analyze_and_track(chat))
        for chat in chats_to_analyze
    ]

    gather_results = await asyncio.gather(*tasks)

    for gather_result in gather_results:
        if isinstance(gather_result, BaseException):
            logger.error(f"Failed to analyze chat: {gather_result}")
            failed += 1
        else:
            results.append(gather_result)

    pbar.close()

    # Save checkpoint with final results
    save_analysis_checkpoint(results, failed)

    # Successful completion - remove checkpoint
    clear_analysis_checkpoint()
    save_results(results, args.output, model=ANALYSIS_MODEL)
    print_summary(results)

    logger.info(f"Done! Successful: {len(results)}, failures: {failed}")

    if failed > 0:
        logger.warning(f"{failed} dialogs were not analyzed due to errors")


def _sync_main(args: argparse.Namespace) -> None:
    """Synchronous entry point for sequential analysis."""
    dataset = load_dataset(args.input)
    chats: list[dict[str, Any]] = dataset["chats"]

    logger.info(f"Loaded {len(chats)} dialogs from {args.input}")
    logger.info(f"Starting analysis (model: {ANALYSIS_MODEL}, seed: {SEED})")

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )

    # Try to load checkpoint
    results: list[dict[str, Any]]
    checkpoint = load_analysis_checkpoint()
    if checkpoint is not None:
        results, failed = checkpoint
        analyzed_ids = {r["chat_id"] for r in results}
        chats_to_analyze = [c for c in chats if c["id"] not in analyzed_ids]
        logger.info(f"Resuming from checkpoint: {len(results)} results already available")
    else:
        results = []
        failed = 0
        chats_to_analyze = chats

    for chat in tqdm(chats_to_analyze, desc="Analyzing dialogs"):
        try:
            result = analyze_single_chat(client, chat)
            results.append(result)

            # Checkpoint every N results
            if len(results) % CHECKPOINT_INTERVAL == 0:
                save_analysis_checkpoint(results, failed)

        except Exception as e:
            logger.error(f"Failed to analyze {chat['id']}: {e}")
            failed += 1
            # Save checkpoint on error
            save_analysis_checkpoint(results, failed)

    # Successful completion - remove checkpoint
    clear_analysis_checkpoint()
    save_results(results, args.output, model=ANALYSIS_MODEL)
    print_summary(results)

    logger.info(f"Done! Successful: {len(results)}, failures: {failed}")

    if failed > 0:
        logger.warning(f"{failed} dialogs were not analyzed due to errors")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CX-Ray support dialogs"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Path to dialogs file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_RESULTS_PATH,
        help=f"Path to results file (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent API requests (default: 1, use 5-10 for faster analysis)",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Create a .env file with the key.")
        sys.exit(1)

    if args.concurrency > 1:
        asyncio.run(_async_main(args))
    else:
        _sync_main(args)


if __name__ == "__main__":
    main()
