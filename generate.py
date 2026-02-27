"""Generate client-agent dialog dataset for CX-Ray support service.

Usage:
    python generate.py [--count 120] [--output data/chats.json] [--seed 42] [--concurrency 5]
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
from tqdm import tqdm

from config import (
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PATH,
    DEFAULT_CHAT_COUNT,
    DEFAULT_OUTPUT_PATH,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    MAX_RETRIES,
    OPENAI_API_KEY,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE,
    RETRY_BACKOFF_MAX,
    SCENARIO_MATRIX,
    SEED,
    VARIATION_CONTEXTS,
)
from prompts.generation import SYSTEM_PROMPT, build_generation_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def save_checkpoint(chats: list[dict[str, Any]], failed_count: int) -> None:
    """Save checkpoint for recovery on failure."""
    checkpoint_data: dict[str, Any] = {
        "chats": chats,
        "failed_count": failed_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Atomic write: write to temp file, then rename to avoid corruption on crash
    fd, tmp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)
        os.replace(tmp_path, CHECKPOINT_PATH)
    except BaseException:
        os.unlink(tmp_path)
        raise
    logger.info(f"Checkpoint saved: {len(chats)} chats, {failed_count} failures")


def load_checkpoint() -> tuple[list[dict[str, Any]], int] | None:
    """Load checkpoint to resume generation."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        logger.info(f"Checkpoint loaded: {len(data.get('chats', []))} chats")
        return data.get("chats", []), data.get("failed_count", 0)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def clear_checkpoint() -> None:
    """Remove checkpoint file after successful completion."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint removed")


def build_scenario_list(count: int = DEFAULT_CHAT_COUNT) -> list[dict[str, Any]]:
    """Build a scenario list of required length from the matrix.

    Takes scenarios from SCENARIO_MATRIX cyclically to ensure coverage
    of all category and case type combinations.
    """
    scenarios: list[dict[str, Any]] = []
    matrix = SCENARIO_MATRIX
    for i in range(count):
        scenarios.append(matrix[i % len(matrix)])
    return scenarios


def parse_chat_response(raw_response: str) -> list[dict[str, str]]:
    """Parse and validate API response with dialog.

    Args:
        raw_response: JSON string from API

    Returns:
        List of dialog messages

    Raises:
        ValueError: if response is invalid
        json.JSONDecodeError: if JSON is malformed
    """
    data: dict[str, Any] = json.loads(raw_response)

    if "messages" not in data:
        raise ValueError("API response does not contain 'messages' field")

    messages: list[dict[str, str]] = data["messages"]
    if not messages:
        raise ValueError("Messages list is empty")

    for msg in messages:
        if "role" not in msg or "text" not in msg:
            raise ValueError(f"Message missing required fields: {msg}")
        if msg["role"] not in ("client", "agent"):
            raise ValueError(f"Unknown role: {msg['role']}")

    return messages


def _build_generation_request(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI API request parameters for dialog generation."""
    variation_index = scenario.get("variation_index", 0)
    contexts = VARIATION_CONTEXTS.get(scenario["category"], [])
    variation_context = contexts[variation_index % len(contexts)] if contexts else None

    user_prompt = build_generation_prompt(
        category=scenario["category"],
        case_type=scenario["case_type"],
        has_hidden_dissatisfaction=scenario.get("has_hidden_dissatisfaction", False),
        agent_mistakes=scenario.get("intended_agent_mistakes", []),
        variation_index=variation_index,
        variation_context=variation_context,
        mixed_intent=scenario.get("mixed_intent"),
    )

    return {
        "model": GENERATION_MODEL,
        "temperature": GENERATION_TEMPERATURE,
        "seed": SEED,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def _process_generation_response(
    raw: str | None, scenario: dict[str, Any], chat_id: str,
) -> dict[str, Any]:
    """Process the API response into a chat dict."""
    if raw is None:
        raise ValueError("API returned empty response")
    messages = parse_chat_response(raw)
    return {
        "id": chat_id,
        "scenario": scenario,
        "messages": messages,
    }


def generate_single_chat(
    client: OpenAI,
    scenario: dict[str, Any],
    chat_id: str,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Generate a single dialog via OpenAI API (synchronous)."""
    request_params = _build_generation_request(scenario)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**request_params)
            return _process_generation_response(
                response.choices[0].message.content, scenario, chat_id,
            )
        except RateLimitError:
            wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
            logger.warning(f"Rate limit, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Parse error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.warning(f"API error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to generate chat {chat_id} after {max_retries} attempts")


async def async_generate_single_chat(
    client: AsyncOpenAI,
    scenario: dict[str, Any],
    chat_id: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Generate a single dialog via OpenAI API (asynchronous)."""
    request_params = _build_generation_request(scenario)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(**request_params)
            return _process_generation_response(
                response.choices[0].message.content, scenario, chat_id,
            )
        except RateLimitError:
            wait = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
            logger.warning(f"Rate limit for {chat_id}, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Parse error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.warning(f"API error for {chat_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to generate chat {chat_id} after {max_retries} attempts")


def save_dataset(
    chats: list[dict[str, Any]],
    output_path: str,
    model: str = GENERATION_MODEL,
    seed: int = SEED,
) -> None:
    """Save dataset to JSON file.

    Args:
        chats: list of dialogs
        output_path: output file path
        model: model name used
        seed: seed for reproducibility
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "total_chats": len(chats),
            "seed": seed,
        },
        "chats": chats,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset saved: {output_path} ({len(chats)} dialogs)")


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point for concurrent generation."""
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )
    scenarios = build_scenario_list(count=args.count)
    semaphore = asyncio.Semaphore(args.concurrency)

    logger.info(
        f"Starting async generation of {args.count} dialogs "
        f"(model: {GENERATION_MODEL}, seed: {args.seed}, concurrency: {args.concurrency})"
    )

    # Try to load checkpoint
    chats: list[dict[str, Any]]
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        chats, failed = checkpoint
        start_index = len(chats)
        logger.info(f"Resuming from checkpoint: {start_index} chats already generated")
    else:
        chats = []
        failed = 0
        start_index = 0

    remaining_scenarios = scenarios[start_index:]
    pbar = tqdm(total=len(remaining_scenarios), desc="Generating dialogs", initial=0)

    async def _generate_and_track(
        scenario: dict[str, Any], chat_id: str,
    ) -> dict[str, Any] | BaseException:
        try:
            result = await async_generate_single_chat(client, scenario, chat_id, semaphore)
            return result
        except BaseException as exc:
            return exc
        finally:
            pbar.update(1)

    # Launch all tasks at once — semaphore controls concurrency
    tasks: list[Any] = []
    for i, scenario in enumerate(remaining_scenarios):
        idx = start_index + i
        chat_id = f"chat_{idx + 1:03d}"
        tasks.append(asyncio.ensure_future(_generate_and_track(scenario, chat_id)))

    gather_results = await asyncio.gather(*tasks)

    for gather_result in gather_results:
        if isinstance(gather_result, BaseException):
            logger.error(f"Failed to generate chat: {gather_result}")
            failed += 1
        else:
            chats.append(gather_result)

    pbar.close()

    # Save checkpoint with final results
    save_checkpoint(chats, failed)

    # Successful completion - remove checkpoint
    clear_checkpoint()
    save_dataset(chats, args.output, model=GENERATION_MODEL, seed=args.seed)

    logger.info(f"Done! Successful: {len(chats)}, failures: {failed}")

    if failed > 0:
        logger.warning(f"{failed} dialogs were not generated due to errors")


def _sync_main(args: argparse.Namespace) -> None:
    """Synchronous entry point for sequential generation."""
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )
    scenarios = build_scenario_list(count=args.count)

    logger.info(f"Starting generation of {args.count} dialogs (model: {GENERATION_MODEL}, seed: {args.seed})")

    # Try to load checkpoint
    chats: list[dict[str, Any]]
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        chats, failed = checkpoint
        start_index = len(chats)
        logger.info(f"Resuming from checkpoint: {start_index} chats already generated")
    else:
        chats = []
        failed = 0
        start_index = 0

    for i, scenario in enumerate(tqdm(scenarios[start_index:], desc="Generating dialogs", initial=start_index)):
        chat_id = f"chat_{i + 1 + start_index:03d}"
        try:
            chat = generate_single_chat(client, scenario, chat_id)
            chats.append(chat)

            # Checkpoint every N chats
            if len(chats) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(chats, failed)

        except Exception as e:
            logger.error(f"Failed to generate {chat_id}: {e}")
            failed += 1
            # Save checkpoint on error
            save_checkpoint(chats, failed)

    # Successful completion - remove checkpoint
    clear_checkpoint()
    save_dataset(chats, args.output, model=GENERATION_MODEL, seed=args.seed)

    logger.info(f"Done! Successful: {len(chats)}, failures: {failed}")

    if failed > 0:
        logger.warning(f"{failed} dialogs were not generated due to errors")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CX-Ray support dialog dataset"
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_CHAT_COUNT,
        help=f"Number of dialogs (default: {DEFAULT_CHAT_COUNT})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Output file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Seed for determinism (default: {SEED})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent API requests (default: 1, use 5-10 for faster generation)",
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
