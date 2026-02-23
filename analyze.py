"""Аналіз діалогів та оцінка якості роботи служби підтримки CloudTask.

Використання:
    python analyze.py [--input data/chats.json] [--output results/analysis.json]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from openai import APIError, OpenAI, RateLimitError
from pydantic import ValidationError
from tqdm import tqdm

from config import (
    ANALYSIS_MODEL,
    CHECKPOINT_ANALYSIS_PATH,
    CHECKPOINT_INTERVAL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RESULTS_PATH,
    OPENAI_API_KEY,
    REQUEST_TIMEOUT,
    SEED,
    TEMPERATURE,
)
from models import AnalysisResult
from prompts.analysis import SYSTEM_PROMPT, build_analysis_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def save_analysis_checkpoint(results: list[dict], failed_count: int) -> None:
    """Збереження checkpoint для аналізу."""
    checkpoint_data = {
        "results": results,
        "failed_count": failed_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(CHECKPOINT_ANALYSIS_PATH), exist_ok=True)
    with open(CHECKPOINT_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    logger.info(f"Checkpoint аналізу збережено: {len(results)} результатів")


def load_analysis_checkpoint() -> tuple[list[dict], int] | None:
    """Завантаження checkpoint для продовження аналізу."""
    if not os.path.exists(CHECKPOINT_ANALYSIS_PATH):
        return None
    try:
        with open(CHECKPOINT_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Завантажено checkpoint аналізу: {len(data.get('results', []))} результатів")
        return data.get("results", []), data.get("failed_count", 0)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Не вдалося завантажити checkpoint: {e}")
        return None


def clear_analysis_checkpoint() -> None:
    """Видалення checkpoint файлу після успішного завершення."""
    if os.path.exists(CHECKPOINT_ANALYSIS_PATH):
        os.remove(CHECKPOINT_ANALYSIS_PATH)
        logger.info("Checkpoint аналізу видалено")


def load_dataset(input_path: str) -> dict[str, Any]:
    """Завантаження датасету з JSON-файлу.

    Args:
        input_path: шлях до файлу з діалогами

    Returns:
        Словник з даними датасету

    Raises:
        FileNotFoundError: якщо файл не існує
        json.JSONDecodeError: якщо JSON некоректний
        ValueError: якщо структура невалідна
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не знайдено: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if "chats" not in data:
        raise ValueError("Файл не містить поле 'chats'")

    return data


def parse_analysis_response(raw_response: str, chat_id: str) -> dict:
    """Парсинг та валідація відповіді API з аналізом.

    Args:
        raw_response: JSON-рядок від API
        chat_id: ID чату для прив'язки

    Returns:
        Словник з результатами аналізу

    Raises:
        ValueError: якщо відповідь невалідна
        json.JSONDecodeError: якщо JSON некоректний
    """
    data = json.loads(raw_response)

    # Перевіряємо обов'язкові поля
    required = ["intent", "satisfaction", "quality_score", "agent_mistakes", "summary"]
    for field in required:
        if field not in data:
            raise ValueError(f"Відповідь не містить обов'язкове поле: '{field}'")

    result_data = {
        "chat_id": chat_id,
        "intent": data["intent"],
        "satisfaction": data["satisfaction"],
        "quality_score": data["quality_score"],
        "agent_mistakes": data["agent_mistakes"],
        "summary": data["summary"],
    }

    # Валідація через Pydantic
    AnalysisResult(**result_data)

    return result_data


def analyze_single_chat(
    client: OpenAI,
    chat: dict,
    max_retries: int = 3,
) -> dict:
    """Аналіз одного діалогу через OpenAI API.

    Args:
        client: екземпляр OpenAI клієнта
        chat: дані діалогу
        max_retries: максимум спроб при помилках

    Returns:
        Словник з результатами аналізу
    """
    chat_id = chat["id"]
    user_prompt = build_analysis_prompt(chat["messages"])

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=ANALYSIS_MODEL,
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("API returned empty response")
            result = parse_analysis_response(raw, chat_id)
            return result

        except RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit, очікування {wait}с (спроба {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logger.warning(f"Помилка парсингу для {chat_id} (спроба {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.warning(f"API помилка для {chat_id} (спроба {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Не вдалося проаналізувати чат {chat_id} після {max_retries} спроб")


def save_results(
    results: list[dict],
    output_path: str,
    model: str = ANALYSIS_MODEL,
) -> None:
    """Збереження результатів аналізу у JSON-файл.

    Args:
        results: список результатів аналізу
        output_path: шлях до вихідного файлу
        model: назва використаної моделі
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "total_analyzed": len(results),
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Результати збережено: {output_path} ({len(results)} аналізів)")


def print_summary(results: list[dict]) -> None:
    """Виведення зведеної статистики по результатах аналізу."""

    total = len(results)
    if total == 0:
        logger.info("Немає результатів для аналізу")
        return

    # Статистика задоволеності
    satisfaction_counts: dict[str, int] = {}
    for r in results:
        s = r["satisfaction"]
        satisfaction_counts[s] = satisfaction_counts.get(s, 0) + 1

    # Статистика intent
    intent_counts: dict[str, int] = {}
    for r in results:
        i = r["intent"]
        intent_counts[i] = intent_counts.get(i, 0) + 1

    # Середня якість
    avg_quality = sum(r["quality_score"] for r in results) / total

    # Помилки агента
    all_mistakes: list[str] = []
    for r in results:
        all_mistakes.extend(r["agent_mistakes"])
    mistake_counts: dict[str, int] = {}
    for m in all_mistakes:
        mistake_counts[m] = mistake_counts.get(m, 0) + 1

    chats_with_mistakes = sum(1 for r in results if r["agent_mistakes"])

    print("\n" + "=" * 60)
    print("ЗВЕДЕНА СТАТИСТИКА АНАЛІЗУ")
    print("=" * 60)

    print(f"\nВсього проаналізовано: {total} діалогів")
    print(f"Середня оцінка якості: {avg_quality:.2f} / 5.00")

    print("\nЗадоволеність клієнтів:")
    for level in ["satisfied", "neutral", "unsatisfied"]:
        count = satisfaction_counts.get(level, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {level:12s}: {count:3d} ({pct:5.1f}%) {bar}")

    print("\nКатегорії звернень:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {intent:20s}: {count:3d} ({pct:5.1f}%)")

    print("\nПомилки агентів:")
    print(f"  Діалогів з помилками: {chats_with_mistakes} / {total} ({chats_with_mistakes / total * 100:.1f}%)")
    if mistake_counts:
        for mistake, count in sorted(mistake_counts.items(), key=lambda x: -x[1]):
            print(f"  {mistake:25s}: {count}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Аналіз діалогів служби підтримки CloudTask"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Шлях до файлу з діалогами (за замовчуванням: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_RESULTS_PATH,
        help=f"Шлях до файлу результатів (за замовчуванням: {DEFAULT_RESULTS_PATH})",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не встановлено. Створіть .env файл з ключем.")
        sys.exit(1)

    dataset = load_dataset(args.input)
    chats = dataset["chats"]

    logger.info(f"Завантажено {len(chats)} діалогів з {args.input}")
    logger.info(f"Початок аналізу (модель: {ANALYSIS_MODEL}, seed: {SEED})")

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )

    # Спроба завантажити checkpoint
    checkpoint = load_analysis_checkpoint()
    if checkpoint is not None:
        results, failed = checkpoint
        analyzed_ids = {r["chat_id"] for r in results}
        chats_to_analyze = [c for c in chats if c["id"] not in analyzed_ids]
        logger.info(f"Продовження з checkpoint: {len(results)} результатів вже є")
    else:
        results = []
        failed = 0
        chats_to_analyze = chats

    for chat in tqdm(chats_to_analyze, desc="Аналіз діалогів"):
        try:
            result = analyze_single_chat(client, chat)
            results.append(result)

            # Checkpoint кожні N результатів
            if len(results) % CHECKPOINT_INTERVAL == 0:
                save_analysis_checkpoint(results, failed)

        except Exception as e:
            logger.error(f"Не вдалося проаналізувати {chat['id']}: {e}")
            failed += 1
            # Зберігаємо checkpoint при помилці
            save_analysis_checkpoint(results, failed)

    # Успішне завершення - видаляємо checkpoint
    clear_analysis_checkpoint()
    save_results(results, args.output, model=ANALYSIS_MODEL)
    print_summary(results)

    logger.info(f"Готово! Успішно: {len(results)}, помилок: {failed}")

    if failed > 0:
        logger.warning(f"{failed} діалогів не було проаналізовано через помилки")


if __name__ == "__main__":
    main()
