"""Генерація датасету діалогів клієнт-агент для служби підтримки CloudTask.

Використання:
    python generate.py [--count 120] [--output data/chats.json] [--seed 42]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm

from config import (
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PATH,
    DEFAULT_CHAT_COUNT,
    DEFAULT_OUTPUT_PATH,
    GENERATION_MODEL,
    OPENAI_API_KEY,
    REQUEST_TIMEOUT,
    SCENARIO_MATRIX,
    SEED,
    TEMPERATURE,
)
from prompts.generation import SYSTEM_PROMPT, build_generation_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def save_checkpoint(chats: list[dict], failed_count: int) -> None:
    """Збереження checkpoint для відновлення при помилці."""
    checkpoint_data = {
        "chats": chats,
        "failed_count": failed_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    logger.info(f"Checkpoint збережено: {len(chats)} чатів, {failed_count} помилок")


def load_checkpoint() -> tuple[list[dict], int] | None:
    """Завантаження checkpoint для продовження генерації."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Завантажено checkpoint: {len(data.get('chats', []))} чатів")
        return data.get("chats", []), data.get("failed_count", 0)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Не вдалося завантажити checkpoint: {e}")
        return None


def clear_checkpoint() -> None:
    """Видалення checkpoint файлу після успішного завершення."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint видалено")


def build_scenario_list(count: int = DEFAULT_CHAT_COUNT) -> list[dict]:
    """Побудова списку сценаріїв потрібної довжини з матриці.

    Бере сценарії з SCENARIO_MATRIX циклічно, щоб забезпечити покриття
    всіх комбінацій категорій та типів кейсів.
    """
    scenarios = []
    matrix = SCENARIO_MATRIX
    for i in range(count):
        scenarios.append(matrix[i % len(matrix)])
    return scenarios


def parse_chat_response(raw_response: str) -> list[dict]:
    """Парсинг та валідація відповіді API з діалогом.

    Args:
        raw_response: JSON-рядок від API

    Returns:
        Список повідомлень діалогу

    Raises:
        ValueError: якщо відповідь невалідна
        json.JSONDecodeError: якщо JSON некоректний
    """
    data = json.loads(raw_response)

    if "messages" not in data:
        raise ValueError("Відповідь API не містить поле 'messages'")

    messages: list[dict[str, str]] = data["messages"]
    if not messages:
        raise ValueError("Список повідомлень порожній")

    for msg in messages:
        if "role" not in msg or "text" not in msg:
            raise ValueError(f"Повідомлення не містить обов'язкових полів: {msg}")
        if msg["role"] not in ("client", "agent"):
            raise ValueError(f"Невідома роль: {msg['role']}")

    return messages


def generate_single_chat(
    client: OpenAI,
    scenario: dict,
    chat_id: str,
    max_retries: int = 3,
) -> dict:
    """Генерація одного діалогу через OpenAI API.

    Args:
        client: екземпляр OpenAI клієнта
        scenario: сценарій для генерації
        chat_id: унікальний ID чату
        max_retries: максимум спроб при помилках

    Returns:
        Словник з даними чату (id, scenario, messages)
    """
    user_prompt = build_generation_prompt(
        category=scenario["category"],
        case_type=scenario["case_type"],
        has_hidden_dissatisfaction=scenario.get("has_hidden_dissatisfaction", False),
        agent_mistakes=scenario.get("intended_agent_mistakes", []),
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=2000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("API returned empty response")
            messages = parse_chat_response(raw)

            return {
                "id": chat_id,
                "scenario": scenario,
                "messages": messages,
            }

        except RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit, очікування {wait}с (спроба {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Помилка парсингу для {chat_id} (спроба {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
        except APIError as e:
            logger.warning(f"API помилка для {chat_id} (спроба {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Не вдалося згенерувати чат {chat_id} після {max_retries} спроб")


def save_dataset(
    chats: list[dict],
    output_path: str,
    model: str = GENERATION_MODEL,
    seed: int = SEED,
) -> None:
    """Збереження датасету у JSON-файл.

    Args:
        chats: список діалогів
        output_path: шлях до вихідного файлу
        model: назва використаної моделі
        seed: seed для відтворюваності
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = {
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

    logger.info(f"Датасет збережено: {output_path} ({len(chats)} діалогів)")


def main():
    parser = argparse.ArgumentParser(
        description="Генерація датасету діалогів служби підтримки CloudTask"
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_CHAT_COUNT,
        help=f"Кількість діалогів (за замовчуванням: {DEFAULT_CHAT_COUNT})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Шлях до вихідного файлу (за замовчуванням: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Seed для детермінованості (за замовчуванням: {SEED})",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не встановлено. Створіть .env файл з ключем.")
        sys.exit(1)

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )
    scenarios = build_scenario_list(count=args.count)

    logger.info(f"Початок генерації {args.count} діалогів (модель: {GENERATION_MODEL}, seed: {args.seed})")

    # Спроба завантажити checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        chats, failed = checkpoint
        start_index = len(chats)
        logger.info(f"Продовження з checkpoint: {start_index} чатів вже згенеровано")
    else:
        chats = []
        failed = 0
        start_index = 0

    for i, scenario in enumerate(tqdm(scenarios[start_index:], desc="Генерація діалогів", initial=start_index)):
        chat_id = f"chat_{i + 1 + start_index:03d}"
        try:
            chat = generate_single_chat(client, scenario, chat_id)
            chats.append(chat)

            # Checkpoint кожні N чатів
            if len(chats) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(chats, failed)

        except Exception as e:
            logger.error(f"Не вдалося згенерувати {chat_id}: {e}")
            failed += 1
            # Зберігаємо checkpoint при помилці
            save_checkpoint(chats, failed)

    # Успішне завершення - видаляємо checkpoint
    clear_checkpoint()
    save_dataset(chats, args.output, model=GENERATION_MODEL, seed=args.seed)

    logger.info(f"Готово! Успішно: {len(chats)}, помилок: {failed}")

    if failed > 0:
        logger.warning(f"{failed} діалогів не було згенеровано через помилки")


if __name__ == "__main__":
    main()
