"""Конфігурація проекту: моделі, параметри, матриця сценаріїв."""

import os

from dotenv import load_dotenv

load_dotenv()

# ── OpenAI API ───────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Модель для генерації діалогів (дешевша, достатня для тексту)
GENERATION_MODEL = "gpt-4o-mini"

# Модель для аналізу діалогів (потужніша, краще розуміє нюанси)
ANALYSIS_MODEL = "gpt-4o"

# Параметри для детермінованості
TEMPERATURE = 0
SEED = 42

# Timeout для API запитів (секунди)
REQUEST_TIMEOUT = 60.0

# Checkpointing: зберігати прогрес кожні N чатів
CHECKPOINT_INTERVAL = 10
CHECKPOINT_PATH = "data/checkpoint.json"
CHECKPOINT_ANALYSIS_PATH = "results/checkpoint_analysis.json"

# ── Датасет ──────────────────────────────────────────────────────────

DEFAULT_CHAT_COUNT = 120
DEFAULT_OUTPUT_PATH = "data/chats.json"
DEFAULT_RESULTS_PATH = "results/analysis.json"

# ── Категорії та типи ────────────────────────────────────────────────

CATEGORIES = [
    "payment_issue",
    "technical_error",
    "account_access",
    "tariff_question",
    "refund_request",
    "other",
]

CASE_TYPES = [
    "successful",
    "problematic",
    "conflict",
    "agent_error",
]

AGENT_MISTAKES = [
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
]

# Опис категорій українською (для промптів)
CATEGORY_DESCRIPTIONS = {
    "payment_issue": "Проблеми з оплатою (картка не проходить, подвійне списання, не зараховано оплату за підписку CloudTask)",
    "technical_error": "Технічні помилки (помилка 500, не працює API-інтеграція, баг в інтерфейсі, не завантажується дашборд)",
    "account_access": "Доступ до акаунту (забутий пароль, заблокований акаунт, проблеми з SSO/двофакторною автентифікацією)",
    "tariff_question": "Питання по тарифу (різниця між Free/Pro/Enterprise, зміна плану, ліміти функцій, умови підписки CloudTask)",
    "refund_request": "Повернення коштів (повернення за невикористаний період підписки, скасування автопродовження, помилкове списання)",
    "other": "Інші звернення (пропозиції по покращенню, загальні питання про функціонал CloudTask, скарги на сервіс)",
}

# Опис типів кейсів (для промптів)
CASE_TYPE_DESCRIPTIONS = {
    "successful": "Успішний кейс: агент швидко розуміє проблему, надає чітке рішення, клієнт задоволений результатом",
    "problematic": "Проблемний кейс: агент потребує кількох уточнень, рішення не ідеальне, клієнт нейтральний або частково задоволений",
    "conflict": "Конфліктний кейс: клієнт емоційний та незадоволений, вимагає ескалацію або компенсацію, агент під тиском",
    "agent_error": "Кейс з помилкою агента: агент допускає конкретні помилки (неправильна інформація, грубий тон, ігнорування питання тощо)",
}

# Опис помилок агента (для промптів)
MISTAKE_DESCRIPTIONS = {
    "ignored_question": "Ігнорування питання — агент не відповідає на конкретне запитання клієнта, переходить до іншої теми",
    "incorrect_info": "Неправильна інформація — агент надає хибну інформацію про тарифи, функції або процедури CloudTask",
    "rude_tone": "Грубий тон — агент відповідає зневажливо, нетерпляче або непрофесійно",
    "no_resolution": "Відсутність рішення — діалог закінчується без реального вирішення проблеми клієнта",
    "unnecessary_escalation": "Непотрібна ескалація — агент перенаправляє на іншого спеціаліста без спроби вирішити самостійно",
}


def _build_scenario_matrix() -> list[dict]:
    """Побудова повної матриці сценаріїв для генерації 120 діалогів.

    Розподіл:
    - 60 основних: 5 категорій × 4 типи × 3 варіації
    - 20 з прихованою незадоволеністю
    - 20 з помилками агента (додаткові)
    - 20 змішаних / edge cases
    """
    scenarios = []

    # Основні категорії (без "other")
    main_categories = [c for c in CATEGORIES if c != "other"]

    # ── Блок 1: Основна матриця (5 категорій × 4 типи × 3 варіації = 60) ──
    for category in main_categories:
        for case_type in CASE_TYPES:
            for variation in range(3):
                scenario = {
                    "category": category,
                    "case_type": case_type,
                    "has_hidden_dissatisfaction": False,
                    "intended_agent_mistakes": [],
                }
                # Для agent_error додаємо конкретні помилки
                if case_type == "agent_error":
                    mistake_idx = variation % len(AGENT_MISTAKES)
                    scenario["intended_agent_mistakes"] = [AGENT_MISTAKES[mistake_idx]]
                scenarios.append(scenario)

    # ── Блок 2: Прихована незадоволеність (20 кейсів) ──
    hidden_configs = [
        # Клієнт формально дякує, але проблема не вирішена
        ("payment_issue", "problematic", ["no_resolution"]),
        ("payment_issue", "successful", []),
        ("technical_error", "problematic", ["no_resolution"]),
        ("technical_error", "successful", []),
        ("account_access", "problematic", ["ignored_question"]),
        ("account_access", "successful", []),
        ("tariff_question", "problematic", ["incorrect_info"]),
        ("tariff_question", "successful", []),
        ("refund_request", "problematic", ["no_resolution"]),
        ("refund_request", "successful", []),
        # Агент дає шаблонну відповідь
        ("payment_issue", "agent_error", ["no_resolution"]),
        ("technical_error", "agent_error", ["ignored_question"]),
        ("account_access", "agent_error", ["unnecessary_escalation"]),
        ("tariff_question", "agent_error", ["incorrect_info"]),
        ("refund_request", "agent_error", ["no_resolution"]),
        # Клієнт сам "здається"
        ("payment_issue", "conflict", []),
        ("technical_error", "conflict", []),
        ("account_access", "conflict", []),
        ("tariff_question", "conflict", []),
        ("refund_request", "conflict", []),
    ]
    for category, case_type, mistakes in hidden_configs:
        scenarios.append({
            "category": category,
            "case_type": case_type,
            "has_hidden_dissatisfaction": True,
            "intended_agent_mistakes": mistakes,
        })

    # ── Блок 3: Додаткові помилки агента (20 кейсів) ──
    mistake_combos = [
        ["ignored_question", "no_resolution"],
        ["incorrect_info", "rude_tone"],
        ["rude_tone", "no_resolution"],
        ["ignored_question", "unnecessary_escalation"],
        ["incorrect_info", "no_resolution"],
    ]
    for category in main_categories:
        for i, combo in enumerate(mistake_combos[:4]):
            scenarios.append({
                "category": category,
                "case_type": "agent_error",
                "has_hidden_dissatisfaction": i % 2 == 0,
                "intended_agent_mistakes": combo,
            })

    # ── Блок 4: Edge cases з категорією "other" (20 кейсів) ──
    for case_type in CASE_TYPES:
        for variation in range(5):
            scenario = {
                "category": "other",
                "case_type": case_type,
                "has_hidden_dissatisfaction": variation % 3 == 0,
                "intended_agent_mistakes": [],
            }
            if case_type == "agent_error":
                mistake_idx = variation % len(AGENT_MISTAKES)
                scenario["intended_agent_mistakes"] = [AGENT_MISTAKES[mistake_idx]]
            scenarios.append(scenario)

    return scenarios


# Повна матриця сценаріїв (детерміновано побудована)
SCENARIO_MATRIX = _build_scenario_matrix()
