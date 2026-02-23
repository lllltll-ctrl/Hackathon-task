"""Промпти для генерації діалогів клієнт-агент."""

from config import CASE_TYPE_DESCRIPTIONS, CATEGORY_DESCRIPTIONS, MISTAKE_DESCRIPTIONS

SYSTEM_PROMPT = """Ти — генератор реалістичних діалогів служби підтримки SaaS-платформи "CloudTask" українською мовою.

CloudTask — це хмарна платформа для управління проєктами та командною роботою.
Тарифні плани: Free (безкоштовний, до 5 користувачів), Pro ($15/міс за користувача), Enterprise (індивідуальна ціна).
Функції: дашборди, канбан-дошки, тайм-трекінг, API-інтеграції, звіти, командний чат.

Правила генерації діалогів:
1. Мова — українська, природна розмовна для клієнта, професійна для агента
2. Клієнт може використовувати розмовний стиль, скорочення, емоційні вирази
3. Агент відповідає ввічливо, структуровано, з конкретними кроками
4. Діалог повинен містити від 6 до 14 реплік (3-7 від кожної сторони)
5. Перша репліка завжди від клієнта
6. Остання репліка завжди від агента
7. Кожна репліка має бути змістовною (не просто "ок" або "дякую")

Формат відповіді — ТІЛЬКИ валідний JSON:
{
  "messages": [
    {"role": "client", "text": "..."},
    {"role": "agent", "text": "..."}
  ]
}"""


def build_generation_prompt(
    category: str,
    case_type: str,
    has_hidden_dissatisfaction: bool,
    agent_mistakes: list[str],
) -> str:
    """Формує промпт для генерації одного діалогу."""

    category_desc = CATEGORY_DESCRIPTIONS.get(category, category)
    case_type_desc = CASE_TYPE_DESCRIPTIONS.get(case_type, case_type)

    prompt_parts = [
        "Згенеруй реалістичний діалог між клієнтом та саппорт-агентом платформи CloudTask.",
        "",
        f"Категорія звернення: {category_desc}",
        f"Тип кейсу: {case_type_desc}",
    ]

    # Додаткові інструкції залежно від типу кейсу
    if case_type == "successful" and not has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "ВАЖЛИВО — Це УСПІШНИЙ кейс. Агент ПОВИНЕН реально вирішити проблему клієнта: "
            "знайти конкретну причину, виконати дію (скинути пароль, повернути кошти, "
            "змінити тариф, виправити помилку) і підтвердити що проблема вирішена. "
            "Клієнт має бути ДІЙСНО задоволений результатом, а не просто отримати загальну пораду."
        )
    elif case_type == "problematic" and not has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "Це проблемний кейс: агент намагається допомогти, але рішення не ідеальне — "
            "потрібні додаткові кроки, очікування, або компроміс. Проблема вирішена ЧАСТКОВО."
        )
    elif case_type == "conflict" and not has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "Це конфліктний кейс: клієнт емоційний, розчарований, може підвищувати тон. "
            "Агент під тиском, ситуація напружена. Клієнт відкрито незадоволений."
        )

    if has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "ВАЖЛИВО — Прихована незадоволеність: клієнт повинен бути формально ввічливим "
            "(дякувати, казати «добре», «зрозумів»), але проблема фактично НЕ вирішена. "
            "Клієнт може:\n"
            "- Сказати «Добре, спробую розібратись сам» (здався)\n"
            "- Подякувати за інформацію, яка насправді не допомогла\n"
            "- Погодитись з відповіддю з легким сарказмом або розчаруванням\n"
            "- Припинити діалог, не отримавши реального рішення"
        )

    if agent_mistakes:
        prompt_parts.append("")
        prompt_parts.append("Агент повинен допустити наступні помилки (природно, не надто очевидно):")
        for mistake in agent_mistakes:
            desc = MISTAKE_DESCRIPTIONS.get(mistake, mistake)
            prompt_parts.append(f"- {desc}")

    prompt_parts.append("")
    prompt_parts.append(
        "Відповідь — ТІЛЬКИ валідний JSON з полем \"messages\". "
        "Кожне повідомлення має поля \"role\" (client/agent) та \"text\"."
    )

    return "\n".join(prompt_parts)
