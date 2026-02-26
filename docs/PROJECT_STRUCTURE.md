# Структура проекту — AI Support Chat Analyzer

## Дерево файлів

```
ai-support-analyzer/
├── generate.py              # Генерація датасету діалогів
├── analyze.py               # Аналіз діалогів та оцінка якості
├── analysis_notebook.ipynb  # Покроковий аналіз та виявлення аномалій (Jupyter)
├── config.py                # Конфігурація: моделі, параметри, константи, варіативні контексти
├── models.py                # Pydantic-моделі для валідації даних
├── validation.py            # Правило-базова пост-валідація результатів аналізу
├── requirements.txt         # Python-залежності
├── README.md                # Інструкція запуску
├── .env.example             # Шаблон змінних середовища
├── Dockerfile               # Контейнеризація
├── docker-compose.yml       # Docker Compose для зручного запуску
├── pyproject.toml           # Конфігурація ruff, mypy, pytest
├── docs/
│   ├── PROJECT_STRUCTURE.md # Цей документ
│   ├── TECHNICAL_DOCS.md    # Технічна документація
│   └── COST_AND_SCALING.md  # Вартість та масштабування
├── data/
│   └── chats.json           # Згенерований датасет діалогів
├── results/
│   └── analysis.json        # Результати аналізу
├── tests/
│   ├── conftest.py          # Фікстури для тестів
│   ├── test_models.py       # Тести Pydantic-моделей
│   ├── test_config.py       # Тести конфігурації
│   ├── test_prompts.py      # Тести промптів
│   ├── test_generate.py     # Тести генерації
│   ├── test_analyze.py      # Тести аналізу
│   ├── test_validation.py   # Тести правило-базової валідації
│   └── test_integration.py  # Інтеграційні тести
└── prompts/
    ├── generation.py        # Промпти для генерації діалогів
    └── analysis.py          # Промпти для аналізу діалогів
```

---

## Опис файлів

| Файл | Призначення |
|------|-------------|
| `generate.py` | Головний скрипт генерації. Створює 120 діалогів англійською мовою клієнт-агент, покриваючи всі сценарії та типи кейсів. Зберігає результат у `data/chats.json` |
| `analyze.py` | Головний скрипт аналізу. Читає діалоги з `data/chats.json`, оцінює кожен через LLM, зберігає результати в `results/analysis.json` |
| `analysis_notebook.ipynb` | Jupyter Notebook з покроковим аналізом датасету: графіки розподілів, виявлення аномалій, патерни помилок агентів |
| `config.py` | Централізована конфігурація: назви моделей, temperature (окремо для генерації та аналізу), seed, категорії, типи кейсів, матриця сценаріїв, варіативні контексти, крос-категорійні сценарії |
| `models.py` | Pydantic v2 моделі: Category, CaseType, AgentMistake, Satisfaction, Message, MixedIntent, Scenario, Chat, AnalysisResult |
| `validation.py` | Правило-базова пост-валідація: перевірка узгодженості quality_score, satisfaction та agent_mistakes після LLM-аналізу |
| `prompts/generation.py` | Промпти для генерації з підтримкою варіативних контекстів, паттернів прихованої незадоволеності та крос-категорійних сценаріїв |
| `prompts/analysis.py` | Промпти для аналізу з семантичними індикаторами поведінки (замість шаблонних фраз) |
| `requirements.txt` | Залежності: `openai`, `pydantic`, `python-dotenv`, `tqdm`, `pandas`, `matplotlib`, `seaborn`, `jupyter`, `pytest`, `ruff`, `mypy` |
| `.env.example` | Шаблон: `OPENAI_API_KEY=your-api-key-here` |
| `tests/` | 166 тестів (pytest): моделі, конфігурація, промпти, генерація, аналіз, валідація, інтеграційні |

---

## Потік даних

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ config.py   │────>│ generate.py  │────>│ data/        │
│ prompts/    │     │              │     │  chats.json  │
│ generation  │     │ OpenAI API   │     │              │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                v
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ prompts/    │────>│ analyze.py   │────>│ validation.py│────>│ results/     │
│ analysis    │     │              │     │ (пост-       │     │ analysis.json│
│             │     │ OpenAI API   │     │  валідація)  │     │              │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                v
                                         ┌──────────────┐
                                         │ analysis_    │
                                         │ notebook.ipynb│
                                         │ (графіки,    │
                                         │  аномалії)   │
                                         └──────────────┘
```

---

## Формат даних

### `data/chats.json` — Датасет діалогів

```json
{
  "metadata": {
    "generated_at": "2026-02-24T12:00:00Z",
    "model": "gpt-4o-mini",
    "total_chats": 120,
    "seed": 42
  },
  "chats": [
    {
      "id": "chat_001",
      "scenario": {
        "category": "payment_issue",
        "case_type": "successful",
        "has_hidden_dissatisfaction": false,
        "intended_agent_mistakes": [],
        "variation_index": 0,
        "mixed_intent": null
      },
      "messages": [
        {
          "role": "client",
          "text": "Hi! I'm trying to subscribe to the Pro plan but my card keeps getting declined..."
        },
        {
          "role": "agent",
          "text": "Hello! Let me help you with that. Could you confirm that your card is active and has sufficient funds?"
        }
      ]
    }
  ]
}
```

**Категорії (`category`):**
| Значення | Опис |
|----------|------|
| `payment_issue` | Проблеми з оплатою |
| `technical_error` | Технічні помилки |
| `account_access` | Доступ до акаунту |
| `tariff_question` | Питання по тарифу |
| `refund_request` | Повернення коштів |
| `other` | Інші звернення |

**Типи кейсів (`case_type`):**
| Значення | Опис |
|----------|------|
| `successful` | Проблема вирішена, клієнт задоволений |
| `problematic` | Проблема вирішена частково або з труднощами |
| `conflict` | Конфліктна ситуація, клієнт незадоволений |
| `agent_error` | Агент допускає помилки |

---

### `results/analysis.json` — Результати аналізу

```json
{
  "metadata": {
    "analyzed_at": "2026-02-24T12:30:00Z",
    "model": "gpt-4o",
    "total_analyzed": 120
  },
  "results": [
    {
      "chat_id": "chat_001",
      "intent": "payment_issue",
      "satisfaction": "satisfied",
      "quality_score": 5,
      "agent_mistakes": [],
      "summary": "Client had a payment issue with card being declined. Agent quickly identified the problem and helped resolve it.",
      "validation_warnings": []
    }
  ]
}
```

**Поля аналізу:**
| Поле | Тип | Значення |
|------|-----|----------|
| `intent` | string | `payment_issue` \| `technical_error` \| `account_access` \| `tariff_question` \| `refund_request` \| `other` |
| `satisfaction` | string | `satisfied` \| `neutral` \| `unsatisfied` |
| `quality_score` | integer | 1–5 (1 = terrible, 5 = excellent) |
| `agent_mistakes` | array | Список з: `ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation` |
| `summary` | string | Короткий опис ситуації (англійською) |
| `validation_warnings` | array | Список попереджень/корекцій від правило-базової пост-валідації |

---

## Розподіл датасету (120 діалогів)

| Категорія | Успішні | Проблемні | Конфліктні | Помилки агента | Всього |
|-----------|---------|-----------|------------|----------------|--------|
| payment_issue | 4 | 3 | 3 | 2 | 12 |
| technical_error | 4 | 3 | 3 | 2 | 12 |
| account_access | 4 | 3 | 3 | 2 | 12 |
| tariff_question | 4 | 3 | 3 | 2 | 12 |
| refund_request | 4 | 3 | 3 | 2 | 12 |
| **Підтипи** | | | | | |
| Прихована незадоволеність | — | 10 | — | 10 | 20 |
| Крос-категорійні (mixed intent) | 3 | 3 | 2 | 2 | 10 |
| Edge cases (other) | 3 | 3 | 2 | 2 | 10 |
| Змішані категорії | 5 | 5 | 5 | 5 | 20 |
| **Всього** | **30** | **35** | **25** | **30** | **120** |

> **Варіативність:** Кожна категорія має 3 варіативні контексти (persona, situation, specific_detail), що забезпечує різноманітність діалогів навіть при однаковому seed.
