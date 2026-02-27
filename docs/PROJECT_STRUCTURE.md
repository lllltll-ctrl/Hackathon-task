# Структура проекту — AI Support Chat Analyzer

## Дерево файлів

```
ai-support-analyzer/
├── generate.py              # Генерація датасету діалогів
├── analyze.py               # Аналіз діалогів та оцінка якості
├── evaluate.py              # Оцінка результатів проти ground truth
├── analysis_notebook.ipynb  # Покроковий аналіз та виявлення аномалій (Jupyter)
├── config.py                # Конфігурація: моделі, параметри, константи, варіативні контексти
├── models.py                # Pydantic-моделі для валідації даних
├── validation.py            # Правило-базова пост-валідація результатів аналізу (7 правил)
├── requirements.txt         # Python-залежності
├── README.md                # Інструкція запуску
├── .env.example             # Шаблон змінних середовища
├── Dockerfile               # Контейнеризація
├── docker-compose.yml       # Docker Compose для зручного запуску (4 сервіси)
├── pyproject.toml           # Конфігурація ruff, mypy, pytest
├── docs/
│   ├── PROJECT_STRUCTURE.md # Цей документ
│   ├── TECHNICAL_DOCS.md    # Технічна документація
│   └── COST_AND_SCALING.md  # Вартість та масштабування
├── data/
│   └── chats.json           # Згенерований датасет діалогів
├── results/
│   ├── analysis.json        # Результати аналізу
│   └── evaluation.json      # Ground truth evaluation
├── tests/
│   ├── conftest.py          # Фікстури для тестів
│   ├── test_models.py       # Тести Pydantic-моделей
│   ├── test_config.py       # Тести конфігурації
│   ├── test_prompts.py      # Тести промптів
│   ├── test_generate.py     # Тести генерації
│   ├── test_analyze.py      # Тести аналізу
│   ├── test_validation.py   # Тести правило-базової валідації (7 правил)
│   ├── test_evaluate.py     # Тести ground truth evaluation
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
| `analyze.py` | Головний скрипт аналізу. Читає діалоги з `data/chats.json`, оцінює кожен через LLM, вбудовує ground truth зі сценарію, зберігає результати в `results/analysis.json` |
| `evaluate.py` | Оцінка результатів проти ground truth: intent accuracy, hidden dissatisfaction detection, mistake precision/recall/F1, quality consistency, confidence calibration, автоматичний грейдинг (PASS/WARN/FAIL) |
| `analysis_notebook.ipynb` | Jupyter Notebook з покроковим аналізом датасету: графіки розподілів, виявлення аномалій, патерни помилок агентів |
| `config.py` | Централізована конфігурація: назви моделей, temperature (окремо для генерації та аналізу), seed, категорії, типи кейсів, матриця сценаріїв, варіативні контексти, крос-категорійні сценарії |
| `models.py` | Pydantic v2 моделі: Category, CaseType, AgentMistake (7 типів), Satisfaction, Message, MixedIntent, Scenario, Chat, AnalysisResult |
| `validation.py` | Правило-базова пост-валідація: 7 правил перевірки узгодженості quality_score, satisfaction та agent_mistakes після LLM-аналізу |
| `prompts/generation.py` | Промпти для генерації з підтримкою варіативних контекстів, паттернів прихованої незадоволеності та крос-категорійних сценаріїв |
| `prompts/analysis.py` | Промпти для аналізу з семантичними індикаторами поведінки, confidence score та mixed intent detection |
| `requirements.txt` | Залежності: `openai`, `pydantic`, `python-dotenv`, `tqdm`, `pandas`, `matplotlib`, `seaborn`, `jupyter`, `pytest`, `ruff`, `mypy` |
| `.env.example` | Шаблон: `OPENAI_API_KEY=your-api-key-here` |
| `tests/` | 196 тестів (pytest): моделі, конфігурація, промпти, генерація, аналіз, валідація, evaluation, інтеграційні |

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
│             │     │ OpenAI API   │     │  валідація)  │     │ (+ground     │
└─────────────┘     └──────────────┘     └──────────────┘     │  truth)      │
                                                               └──────┬───────┘
                                                                      │
                                          ┌───────────────────────────┤
                                          v                           v
                                   ┌──────────────┐           ┌──────────────┐
                                   │ evaluate.py  │           │ analysis_    │
                                   │ (ground truth│           │ notebook.ipynb│
                                   │  evaluation) │           │ (графіки,    │
                                   │              │           │  аномалії)   │
                                   └──────┬───────┘           └──────────────┘
                                          v
                                   ┌──────────────┐
                                   │ results/     │
                                   │ evaluation.  │
                                   │ json         │
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
      "confidence": 0.92,
      "validation_warnings": [],
      "ground_truth": {
        "expected_intent": "payment_issue",
        "has_hidden_dissatisfaction": false,
        "intended_agent_mistakes": [],
        "case_type": "successful",
        "mixed_intent": null
      }
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
| `agent_mistakes` | array | Список з: `ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation`, `slow_response`, `generic_response` |
| `summary` | string | Короткий опис ситуації (англійською) |
| `confidence` | float | 0.0–1.0, впевненість моделі в оцінці satisfaction |
| `validation_warnings` | array | Список попереджень/корекцій від правило-базової пост-валідації |
| `ground_truth` | object | Вбудовані дані сценарію для evaluation |

---

### `results/evaluation.json` — Ground truth evaluation

```json
{
  "metadata": {
    "evaluated_at": "2026-02-24T13:00:00Z"
  },
  "evaluation": {
    "intent_accuracy": {
      "total": 120,
      "correct": 108,
      "accuracy": 0.9,
      "per_category_accuracy": { "payment_issue": 0.95, "..." : "..." }
    },
    "hidden_dissatisfaction": {
      "total_hidden": 20,
      "detected": 17,
      "detection_rate": 0.85
    },
    "mistake_detection": {
      "precision": 0.82,
      "recall": 0.75,
      "f1_score": 0.78
    },
    "quality_consistency": {
      "average_score_by_case_type": { "successful": 4.5, "agent_error": 1.8 }
    },
    "confidence_calibration": {
      "avg_confidence_correct": 0.88,
      "avg_confidence_incorrect": 0.52,
      "calibration_gap": 0.36
    },
    "grades": {
      "intent_accuracy": "PASS",
      "hidden_dissatisfaction_detection": "PASS",
      "mistake_recall": "PASS"
    }
  }
}
```

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
