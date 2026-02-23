# Структура проекту — AI Support Chat Analyzer

## Дерево файлів

```
ai-support-analyzer/
├── generate.py              # Генерація датасету діалогів
├── analyze.py               # Аналіз діалогів та оцінка якості
├── config.py                # Конфігурація: моделі, параметри, константи
├── requirements.txt         # Python-залежності
├── README.md                # Інструкція запуску
├── .env.example             # Шаблон змінних середовища
├── Dockerfile               # (опціонально) контейнеризація
├── docs/
│   ├── PROJECT_STRUCTURE.md # Цей документ
│   ├── TECHNICAL_DOCS.md    # Технічна документація
│   └── COST_AND_SCALING.md  # Вартість та масштабування
├── data/
│   └── chats.json           # Згенерований датасет діалогів
├── results/
│   └── analysis.json        # Результати аналізу
└── prompts/
    ├── generation.py        # Промпти для генерації діалогів
    └── analysis.py          # Промпти для аналізу діалогів
```

---

## Опис файлів

| Файл | Призначення |
|------|-------------|
| `generate.py` | Головний скрипт генерації. Створює 100+ діалогів клієнт-агент, покриваючи всі сценарії та типи кейсів. Зберігає результат у `data/chats.json` |
| `analyze.py` | Головний скрипт аналізу. Читає діалоги з `data/chats.json`, оцінює кожен через LLM, зберігає результати в `results/analysis.json` |
| `config.py` | Централізована конфігурація: назви моделей, temperature, seed, кількість діалогів, категорії, типи кейсів |
| `prompts/generation.py` | Шаблони промптів для генерації діалогів різних типів та категорій |
| `prompts/analysis.py` | Шаблони промптів для аналізу діалогів (визначення intent, satisfaction, quality, mistakes) |
| `requirements.txt` | Залежності: `openai`, `pydantic`, `python-dotenv`, `tqdm` |
| `.env.example` | Шаблон: `OPENAI_API_KEY=your-key-here` |

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
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ prompts/    │────>│ analyze.py   │────>│ results/     │
│ analysis    │     │              │     │ analysis.json│
│             │     │ OpenAI API   │     │              │
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## Формат даних

### `data/chats.json` — Датасет діалогів

```json
{
  "metadata": {
    "generated_at": "2026-02-23T12:00:00Z",
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
        "intended_agent_mistakes": []
      },
      "messages": [
        {
          "role": "client",
          "text": "Доброго дня! У мене не проходить оплата картою..."
        },
        {
          "role": "agent",
          "text": "Вітаю! Давайте розберемось з цією ситуацією..."
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
    "analyzed_at": "2026-02-23T12:30:00Z",
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
      "summary": "Клієнт звернувся з проблемою оплати. Агент оперативно допоміг вирішити питання."
    },
    {
      "chat_id": "chat_015",
      "intent": "refund_request",
      "satisfaction": "unsatisfied",
      "quality_score": 2,
      "agent_mistakes": ["ignored_question", "no_resolution"],
      "summary": "Клієнт просив повернення коштів. Агент проігнорував ключове питання і не запропонував рішення."
    }
  ]
}
```

**Поля аналізу:**
| Поле | Тип | Значення |
|------|-----|----------|
| `intent` | string | `payment_issue` \| `technical_error` \| `account_access` \| `tariff_question` \| `refund_request` \| `other` |
| `satisfaction` | string | `satisfied` \| `neutral` \| `unsatisfied` |
| `quality_score` | integer | 1–5 (1 = жахливо, 5 = відмінно) |
| `agent_mistakes` | array | Список з: `ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation` |
| `summary` | string | Короткий опис ситуації та висновок |

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
| Змішані категорії | 5 | 5 | 5 | 5 | 20 |
| Edge cases (other) | 5 | 5 | 5 | 5 | 20 |
| **Всього** | **30** | **35** | **25** | **30** | **120** |
