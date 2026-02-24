# Технічна документація — AI Support Chat Analyzer

## 1. Архітектура рішення

Система складається з двох незалежних pipeline + Jupyter Notebook для аналізу:

```
Pipeline 1: ГЕНЕРАЦІЯ
┌──────────┐    ┌───────────────┐    ┌───────────┐    ┌────────────┐
│ Матриця  │───>│ Формування   │───>│ OpenAI    │───>│ Валідація  │───> chats.json
│ сценаріїв│    │ промпту      │    │ API Call  │    │ (Pydantic) │
└──────────┘    └───────────────┘    └───────────┘    └────────────┘

Pipeline 2: АНАЛІЗ
┌────────────┐    ┌───────────────┐    ┌───────────┐    ┌────────────┐
│ chats.json │───>│ Формування   │───>│ OpenAI    │───>│ Валідація  │───> analysis.json
│            │    │ промпту      │    │ API Call  │    │ (Pydantic) │
└────────────┘    └───────────────┘    └───────────┘    └────────────┘

Pipeline 3: NOTEBOOK АНАЛІЗ
┌────────────┐    ┌────────────────┐    ┌────────────────┐
│ chats.json │───>│ analysis_      │───>│ Графіки,       │
│ analysis.  │    │ notebook.ipynb │    │ аномалії,      │
│ json       │    │ (pandas,       │    │ висновки       │
│            │    │  matplotlib)   │    │                │
└────────────┘    └────────────────┘    └────────────────┘
```

### Принцип поділу відповідальності

| Компонент | Відповідальність |
|-----------|-----------------|
| `config.py` | Всі константи, параметри моделей, seed, категорії, матриця сценаріїв |
| `models.py` | Pydantic v2 моделі для валідації всіх даних |
| `prompts/generation.py` | Шаблони промптів для генерації діалогів (англійською) |
| `prompts/analysis.py` | Шаблони промптів для аналізу діалогів (англійською) |
| `generate.py` | Оркестрація генерації: цикл по сценаріях, виклик API, checkpoint, збереження |
| `analyze.py` | Оркестрація аналізу: читання чатів, виклик API, checkpoint, збереження результатів |
| `analysis_notebook.ipynb` | Покроковий аналіз: графіки, розподіли, виявлення аномалій |

---

## 2. Моделі OpenAI та параметри

### Генерація діалогів
| Параметр | Значення | Обґрунтування |
|----------|----------|---------------|
| **Модель** | `gpt-4o-mini` | Достатня якість для генерації діалогів, низька вартість |
| **temperature** | `0` | Детермінованість результатів |
| **seed** | `42` | Фіксований seed для відтворюваності |
| **max_tokens** | `2000` | Достатньо для діалогу з 6-14 реплік |
| **response_format** | `{ "type": "json_object" }` | Гарантований JSON-вивід |

### Аналіз діалогів
| Параметр | Значення | Обґрунтування |
|----------|----------|---------------|
| **Модель** | `gpt-4o` | Вища якість аналізу, краще розуміння нюансів |
| **temperature** | `0` | Детермінованість |
| **seed** | `42` | Відтворюваність |
| **max_tokens** | `500` | Достатньо для JSON з результатами аналізу |
| **response_format** | `{ "type": "json_object" }` | Гарантований JSON-вивід |

> **Чому різні моделі?** Генерація потребує об'єму (багато тексту), але не найвищої якості аналізу — `gpt-4o-mini` справляється і коштує в 30x дешевше. Аналіз потребує глибокого розуміння контексту, особливо для прихованої незадоволеності — тут варто використати `gpt-4o`.

---

## 3. Матриця сценаріїв

### 3.1 Основні категорії (intent)

| # | Категорія | Ключ | Приклади ситуацій |
|---|-----------|------|-------------------|
| 1 | Проблеми з оплатою | `payment_issue` | Card declined, double charge, payment not credited |
| 2 | Технічні помилки | `technical_error` | Error 500, API integration failure, UI bug |
| 3 | Доступ до акаунту | `account_access` | Forgotten password, locked account, 2FA issues |
| 4 | Питання по тарифу | `tariff_question` | Plan differences, plan change, feature limits |
| 5 | Повернення коштів | `refund_request` | Refund for unused period, cancel auto-renewal |
| 6 | Інше | `other` | Feature suggestions, general questions, complaints |

### 3.2 Типи кейсів

| Тип | Опис | Характерні ознаки |
|-----|------|-------------------|
| **Успішний** | Проблема вирішена повністю | Агент швидко зрозумів проблему, запропонував дієве рішення, клієнт задоволений |
| **Проблемний** | Проблема вирішена частково | Агент потребував уточнень, рішення не ідеальне, клієнт нейтральний |
| **Конфліктний** | Клієнт незадоволений | Емоційний діалог, клієнт вимагає ескалацію, агент під тиском |
| **Помилка агента** | Агент допускає помилки | Неправильна інформація, грубий тон, ігнорування питання |

### 3.3 Спеціальні сценарії

#### Прихована незадоволеність (20 діалогів)
Клієнт формально ввічливий, дякує агенту, але:
- Проблема фактично не вирішена
- Клієнт отримав відписку замість рішення
- Агент перенаправив на FAQ замість реальної допомоги
- Клієнт погодився, але з сарказмом або пасивною агресією

**Приклад:**
```
Client: "Alright, I'll try to figure it out myself. Thanks for your time."
(Problem: client didn't get a real solution, just gave up)
```

#### Помилки агента

| Помилка | Ключ | Опис |
|---------|------|------|
| Ігнорування питання | `ignored_question` | Агент не відповів на конкретне запитання клієнта |
| Неправильна інформація | `incorrect_info` | Агент надав хибну інформацію про тарифи, процеси тощо |
| Грубий тон | `rude_tone` | Зневажливий, нетерплячий або непрофесійний тон |
| Відсутність рішення | `no_resolution` | Діалог закінчився без вирішення проблеми |
| Непотрібна ескалація | `unnecessary_escalation` | Агент перенаправив на іншого спеціаліста без потреби |

---

## 4. Промпти

### 4.1 Промпт генерації діалогу

```
System role:
You are a generator of realistic support dialogs for the SaaS platform
"CX-Ray" in English.

User prompt:
Generate a realistic dialog between a client and a support agent of the
CX-Ray platform.

Request category: {category_description}
Case type: {case_type_description}

[Conditional instructions based on case type: successful/problematic/conflict]
[Hidden dissatisfaction instructions if applicable]
[Agent mistake instructions if applicable]

Response — ONLY valid JSON with a "messages" field.
Each message has fields "role" (client/agent) and "text".
```

### 4.2 Промпт аналізу діалогу

```
System role:
You are an expert in evaluating SaaS platform support quality.
CRITICALLY IMPORTANT — detecting HIDDEN dissatisfaction:
- Client says 'okay, I'll try to figure it out myself' — gave up
- 'Thanks for the information' — when info doesn't actually help
- Passive aggression, sarcasm
If problem NOT resolved — satisfaction = "unsatisfied", even if client is polite.

User prompt:
Analyze the following dialog between a client and a support agent:
{dialogue}

Determine:
1. intent — request category (payment_issue, technical_error, etc.)
2. satisfaction — REAL satisfaction level (satisfied/neutral/unsatisfied)
3. quality_score — agent quality 1-5
4. agent_mistakes — list of mistakes
5. summary — brief description (1-2 sentences in English)

Response — ONLY valid JSON.
```

---

## 5. Логіка детермінованості

Для забезпечення відтворюваних результатів:

1. **`temperature=0`** — модель завжди обирає найімовірніший токен
2. **`seed=42`** — фіксований seed для API (OpenAI підтримує з листопада 2023)
3. **Фіксований порядок сценаріїв** — матриця генерується детерміновано з config.py
4. **JSON mode** — `response_format={"type": "json_object"}` гарантує валідний JSON
5. **Pydantic-валідація** — кожна відповідь перевіряється на відповідність схемі

> **Примітка:** OpenAI зазначає, що `seed` забезпечує "mostly deterministic" результати. Для 100% відтворюваності варто кешувати результати API-викликів.

---

## 6. Pydantic-моделі (валідація)

```python
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class Category(str, Enum):
    PAYMENT_ISSUE = "payment_issue"
    TECHNICAL_ERROR = "technical_error"
    ACCOUNT_ACCESS = "account_access"
    TARIFF_QUESTION = "tariff_question"
    REFUND_REQUEST = "refund_request"
    OTHER = "other"

class CaseType(str, Enum):
    SUCCESSFUL = "successful"
    PROBLEMATIC = "problematic"
    CONFLICT = "conflict"
    AGENT_ERROR = "agent_error"

class AgentMistake(str, Enum):
    IGNORED_QUESTION = "ignored_question"
    INCORRECT_INFO = "incorrect_info"
    RUDE_TONE = "rude_tone"
    NO_RESOLUTION = "no_resolution"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"

class Satisfaction(str, Enum):
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    UNSATISFIED = "unsatisfied"

class Message(BaseModel):
    role: Literal["client", "agent"]
    text: str = Field(min_length=1)

class Scenario(BaseModel):
    category: Category
    case_type: CaseType
    has_hidden_dissatisfaction: bool = False
    intended_agent_mistakes: list[AgentMistake] = []

class Chat(BaseModel):
    id: str
    scenario: Scenario
    messages: list[Message] = Field(min_length=4, max_length=20)

class AnalysisResult(BaseModel):
    chat_id: str
    intent: Category
    satisfaction: Satisfaction
    quality_score: int = Field(ge=1, le=5)
    agent_mistakes: list[AgentMistake]
    summary: str
```

---

## 7. Checkpointing

Для великих датасетів система зберігає прогрес кожні N чатів (за замовчуванням 10):

| Параметр | Значення |
|----------|----------|
| `CHECKPOINT_INTERVAL` | 10 |
| `CHECKPOINT_PATH` | `data/checkpoint.json` |
| `CHECKPOINT_ANALYSIS_PATH` | `results/checkpoint_analysis.json` |

**Логіка:**
1. Кожні 10 успішних чатів — зберігається checkpoint
2. При помилці — checkpoint зберігається одразу
3. При повторному запуску — генерація/аналіз продовжується з checkpoint
4. Після успішного завершення — checkpoint видаляється

---

## 8. Обробка помилок

| Ситуація | Стратегія |
|----------|-----------|
| API rate limit (429) | Exponential backoff: 1s → 2s → 4s, макс 3 спроби |
| Невалідний JSON від API | Повторний запит (макс 3 спроби), логування помилки |
| Pydantic validation error | Повторний запит, логування |
| API timeout | Таймаут 60с, повтор через backoff |
| API returned empty response | ValueError, повторний запит |

---

## 9. CLI-інтерфейс

### generate.py
```bash
python generate.py [--count 120] [--output data/chats.json] [--seed 42]
```

| Аргумент | Опис | За замовчуванням |
|----------|------|-----------------|
| `--count` | Кількість діалогів | 120 |
| `--output` | Шлях до вихідного файлу | `data/chats.json` |
| `--seed` | Seed для детермінованості | 42 |

### analyze.py
```bash
python analyze.py [--input data/chats.json] [--output results/analysis.json]
```

| Аргумент | Опис | За замовчуванням |
|----------|------|-----------------|
| `--input` | Шлях до файлу діалогів | `data/chats.json` |
| `--output` | Шлях до файлу результатів | `results/analysis.json` |

### analysis_notebook.ipynb
```bash
jupyter notebook analysis_notebook.ipynb
```

Notebook автоматично завантажує `data/chats.json` та `results/analysis.json` і виконує покроковий аналіз з візуалізаціями.

---

## 10. Jupyter Notebook — аналіз та аномалії

`analysis_notebook.ipynb` містить 8 секцій:

| # | Секція | Опис |
|---|--------|------|
| 1 | Завантаження даних | Load chats.json + analysis.json, створення DataFrame |
| 2 | Огляд датасету | Розподіл по категоріях, case types, довжина діалогів |
| 3 | Розподіл оцінок | Гістограма quality_score, boxplot по case_type |
| 4 | Аналіз задоволеності | Pie chart, stacked bar по категоріях |
| 5 | Виявлення аномалій | Hidden dissatisfaction detection accuracy, score inconsistencies |
| 6 | Помилки агентів | Частота помилок, кореляція з quality_score |
| 7 | Аналіз інтентів | Розподіл інтентів, середня оцінка по інтентах |
| 8 | Підсумки | Загальна статистика, ключові знахідки |

**Бібліотеки:** pandas, matplotlib, seaborn
