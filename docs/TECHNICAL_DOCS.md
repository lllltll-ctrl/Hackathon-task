# Технічна документація — AI Support Chat Analyzer

## 1. Архітектура рішення

Система складається з трьох незалежних pipeline + Jupyter Notebook для аналізу:

```
Pipeline 1: ГЕНЕРАЦІЯ
┌──────────┐    ┌───────────────┐    ┌───────────┐    ┌────────────┐
│ Матриця  │───>│ Формування   │───>│ OpenAI    │───>│ Валідація  │───> chats.json
│ сценаріїв│    │ промпту      │    │ API Call  │    │ (Pydantic) │
└──────────┘    └───────────────┘    └───────────┘    └────────────┘

Pipeline 2: АНАЛІЗ
┌────────────┐    ┌───────────────┐    ┌───────────┐    ┌────────────┐    ┌─────────────┐
│ chats.json │───>│ Формування   │───>│ OpenAI    │───>│ Валідація  │───>│ Пост-       │───> analysis.json
│            │    │ промпту      │    │ API Call  │    │ (Pydantic) │    │ валідація   │    (+ground truth,
└────────────┘    └───────────────┘    └───────────┘    └────────────┘    │(validation. │     confidence)
                                                                          │ py — 7 правил│
                                                                          └─────────────┘

Pipeline 3: EVALUATION
┌────────────┐    ┌───────────────┐    ┌─────────────────┐
│ analysis.  │───>│ evaluate.py   │───>│ evaluation.json  │
│ json       │    │ (порівняння   │    │ (метрики,        │
│ chats.json │    │  з ground     │    │  грейдинг        │
│            │    │  truth)       │    │  PASS/WARN/FAIL) │
└────────────┘    └───────────────┘    └─────────────────┘

Pipeline 4: NOTEBOOK АНАЛІЗ
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
| `config.py` | Всі константи, параметри моделей, seed, категорії, матриця сценаріїв, варіативні контексти, крос-категорійні сценарії |
| `models.py` | Pydantic v2 моделі для валідації всіх даних (включно з MixedIntent) |
| `validation.py` | Правило-базова пост-валідація: 7 детермінованих правил корекції після LLM-аналізу |
| `prompts/generation.py` | Промпти для генерації з варіативними контекстами та паттернами поведінки |
| `prompts/analysis.py` | Промпти для аналізу з семантичними індикаторами, confidence score, mixed intent detection |
| `generate.py` | Оркестрація генерації: цикл по сценаріях, виклик API, checkpoint, збереження |
| `analyze.py` | Оркестрація аналізу: читання чатів, виклик API, пост-валідація, ground truth embedding, checkpoint, збереження |
| `evaluate.py` | Ground truth evaluation: intent accuracy, hidden dissatisfaction detection, mistake precision/recall/F1, confidence calibration, threshold grading |
| `analysis_notebook.ipynb` | Покроковий аналіз: графіки, розподіли, виявлення аномалій |

---

## 2. Моделі OpenAI та параметри

### Генерація діалогів
| Параметр | Значення | Обґрунтування |
|----------|----------|---------------|
| **Модель** | `gpt-4o-mini` | Достатня якість для генерації діалогів, низька вартість |
| **temperature** | `0.3` | Невелика варіативність для різноманітності діалогів |
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
Клієнт формально ввічливий, але проблема не вирішена. Використовуються 3 різні поведінкові паттерни:

| # | Паттерн | Опис |
|---|---------|------|
| 0 | Відступлення | Клієнт здається і бере відповідальність на себе |
| 1 | Формальна ввічливість | Клієнт дякує, але уникає подальшого обговорення |
| 2 | Перекладання | Клієнт планує звернутись через інший канал |

Паттерн обирається через `variation_index % 3`, що забезпечує рівномірний розподіл.

#### Помилки агента (7 типів)

| Помилка | Ключ | Опис |
|---------|------|------|
| Ігнорування питання | `ignored_question` | Агент не відповів на конкретне запитання клієнта |
| Неправильна інформація | `incorrect_info` | Агент надав хибну інформацію про тарифи, процеси тощо |
| Грубий тон | `rude_tone` | Зневажливий, нетерплячий або непрофесійний тон |
| Відсутність рішення | `no_resolution` | Діалог закінчився без вирішення проблеми |
| Непотрібна ескалація | `unnecessary_escalation` | Агент перенаправив на іншого спеціаліста без потреби |
| Повільна реакція | `slow_response` | Агент затягує відповідь, просить чекати без конкретних дій |
| Шаблонна відповідь | `generic_response` | Агент дає FAQ/шаблонні відповіді замість конкретного рішення |

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

[Варіативний контекст: persona, situation, specific_detail]
[Conditional instructions based on case type: successful/problematic/conflict]
[Hidden dissatisfaction — один з 3 різних поведінкових паттернів]
[Agent mistake instructions if applicable]
[Mixed intent instructions if applicable — apparent vs actual category]

Response — ONLY valid JSON with a "messages" field.
Each message has fields "role" (client/agent) and "text".
```

**Варіативність промптів:**
- Кожен сценарій має `variation_index` (0, 1, 2), що обирає один з 3 контекстів
- Для прихованої незадоволеності використовуються 3 різні поведінкові паттерни
  (відступлення, формальна ввічливість, перекладання відповідальності)
- Для крос-категорійних сценаріїв — інструкції щодо apparent vs actual category
- Повідомлення мають варіюватись по довжині (1-3 речення)

### 4.2 Промпт аналізу діалогу

```
System role:
You are an expert in evaluating SaaS platform support quality.

CRITICALLY IMPORTANT — detecting HIDDEN dissatisfaction using
SEMANTIC BEHAVIORAL INDICATORS:
- The client's original problem was NOT resolved
- The client stops asking follow-up questions and disengages
- The client takes responsibility when the agent should have resolved it
- "I'll figure it out myself" / "I'll try again later" = resignation, NOT satisfaction
- Mismatch between initial urgency and brief, resigned closing messages
- No concrete solution was provided (only FAQ/generic advice)
If problem NOT resolved — satisfaction = "unsatisfied", even if client is polite.

MIXED INTENT detection:
If the client starts with one problem but the conversation reveals a
different underlying issue, classify by the ACTUAL root cause.

User prompt:
Analyze the following dialog between a client and a support agent:
{dialogue}

Determine:
1. intent — request category (payment_issue, technical_error, etc.)
2. satisfaction — REAL satisfaction level (satisfied/neutral/unsatisfied)
3. quality_score — agent quality 1-5
4. agent_mistakes — list of mistakes (7 types)
5. summary — brief description (1-2 sentences in English)
6. confidence — confidence in satisfaction assessment (0.0 to 1.0)

Response — ONLY valid JSON.
```

**Ключова різниця від попереднього підходу:** Промпт аналізу використовує
семантичні індикатори поведінки (disengagement, tone shift, unresolved outcome)
замість точних фраз. Це усуває тавтологічну валідацію, де генерація та аналіз
використовували однакові маркери.

---

## 5. Логіка детермінованості та варіативності

Система використовує **різні стратегії для генерації та аналізу**:

### Генерація — контрольована варіативність
1. **`temperature=0.3`** — невелика варіативність для різноманітності діалогів
2. **`seed=42`** — фіксований seed для відтворюваності
3. **Варіативні контексти** — 3 унікальні контексти (persona, situation, specific_detail) на кожну категорію
4. **3 поведінкові паттерни** — для прихованої незадоволеності (обираються через `variation_index`)

### Аналіз — максимальна детермінованість
1. **`temperature=0`** — модель завжди обирає найімовірніший токен
2. **`seed=42`** — фіксований seed для API
3. **Правило-базова пост-валідація** — 7 детермінованих правил корекції після LLM-аналізу

### Спільне
1. **Фіксований порядок сценаріїв** — матриця генерується детерміновано з config.py
2. **JSON mode** — `response_format={"type": "json_object"}` гарантує валідний JSON
3. **Pydantic-валідація** — кожна відповідь перевіряється на відповідність схемі

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
    SLOW_RESPONSE = "slow_response"
    GENERIC_RESPONSE = "generic_response"

class Satisfaction(str, Enum):
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    UNSATISFIED = "unsatisfied"

class Message(BaseModel):
    role: Literal["client", "agent"]
    text: str = Field(min_length=1)

class MixedIntent(BaseModel):
    apparent_category: Category
    actual_category: Category
    description: str

class Scenario(BaseModel):
    category: Category
    case_type: CaseType
    has_hidden_dissatisfaction: bool = False
    intended_agent_mistakes: list[AgentMistake] = []
    variation_index: int = 0
    mixed_intent: MixedIntent | None = None

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
    validation_warnings: list[str] = Field(default_factory=list)
```

---

## 7. Правило-базова пост-валідація (validation.py)

Після LLM-аналізу кожен результат проходить через 7 детермінованих правил:

| # | Правило | Дія |
|---|---------|-----|
| 1 | Є помилки агента → `quality_score ≤ 3` | Автокорекція score до 3 |
| 2 | Є помилка `rude_tone` → `quality_score ≤ 2` | Автокорекція score до 2 |
| 3 | Є помилка `no_resolution` → `satisfaction ≠ satisfied` | Корекція satisfaction на `unsatisfied` |
| 4 | `satisfied` + `quality_score ≤ 2` | Попередження-аномалія (без корекції) |
| 5 | Немає помилок + `quality_score ≤ 2` | Попередження-аномалія (без корекції) |
| 6 | `unsatisfied` + `quality_score ≥ 4` | Автокорекція score до 3 |
| 7 | 3+ помилок агента | Автокорекція score до 1 (критичний збій) |

**Принцип:** Правила 1-3, 6-7 автоматично коригують результат. Правила 4-5 лише додають `validation_warnings` без зміни даних (аномалії для ручної перевірки).

Результат `validation_warnings` зберігається в кожному `AnalysisResult`:
```json
{
  "validation_warnings": [
    "ValidationWarning(quality_score: mistakes_present_but_high_score, 4 -> 3)",
    "ValidationWarning(satisfaction: satisfied_but_low_score, value=satisfied)"
  ]
}
```

---

## 8. Ground Truth Evaluation (evaluate.py)

### 8.1 Метрики

| Метрика | Опис |
|---------|------|
| **Intent Accuracy** | Точність визначення категорії з per-category breakdown та confusion matrix |
| **Hidden Dissatisfaction Detection** | Detection rate + false positive rate для successful кейсів |
| **Mistake Detection** | Precision, recall, F1 per mistake type. **Примітка:** Precision розраховується тільки по чатах з ground truth помилками. Низька precision (40-50%) є очікуваною — GPT-4o знаходить додаткові реальні помилки, яких немає в ground truth (наприклад, `no_resolution` в чаті де intended тільки `ignored_question`). Тому грейдинг використовує **recall** як основну метрику |
| **Quality Consistency** | Середній score по case_type (successful має бути вищий за agent_error) |
| **Confidence Calibration** | Чи корелює confidence з правильністю (gap між correct/incorrect) |

### 8.2 Автоматичний грейдинг

| Метрика | PASS | WARN | FAIL |
|---------|------|------|------|
| Intent accuracy | >= 85% | >= 70% | < 70% |
| Hidden dissatisfaction detection | >= 75% | >= 50% | < 50% |
| Mistake Recall | >= 70% | >= 50% | < 50% |

### 8.3 Ground Truth Embedding

`analyze.py` автоматично вбудовує ground truth з кожного scenario в результат аналізу:
```json
{
  "ground_truth": {
    "expected_intent": "payment_issue",
    "has_hidden_dissatisfaction": true,
    "intended_agent_mistakes": ["no_resolution"],
    "case_type": "problematic",
    "mixed_intent": null
  }
}
```

Це дозволяє `evaluate.py` порівнювати LLM-аналіз з очікуваними значеннями без додаткових файлів.

---

## 9. Checkpointing

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

## 10. Обробка помилок

| Ситуація | Стратегія |
|----------|-----------|
| API rate limit (429) | Exponential backoff: 2^n секунд, макс 30с між спробами, макс 5 спроб (конфігурується через `MAX_RETRIES`, `RETRY_BACKOFF_BASE`, `RETRY_BACKOFF_MAX`) |
| Невалідний JSON від API | Повторний запит (макс 5 спроб), логування помилки |
| Pydantic validation error | Повторний запит, логування |
| API timeout | Таймаут 60с, повтор через backoff |
| API returned empty response | ValueError, повторний запит |

---

## 11. CLI-інтерфейс

### generate.py
```bash
python generate.py [--count 120] [--output data/chats.json] [--seed 42] [--concurrency 5]
```

| Аргумент | Опис | За замовчуванням |
|----------|------|-----------------|
| `--count` | Кількість діалогів | 120 |
| `--output` | Шлях до вихідного файлу | `data/chats.json` |
| `--seed` | Seed для детермінованості | 42 |
| `--concurrency` | Кількість паралельних запитів до API | 1 |

### analyze.py
```bash
python analyze.py [--input data/chats.json] [--output results/analysis.json] [--concurrency 5]
```

| Аргумент | Опис | За замовчуванням |
|----------|------|-----------------|
| `--input` | Шлях до файлу діалогів | `data/chats.json` |
| `--output` | Шлях до файлу результатів | `results/analysis.json` |
| `--concurrency` | Кількість паралельних запитів до API | 1 |

### evaluate.py
```bash
python evaluate.py [--chats data/chats.json] [--analysis results/analysis.json] [--output results/evaluation.json]
```

| Аргумент | Опис | За замовчуванням |
|----------|------|-----------------|
| `--chats` | Шлях до файлу діалогів (fallback ground truth) | `data/chats.json` |
| `--analysis` | Шлях до файлу результатів аналізу | `results/analysis.json` |
| `--output` | Шлях до файлу evaluation | `results/evaluation.json` |

### analysis_notebook.ipynb
```bash
jupyter notebook analysis_notebook.ipynb
```

Notebook автоматично завантажує `data/chats.json` та `results/analysis.json` і виконує покроковий аналіз з візуалізаціями.

---

## 12. Jupyter Notebook — аналіз та аномалії

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
