"""Pydantic-моделі для валідації даних діалогів та результатів аналізу."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    """Категорії звернень клієнтів."""
    PAYMENT_ISSUE = "payment_issue"
    TECHNICAL_ERROR = "technical_error"
    ACCOUNT_ACCESS = "account_access"
    TARIFF_QUESTION = "tariff_question"
    REFUND_REQUEST = "refund_request"
    OTHER = "other"


class CaseType(str, Enum):
    """Типи кейсів діалогів."""
    SUCCESSFUL = "successful"
    PROBLEMATIC = "problematic"
    CONFLICT = "conflict"
    AGENT_ERROR = "agent_error"


class AgentMistake(str, Enum):
    """Можливі помилки саппорт-агента."""
    IGNORED_QUESTION = "ignored_question"
    INCORRECT_INFO = "incorrect_info"
    RUDE_TONE = "rude_tone"
    NO_RESOLUTION = "no_resolution"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"


class Satisfaction(str, Enum):
    """Рівень задоволеності клієнта."""
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    UNSATISFIED = "unsatisfied"


class Message(BaseModel):
    """Одне повідомлення в діалозі."""
    role: Literal["client", "agent"]
    text: str = Field(min_length=1)


class Scenario(BaseModel):
    """Сценарій для генерації діалогу."""
    category: Category
    case_type: CaseType
    has_hidden_dissatisfaction: bool = False
    intended_agent_mistakes: list[AgentMistake] = []


class Chat(BaseModel):
    """Один діалог між клієнтом та саппорт-агентом."""
    id: str
    scenario: Scenario
    messages: list[Message] = Field(min_length=4, max_length=20)


class AnalysisResult(BaseModel):
    """Результат аналізу одного діалогу."""
    chat_id: str
    intent: Category
    satisfaction: Satisfaction
    quality_score: int = Field(ge=1, le=5)
    agent_mistakes: list[AgentMistake]
    summary: str
