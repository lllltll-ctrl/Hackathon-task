"""Pydantic models for dialog data and analysis results validation."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    """Client request categories."""
    PAYMENT_ISSUE = "payment_issue"
    TECHNICAL_ERROR = "technical_error"
    ACCOUNT_ACCESS = "account_access"
    TARIFF_QUESTION = "tariff_question"
    REFUND_REQUEST = "refund_request"
    OTHER = "other"


class CaseType(str, Enum):
    """Dialog case types."""
    SUCCESSFUL = "successful"
    PROBLEMATIC = "problematic"
    CONFLICT = "conflict"
    AGENT_ERROR = "agent_error"


class AgentMistake(str, Enum):
    """Possible support agent mistakes."""
    IGNORED_QUESTION = "ignored_question"
    INCORRECT_INFO = "incorrect_info"
    RUDE_TONE = "rude_tone"
    NO_RESOLUTION = "no_resolution"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"


class Satisfaction(str, Enum):
    """Client satisfaction level."""
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    UNSATISFIED = "unsatisfied"


class Message(BaseModel):
    """A single message in a dialog."""
    role: Literal["client", "agent"]
    text: str = Field(min_length=1)


class MixedIntent(BaseModel):
    """Cross-category intent descriptor for mixed-intent scenarios."""
    apparent_category: Category
    actual_category: Category
    description: str


class Scenario(BaseModel):
    """Scenario for dialog generation."""
    category: Category
    case_type: CaseType
    has_hidden_dissatisfaction: bool = False
    intended_agent_mistakes: list[AgentMistake] = Field(default_factory=list)
    variation_index: int = 0
    mixed_intent: MixedIntent | None = None


class Chat(BaseModel):
    """A single dialog between client and support agent."""
    id: str
    scenario: Scenario
    messages: list[Message] = Field(min_length=4, max_length=20)


class AnalysisResult(BaseModel):
    """Analysis result for a single dialog."""
    chat_id: str
    intent: Category
    satisfaction: Satisfaction
    quality_score: int = Field(ge=1, le=5)
    agent_mistakes: list[AgentMistake]
    summary: str
    validation_warnings: list[str] = Field(default_factory=list)
