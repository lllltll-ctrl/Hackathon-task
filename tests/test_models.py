"""Тести для Pydantic-моделей даних."""

import pytest
from pydantic import ValidationError

from models import (
    AgentMistake,
    AnalysisResult,
    CaseType,
    Category,
    Chat,
    Message,
    MixedIntent,
    Satisfaction,
    Scenario,
)

# ── Message ──────────────────────────────────────────────────────────

class TestMessage:
    def test_valid_client_message(self, sample_message_client):
        msg = Message(**sample_message_client)
        assert msg.role == "client"
        assert "оплата" in msg.text

    def test_valid_agent_message(self, sample_message_agent):
        msg = Message(**sample_message_agent)
        assert msg.role == "agent"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="manager", text="Привіт")

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="client", text="")

    def test_missing_text_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="client")


# ── Category Enum ────────────────────────────────────────────────────

class TestCategory:
    EXPECTED_CATEGORIES = [
        "payment_issue",
        "technical_error",
        "account_access",
        "tariff_question",
        "refund_request",
        "other",
    ]

    def test_all_categories_exist(self):
        for cat in self.EXPECTED_CATEGORIES:
            assert Category(cat) is not None

    def test_exactly_six_categories(self):
        assert len(Category) == 6

    def test_invalid_category_rejected(self):
        with pytest.raises(ValueError):
            Category("nonexistent_category")


# ── CaseType Enum ────────────────────────────────────────────────────

class TestCaseType:
    EXPECTED_TYPES = ["successful", "problematic", "conflict", "agent_error"]

    def test_all_case_types_exist(self):
        for ct in self.EXPECTED_TYPES:
            assert CaseType(ct) is not None

    def test_exactly_four_types(self):
        assert len(CaseType) == 4


# ── AgentMistake Enum ────────────────────────────────────────────────

class TestAgentMistake:
    EXPECTED_MISTAKES = [
        "ignored_question",
        "incorrect_info",
        "rude_tone",
        "no_resolution",
        "unnecessary_escalation",
        "slow_response",
        "generic_response",
    ]

    def test_all_mistakes_exist(self):
        for m in self.EXPECTED_MISTAKES:
            assert AgentMistake(m) is not None

    def test_exactly_seven_mistakes(self):
        assert len(AgentMistake) == 7


# ── Satisfaction Enum ────────────────────────────────────────────────

class TestSatisfaction:
    def test_three_levels(self):
        assert len(Satisfaction) == 3

    def test_valid_values(self):
        for val in ["satisfied", "neutral", "unsatisfied"]:
            assert Satisfaction(val) is not None


# ── Scenario ─────────────────────────────────────────────────────────

class TestScenario:
    def test_valid_scenario(self, sample_scenario_successful):
        scenario = Scenario(**sample_scenario_successful)
        assert scenario.category == Category.PAYMENT_ISSUE
        assert scenario.case_type == CaseType.SUCCESSFUL
        assert scenario.has_hidden_dissatisfaction is False
        assert scenario.intended_agent_mistakes == []

    def test_scenario_with_mistakes(self, sample_scenario_with_mistakes):
        scenario = Scenario(**sample_scenario_with_mistakes)
        assert scenario.has_hidden_dissatisfaction is True
        assert len(scenario.intended_agent_mistakes) == 2
        assert AgentMistake.IGNORED_QUESTION in scenario.intended_agent_mistakes

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            Scenario(
                category="invalid",
                case_type="successful",
                has_hidden_dissatisfaction=False,
                intended_agent_mistakes=[],
            )

    def test_invalid_case_type_rejected(self):
        with pytest.raises(ValidationError):
            Scenario(
                category="payment_issue",
                case_type="invalid_type",
                has_hidden_dissatisfaction=False,
                intended_agent_mistakes=[],
            )

    def test_default_values(self):
        scenario = Scenario(category="payment_issue", case_type="successful")
        assert scenario.has_hidden_dissatisfaction is False
        assert scenario.intended_agent_mistakes == []
        assert scenario.variation_index == 0
        assert scenario.mixed_intent is None

    def test_scenario_with_variation_index(self):
        scenario = Scenario(
            category="payment_issue",
            case_type="successful",
            variation_index=2,
        )
        assert scenario.variation_index == 2

    def test_scenario_with_mixed_intent(self):
        scenario = Scenario(
            category="technical_error",
            case_type="problematic",
            mixed_intent={
                "apparent_category": "payment_issue",
                "actual_category": "technical_error",
                "description": "Test mixed intent",
            },
        )
        assert scenario.mixed_intent is not None
        assert scenario.mixed_intent.apparent_category == Category.PAYMENT_ISSUE
        assert scenario.mixed_intent.actual_category == Category.TECHNICAL_ERROR


# ── MixedIntent ─────────────────────────────────────────────────────

class TestMixedIntent:
    def test_valid_mixed_intent(self):
        mi = MixedIntent(
            apparent_category="payment_issue",
            actual_category="technical_error",
            description="Test",
        )
        assert mi.apparent_category == Category.PAYMENT_ISSUE
        assert mi.actual_category == Category.TECHNICAL_ERROR

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            MixedIntent(
                apparent_category="invalid",
                actual_category="payment_issue",
                description="Test",
            )


# ── Chat ─────────────────────────────────────────────────────────────

class TestChat:
    def test_valid_chat(self, sample_chat):
        chat = Chat(**sample_chat)
        assert chat.id == "chat_001"
        assert len(chat.messages) == 6

    def test_chat_minimum_messages(self):
        """Діалог повинен мати мінімум 4 повідомлення."""
        chat = Chat(
            id="chat_min",
            scenario={
                "category": "payment_issue",
                "case_type": "successful",
            },
            messages=[
                {"role": "client", "text": "Питання"},
                {"role": "agent", "text": "Відповідь"},
                {"role": "client", "text": "Уточнення"},
                {"role": "agent", "text": "Рішення"},
            ],
        )
        assert len(chat.messages) == 4

    def test_chat_too_few_messages_rejected(self):
        """Менше 4 повідомлень — помилка валідації."""
        with pytest.raises(ValidationError):
            Chat(
                id="chat_short",
                scenario={
                    "category": "payment_issue",
                    "case_type": "successful",
                },
                messages=[
                    {"role": "client", "text": "Привіт"},
                    {"role": "agent", "text": "Вітаю"},
                ],
            )

    def test_chat_too_many_messages_rejected(self):
        """Більше 20 повідомлень — помилка валідації."""
        messages = [
            {"role": "client" if i % 2 == 0 else "agent", "text": f"Повідомлення {i}"}
            for i in range(22)
        ]
        with pytest.raises(ValidationError):
            Chat(
                id="chat_long",
                scenario={
                    "category": "payment_issue",
                    "case_type": "successful",
                },
                messages=messages,
            )

    def test_chat_id_required(self):
        with pytest.raises(ValidationError):
            Chat(
                scenario={
                    "category": "payment_issue",
                    "case_type": "successful",
                },
                messages=[
                    {"role": "client", "text": "Текст"},
                    {"role": "agent", "text": "Текст"},
                    {"role": "client", "text": "Текст"},
                    {"role": "agent", "text": "Текст"},
                ],
            )


# ── AnalysisResult ───────────────────────────────────────────────────

class TestAnalysisResult:
    def test_valid_result(self, sample_analysis_result):
        result = AnalysisResult(**sample_analysis_result)
        assert result.intent == Category.PAYMENT_ISSUE
        assert result.satisfaction == Satisfaction.SATISFIED
        assert result.quality_score == 5
        assert result.agent_mistakes == []

    def test_result_with_mistakes(self, sample_analysis_result_negative):
        result = AnalysisResult(**sample_analysis_result_negative)
        assert result.satisfaction == Satisfaction.UNSATISFIED
        assert result.quality_score == 2
        assert len(result.agent_mistakes) == 2

    def test_quality_score_min_boundary(self):
        """quality_score = 1 повинен бути валідним."""
        result = AnalysisResult(
            chat_id="test",
            intent="payment_issue",
            satisfaction="unsatisfied",
            quality_score=1,
            agent_mistakes=["rude_tone"],
            summary="Тест",
        )
        assert result.quality_score == 1

    def test_quality_score_max_boundary(self):
        """quality_score = 5 повинен бути валідним."""
        result = AnalysisResult(
            chat_id="test",
            intent="payment_issue",
            satisfaction="satisfied",
            quality_score=5,
            agent_mistakes=[],
            summary="Тест",
        )
        assert result.quality_score == 5

    def test_quality_score_below_min_rejected(self):
        """quality_score = 0 повинен бути відхилений."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                chat_id="test",
                intent="payment_issue",
                satisfaction="satisfied",
                quality_score=0,
                agent_mistakes=[],
                summary="Тест",
            )

    def test_quality_score_above_max_rejected(self):
        """quality_score = 6 повинен бути відхилений."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                chat_id="test",
                intent="payment_issue",
                satisfaction="satisfied",
                quality_score=6,
                agent_mistakes=[],
                summary="Тест",
            )

    def test_invalid_satisfaction_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                chat_id="test",
                intent="payment_issue",
                satisfaction="happy",
                quality_score=5,
                agent_mistakes=[],
                summary="Тест",
            )

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                chat_id="test",
                intent="invalid_intent",
                satisfaction="satisfied",
                quality_score=5,
                agent_mistakes=[],
                summary="Тест",
            )

    def test_invalid_agent_mistake_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                chat_id="test",
                intent="payment_issue",
                satisfaction="satisfied",
                quality_score=3,
                agent_mistakes=["nonexistent_mistake"],
                summary="Тест",
            )

    def test_validation_warnings_default_empty(self):
        result = AnalysisResult(
            chat_id="test",
            intent="payment_issue",
            satisfaction="satisfied",
            quality_score=5,
            agent_mistakes=[],
            summary="Тест",
        )
        assert result.validation_warnings == []

    def test_validation_warnings_accepted(self):
        result = AnalysisResult(
            chat_id="test",
            intent="payment_issue",
            satisfaction="unsatisfied",
            quality_score=2,
            agent_mistakes=["rude_tone"],
            summary="Тест",
            validation_warnings=["rude_tone_but_score_above_2"],
        )
        assert len(result.validation_warnings) == 1
