"""Tests for generation and analysis prompts."""


from prompts.analysis import SYSTEM_PROMPT as ANALYSIS_SYSTEM
from prompts.analysis import build_analysis_prompt
from prompts.generation import SYSTEM_PROMPT as GEN_SYSTEM
from prompts.generation import build_generation_prompt


class TestGenerationPrompts:
    """Tests for dialog generation prompts."""

    def test_system_prompt_is_not_empty(self):
        assert len(GEN_SYSTEM) > 50

    def test_system_prompt_mentions_english(self):
        assert "english" in GEN_SYSTEM.lower()

    def test_build_prompt_returns_string(self):
        prompt = build_generation_prompt(
            category="payment_issue",
            case_type="successful",
            has_hidden_dissatisfaction=False,
            agent_mistakes=[],
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_category(self):
        prompt = build_generation_prompt(
            category="technical_error",
            case_type="conflict",
            has_hidden_dissatisfaction=False,
            agent_mistakes=[],
        )
        assert "technical" in prompt.lower() or "technical_error" in prompt

    def test_prompt_contains_case_type(self):
        prompt = build_generation_prompt(
            category="payment_issue",
            case_type="conflict",
            has_hidden_dissatisfaction=False,
            agent_mistakes=[],
        )
        assert "conflict" in prompt.lower()

    def test_prompt_includes_hidden_dissatisfaction_flag(self):
        prompt = build_generation_prompt(
            category="payment_issue",
            case_type="problematic",
            has_hidden_dissatisfaction=True,
            agent_mistakes=[],
        )
        lower = prompt.lower()
        assert "hidden" in lower or "dissatisfaction" in lower

    def test_prompt_includes_agent_mistakes(self):
        prompt = build_generation_prompt(
            category="payment_issue",
            case_type="agent_error",
            has_hidden_dissatisfaction=False,
            agent_mistakes=["rude_tone", "incorrect_info"],
        )
        assert "rude" in prompt.lower()
        assert "incorrect" in prompt.lower()

    def test_prompt_requests_json_format(self):
        prompt = build_generation_prompt(
            category="payment_issue",
            case_type="successful",
            has_hidden_dissatisfaction=False,
            agent_mistakes=[],
        )
        assert "json" in prompt.lower() or "JSON" in prompt


class TestAnalysisPrompts:
    """Tests for dialog analysis prompts."""

    def test_system_prompt_is_not_empty(self):
        assert len(ANALYSIS_SYSTEM) > 50

    def test_system_prompt_mentions_hidden_dissatisfaction(self):
        lower = ANALYSIS_SYSTEM.lower()
        assert "hidden" in lower or "dissatisfaction" in lower

    def test_build_prompt_returns_string(self):
        dialogue = [
            {"role": "client", "text": "Hello"},
            {"role": "agent", "text": "Hi there"},
            {"role": "client", "text": "Thanks"},
            {"role": "agent", "text": "You're welcome"},
        ]
        prompt = build_analysis_prompt(dialogue)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_dialogue_text(self):
        dialogue = [
            {"role": "client", "text": "My payment is not going through"},
            {"role": "agent", "text": "Let me check that for you"},
            {"role": "client", "text": "Thanks"},
            {"role": "agent", "text": "Happy to help"},
        ]
        prompt = build_analysis_prompt(dialogue)
        assert "payment is not going through" in prompt

    def test_prompt_specifies_all_intents(self):
        dialogue = [
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
        ]
        prompt = build_analysis_prompt(dialogue)
        for intent in ["payment_issue", "technical_error", "account_access",
                        "tariff_question", "refund_request", "other"]:
            assert intent in prompt, f"Prompt does not contain intent: {intent}"

    def test_prompt_specifies_satisfaction_levels(self):
        dialogue = [
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
        ]
        prompt = build_analysis_prompt(dialogue)
        for level in ["satisfied", "neutral", "unsatisfied"]:
            assert level in prompt, f"Prompt does not contain satisfaction level: {level}"

    def test_prompt_specifies_quality_scale(self):
        dialogue = [
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
        ]
        prompt = build_analysis_prompt(dialogue)
        assert "1" in prompt and "5" in prompt

    def test_prompt_specifies_agent_mistakes(self):
        dialogue = [
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
        ]
        prompt = build_analysis_prompt(dialogue)
        for mistake in ["ignored_question", "incorrect_info", "rude_tone",
                         "no_resolution", "unnecessary_escalation"]:
            assert mistake in prompt, f"Prompt does not contain mistake: {mistake}"

    def test_prompt_requests_json_format(self):
        dialogue = [
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
            {"role": "client", "text": "Text"},
            {"role": "agent", "text": "Text"},
        ]
        prompt = build_analysis_prompt(dialogue)
        assert "json" in prompt.lower() or "JSON" in prompt
