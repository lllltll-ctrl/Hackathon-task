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

    def test_prompt_includes_variation_context(self):
        """Variation context should be injected into the prompt."""
        ctx = {
            "persona": "a startup founder managing 12 users",
            "specific_detail": "payment error code E-4012",
            "situation": "team losing access to projects",
        }
        prompt = build_generation_prompt(
            category="payment_issue",
            case_type="successful",
            has_hidden_dissatisfaction=False,
            agent_mistakes=[],
            variation_context=ctx,
        )
        assert "startup founder" in prompt
        assert "E-4012" in prompt
        assert "losing access" in prompt

    def test_prompts_differ_across_variations(self):
        """Different variation_index + context should produce different prompts."""
        from config import VARIATION_CONTEXTS
        contexts = VARIATION_CONTEXTS["payment_issue"]
        prompts = []
        for i in range(3):
            p = build_generation_prompt(
                category="payment_issue",
                case_type="successful",
                has_hidden_dissatisfaction=False,
                agent_mistakes=[],
                variation_index=i,
                variation_context=contexts[i],
            )
            prompts.append(p)
        # All 3 prompts should be different
        assert prompts[0] != prompts[1]
        assert prompts[1] != prompts[2]
        assert prompts[0] != prompts[2]

    def test_hidden_dissatisfaction_varies_by_index(self):
        """Different variation_index should produce different dissatisfaction patterns."""
        prompts = []
        for i in range(3):
            p = build_generation_prompt(
                category="payment_issue",
                case_type="problematic",
                has_hidden_dissatisfaction=True,
                agent_mistakes=[],
                variation_index=i,
            )
            prompts.append(p)
        assert prompts[0] != prompts[1]
        assert prompts[1] != prompts[2]

    def test_prompt_includes_mixed_intent(self):
        """Mixed intent scenario details should be in the prompt."""
        mixed = {
            "apparent_category": "payment_issue",
            "actual_category": "technical_error",
            "description": "Client reports payment failure, but root cause is a UI bug",
        }
        prompt = build_generation_prompt(
            category="technical_error",
            case_type="problematic",
            has_hidden_dissatisfaction=False,
            agent_mistakes=[],
            mixed_intent=mixed,
        )
        assert "mixed intent" in prompt.lower() or "Mixed intent" in prompt
        assert "UI bug" in prompt


class TestAnalysisPrompts:
    """Tests for dialog analysis prompts."""

    def test_system_prompt_is_not_empty(self):
        assert len(ANALYSIS_SYSTEM) > 50

    def test_system_prompt_mentions_hidden_dissatisfaction(self):
        lower = ANALYSIS_SYSTEM.lower()
        assert "hidden" in lower or "dissatisfaction" in lower

    def test_system_prompt_uses_semantic_indicators(self):
        """Analysis prompt should use behavioral indicators, not exact phrases."""
        lower = ANALYSIS_SYSTEM.lower()
        # Should contain semantic analysis terms
        assert "behavioral" in lower or "outcome" in lower or "disengages" in lower
        # Should NOT contain the exact generation phrases (tautological markers)
        assert "i'll try to figure it out myself" not in lower
        assert "thanks for the information" not in lower

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
