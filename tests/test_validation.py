"""Тести для rule-based валідації результатів аналізу."""

from validation import ValidationWarning, validate_analysis_result


class TestValidateConsistentResult:
    """Результат без інконсистентностей проходить без змін."""

    def test_no_corrections_for_clean_result(self):
        result = {
            "chat_id": "chat_001",
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 5,
            "agent_mistakes": [],
            "summary": "Problem resolved.",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected == result
        assert warnings == []

    def test_no_corrections_for_consistent_negative_result(self):
        result = {
            "chat_id": "chat_002",
            "intent": "technical_error",
            "satisfaction": "unsatisfied",
            "quality_score": 2,
            "agent_mistakes": ["no_resolution", "ignored_question"],
            "summary": "Agent failed to resolve.",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 2
        assert corrected["satisfaction"] == "unsatisfied"
        assert warnings == []


class TestRule1MistakesPresentHighScore:
    """Rule 1: mistakes present → quality_score should be ≤ 3."""

    def test_score_5_with_mistakes_corrected_to_3(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "neutral",
            "quality_score": 5,
            "agent_mistakes": ["no_resolution"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 3
        assert len(warnings) >= 1
        assert any(w.rule == "mistakes_present_but_high_score" for w in warnings)

    def test_score_4_with_mistakes_corrected_to_3(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "neutral",
            "quality_score": 4,
            "agent_mistakes": ["ignored_question"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 3

    def test_score_3_with_mistakes_not_changed(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "neutral",
            "quality_score": 3,
            "agent_mistakes": ["no_resolution"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 3


class TestRule2RudeToneCapsScore:
    """Rule 2: rude_tone → quality_score should be ≤ 2."""

    def test_rude_tone_caps_score_at_2(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "unsatisfied",
            "quality_score": 4,
            "agent_mistakes": ["rude_tone"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 2
        assert any(w.rule == "rude_tone_but_score_above_2" for w in warnings)

    def test_rude_tone_score_1_not_changed(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "unsatisfied",
            "quality_score": 1,
            "agent_mistakes": ["rude_tone"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 1


class TestRule3NoResolutionSatisfaction:
    """Rule 3: no_resolution → satisfaction ≠ 'satisfied'."""

    def test_no_resolution_forces_unsatisfied(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 3,
            "agent_mistakes": ["no_resolution"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["satisfaction"] == "unsatisfied"
        assert any(w.rule == "no_resolution_but_satisfied" for w in warnings)

    def test_no_resolution_neutral_not_changed(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "neutral",
            "quality_score": 3,
            "agent_mistakes": ["no_resolution"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["satisfaction"] == "neutral"


class TestRule4SatisfiedLowScoreAnomaly:
    """Rule 4: satisfied + low score → anomaly flag (no auto-correction)."""

    def test_satisfied_low_score_flagged(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 2,
            "agent_mistakes": [],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        # No auto-correction for satisfaction
        assert corrected["satisfaction"] == "satisfied"
        assert any(w.rule == "satisfied_but_low_score" for w in warnings)


class TestRule5NoMistakesLowScoreAnomaly:
    """Rule 5: no mistakes + low score → anomaly flag (no auto-correction)."""

    def test_no_mistakes_low_score_flagged(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "neutral",
            "quality_score": 1,
            "agent_mistakes": [],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        assert corrected["quality_score"] == 1  # Not auto-corrected
        assert any(w.rule == "no_mistakes_but_low_score" for w in warnings)


class TestMultipleRulesApplied:
    """Multiple rules applied together."""

    def test_rude_tone_and_no_resolution_combined(self):
        result = {
            "chat_id": "test",
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 5,
            "agent_mistakes": ["rude_tone", "no_resolution"],
            "summary": "Test",
        }
        corrected, warnings = validate_analysis_result(result)
        # Rule 1: 5 -> 3, Rule 2: 3 -> 2, Rule 3: satisfied -> unsatisfied
        assert corrected["quality_score"] == 2
        assert corrected["satisfaction"] == "unsatisfied"
        assert len(warnings) >= 3


class TestValidationWarningRepr:
    """Test ValidationWarning string representation."""

    def test_repr_with_correction(self):
        w = ValidationWarning("quality_score", "test_rule", 5, 3)
        s = repr(w)
        assert "quality_score" in s
        assert "test_rule" in s

    def test_repr_without_correction(self):
        w = ValidationWarning("satisfaction", "anomaly", "satisfied", None)
        s = repr(w)
        assert "satisfaction" in s
        assert "anomaly" in s
