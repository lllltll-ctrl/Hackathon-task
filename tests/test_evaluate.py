"""Тести для evaluate.py — ground truth evaluation."""


from evaluate import (
    evaluate_confidence_calibration,
    evaluate_hidden_dissatisfaction,
    evaluate_intent_accuracy,
    evaluate_mistake_detection,
    evaluate_quality_consistency,
    grade_evaluation,
    run_evaluation,
)


def _make_result(
    chat_id: str,
    intent: str,
    satisfaction: str,
    quality_score: int,
    agent_mistakes: list[str],
    ground_truth: dict | None = None,
) -> dict:
    r = {
        "chat_id": chat_id,
        "intent": intent,
        "satisfaction": satisfaction,
        "quality_score": quality_score,
        "agent_mistakes": agent_mistakes,
        "summary": "Test",
    }
    if ground_truth is not None:
        r["ground_truth"] = ground_truth
    return r


class TestIntentAccuracy:
    """Test intent detection accuracy calculation."""

    def test_perfect_accuracy(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 5, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
            _make_result("c2", "technical_error", "neutral", 3, [],
                         {"expected_intent": "technical_error", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "problematic", "mixed_intent": None}),
        ]
        report = evaluate_intent_accuracy(results)
        assert report["accuracy"] == 1.0
        assert report["correct"] == 2
        assert report["total"] == 2
        assert report["mismatches"] == []

    def test_partial_accuracy(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 5, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
            _make_result("c2", "payment_issue", "neutral", 3, [],
                         {"expected_intent": "technical_error", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "problematic", "mixed_intent": None}),
        ]
        report = evaluate_intent_accuracy(results)
        assert report["accuracy"] == 0.5
        assert report["correct"] == 1
        assert len(report["mismatches"]) == 1

    def test_no_ground_truth_skipped(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 5, []),
        ]
        report = evaluate_intent_accuracy(results)
        assert report["total"] == 0
        assert report["accuracy"] == 0.0

    def test_mixed_intent_mismatch_flagged(self):
        results = [
            _make_result("c1", "payment_issue", "neutral", 3, [],
                         {"expected_intent": "technical_error", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "problematic",
                          "mixed_intent": {"apparent_category": "payment_issue", "actual_category": "technical_error"}}),
        ]
        report = evaluate_intent_accuracy(results)
        assert report["mismatches"][0]["has_mixed_intent"] is True


class TestHiddenDissatisfactionDetection:
    """Test hidden dissatisfaction detection evaluation."""

    def test_detected_correctly(self):
        results = [
            _make_result("c1", "payment_issue", "unsatisfied", 2, ["no_resolution"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": True,
                          "intended_agent_mistakes": ["no_resolution"], "case_type": "problematic", "mixed_intent": None}),
        ]
        report = evaluate_hidden_dissatisfaction(results)
        assert report["total_hidden"] == 1
        assert report["detected"] == 1
        assert report["detection_rate"] == 1.0

    def test_neutral_counts_as_detected(self):
        results = [
            _make_result("c1", "payment_issue", "neutral", 3, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": True,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
        ]
        report = evaluate_hidden_dissatisfaction(results)
        assert report["detected"] == 1

    def test_missed_hidden_dissatisfaction(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 4, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": True,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
        ]
        report = evaluate_hidden_dissatisfaction(results)
        assert report["detected"] == 0
        assert report["detection_rate"] == 0.0
        assert len(report["missed_cases"]) == 1

    def test_false_positive_tracking(self):
        results = [
            _make_result("c1", "payment_issue", "unsatisfied", 2, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
        ]
        report = evaluate_hidden_dissatisfaction(results)
        assert report["false_positives"] == 1
        assert report["false_positive_rate"] == 1.0


class TestMistakeDetection:
    """Test agent mistake detection precision/recall."""

    def test_perfect_detection(self):
        results = [
            _make_result("c1", "payment_issue", "unsatisfied", 2,
                         ["no_resolution", "ignored_question"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": ["no_resolution", "ignored_question"],
                          "case_type": "agent_error", "mixed_intent": None}),
        ]
        report = evaluate_mistake_detection(results)
        assert report["precision"] == 1.0
        assert report["recall"] == 1.0
        assert report["f1_score"] == 1.0

    def test_partial_recall(self):
        results = [
            _make_result("c1", "payment_issue", "unsatisfied", 2,
                         ["no_resolution"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": ["no_resolution", "ignored_question"],
                          "case_type": "agent_error", "mixed_intent": None}),
        ]
        report = evaluate_mistake_detection(results)
        assert report["recall"] == 0.5
        assert report["precision"] == 1.0

    def test_extra_detected_lowers_precision(self):
        results = [
            _make_result("c1", "payment_issue", "unsatisfied", 2,
                         ["no_resolution", "rude_tone"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": ["no_resolution"],
                          "case_type": "agent_error", "mixed_intent": None}),
        ]
        report = evaluate_mistake_detection(results)
        assert report["recall"] == 1.0
        assert report["precision"] == 0.5

    def test_no_expected_mistakes(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 5, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
        ]
        report = evaluate_mistake_detection(results)
        assert report["total_expected_mistakes"] == 0
        assert report["detected_in_gt_chats"] == 0

    def test_extra_mistakes_in_non_gt_chats_ignored_for_precision(self):
        """Detected mistakes in chats without intended mistakes don't affect precision."""
        results = [
            # Chat WITH intended mistakes — precision counts here
            _make_result("c1", "payment_issue", "unsatisfied", 2,
                         ["no_resolution"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": ["no_resolution"],
                          "case_type": "agent_error", "mixed_intent": None}),
            # Chat WITHOUT intended mistakes but GPT found some — should NOT lower precision
            _make_result("c2", "payment_issue", "neutral", 3,
                         ["generic_response", "no_resolution"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [],
                          "case_type": "problematic", "mixed_intent": None}),
        ]
        report = evaluate_mistake_detection(results)
        assert report["recall"] == 1.0
        assert report["precision"] == 1.0  # Only c1 counts for precision


class TestQualityConsistency:
    """Test quality score consistency by case type."""

    def test_score_averages_by_case_type(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 5, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
            _make_result("c2", "payment_issue", "satisfied", 4, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
            _make_result("c3", "payment_issue", "unsatisfied", 1, ["rude_tone"],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": ["rude_tone"], "case_type": "agent_error", "mixed_intent": None}),
        ]
        report = evaluate_quality_consistency(results)
        assert report["average_score_by_case_type"]["successful"] == 4.5
        assert report["average_score_by_case_type"]["agent_error"] == 1.0
        assert report["sample_counts"]["successful"] == 2


class TestConfidenceCalibration:
    """Test confidence calibration metrics."""

    def test_correct_higher_confidence(self):
        results = [
            {**_make_result("c1", "payment_issue", "satisfied", 5, [],
                            {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                             "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
             "confidence": 0.95},
            {**_make_result("c2", "technical_error", "neutral", 3, [],
                            {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                             "intended_agent_mistakes": [], "case_type": "problematic", "mixed_intent": None}),
             "confidence": 0.4},
        ]
        report = evaluate_confidence_calibration(results)
        assert report["avg_confidence_correct"] == 0.95
        assert report["avg_confidence_incorrect"] == 0.4
        assert report["calibration_gap"] > 0

    def test_no_ground_truth_empty(self):
        results = [_make_result("c1", "payment_issue", "satisfied", 5, [])]
        report = evaluate_confidence_calibration(results)
        assert report["total_correct"] == 0
        assert report["total_incorrect"] == 0


class TestGradeEvaluation:
    """Test threshold grading."""

    def test_all_pass(self):
        evaluation = {
            "intent_accuracy": {"accuracy": 0.90},
            "hidden_dissatisfaction": {"detection_rate": 0.80},
            "mistake_detection": {"recall": 0.75},
        }
        grades = grade_evaluation(evaluation)
        assert grades["intent_accuracy"] == "PASS"
        assert grades["hidden_dissatisfaction_detection"] == "PASS"
        assert grades["mistake_recall"] == "PASS"

    def test_warn_and_fail(self):
        evaluation = {
            "intent_accuracy": {"accuracy": 0.72},
            "hidden_dissatisfaction": {"detection_rate": 0.40},
            "mistake_detection": {"recall": 0.55},
        }
        grades = grade_evaluation(evaluation)
        assert grades["intent_accuracy"] == "WARN"
        assert grades["hidden_dissatisfaction_detection"] == "FAIL"
        assert grades["mistake_recall"] == "WARN"


class TestEdgeCases:
    """Edge cases for evaluation functions."""

    def test_empty_results(self):
        report = evaluate_intent_accuracy([])
        assert report["total"] == 0
        assert report["accuracy"] == 0.0

    def test_empty_results_hidden(self):
        report = evaluate_hidden_dissatisfaction([])
        assert report["total_hidden"] == 0
        assert report["detection_rate"] == 0.0

    def test_empty_results_mistakes(self):
        report = evaluate_mistake_detection([])
        assert report["total_expected_mistakes"] == 0
        assert report["f1_score"] == 0.0

    def test_empty_results_quality(self):
        report = evaluate_quality_consistency([])
        assert report["average_score_by_case_type"] == {}


class TestRunEvaluation:
    """Test full evaluation pipeline."""

    def test_returns_all_sections(self):
        results = [
            _make_result("c1", "payment_issue", "satisfied", 5, [],
                         {"expected_intent": "payment_issue", "has_hidden_dissatisfaction": False,
                          "intended_agent_mistakes": [], "case_type": "successful", "mixed_intent": None}),
        ]
        evaluation = run_evaluation(results)
        assert "intent_accuracy" in evaluation
        assert "hidden_dissatisfaction" in evaluation
        assert "mistake_detection" in evaluation
        assert "quality_consistency" in evaluation
        assert "confidence_calibration" in evaluation
        assert "grades" in evaluation
