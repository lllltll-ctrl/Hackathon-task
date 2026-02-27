"""Evaluate analysis results against scenario ground truth.

Compares LLM analysis output with known scenario parameters to measure
detection accuracy for intent, satisfaction, and agent mistakes.

Usage:
    python evaluate.py [--chats data/chats.json] [--analysis results/analysis.json] [--output results/evaluation.json]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from config import CATEGORIES, DEFAULT_OUTPUT_PATH, DEFAULT_RESULTS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_EVALUATION_PATH: str = "results/evaluation.json"


def load_analysis_results(path: str) -> list[dict[str, Any]]:
    """Load analysis results with embedded ground truth."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    if "results" not in data:
        raise ValueError("File does not contain 'results' field")
    return data["results"]


def load_chats_dataset(path: str) -> dict[str, dict[str, Any]]:
    """Load chats dataset and index by chat id for ground truth lookup."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    if "chats" not in data:
        raise ValueError("File does not contain 'chats' field")
    return {chat["id"]: chat for chat in data["chats"]}


def _get_ground_truth(
    result: dict[str, Any],
    chats_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Get ground truth from result or from chats dataset."""
    gt = result.get("ground_truth")
    if gt is not None:
        return gt
    if chats_index is None:
        return None
    chat = chats_index.get(result["chat_id"])
    if chat is None:
        return None
    scenario = chat.get("scenario")
    if scenario is None:
        return None
    return {
        "expected_intent": scenario.get("category"),
        "has_hidden_dissatisfaction": scenario.get("has_hidden_dissatisfaction", False),
        "intended_agent_mistakes": scenario.get("intended_agent_mistakes", []),
        "case_type": scenario.get("case_type"),
        "mixed_intent": scenario.get("mixed_intent"),
    }


def evaluate_intent_accuracy(
    results: list[dict[str, Any]],
    chats_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate intent detection accuracy and per-category breakdown.

    For mixed intent scenarios, the actual_category from the scenario
    is considered the correct intent.
    """
    total = 0
    correct = 0
    per_category: dict[str, dict[str, int]] = {
        cat: {"total": 0, "correct": 0} for cat in CATEGORIES
    }
    confusion: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, str]] = []

    for result in results:
        gt = _get_ground_truth(result, chats_index)
        if gt is None:
            continue

        expected = gt["expected_intent"]
        predicted = result["intent"]
        total += 1

        if expected not in confusion:
            confusion[expected] = {}
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

        if expected in per_category:
            per_category[expected]["total"] += 1

        if predicted == expected:
            correct += 1
            if expected in per_category:
                per_category[expected]["correct"] += 1
        else:
            mismatches.append({
                "chat_id": result["chat_id"],
                "expected": expected,
                "predicted": predicted,
                "has_mixed_intent": gt.get("mixed_intent") is not None,
            })

    accuracy = correct / total if total > 0 else 0.0

    per_category_accuracy: dict[str, float | str] = {}
    for cat, counts in per_category.items():
        if counts["total"] > 0:
            per_category_accuracy[cat] = round(counts["correct"] / counts["total"], 3)
        else:
            per_category_accuracy[cat] = "N/A"

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "per_category_accuracy": per_category_accuracy,
        "confusion_matrix": confusion,
        "mismatches": mismatches[:20],
    }


def evaluate_hidden_dissatisfaction(
    results: list[dict[str, Any]],
    chats_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate ability to detect hidden dissatisfaction.

    Ground truth: scenario.has_hidden_dissatisfaction == True
    Expected: satisfaction should NOT be 'satisfied'
    """
    total_hidden = 0
    detected = 0
    missed: list[dict[str, str]] = []

    total_normal = 0
    false_positives = 0

    for result in results:
        gt = _get_ground_truth(result, chats_index)
        if gt is None:
            continue

        satisfaction = result["satisfaction"]

        if gt["has_hidden_dissatisfaction"]:
            total_hidden += 1
            if satisfaction != "satisfied":
                detected += 1
            else:
                missed.append({
                    "chat_id": result["chat_id"],
                    "satisfaction": satisfaction,
                    "case_type": gt.get("case_type", ""),
                })
        else:
            # For successful cases without hidden dissatisfaction,
            # being satisfied is the expected outcome
            if gt.get("case_type") == "successful":
                total_normal += 1
                if satisfaction == "unsatisfied":
                    false_positives += 1

    detection_rate = detected / total_hidden if total_hidden > 0 else 0.0
    false_positive_rate = false_positives / total_normal if total_normal > 0 else 0.0

    return {
        "total_hidden": total_hidden,
        "detected": detected,
        "detection_rate": round(detection_rate, 3),
        "missed_cases": missed[:20],
        "total_normal_successful": total_normal,
        "false_positives": false_positives,
        "false_positive_rate": round(false_positive_rate, 3),
    }


def evaluate_mistake_detection(
    results: list[dict[str, Any]],
    chats_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate agent mistake detection recall and precision.

    Recall: of all intended mistakes, how many were detected?
    Precision: computed only over chats that HAVE intended mistakes,
    because chats with no intended mistakes (successful, problematic, conflict)
    can legitimately surface additional issues like no_resolution or generic_response
    that are not false positives but emergent findings.
    """
    total_expected = 0
    true_positives = 0
    # For precision: only count detected mistakes in chats that have ground truth mistakes
    detected_in_gt_chats = 0
    per_mistake: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []

    for result in results:
        gt = _get_ground_truth(result, chats_index)
        if gt is None:
            continue

        expected_mistakes = set(gt.get("intended_agent_mistakes", []))
        detected_mistakes = set(result.get("agent_mistakes", []))

        total_expected += len(expected_mistakes)

        # Only count detected mistakes toward precision when ground truth has mistakes
        if expected_mistakes:
            detected_in_gt_chats += len(detected_mistakes)

        for mistake in expected_mistakes:
            if mistake not in per_mistake:
                per_mistake[mistake] = {"expected": 0, "detected": 0}
            per_mistake[mistake]["expected"] += 1
            if mistake in detected_mistakes:
                true_positives += 1
                per_mistake[mistake]["detected"] += 1

        if expected_mistakes != detected_mistakes:
            mismatches.append({
                "chat_id": result["chat_id"],
                "expected": sorted(expected_mistakes),
                "detected": sorted(detected_mistakes),
            })

    recall = true_positives / total_expected if total_expected > 0 else 0.0
    precision = true_positives / detected_in_gt_chats if detected_in_gt_chats > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_mistake_recall: dict[str, float | str] = {}
    for mistake, counts in per_mistake.items():
        if counts["expected"] > 0:
            per_mistake_recall[mistake] = round(counts["detected"] / counts["expected"], 3)
        else:
            per_mistake_recall[mistake] = "N/A"

    return {
        "total_expected_mistakes": total_expected,
        "detected_in_gt_chats": detected_in_gt_chats,
        "true_positives": true_positives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "per_mistake_recall": per_mistake_recall,
        "mismatches": mismatches[:20],
    }


def evaluate_quality_consistency(
    results: list[dict[str, Any]],
    chats_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate quality score consistency with case types.

    Successful cases should have higher scores, agent_error lower.
    """
    by_case_type: dict[str, list[int]] = {}

    for result in results:
        gt = _get_ground_truth(result, chats_index)
        if gt is None:
            continue

        case_type = gt.get("case_type", "unknown")
        score = result["quality_score"]

        if case_type not in by_case_type:
            by_case_type[case_type] = []
        by_case_type[case_type].append(score)

    averages: dict[str, float] = {}
    for case_type, scores in by_case_type.items():
        averages[case_type] = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "average_score_by_case_type": averages,
        "sample_counts": {ct: len(scores) for ct, scores in by_case_type.items()},
    }


def evaluate_confidence_calibration(
    results: list[dict[str, Any]],
    chats_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate whether confidence scores correlate with correctness.

    Lower confidence should correlate with cases where the model is wrong
    (especially hidden dissatisfaction misses).
    """
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []

    for result in results:
        gt = _get_ground_truth(result, chats_index)
        if gt is None:
            continue

        confidence = result.get("confidence", 0.8)
        is_correct = result["intent"] == gt["expected_intent"]

        if gt["has_hidden_dissatisfaction"]:
            is_correct = is_correct and result["satisfaction"] != "satisfied"

        if is_correct:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

    avg_correct = (
        round(sum(correct_confidences) / len(correct_confidences), 3)
        if correct_confidences else 0.0
    )
    avg_incorrect = (
        round(sum(incorrect_confidences) / len(incorrect_confidences), 3)
        if incorrect_confidences else 0.0
    )

    return {
        "avg_confidence_correct": avg_correct,
        "avg_confidence_incorrect": avg_incorrect,
        "calibration_gap": round(avg_correct - avg_incorrect, 3),
        "total_correct": len(correct_confidences),
        "total_incorrect": len(incorrect_confidences),
    }


# ── Threshold grading ────────────────────────────────────────────────

THRESHOLDS: dict[str, dict[str, float]] = {
    "intent_accuracy": {"pass": 0.85, "warn": 0.70},
    "hidden_dissatisfaction_detection": {"pass": 0.75, "warn": 0.50},
    "mistake_recall": {"pass": 0.70, "warn": 0.50},
}


def _grade(value: float, metric: str) -> str:
    """Return PASS / WARN / FAIL for a metric value."""
    t = THRESHOLDS.get(metric)
    if t is None:
        return "N/A"
    if value >= t["pass"]:
        return "PASS"
    if value >= t["warn"]:
        return "WARN"
    return "FAIL"


def grade_evaluation(evaluation: dict[str, Any]) -> dict[str, str]:
    """Apply threshold grading to evaluation results.

    For mistake detection, recall is the primary metric because the analyzer
    legitimately finds additional mistakes beyond what was seeded in scenarios
    (e.g. no_resolution in agent_error chats). High recall means the intended
    mistakes are reliably detected.
    """
    return {
        "intent_accuracy": _grade(
            evaluation["intent_accuracy"]["accuracy"], "intent_accuracy",
        ),
        "hidden_dissatisfaction_detection": _grade(
            evaluation["hidden_dissatisfaction"]["detection_rate"],
            "hidden_dissatisfaction_detection",
        ),
        "mistake_recall": _grade(
            evaluation["mistake_detection"]["recall"], "mistake_recall",
        ),
    }


def run_evaluation(
    results: list[dict[str, Any]],
    chats_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all evaluation metrics."""
    evaluation = {
        "intent_accuracy": evaluate_intent_accuracy(results, chats_index),
        "hidden_dissatisfaction": evaluate_hidden_dissatisfaction(results, chats_index),
        "mistake_detection": evaluate_mistake_detection(results, chats_index),
        "quality_consistency": evaluate_quality_consistency(results, chats_index),
        "confidence_calibration": evaluate_confidence_calibration(results, chats_index),
    }
    evaluation["grades"] = grade_evaluation(evaluation)
    return evaluation


def save_evaluation(evaluation: dict[str, Any], output_path: str) -> None:
    """Save evaluation report to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output: dict[str, Any] = {
        "metadata": {
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "evaluation": evaluation,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation saved: {output_path}")


def print_evaluation(evaluation: dict[str, Any]) -> None:
    """Print human-readable evaluation report."""
    print("\n" + "=" * 60)
    print("GROUND TRUTH EVALUATION REPORT")
    print("=" * 60)

    # Intent accuracy
    intent = evaluation["intent_accuracy"]
    print(f"\nIntent Detection Accuracy: {intent['correct']}/{intent['total']} ({intent['accuracy']:.1%})")
    print("  Per category:")
    for cat, acc in intent["per_category_accuracy"].items():
        if acc != "N/A":
            print(f"    {cat:20s}: {acc:.1%}")

    if intent["mismatches"]:
        mixed = sum(1 for m in intent["mismatches"] if m.get("has_mixed_intent"))
        print(f"  Mismatches: {len(intent['mismatches'])} total ({mixed} from mixed-intent scenarios)")

    # Hidden dissatisfaction
    hd = evaluation["hidden_dissatisfaction"]
    print(f"\nHidden Dissatisfaction Detection: {hd['detected']}/{hd['total_hidden']} ({hd['detection_rate']:.1%})")
    if hd["total_normal_successful"] > 0:
        print(f"  False positive rate (successful cases): {hd['false_positive_rate']:.1%}")
    if hd["missed_cases"]:
        print(f"  Missed cases: {len(hd['missed_cases'])}")

    # Mistake detection
    md = evaluation["mistake_detection"]
    print("\nAgent Mistake Detection:")
    print(f"  Precision: {md['precision']:.1%}")
    print(f"  Recall:    {md['recall']:.1%}")
    print(f"  F1 Score:  {md['f1_score']:.1%}")
    if md["per_mistake_recall"]:
        print("  Per mistake recall:")
        for mistake, recall in md["per_mistake_recall"].items():
            if recall != "N/A":
                print(f"    {mistake:25s}: {recall:.1%}")

    # Quality consistency
    qc = evaluation["quality_consistency"]
    print("\nAverage Quality Score by Case Type:")
    for ct, avg in qc["average_score_by_case_type"].items():
        count = qc["sample_counts"].get(ct, 0)
        print(f"  {ct:15s}: {avg:.2f} (n={count})")

    # Confidence calibration
    if "confidence_calibration" in evaluation:
        cc = evaluation["confidence_calibration"]
        print("\nConfidence Calibration:")
        print(f"  Avg confidence (correct):   {cc['avg_confidence_correct']:.3f}")
        print(f"  Avg confidence (incorrect): {cc['avg_confidence_incorrect']:.3f}")
        print(f"  Calibration gap:            {cc['calibration_gap']:.3f}")

    # Grades
    if "grades" in evaluation:
        grades = evaluation["grades"]
        print("\nOverall Grades:")
        for metric, grade in grades.items():
            status = {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]"}.get(grade, "[--]")
            print(f"  {status} {metric}: {grade}")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CX-Ray analysis results against ground truth"
    )
    parser.add_argument(
        "--chats", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Path to chats file for ground truth (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--analysis", type=str, default=DEFAULT_RESULTS_PATH,
        help=f"Path to analysis results (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_EVALUATION_PATH,
        help=f"Path to evaluation output (default: {DEFAULT_EVALUATION_PATH})",
    )
    args = parser.parse_args()

    try:
        results = load_analysis_results(args.analysis)
    except FileNotFoundError:
        logger.error(f"Analysis results not found: {args.analysis}")
        logger.error("Run analyze.py first to generate analysis results.")
        sys.exit(1)

    # Try to load chats for fallback ground truth
    chats_index: dict[str, dict[str, Any]] | None = None
    try:
        chats_index = load_chats_dataset(args.chats)
    except FileNotFoundError:
        logger.warning(f"Chats file not found: {args.chats}. Using embedded ground truth only.")

    # Check if any ground truth is available
    has_gt = any(
        r.get("ground_truth") is not None
        or (chats_index and r["chat_id"] in chats_index)
        for r in results
    )
    if not has_gt:
        logger.error("No ground truth available. Ensure analysis was run with scenario data.")
        sys.exit(1)

    evaluation = run_evaluation(results, chats_index)
    save_evaluation(evaluation, args.output)
    print_evaluation(evaluation)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
