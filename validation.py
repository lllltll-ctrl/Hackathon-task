"""Post-processing validation layer for analysis results consistency.

Applies deterministic rules to ensure logical consistency between
quality_score, agent_mistakes, and satisfaction fields.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidationWarning:
    """A single validation inconsistency."""

    def __init__(
        self,
        field: str,
        rule: str,
        original_value: Any,
        corrected_value: Any | None = None,
    ):
        self.field = field
        self.rule = rule
        self.original_value = original_value
        self.corrected_value = corrected_value

    def __repr__(self) -> str:
        if self.corrected_value is not None:
            return f"ValidationWarning({self.field}: {self.rule}, {self.original_value} -> {self.corrected_value})"
        return f"ValidationWarning({self.field}: {self.rule}, value={self.original_value})"


def validate_analysis_result(
    result: dict[str, Any],
) -> tuple[dict[str, Any], list[ValidationWarning]]:
    """Validate and correct logical inconsistencies in analysis result.

    Rules applied (in order):
    1. If agent_mistakes is non-empty, quality_score should be <= 3
    2. If 'rude_tone' in mistakes, quality_score should be <= 2
    3. If 'no_resolution' in mistakes, satisfaction should NOT be 'satisfied'
    4. If satisfaction == 'satisfied' and quality_score <= 2, flag anomaly
    5. If no mistakes and quality_score <= 2, flag anomaly
    6. If quality_score >= 4 and satisfaction == 'unsatisfied', cap score at 3
    7. If 3 or more agent_mistakes, cap quality_score at 1

    Args:
        result: analysis result dict from LLM

    Returns:
        Tuple of (corrected_result, list_of_warnings)
    """
    corrected = dict(result)
    warnings: list[ValidationWarning] = []

    mistakes: list[str] = corrected.get("agent_mistakes", [])
    score: int = corrected.get("quality_score", 3)
    satisfaction: str = corrected.get("satisfaction", "neutral")

    # Rule 1: mistakes present -> score cannot be 4 or 5
    if mistakes and score > 3:
        warnings.append(ValidationWarning(
            "quality_score", "mistakes_present_but_high_score",
            score, min(score, 3),
        ))
        corrected["quality_score"] = min(score, 3)
        score = corrected["quality_score"]

    # Rule 2: rude_tone -> score <= 2
    if "rude_tone" in mistakes and score > 2:
        warnings.append(ValidationWarning(
            "quality_score", "rude_tone_but_score_above_2",
            score, 2,
        ))
        corrected["quality_score"] = 2
        score = corrected["quality_score"]

    # Rule 3: no_resolution -> satisfaction != satisfied
    if "no_resolution" in mistakes and satisfaction == "satisfied":
        warnings.append(ValidationWarning(
            "satisfaction", "no_resolution_but_satisfied",
            satisfaction, "unsatisfied",
        ))
        corrected["satisfaction"] = "unsatisfied"
        satisfaction = corrected["satisfaction"]

    # Rule 4: satisfied but low score -> anomaly flag (no auto-correction)
    if satisfaction == "satisfied" and score <= 2:
        warnings.append(ValidationWarning(
            "satisfaction", "satisfied_but_low_score",
            satisfaction, None,
        ))

    # Rule 5: no mistakes but very low score -> anomaly flag (no auto-correction)
    if not mistakes and score <= 2:
        warnings.append(ValidationWarning(
            "quality_score", "no_mistakes_but_low_score",
            score, None,
        ))

    # Rule 6: high score but unsatisfied -> cap at 3
    if score >= 4 and satisfaction == "unsatisfied":
        warnings.append(ValidationWarning(
            "quality_score", "high_score_but_unsatisfied",
            score, 3,
        ))
        corrected["quality_score"] = 3
        score = corrected["quality_score"]

    # Rule 7: 3+ mistakes -> critical failure, cap at 1
    if len(mistakes) >= 3 and score > 1:
        warnings.append(ValidationWarning(
            "quality_score", "multiple_mistakes_critical",
            score, 1,
        ))
        corrected["quality_score"] = 1

    return corrected, warnings
