"""Project configuration: models, parameters, scenario matrix."""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── OpenAI API ───────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model for dialog generation (cheaper, sufficient for text)
GENERATION_MODEL: str = "gpt-4o-mini"

# Model for dialog analysis (more powerful, better understanding of nuances)
ANALYSIS_MODEL: str = "gpt-4o"

# Parameters for determinism
TEMPERATURE: int = 0
SEED: int = 42

# Timeout for API requests (seconds)
REQUEST_TIMEOUT: float = 60.0

# Checkpointing: save progress every N chats
CHECKPOINT_INTERVAL: int = 10
CHECKPOINT_PATH: str = "data/checkpoint.json"
CHECKPOINT_ANALYSIS_PATH: str = "results/checkpoint_analysis.json"

# ── Dataset ──────────────────────────────────────────────────────────

DEFAULT_CHAT_COUNT: int = 120
DEFAULT_OUTPUT_PATH: str = "data/chats.json"
DEFAULT_RESULTS_PATH: str = "results/analysis.json"

# ── Categories and types ─────────────────────────────────────────────

CATEGORIES: list[str] = [
    "payment_issue",
    "technical_error",
    "account_access",
    "tariff_question",
    "refund_request",
    "other",
]

CASE_TYPES: list[str] = [
    "successful",
    "problematic",
    "conflict",
    "agent_error",
]

AGENT_MISTAKES: list[str] = [
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
]

# Category descriptions (for prompts)
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "payment_issue": "Payment issues (card not going through, double charge, subscription payment not credited for CloudTask)",
    "technical_error": "Technical errors (error 500, API integration not working, UI bug, dashboard not loading)",
    "account_access": "Account access (forgotten password, locked account, SSO/2FA authentication issues)",
    "tariff_question": "Plan/pricing questions (difference between Free/Pro/Enterprise, plan change, feature limits, CloudTask subscription terms)",
    "refund_request": "Refund requests (refund for unused subscription period, auto-renewal cancellation, accidental charge)",
    "other": "Other inquiries (improvement suggestions, general questions about CloudTask features, service complaints)",
}

# Case type descriptions (for prompts)
CASE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "successful": "Successful case: agent quickly understands the problem, provides a clear solution, client is satisfied with the result",
    "problematic": "Problematic case: agent needs several clarifications, solution is not ideal, client is neutral or partially satisfied",
    "conflict": "Conflict case: client is emotional and dissatisfied, demands escalation or compensation, agent is under pressure",
    "agent_error": "Agent error case: agent makes specific mistakes (incorrect information, rude tone, ignoring questions, etc.)",
}

# Agent mistake descriptions (for prompts)
MISTAKE_DESCRIPTIONS: dict[str, str] = {
    "ignored_question": "Ignoring question — agent doesn't answer the client's specific question, changes the topic",
    "incorrect_info": "Incorrect information — agent provides false information about CloudTask plans, features, or procedures",
    "rude_tone": "Rude tone — agent responds dismissively, impatiently, or unprofessionally",
    "no_resolution": "No resolution — dialog ends without actually resolving the client's problem",
    "unnecessary_escalation": "Unnecessary escalation — agent redirects to another specialist without attempting to solve it themselves",
}


def _build_scenario_matrix() -> list[dict[str, Any]]:
    """Build the full scenario matrix for generating 120 dialogs.

    Distribution:
    - 60 main: 5 categories × 4 types × 3 variations
    - 20 with hidden dissatisfaction
    - 20 with agent mistakes (additional)
    - 20 mixed / edge cases
    """
    scenarios: list[dict[str, Any]] = []

    # Main categories (without "other")
    main_categories: list[str] = [c for c in CATEGORIES if c != "other"]

    # ── Block 1: Main matrix (5 categories × 4 types × 3 variations = 60) ──
    for category in main_categories:
        for case_type in CASE_TYPES:
            for variation in range(3):
                scenario: dict[str, Any] = {
                    "category": category,
                    "case_type": case_type,
                    "has_hidden_dissatisfaction": False,
                    "intended_agent_mistakes": [],
                }
                # For agent_error add specific mistakes
                if case_type == "agent_error":
                    mistake_idx = variation % len(AGENT_MISTAKES)
                    scenario["intended_agent_mistakes"] = [AGENT_MISTAKES[mistake_idx]]
                scenarios.append(scenario)

    # ── Block 2: Hidden dissatisfaction (20 cases) ──
    hidden_configs: list[tuple[str, str, list[str]]] = [
        # Client formally thanks, but problem not resolved
        ("payment_issue", "problematic", ["no_resolution"]),
        ("payment_issue", "successful", []),
        ("technical_error", "problematic", ["no_resolution"]),
        ("technical_error", "successful", []),
        ("account_access", "problematic", ["ignored_question"]),
        ("account_access", "successful", []),
        ("tariff_question", "problematic", ["incorrect_info"]),
        ("tariff_question", "successful", []),
        ("refund_request", "problematic", ["no_resolution"]),
        ("refund_request", "successful", []),
        # Agent gives template response
        ("payment_issue", "agent_error", ["no_resolution"]),
        ("technical_error", "agent_error", ["ignored_question"]),
        ("account_access", "agent_error", ["unnecessary_escalation"]),
        ("tariff_question", "agent_error", ["incorrect_info"]),
        ("refund_request", "agent_error", ["no_resolution"]),
        # Client "gives up"
        ("payment_issue", "conflict", []),
        ("technical_error", "conflict", []),
        ("account_access", "conflict", []),
        ("tariff_question", "conflict", []),
        ("refund_request", "conflict", []),
    ]
    for category, case_type, mistakes in hidden_configs:
        scenarios.append({
            "category": category,
            "case_type": case_type,
            "has_hidden_dissatisfaction": True,
            "intended_agent_mistakes": mistakes,
        })

    # ── Block 3: Additional agent mistakes (20 cases) ──
    mistake_combos: list[list[str]] = [
        ["ignored_question", "no_resolution"],
        ["incorrect_info", "rude_tone"],
        ["rude_tone", "no_resolution"],
        ["ignored_question", "unnecessary_escalation"],
        ["incorrect_info", "no_resolution"],
    ]
    for category in main_categories:
        for i, combo in enumerate(mistake_combos[:4]):
            scenarios.append({
                "category": category,
                "case_type": "agent_error",
                "has_hidden_dissatisfaction": i % 2 == 0,
                "intended_agent_mistakes": combo,
            })

    # ── Block 4: Edge cases with "other" category (20 cases) ──
    for case_type in CASE_TYPES:
        for variation in range(5):
            scenario = {
                "category": "other",
                "case_type": case_type,
                "has_hidden_dissatisfaction": variation % 3 == 0,
                "intended_agent_mistakes": [],
            }
            if case_type == "agent_error":
                mistake_idx = variation % len(AGENT_MISTAKES)
                scenario["intended_agent_mistakes"] = [AGENT_MISTAKES[mistake_idx]]
            scenarios.append(scenario)

    return scenarios


# Full scenario matrix (deterministically built)
SCENARIO_MATRIX: list[dict[str, Any]] = _build_scenario_matrix()
