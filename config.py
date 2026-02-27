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
GENERATION_TEMPERATURE: float = 0.3  # Slight variation for dialog diversity
ANALYSIS_TEMPERATURE: float = 0.0    # Strict determinism for analysis
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
    "slow_response",
    "generic_response",
]

# Category descriptions (for prompts)
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "payment_issue": "Payment issues (card not going through, double charge, subscription payment not credited for CX-Ray)",
    "technical_error": "Technical errors (error 500, API integration not working, UI bug, dashboard not loading)",
    "account_access": "Account access (forgotten password, locked account, SSO/2FA authentication issues)",
    "tariff_question": "Plan/pricing questions (difference between Free/Pro/Enterprise, plan change, feature limits, CX-Ray subscription terms)",
    "refund_request": "Refund requests (refund for unused subscription period, auto-renewal cancellation, accidental charge)",
    "other": "Other inquiries (improvement suggestions, general questions about CX-Ray features, service complaints)",
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
    "incorrect_info": "Incorrect information — agent provides false information about CX-Ray plans, features, or procedures",
    "rude_tone": "Rude tone — agent responds dismissively, impatiently, or unprofessionally",
    "no_resolution": "No resolution — dialog ends without actually resolving the client's problem",
    "unnecessary_escalation": "Unnecessary escalation — agent redirects to another specialist without attempting to solve it themselves",
    "slow_response": "Slow response — agent takes unreasonably long to respond, asks client to wait multiple times, or delays action without clear reason",
    "generic_response": "Generic response — agent gives template/canned answers, links to FAQ, or provides general advice instead of addressing the client's specific situation",
}

# ── Variation contexts for dialog diversity ──────────────────────────

VARIATION_CONTEXTS: dict[str, list[dict[str, str]]] = {
    "payment_issue": [
        {
            "persona": "a startup founder managing a team of 12 on the Pro plan",
            "specific_detail": "recurring monthly payment of $180 keeps failing with error code E-4012",
            "situation": "has been trying to pay for 3 days and team members are losing access to projects",
        },
        {
            "persona": "a freelance designer who just upgraded from Free to Pro",
            "specific_detail": "first payment went through twice, charged $30 instead of $15",
            "situation": "noticed the double charge on bank statement this morning",
        },
        {
            "persona": "an IT admin at a mid-size company on the Enterprise plan",
            "specific_detail": "annual invoice payment of $5,400 shows as pending for 2 weeks",
            "situation": "finance department is asking for payment confirmation for quarterly audit",
        },
    ],
    "technical_error": [
        {
            "persona": "a project manager who relies on dashboards for daily standups",
            "specific_detail": "the Kanban board shows error 500 when dragging cards between columns",
            "situation": "team standup is in 30 minutes and the board is completely unusable",
        },
        {
            "persona": "a developer integrating CX-Ray API into their CI/CD pipeline",
            "specific_detail": "the REST API /projects/{id}/tasks endpoint returns 403 despite valid token",
            "situation": "deployment pipeline has been broken since last night, blocking releases",
        },
        {
            "persona": "a team lead who uses time tracking for client billing",
            "specific_detail": "time tracking export to CSV generates empty files since the last platform update",
            "situation": "monthly client invoice is due tomorrow and billing data cannot be exported",
        },
    ],
    "account_access": [
        {
            "persona": "a new employee onboarded to the company's CX-Ray workspace last week",
            "specific_detail": "SSO login redirects to a blank page, no error message shown",
            "situation": "cannot access any team projects and first sprint planning is today",
        },
        {
            "persona": "a remote contractor who uses CX-Ray across multiple client workspaces",
            "specific_detail": "2FA codes from authenticator app are rejected as invalid after phone reset",
            "situation": "locked out of 3 client workspaces with urgent deliverables due this week",
        },
        {
            "persona": "a department head who hasn't logged in for 2 months during leave",
            "specific_detail": "account shows 'suspended due to inactivity' with no reactivation option",
            "situation": "returned from leave and needs immediate access to ongoing project data",
        },
    ],
    "tariff_question": [
        {
            "persona": "a CTO evaluating CX-Ray for company-wide adoption (200+ employees)",
            "specific_detail": "wants to understand Enterprise vs Pro feature differences for large teams",
            "situation": "preparing a comparison report for the board meeting next week",
        },
        {
            "persona": "a small agency owner currently on the Free plan with 4 team members",
            "specific_detail": "considering Pro plan but unsure about API access limits and storage quotas",
            "situation": "client workload is growing and Free plan limits are becoming a bottleneck",
        },
        {
            "persona": "a university professor using CX-Ray for student group projects",
            "specific_detail": "asking about educational discounts and whether student accounts count toward user limits",
            "situation": "new semester starts in 2 weeks and needs to set up 30 student accounts",
        },
    ],
    "refund_request": [
        {
            "persona": "a solo entrepreneur who signed up for Pro but found the tool too complex",
            "specific_detail": "used the service for only 2 days out of the monthly billing cycle",
            "situation": "already switched to a competitor and wants a full refund of $15",
        },
        {
            "persona": "a marketing manager whose team was auto-renewed without approval",
            "specific_detail": "annual plan auto-renewed for $2,160 despite cancellation email sent 3 weeks ago",
            "situation": "budget for the tool was reallocated and finance is demanding the charge be reversed",
        },
        {
            "persona": "a developer who was charged for an Enterprise trial that was supposed to be free",
            "specific_detail": "credit card was charged $499 for Enterprise features during what was marketed as a 14-day free trial",
            "situation": "trial signup page showed 'no credit card required' but card was charged anyway",
        },
    ],
    "other": [
        {
            "persona": "a product manager who has been using CX-Ray for 2 years",
            "specific_detail": "suggesting a dark mode feature and asking about the product roadmap",
            "situation": "team has been requesting dark mode for months and wants to know if it's planned",
        },
        {
            "persona": "a data analyst who needs to generate compliance reports from CX-Ray data",
            "specific_detail": "asking about GDPR data export capabilities and data retention policies",
            "situation": "company audit is coming up and needs documentation of data handling practices",
        },
        {
            "persona": "a frustrated long-time customer considering cancellation",
            "specific_detail": "general complaint about declining service quality and slow feature updates",
            "situation": "competitor launched similar features at a lower price point last month",
        },
    ],
}

# ── Mixed intent scenarios (cross-category) ──────────────────────────

MIXED_INTENT_SCENARIOS: list[dict[str, Any]] = [
    {
        "apparent_category": "payment_issue",
        "actual_category": "technical_error",
        "description": "Client reports payment failure, but the root cause is a UI bug in the checkout flow that corrupts the payment form",
    },
    {
        "apparent_category": "technical_error",
        "actual_category": "account_access",
        "description": "Client reports 'error loading dashboard' but actually their session expired due to SSO token invalidation",
    },
    {
        "apparent_category": "tariff_question",
        "actual_category": "refund_request",
        "description": "Client asks about plan differences but actually wants a refund because they were auto-upgraded to a higher plan without consent",
    },
    {
        "apparent_category": "account_access",
        "actual_category": "payment_issue",
        "description": "Client cannot log in, but it turns out the account was suspended because their payment method expired",
    },
    {
        "apparent_category": "refund_request",
        "actual_category": "tariff_question",
        "description": "Client demands a refund but after discussion realizes they actually just want to downgrade their plan and keep using the service",
    },
]


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
                    "variation_index": variation,
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
    for i, (category, case_type, mistakes) in enumerate(hidden_configs):
        scenarios.append({
            "category": category,
            "case_type": case_type,
            "has_hidden_dissatisfaction": True,
            "intended_agent_mistakes": mistakes,
            "variation_index": i % 3,
        })

    # ── Block 3: Additional agent mistakes (20 cases) ──
    mistake_combos: list[list[str]] = [
        ["ignored_question", "no_resolution"],
        ["incorrect_info", "rude_tone"],
        ["rude_tone", "no_resolution"],
        ["ignored_question", "unnecessary_escalation"],
        ["incorrect_info", "no_resolution"],
        ["generic_response", "no_resolution"],
        ["slow_response", "ignored_question"],
        ["generic_response", "slow_response"],
    ]
    combo_idx = 0
    for category in main_categories:
        for _ in range(4):
            combo = mistake_combos[combo_idx % len(mistake_combos)]
            scenarios.append({
                "category": category,
                "case_type": "agent_error",
                "has_hidden_dissatisfaction": combo_idx % 2 == 0,
                "intended_agent_mistakes": combo,
                "variation_index": combo_idx % 3,
            })
            combo_idx += 1

    # ── Block 4: Mixed intent + "other" edge cases (20 cases) ──
    # First 10: cross-category mixed intent scenarios
    for i, mixed in enumerate(MIXED_INTENT_SCENARIOS):
        for case_type in ["problematic", "successful"]:
            scenarios.append({
                "category": mixed["actual_category"],
                "case_type": case_type,
                "has_hidden_dissatisfaction": i % 2 == 0,
                "intended_agent_mistakes": [],
                "variation_index": i % 3,
                "mixed_intent": mixed,
            })
    # Remaining 10: "other" category edge cases
    other_case_types = CASE_TYPES + CASE_TYPES + ["successful", "problematic"]
    for j in range(10):
        ct = other_case_types[j % len(other_case_types)]
        scenario = {
            "category": "other",
            "case_type": ct,
            "has_hidden_dissatisfaction": j % 3 == 0,
            "intended_agent_mistakes": [],
            "variation_index": j % 3,
        }
        if ct == "agent_error":
            mistake_idx = j % len(AGENT_MISTAKES)
            scenario["intended_agent_mistakes"] = [AGENT_MISTAKES[mistake_idx]]
        scenarios.append(scenario)

    return scenarios


# Full scenario matrix (deterministically built)
SCENARIO_MATRIX: list[dict[str, Any]] = _build_scenario_matrix()
