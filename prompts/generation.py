"""Prompts for generating client-agent support dialogs."""

from config import CASE_TYPE_DESCRIPTIONS, CATEGORY_DESCRIPTIONS, MISTAKE_DESCRIPTIONS

SYSTEM_PROMPT: str = """You are a generator of realistic support dialogs for the SaaS platform "CX-Ray" in English.

CX-Ray is a cloud-based project management and team collaboration platform.
Pricing plans: Free (up to 5 users), Pro ($15/month per user), Enterprise (custom pricing).
Features: dashboards, kanban boards, time tracking, API integrations, reports, team chat.

Dialog generation rules:
1. Language — English, natural conversational style for the client, professional for the agent
2. The client may use casual style, abbreviations, emotional expressions
3. The agent responds politely, in a structured manner, with specific action steps
4. The dialog must contain 6 to 14 messages (3-7 from each side)
5. The first message is always from the client
6. The last message is always from the agent
7. Each message must be meaningful (not just "ok" or "thanks")
8. Messages should vary in length — some short (1 sentence), some longer (2-3 sentences)

Response format — ONLY valid JSON:
{
  "messages": [
    {"role": "client", "text": "..."},
    {"role": "agent", "text": "..."}
  ]
}"""

# Different behavioral patterns for hidden dissatisfaction (avoids tautological detection)
_HIDDEN_DISSATISFACTION_PATTERNS: list[str] = [
    (
        "IMPORTANT — Hidden dissatisfaction: the client is formally polite and accepts "
        "the agent's response, but their actual problem remains unresolved. The client "
        "gradually disengages from the conversation, giving shorter responses and not "
        "pushing further. They leave without a real solution but avoid confrontation."
    ),
    (
        "IMPORTANT — Hidden dissatisfaction: the client appears agreeable on the surface "
        "but subtly signals frustration through tone shifts — moving from detailed "
        "explanations to brief, resigned replies. They may redirect blame to themselves "
        "('maybe I'm doing something wrong') when actually the support was inadequate."
    ),
    (
        "IMPORTANT — Hidden dissatisfaction: the client maintains professional courtesy "
        "throughout but the dialog ends with them having to find a workaround on their "
        "own. They may express understanding of 'limitations' or 'policies' while clearly "
        "not having their need met. Their final messages carry a tone of polite resignation."
    ),
]


def build_generation_prompt(
    category: str,
    case_type: str,
    has_hidden_dissatisfaction: bool,
    agent_mistakes: list[str],
    variation_index: int = 0,
    variation_context: dict[str, str] | None = None,
    mixed_intent: dict[str, str] | None = None,
) -> str:
    """Build a prompt for generating a single dialog.

    Args:
        category: request category key (e.g. 'payment_issue')
        case_type: case type key (e.g. 'successful')
        has_hidden_dissatisfaction: whether to include hidden dissatisfaction
        agent_mistakes: list of agent mistake keys to include
        variation_index: index for selecting variation-specific context
        variation_context: persona/situation details for diversity
        mixed_intent: cross-category scenario descriptor

    Returns:
        Formatted generation prompt string
    """

    category_desc: str = CATEGORY_DESCRIPTIONS.get(category, category)
    case_type_desc: str = CASE_TYPE_DESCRIPTIONS.get(case_type, case_type)

    prompt_parts: list[str] = [
        "Generate a realistic dialog between a client and a support agent of the CX-Ray platform.",
        "",
        f"Request category: {category_desc}",
        f"Case type: {case_type_desc}",
    ]

    # Inject variation context for diversity
    if variation_context:
        prompt_parts.append("")
        prompt_parts.append(f"Client profile: {variation_context['persona']}.")
        prompt_parts.append(f"Specific situation: {variation_context['specific_detail']}.")
        prompt_parts.append(f"Context: {variation_context['situation']}.")

    # Mixed intent — cross-category scenario
    if mixed_intent:
        apparent_desc = CATEGORY_DESCRIPTIONS.get(
            mixed_intent["apparent_category"], mixed_intent["apparent_category"]
        )
        prompt_parts.append("")
        prompt_parts.append(
            f"IMPORTANT — Mixed intent scenario: The client initially presents their problem "
            f"as a {apparent_desc} issue. "
            f"However, during the conversation it becomes clear that the actual issue is: "
            f"{mixed_intent['description']}. "
            f"The agent should {'correctly identify the real issue and address it' if case_type == 'successful' else 'struggle to identify or properly address the real underlying issue'}."
        )

    # Additional instructions depending on case type
    if case_type == "successful" and not has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "IMPORTANT — This is a SUCCESSFUL case. The agent MUST actually solve the client's problem: "
            "find the specific cause, take action (reset password, issue refund, "
            "change plan, fix the bug) and confirm the problem is resolved. "
            "The client must be GENUINELY satisfied with the result, not just receive generic advice."
        )
    elif case_type == "problematic" and not has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "This is a problematic case: the agent tries to help but the solution is not ideal — "
            "additional steps, waiting, or a compromise is needed. The problem is PARTIALLY resolved."
        )
    elif case_type == "conflict" and not has_hidden_dissatisfaction:
        prompt_parts.append("")
        prompt_parts.append(
            "This is a conflict case: the client is emotional, frustrated, may raise their tone. "
            "The agent is under pressure, the situation is tense. The client is openly dissatisfied."
        )

    if has_hidden_dissatisfaction:
        prompt_parts.append("")
        pattern = _HIDDEN_DISSATISFACTION_PATTERNS[variation_index % len(_HIDDEN_DISSATISFACTION_PATTERNS)]
        prompt_parts.append(pattern)

    if agent_mistakes:
        prompt_parts.append("")
        prompt_parts.append("The agent must make the following mistakes (naturally, not too obviously):")
        for mistake in agent_mistakes:
            desc: str = MISTAKE_DESCRIPTIONS.get(mistake, mistake)
            prompt_parts.append(f"- {desc}")

    prompt_parts.append("")
    prompt_parts.append(
        "Response — ONLY valid JSON with a \"messages\" field. "
        "Each message has fields \"role\" (client/agent) and \"text\"."
    )

    return "\n".join(prompt_parts)
