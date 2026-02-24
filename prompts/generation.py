"""Prompts for generating client-agent support dialogs."""

from config import CASE_TYPE_DESCRIPTIONS, CATEGORY_DESCRIPTIONS, MISTAKE_DESCRIPTIONS

SYSTEM_PROMPT: str = """You are a generator of realistic support dialogs for the SaaS platform "CloudTask" in English.

CloudTask is a cloud-based project management and team collaboration platform.
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

Response format — ONLY valid JSON:
{
  "messages": [
    {"role": "client", "text": "..."},
    {"role": "agent", "text": "..."}
  ]
}"""


def build_generation_prompt(
    category: str,
    case_type: str,
    has_hidden_dissatisfaction: bool,
    agent_mistakes: list[str],
) -> str:
    """Build a prompt for generating a single dialog.

    Args:
        category: request category key (e.g. 'payment_issue')
        case_type: case type key (e.g. 'successful')
        has_hidden_dissatisfaction: whether to include hidden dissatisfaction
        agent_mistakes: list of agent mistake keys to include

    Returns:
        Formatted generation prompt string
    """

    category_desc: str = CATEGORY_DESCRIPTIONS.get(category, category)
    case_type_desc: str = CASE_TYPE_DESCRIPTIONS.get(case_type, case_type)

    prompt_parts: list[str] = [
        "Generate a realistic dialog between a client and a support agent of the CloudTask platform.",
        "",
        f"Request category: {category_desc}",
        f"Case type: {case_type_desc}",
    ]

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
        prompt_parts.append(
            "IMPORTANT — Hidden dissatisfaction: the client must be formally polite "
            "(saying 'thank you', 'okay', 'I understand'), but the problem is NOT actually resolved. "
            "The client may:\n"
            "- Say 'Alright, I'll try to figure it out myself' (gave up)\n"
            "- Thank for information that didn't actually help\n"
            "- Agree with the response with slight sarcasm or disappointment\n"
            "- End the dialog without receiving a real solution"
        )

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
