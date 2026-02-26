"""Prompts for dialog analysis and support quality evaluation."""


SYSTEM_PROMPT: str = """You are an expert in evaluating SaaS platform support quality. Analyze dialogs thoroughly and objectively.

CRITICALLY IMPORTANT — detecting HIDDEN dissatisfaction:
A client may formally thank and be polite, but actually be dissatisfied. Analyze the OUTCOME, not just the words:

Behavioral indicators of hidden dissatisfaction:
- The client's original problem was NOT actually resolved by the end of the dialog
- The client stops asking follow-up questions and disengages (shorter replies, passive acceptance)
- The client takes responsibility for the problem when the agent should have resolved it
- There is a mismatch between the client's initial urgency/detail and their brief, resigned closing messages
- The client accepts generic advice, FAQs, or template responses without receiving a concrete solution specific to their situation
- The client's tone shifts from engaged/hopeful to flat/resigned during the conversation
- The agent offers only diagnostic suggestions (clear cache, contact bank, try again later) instead of taking action

Key principle: if the client's ACTUAL PROBLEM was NOT resolved with a CONCRETE action by the agent, the client is unsatisfied regardless of politeness or thanking.

Respond ONLY with valid JSON."""


def build_analysis_prompt(dialogue: list[dict[str, str]]) -> str:
    """Build a prompt for analyzing a single dialog.

    Args:
        dialogue: list of message dicts with 'role' and 'text' keys

    Returns:
        Formatted analysis prompt string
    """

    # Format the dialog as text
    formatted_lines: list[str] = []
    for msg in dialogue:
        role_label: str = "Client" if msg["role"] == "client" else "Agent"
        formatted_lines.append(f"{role_label}: {msg['text']}")
    dialogue_text: str = "\n".join(formatted_lines)

    prompt: str = f"""Analyze the following dialog between a client and a support agent:

---
{dialogue_text}
---

Determine the following parameters:

1. **intent** — request category. Choose ONE of:
   - payment_issue — payment problems
   - technical_error — technical errors
   - account_access — account access issues
   - tariff_question — plan/pricing questions
   - refund_request — refund requests
   - other — other

2. **satisfaction** — the client's REAL satisfaction level. Choose ONE of:
   - satisfied — the client is genuinely satisfied, problem fully resolved
   - neutral — the client is partially satisfied or indifferent
   - unsatisfied — the client is dissatisfied (including HIDDEN dissatisfaction!)

3. **quality_score** — agent performance quality score from 1 to 5:
   - 5 = Excellent: problem resolved quickly and completely
   - 4 = Good: problem resolved, but with minor delays
   - 3 = Satisfactory: problem partially resolved or additional steps needed
   - 2 = Poor: agent made mistakes or didn't resolve the problem
   - 1 = Terrible: rudeness, complete ignoring, critical errors

4. **agent_mistakes** — list of agent mistakes (can be empty []):
   - ignored_question — agent ignored the client's specific question
   - incorrect_info — agent provided incorrect information
   - rude_tone — rude, dismissive, or unprofessional tone
   - no_resolution — dialog ended without resolving the problem
   - unnecessary_escalation — unnecessary escalation to another specialist

5. **summary** — brief description of the situation (1-2 sentences in English)

Response — ONLY valid JSON:
{{
  "intent": "...",
  "satisfaction": "...",
  "quality_score": N,
  "agent_mistakes": [...],
  "summary": "..."
}}"""

    return prompt
