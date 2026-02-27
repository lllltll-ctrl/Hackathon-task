"""Prompts for dialog analysis and support quality evaluation."""


SYSTEM_PROMPT: str = """You are an expert in evaluating SaaS platform support quality. Analyze dialogs thoroughly and objectively.

CRITICALLY IMPORTANT — detecting HIDDEN dissatisfaction:
A client may formally thank and be polite, but actually be dissatisfied. Analyze the OUTCOME, not just the words:

Behavioral indicators of hidden dissatisfaction:
- The client's original problem was NOT actually resolved by the end of the dialog
- The client stops asking follow-up questions and disengages (shorter replies, passive acceptance)
- The client takes responsibility for the problem when the agent should have resolved it ("maybe I'm doing something wrong")
- There is a mismatch between the client's initial urgency/detail and their brief, resigned closing messages
- The client accepts generic advice, FAQs, or template responses without receiving a concrete solution specific to their situation
- The client's tone shifts from engaged/hopeful to flat/resigned during the conversation
- The agent offers only diagnostic suggestions (clear cache, contact bank, try again later) instead of taking action
- The client says "I'll figure it out myself" or "I'll try again later" — these are resignation signals, NOT satisfaction

Key principle: if the client's ACTUAL PROBLEM was NOT resolved with a CONCRETE action by the agent, the client is unsatisfied regardless of politeness or thanking.

MIXED INTENT detection:
If the client starts with one problem but the conversation reveals a different underlying issue, classify by the ACTUAL root cause, not the initial complaint.

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
   - slow_response — agent takes too long, keeps saying "one moment please", "just a second", "let me check" multiple times (2+) without actual progress or resolution
   - generic_response — agent gives template/FAQ answers instead of addressing the specific situation

   Examples of slow_response:
   - Agent says "one moment please" 3+ times without solving
   - Agent repeatedly asks to wait without giving a timeline
   - Agent delays action while claiming to "check" multiple times

5. **summary** — brief description of the situation (1-2 sentences in English)

6. **confidence** — your confidence in the satisfaction assessment (0.0 to 1.0).
   Calibration guide — be HONEST, do not default to high values:
   - 0.9-1.0 — satisfaction is obvious and unambiguous (client explicitly states it)
   - 0.7-0.8 — likely correct but some ambiguity exists
   - 0.5-0.6 — genuinely uncertain, could go either way (e.g., polite client but unresolved issue)
   - 0.3-0.4 — low confidence, satisfaction is very hard to determine
   IMPORTANT: If the client is polite but the problem may not be fully resolved, confidence should be 0.5-0.7, NOT 0.9+.

Response — ONLY valid JSON:
{{
  "intent": "...",
  "satisfaction": "...",
  "quality_score": N,
  "agent_mistakes": [...],
  "summary": "...",
  "confidence": 0.0
}}"""

    return prompt
