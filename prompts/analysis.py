"""Prompts for dialog analysis and support quality evaluation."""

SYSTEM_PROMPT = """You are an expert in evaluating SaaS platform support quality. Analyze dialogs thoroughly and objectively.

CRITICALLY IMPORTANT — detecting HIDDEN dissatisfaction:
A client may formally thank and be polite, but actually be dissatisfied. Signs:
- Client says 'okay, I'll try to figure it out myself' — they did NOT get a solution and gave up
- 'Thanks for the information' — when the information doesn't actually solve the problem
- 'I see, I'll look into it' — the client didn't receive concrete help
- Passive aggression: 'Well, okay, I guess that's how it is'
- Sarcasm: 'Great, very helpful' (when they weren't helped)
- The client stopped insisting — this does NOT mean they are satisfied

If the client's problem was NOT actually resolved — satisfaction = "unsatisfied",
even if the client is formally polite.

Respond ONLY with valid JSON."""


def build_analysis_prompt(dialogue: list[dict]) -> str:
    """Build a prompt for analyzing a single dialog."""

    # Format the dialog as text
    formatted_lines = []
    for msg in dialogue:
        role_label = "Client" if msg["role"] == "client" else "Agent"
        formatted_lines.append(f"{role_label}: {msg['text']}")
    dialogue_text = "\n".join(formatted_lines)

    prompt = f"""Analyze the following dialog between a client and a support agent:

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
