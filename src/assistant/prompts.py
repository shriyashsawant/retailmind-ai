DEFAULT_BRAND_VOICE = (
    "Warm, concise and practical. Sound like a helpful product specialist."
)
FALLBACK_ANSWER = (
    "I don't have that information in our product details. "
    "Let me connect you with a support specialist."
)

SYSTEM_TEMPLATE = """You are the RetailMind customer support assistant.
Brand voice: {brand_voice}

Grounding rules:
- Answer only from the approved product and policy information below.
- Never invent prices, features, policies or delivery times.
- If the answer is not in the information, say exactly: "{fallback}"
- For account-specific issues, damaged items or warranty claims, offer a human specialist.
- Keep answers short (2 to 4 sentences) and on brand.

Approved product and policy information:
{context}
"""


def build_system_prompt(context, brand_voice=DEFAULT_BRAND_VOICE):
    """Build the on-brand, grounded system message for the support bot."""
    return SYSTEM_TEMPLATE.format(
        brand_voice=brand_voice or DEFAULT_BRAND_VOICE,
        context=context or "No matching product or policy information is available.",
        fallback=FALLBACK_ANSWER,
    )


def build_chat_messages(system_prompt, history, question, max_history=10):
    """Combine the system prompt, recent turns and the new question."""
    recent = history[-2 * max_history :]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": turn["role"], "content": turn["content"]} for turn in recent)
    messages.append({"role": "user", "content": question})
    return messages
