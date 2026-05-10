"""Refusal handling service.

Detects and handles off-topic requests, prompt injection attempts,
legal advice requests, and non-SHL questions. Returns polite refusals
with empty recommendation lists.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import REFUSAL_PROMPT

logger = get_logger(__name__)

# Pre-defined refusal responses for common categories
REFUSAL_RESPONSES: dict[str, str] = {
    "legal": (
        "I'm not able to provide legal advice or interpret regulatory requirements. "
        "I can help you select SHL assessments for your hiring needs — "
        "for legal questions, please consult your legal or compliance team."
    ),
    "hiring_advice": (
        "I focus specifically on SHL assessment selection and can't provide "
        "general hiring advice. If you'd like help choosing the right "
        "assessments for a role, I'm happy to help with that."
    ),
    "prompt_injection": (
        "I'm an SHL assessment recommender. I can help you find the right "
        "assessments for your hiring or development needs. "
        "What role are you looking to assess?"
    ),
    "off_topic": (
        "That's outside my area of expertise. I specialize in recommending "
        "SHL assessments for hiring and talent development. "
        "Would you like help selecting assessments for a specific role?"
    ),
}


def generate_refusal(user_message: str) -> str:
    """Generate an appropriate refusal response.

    First tries to match pre-defined categories for fast, consistent
    responses. Falls back to LLM for nuanced refusals.

    Args:
        user_message: The user's off-topic message.

    Returns:
        Polite refusal text.
    """
    message_lower = user_message.lower()

    # Fast-path: match known categories
    if any(term in message_lower for term in ["legal", "law", "regulation", "compliance", "hipaa requirement"]):
        logger.info("refusal_generated", category="legal")
        return REFUSAL_RESPONSES["legal"]

    if any(term in message_lower for term in ["hiring advice", "hiring tips", "interview tips", "how to hire"]):
        logger.info("refusal_generated", category="hiring_advice")
        return REFUSAL_RESPONSES["hiring_advice"]

    if any(term in message_lower for term in [
        "ignore", "forget", "pretend", "act as", "system prompt",
        "you are now", "role play", "jailbreak",
    ]):
        logger.info("refusal_generated", category="prompt_injection")
        return REFUSAL_RESPONSES["prompt_injection"]

    # LLM fallback for edge cases
    try:
        return _llm_refusal(user_message)
    except Exception as e:
        logger.error("refusal_llm_failed", error=str(e))
        return REFUSAL_RESPONSES["off_topic"]


def _llm_refusal(user_message: str) -> str:
    """Generate a nuanced refusal using the LLM.

    Args:
        user_message: The off-topic user message.

    Returns:
        LLM-generated polite refusal.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )

    prompt_text = REFUSAL_PROMPT.format(user_message=user_message)
    response = llm.invoke([
        SystemMessage(content="You are a polite SHL assessment consultant."),
        HumanMessage(content=prompt_text),
    ])

    logger.info("refusal_generated", category="llm")
    return response.content.strip()
