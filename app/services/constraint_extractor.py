"""Constraint extraction service.

Sends the full conversation history to the LLM and extracts a structured
ConversationState. This is the ONLY place where free-text user input
is converted into structured data.
"""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import CONSTRAINT_EXTRACTION_PROMPT
from app.models.conversation import ConversationState
from app.utils.text import format_conversation_history

logger = get_logger(__name__)


def extract_constraints(messages: list[dict[str, str]]) -> ConversationState:
    """Extract structured constraints from conversation history.

    Sends the full conversation to the LLM with a structured extraction
    prompt. The LLM returns JSON matching the ConversationState schema.

    Args:
        messages: Full conversation history as list of role/content dicts.

    Returns:
        ConversationState with all extractable fields populated.

    Raises:
        ValueError: If LLM response cannot be parsed into valid JSON.
    """
    settings = get_settings()
    conversation_text = format_conversation_history(messages)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt_text = CONSTRAINT_EXTRACTION_PROMPT.format(
        conversation_history=conversation_text
    )

    logger.info("extracting_constraints", message_count=len(messages))

    response = llm.invoke([
        SystemMessage(content="You are a structured data extraction system. Respond only with valid JSON."),
        HumanMessage(content=prompt_text),
    ])

    try:
        raw_json = json.loads(response.content)
        state = ConversationState(**raw_json)
        logger.info(
            "constraints_extracted",
            role=state.role,
            seniority=state.seniority,
            skills_count=len(state.technical_skills),
            intent_signals={
                "personality": state.personality_needed,
                "cognitive": state.cognitive_needed,
                "confirmed": state.confirmed,
            },
        )
        return state
    except (json.JSONDecodeError, Exception) as e:
        logger.error("constraint_extraction_failed", error=str(e))
        # Fallback: return minimal state with just the raw query
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        return ConversationState(raw_query=last_user_msg)
