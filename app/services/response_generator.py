"""Response generation service.

Produces grounded natural language responses using the LLM.
The LLM NEVER generates assessment names or URLs — those come
from the catalog objects. The LLM only explains and narrates.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import (
    RESPONSE_GENERATION_PROMPT,
    CLARIFICATION_PROMPT,
    REFINEMENT_PROMPT,
)
from app.models.catalog import CatalogEntry
from app.models.conversation import ConversationState
from app.utils.text import format_conversation_history

logger = get_logger(__name__)


def generate_recommendation_response(
    messages: list[dict[str, str]],
    state: ConversationState,
    assessments: list[CatalogEntry],
) -> str:
    """Generate a response explaining assessment recommendations.

    The LLM receives the ranked assessments and explains why
    they fit. It NEVER invents assessment data.

    Args:
        messages: Conversation history.
        state: Extracted conversation constraints.
        assessments: Ranked catalog entries to recommend.

    Returns:
        Natural language explanation of the recommendations.
    """
    settings = get_settings()
    conversation_text = format_conversation_history(messages)
    assessments_text = _format_assessments(assessments)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )

    prompt_text = RESPONSE_GENERATION_PROMPT.format(
        conversation_history=conversation_text,
        constraints_summary=state.constraints_summary,
        assessments_text=assessments_text,
    )

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a helpful SHL assessment consultant. "
                "Only reference assessments from the provided catalog data. "
                "Never invent names, URLs, or descriptions."
            )
        ),
        HumanMessage(content=prompt_text),
    ])

    logger.info("recommendation_response_generated", assessment_count=len(assessments))
    return response.content.strip()


def generate_clarification_response(
    messages: list[dict[str, str]],
    state: ConversationState,
) -> str:
    """Generate a clarifying question response.

    Asks the user for missing information needed to make
    a recommendation.

    Args:
        messages: Conversation history.
        state: Extracted constraints with missing fields.

    Returns:
        Clarifying question text.
    """
    settings = get_settings()
    conversation_text = format_conversation_history(messages)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )

    prompt_text = CLARIFICATION_PROMPT.format(
        conversation_history=conversation_text,
        constraints_summary=state.constraints_summary,
        missing_fields=", ".join(state.missing_fields) if state.missing_fields else "specific requirements",
    )

    response = llm.invoke([
        SystemMessage(content="You are a helpful SHL assessment consultant."),
        HumanMessage(content=prompt_text),
    ])

    logger.info("clarification_response_generated", missing=state.missing_fields)
    return response.content.strip()


def generate_refinement_response(
    messages: list[dict[str, str]],
    state: ConversationState,
    assessments: list[CatalogEntry],
) -> str:
    """Generate a response for refined recommendations.

    Acknowledges the changes the user requested and explains
    the updated recommendation set.

    Args:
        messages: Conversation history.
        state: Updated constraints.
        assessments: Updated ranked catalog entries.

    Returns:
        Response explaining the refinement.
    """
    settings = get_settings()
    conversation_text = format_conversation_history(messages)
    assessments_text = _format_assessments(assessments)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )

    prompt_text = REFINEMENT_PROMPT.format(
        conversation_history=conversation_text,
        constraints_summary=state.constraints_summary,
        assessments_text=assessments_text,
    )

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a helpful SHL assessment consultant. "
                "Acknowledge what changed and explain the updated list."
            )
        ),
        HumanMessage(content=prompt_text),
    ])

    logger.info("refinement_response_generated", assessment_count=len(assessments))
    return response.content.strip()


def _format_assessments(assessments: list[CatalogEntry]) -> str:
    """Format assessments for inclusion in LLM prompts.

    Args:
        assessments: List of catalog entries.

    Returns:
        Numbered list of assessment details.
    """
    if not assessments:
        return "No assessments matched the criteria."

    parts: list[str] = []
    for i, entry in enumerate(assessments, 1):
        part = (
            f"{i}. {entry.name}\n"
            f"   Type: {entry.test_type_codes} ({', '.join(entry.keys)})\n"
            f"   Duration: {entry.duration or 'Not specified'}\n"
            f"   Description: {entry.description[:200]}\n"
            f"   Job Levels: {', '.join(entry.job_levels[:3])}"
        )
        parts.append(part)
    return "\n\n".join(parts)
