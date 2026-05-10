"""Assessment comparison service.

Handles explicit comparison requests where the user asks about
differences between specific assessments. All data comes from
the catalog — never hallucinated.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import COMPARISON_PROMPT
from app.models.catalog import CatalogEntry
from app.services.retrieval import find_entry_by_name, get_all_catalog_entries
from app.utils.text import format_conversation_history

logger = get_logger(__name__)


def find_comparison_candidates(
    messages: list[dict[str, str]],
) -> list[CatalogEntry]:
    """Identify which assessments the user wants to compare.

    Searches the latest user message for assessment names
    that match catalog entries.

    Args:
        messages: Full conversation history.

    Returns:
        List of CatalogEntry objects to compare.
    """
    # Get the last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    # Also check recent assistant messages for assessment names
    recent_assessments: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Extract assessment names mentioned in assistant responses
            for entry in get_all_catalog_entries():
                if entry.name.lower() in content.lower():
                    recent_assessments.append(entry.name)

    # Search for catalog entries mentioned in the user's comparison request
    found: list[CatalogEntry] = []
    msg_lower = last_user_msg.lower()

    for entry in get_all_catalog_entries():
        name_lower = entry.name.lower()
        if name_lower in msg_lower:
            found.append(entry)

    # If user references assessments from prior turns
    if len(found) < 2:
        for name in recent_assessments:
            entry = find_entry_by_name(name)
            if entry and entry not in found:
                found.append(entry)
            if len(found) >= 2:
                break

    logger.info(
        "comparison_candidates_found",
        count=len(found),
        names=[e.name for e in found],
    )
    return found


def generate_comparison(
    messages: list[dict[str, str]],
    entries: list[CatalogEntry],
) -> str:
    """Generate a grounded comparison of assessment products.

    Uses the LLM to create a natural language comparison,
    but ONLY from catalog data provided in the prompt.

    Args:
        messages: Conversation history for context.
        entries: CatalogEntry objects to compare.

    Returns:
        Comparison text grounded in catalog data.
    """
    if not entries:
        return "I couldn't identify specific assessments to compare. Could you name the assessments you'd like me to compare?"

    # Build assessment text from catalog data
    assessments_text = _format_entries_for_prompt(entries)
    conversation_text = format_conversation_history(messages)

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
    )

    prompt_text = COMPARISON_PROMPT.format(
        conversation_history=conversation_text,
        assessments_text=assessments_text,
    )

    response = llm.invoke([
        SystemMessage(content="You are an SHL assessment expert. Only use the catalog data provided."),
        HumanMessage(content=prompt_text),
    ])

    logger.info("comparison_generated", entries_compared=len(entries))
    return response.content.strip()


def _format_entries_for_prompt(entries: list[CatalogEntry]) -> str:
    """Format catalog entries for inclusion in comparison prompt.

    Args:
        entries: CatalogEntry objects to format.

    Returns:
        Formatted text block with all relevant catalog fields.
    """
    parts: list[str] = []
    for i, entry in enumerate(entries, 1):
        part = (
            f"Assessment {i}: {entry.name}\n"
            f"  Description: {entry.description}\n"
            f"  Test Type: {entry.test_type_codes}\n"
            f"  Keys: {', '.join(entry.keys)}\n"
            f"  Duration: {entry.duration or 'Not specified'}\n"
            f"  Job Levels: {', '.join(entry.job_levels)}\n"
            f"  Languages: {', '.join(entry.languages[:5]) if entry.languages else 'Not specified'}\n"
            f"  Remote: {entry.remote}\n"
            f"  Adaptive: {entry.adaptive}\n"
            f"  URL: {entry.link}"
        )
        parts.append(part)
    return "\n\n".join(parts)
