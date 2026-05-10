"""LangGraph state definition.

Defines the TypedDict that flows through the graph.
This is the data structure that each node reads from and writes to.
"""

from __future__ import annotations

from typing import TypedDict

from app.models.catalog import CatalogEntry
from app.models.conversation import ConversationState


class GraphState(TypedDict):
    """State object flowing through the LangGraph workflow.

    Each node reads what it needs and writes its outputs.
    The state is created fresh for every request — no persistence.
    """

    # Input — set before graph execution
    messages: list[dict[str, str]]

    # Extracted by constraint_extractor node
    conversation_state: ConversationState | None

    # Set by intent_classifier node
    intent: str

    # Set by retrieval + ranking nodes
    ranked_assessments: list[CatalogEntry]

    # Set by response generator / refusal / comparison nodes
    reply: str
    recommendations: list[dict[str, str]]
    end_of_conversation: bool
