"""LangGraph workflow construction.

Builds the deterministic orchestration graph. No loops,
no tool-calling, no autonomous agents. Simple conditional
routing based on classified intent.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.logging import get_logger
from app.graph.state import GraphState
from app.graph.nodes import (
    extract_constraints_node,
    classify_intent_node,
    handle_refusal_node,
    handle_clarification_node,
    retrieve_and_rank_node,
    handle_comparison_node,
    generate_response_node,
    route_by_intent,
)

logger = get_logger(__name__)

# Module-level compiled graph singleton
_compiled_graph = None


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow.

    Flow:
        extract_constraints → classify_intent → (conditional routing)
            → refuse:   handle_refusal → END
            → clarify:  handle_clarification → END
            → recommend: retrieve_and_rank → generate_response → END
            → refine:   retrieve_and_rank → generate_response → END
            → compare:  handle_comparison → END

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("extract_constraints", extract_constraints_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("handle_refusal", handle_refusal_node)
    graph.add_node("handle_clarification", handle_clarification_node)
    graph.add_node("retrieve_and_rank", retrieve_and_rank_node)
    graph.add_node("handle_comparison", handle_comparison_node)
    graph.add_node("generate_response", generate_response_node)

    # Define flow
    graph.set_entry_point("extract_constraints")
    graph.add_edge("extract_constraints", "classify_intent")

    # Conditional routing based on intent
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "refuse": "handle_refusal",
            "clarify": "handle_clarification",
            "recommend": "retrieve_and_rank",
            "refine": "retrieve_and_rank",
            "compare": "handle_comparison",
        },
    )

    # Terminal edges
    graph.add_edge("handle_refusal", END)
    graph.add_edge("handle_clarification", END)
    graph.add_edge("retrieve_and_rank", "generate_response")
    graph.add_edge("handle_comparison", END)
    graph.add_edge("generate_response", END)

    logger.info("graph_built", nodes=7, edges=9)
    return graph


def get_compiled_graph():
    """Get or create the compiled graph singleton.

    Compiles the graph once and caches it for reuse.

    Returns:
        Compiled LangGraph ready for .invoke().
    """
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_graph()
        _compiled_graph = graph.compile()
        logger.info("graph_compiled")
    return _compiled_graph
