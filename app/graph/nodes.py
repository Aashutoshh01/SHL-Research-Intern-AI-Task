"""LangGraph node functions.

Each function is a graph node that reads from and writes to GraphState.
Nodes are thin wrappers that delegate to service modules.
No business logic lives here — only orchestration glue.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.graph.state import GraphState
import re

CONFIRM_QUICK_CHECK = [
    r"\b(perfect|confirmed?|that'?s?\s+(it|good)|thanks?|thank\s+you|looks?\s+good|that\s+works?)\b",
    r"\b(we'?ll?\s+(go|use|take)\s+with|approved?|finalized?|lock(ing)?\s+in)\b",
]

def _is_confirmation(messages: list[dict]) -> bool:
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "").lower()
            break
    return any(re.search(p, last_user) for p in CONFIRM_QUICK_CHECK)
from app.services.constraint_extractor import extract_constraints
from app.services.intent_classifier import classify_intent
from app.services.retrieval import retrieve
from app.services.metadata_filter import apply_metadata_filters
from app.services.ranking import rank_assessments
from app.services.response_generator import (
    generate_recommendation_response,
    generate_clarification_response,
    generate_refinement_response,
)
from app.services.refusal import generate_refusal
from app.services.comparison import find_comparison_candidates, generate_comparison

logger = get_logger(__name__)


def extract_constraints_node(state: GraphState) -> dict:
    """Extract structured constraints from conversation history.

    Reads: messages
    Writes: conversation_state
    """
    logger.info("node_executing", node="extract_constraints")
    conversation_state = extract_constraints(state["messages"])
    return {"conversation_state": conversation_state}


def classify_intent_node(state: GraphState) -> dict:
    """Classify the user's current intent.

    Reads: messages, conversation_state
    Writes: intent
    """
    logger.info("node_executing", node="classify_intent")
    
    # Circuit Breaker: Enforce the 8-turn max cap.
    # If we hit 7 messages (User-Assist-User-Assist-User-Assist-User), 
    # force a recommendation to end the conversation gracefully.
    messages = state.get("messages", [])
    at_cap = len(messages) >= 7 and messages[-1].get("role") == "user"
    if at_cap:
        logger.warning("turn_cap_reached", message_count=len(messages))
        return {"intent": "recommend"}
        
    intent = classify_intent(state["messages"], state["conversation_state"])
    return {"intent": intent}


def handle_refusal_node(state: GraphState) -> dict:
    """Generate a refusal response for off-topic requests.

    Reads: conversation_state
    Writes: reply, recommendations, end_of_conversation
    """
    logger.info("node_executing", node="handle_refusal")
    raw_query = state["conversation_state"].raw_query if state["conversation_state"] else ""
    reply = generate_refusal(raw_query)
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def handle_clarification_node(state: GraphState) -> dict:
    """Generate a clarifying question.

    Reads: messages, conversation_state
    Writes: reply, recommendations, end_of_conversation
    """
    logger.info("node_executing", node="handle_clarification")
    reply = generate_clarification_response(
        state["messages"],
        state["conversation_state"],
    )
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def retrieve_and_rank_node(state: GraphState) -> dict:
    """Retrieve, filter, rank, and inject domain-relevant assessments.

    Includes:
    1. Domain-aware candidate injection: ensures FAISS retrieval gaps
       for domain-critical assessments (DSI, Financial Accounting, etc.)
       are filled before ranking.
    2. Personality injection: ensures OPQ/behavioral assessments appear
       for leadership roles.

    Reads: conversation_state
    Writes: ranked_assessments
    """
    logger.info("node_executing", node="retrieve_and_rank")
    conv_state = state["conversation_state"]

    # Build search query from constraints
    query = conv_state.search_query
    if not query.strip():
        query = conv_state.raw_query

    # Retrieve candidates from vector store
    candidates = retrieve(query)

    # --- Domain injection: fill retrieval gaps ---
    candidates = _inject_domain_candidates(candidates, conv_state)

    # Apply metadata filters
    filtered = apply_metadata_filters(candidates, conv_state)

    # Rank deterministically
    ranked = rank_assessments(filtered, conv_state)

    # --- Personality injection post-processing ---
    ranked = _inject_personality_if_needed(ranked, conv_state)

    logger.info(
        "retrieve_and_rank_complete",
        retrieved=len(candidates),
        filtered=len(filtered),
        ranked=len(ranked),
    )
    return {"ranked_assessments": ranked}


# Domain-aware candidate injection

# Maps domain signal keywords to assessment name substrings that MUST
# appear in the candidate pool if any signal keyword matches.
_DOMAIN_INJECTIONS: list[tuple[set[str], list[str]]] = [
    # Safety / Industrial — DSI often missed by semantic search
    (
        {"safety", "plant", "chemical", "hazard", "operator", "compliance",
         "dependability", "reliable", "reliability", "safe work"},
        ["dependability and safety", "workplace health and safety"],
    ),
    # Finance / Accounting — Financial Accounting missed when aptitude dominates
    (
        {"financial", "finance", "accounting", "bookkeeping", "audit",
         "financial analyst", "banking", "actuary"},
        ["Financial Accounting (New)", "basic statistics", "financial and banking"],
    ),
    # Contact Centre
    (
        {"contact center", "contact centre", "call center", "call centre",
         "inbound", "customer service agent"},
        ["contact center call simulation", "entry level customer serv",
         "svar - spoken english"],
    ),
    # Admin / Clerical
    (
        {"admin assistant", "administrative", "clerical", "receptionist",
         "secretary", "office assistant"},
        ["ms excel", "ms word", "proofreading", "data entry"],
    ),
]


def _inject_domain_candidates(
    candidates: list[tuple], conv_state
) -> list[tuple]:
    """Inject domain-critical assessments that FAISS may have missed.

    Scans the conversation state for domain signal keywords, then
    finds matching assessments in the full catalog and adds them
    to the candidate set with a high score so they participate
    in ranking.

    Args:
        candidates: Current FAISS-retrieved candidates.
        conv_state: Conversation state.

    Returns:
        Candidates with domain-critical assessments injected.
    """
    from app.services.retrieval import get_all_catalog_entries

    # Build context from all relevant state fields
    context_parts = []
    if conv_state.domain:
        context_parts.append(conv_state.domain.lower())
    if conv_state.role:
        context_parts.append(conv_state.role.lower())
    if conv_state.raw_query:
        context_parts.append(conv_state.raw_query.lower())
    context = " ".join(context_parts)

    if not context:
        return candidates

    # Find which domain injections match
    names_to_inject: list[str] = []
    for signal_keywords, target_names in _DOMAIN_INJECTIONS:
        if any(kw in context for kw in signal_keywords):
            names_to_inject.extend(target_names)

    if not names_to_inject:
        return candidates

    # Get existing candidate entity IDs to avoid duplicates
    existing_ids = {entry.entity_id for entry, _ in candidates}

    # Find matching catalog entries
    all_entries = get_all_catalog_entries()
    injected_count = 0
    for entry in all_entries:
        if entry.entity_id in existing_ids:
            continue
        entry_name_lower = entry.name.lower()
        for target_name in names_to_inject:
            if target_name.lower() in entry_name_lower:
                # Inject with a high score so it participates in ranking
                candidates.append((entry, 0.85))
                existing_ids.add(entry.entity_id)
                injected_count += 1
                break

    if injected_count > 0:
        logger.info("domain_candidates_injected", count=injected_count)

    return candidates


# Roles that imply personality/behavioral assessment is needed
_LEADERSHIP_ROLES = {
    "manager", "leader", "executive", "director", "vp", "cxo",
    "head", "chief", "president", "supervisor", "team lead",
}

# Personality assessment names to look for in catalog
_PERSONALITY_NAMES = [
    "occupational personality",
    "opq",
    "personality",
    "behavioral",
]


def _inject_personality_if_needed(
    ranked: list, conv_state
) -> list:
    """Inject personality assessments if needed but missing from results.

    Triggers when:
    1. personality_needed is True, OR
    2. The role is leadership/management/executive

    AND no personality assessment is already in the ranked list.

    Args:
        ranked: Current ranked CatalogEntry list.
        conv_state: Conversation state.

    Returns:
        Ranked list with personality entries injected if needed.
    """
    from app.services.retrieval import get_all_catalog_entries

    # Determine if personality injection is needed
    needs_personality = conv_state.personality_needed
    if not needs_personality and conv_state.role:
        role_lower = conv_state.role.lower()
        needs_personality = any(
            kw in role_lower for kw in _LEADERSHIP_ROLES
        )
    if not needs_personality and conv_state.seniority:
        seniority_lower = conv_state.seniority.lower()
        needs_personality = seniority_lower in {
            "executive", "director", "cxo", "vp", "c-level"
        }

    if not needs_personality:
        return ranked

    # Check if any personality assessment is already present
    has_personality = any(
        "Personality & Behavior" in entry.keys
        for entry in ranked
    )
    if has_personality:
        return ranked

    # Find personality assessments from catalog
    all_entries = get_all_catalog_entries()
    personality_entries = []
    for entry in all_entries:
        if "Personality & Behavior" in entry.keys:
            name_lower = entry.name.lower()
            if any(pn in name_lower for pn in _PERSONALITY_NAMES):
                personality_entries.append(entry)

    if not personality_entries:
        return ranked

    # Inject the best personality entry (OPQ preferred)
    best = None
    for pe in personality_entries:
        if "opq" in pe.name.lower():
            best = pe
            break
    if not best:
        best = personality_entries[0]

    # Check it's not already in ranked
    ranked_ids = {e.entity_id for e in ranked}
    if best.entity_id not in ranked_ids:
        # Insert after the first 2 entries (so it doesn't dominate)
        insert_pos = min(2, len(ranked))
        ranked.insert(insert_pos, best)
        logger.info("personality_injected", assessment=best.name)

    return ranked


def handle_comparison_node(state: GraphState) -> dict:
    """Handle assessment comparison requests.

    Reads: messages
    Writes: reply, recommendations, end_of_conversation
    """
    logger.info("node_executing", node="handle_comparison")
    entries = find_comparison_candidates(state["messages"])
    comparison_text = generate_comparison(state["messages"], entries)

    # Include compared assessments as recommendations
    recommendations = [e.to_recommendation_dict() for e in entries]

    return {
        "reply": comparison_text,
        "recommendations": recommendations,
        "end_of_conversation": False,
        "ranked_assessments": entries,
    }


def generate_response_node(state: GraphState) -> dict:
    """Generate the final natural language response.

    Reads: messages, conversation_state, intent, ranked_assessments
    Writes: reply, recommendations, end_of_conversation
    """
    logger.info("node_executing", node="generate_response")
    intent = state.get("intent", "recommend")
    conv_state = state["conversation_state"]
    assessments = state.get("ranked_assessments", [])

    # Generate response based on intent
    if intent == "refine":
        reply = generate_refinement_response(
            state["messages"], conv_state, assessments
        )
    elif intent == "clarify":
        # Already handled by clarification node, but fallback
        reply = generate_clarification_response(
            state["messages"], conv_state
        )
    else:
        reply = generate_recommendation_response(
            state["messages"], conv_state, assessments
        )

    # Build recommendation dicts from catalog entries
    recommendations = [a.to_recommendation_dict() for a in assessments]

    # Determine if conversation should end
    end_of_conversation = (
        (conv_state.confirmed if conv_state else False) or
        _is_confirmation(state["messages"])
    )

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }


def route_by_intent(state: GraphState) -> str:
    """Route to the correct handler based on classified intent.

    This is the conditional edge function for LangGraph.

    Args:
        state: Current graph state with intent set.

    Returns:
        Name of the next node to execute.
    """
    intent = state.get("intent", "clarify")
    logger.info("routing", intent=intent)
    return intent
