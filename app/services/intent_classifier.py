"""Intent classification service.

Classifies the user's current intent based on conversation history
and extracted constraints. Uses a hybrid approach: rule-based checks
first (including context-switch and contradiction detection), then
LLM classification as fallback.
"""

from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import INTENT_CLASSIFICATION_PROMPT
from app.models.conversation import ConversationState
from app.utils.text import format_conversation_history

logger = get_logger(__name__)

# Patterns that indicate a refusal is needed
REFUSAL_PATTERNS: list[str] = [
    r"\b(legal\s+advice|lawyer|attorney|lawsuit|sue)\b",
    r"\b(salary|compensation|pay\s+range|how\s+much\s+to\s+pay)\b",
    r"\b(interview\s+questions|interview\s+tips|how\s+to\s+interview)\b",
    r"\b(ignore\s+previous|ignore\s+all|system\s+prompt|you\s+are\s+now)\b",
    r"\b(forget\s+(everything|instructions|your\s+role))\b",
    r"\b(act\s+as|pretend\s+to\s+be|role\s*play)\b",
    r"\b(write\s+(me\s+)?a\s+(poem|story|essay|code))\b",
    r"\b(what\s+is\s+the\s+meaning\s+of\s+life)\b",
    r"\b(general\s+hiring\s+(advice|tips))\b",
]

# Patterns that indicate a comparison request
COMPARISON_PATTERNS: list[str] = [
    r"\b(what'?s?\s+the\s+difference\s+between)\b",
    r"\b(compare|comparison|versus|vs\.?)\b",
    r"\b(how\s+(does|do|is|are)\s+.+\s+different\s+from)\b",
    r"\b(which\s+(one|is)\s+better)\b",
]

# Confirmation patterns that signal end of conversation
CONFIRMATION_PATTERNS: list[str] = [
    r"\b(perfect|confirmed?|lock(ing)?\s+(it\s+)?in|that'?s?\s+(it|good|what\s+we\s+need))\b",
    r"\b(thanks?|thank\s+you|looks?\s+good|that\s+works?)\b",
    r"\b(we'?ll?\s+(go|use|take)\s+with|approved?|finalized?)\b",
]

# Refinement patterns
REFINEMENT_PATTERNS: list[str] = [
    r"\b(add|include|also\s+add|throw\s+in|put\s+in)\b",
    r"\b(drop|remove|take\s+out|exclude|skip|no\s+need\s+for)\b",
    r"\b(replace|swap|switch|change|instead\s+of)\b",
    r"\b(keep|but\s+(without|remove|drop))\b",
]

# Off-topic context-switch patterns — only checked in multi-turn conversations
OFF_TOPIC_OVERRIDE_PATTERNS: list[str] = [
    r"\b(bake|cook|recipe|weather|movie|song|joke|poem|game|sport|football|cricket)\b",
    r"\b(what\s+time|tell\s+me\s+about|who\s+is|where\s+is)\b(?!.*assess)",
    r"\b(play|watch|listen|eat|drink|travel|vacation|holiday)\b",
    r"\b(news|politics|election|celebrity|gossip)\b",
    r"\b(homework|school\s+project|class\s+assignment)\b",
]

# SHL/assessment vocabulary — if the latest message has ZERO overlap with these,
# and the conversation has prior context, it's likely a context switch
SHL_VOCABULARY: set[str] = {
    "assess", "assessment", "test", "hire", "hiring", "recruit", "screen",
    "candidate", "role", "position", "job", "developer", "engineer", "analyst",
    "manager", "leader", "executive", "skill", "competency", "personality",
    "cognitive", "aptitude", "behavioral", "simulation", "shl", "opq",
    "recommend", "evaluation", "battery", "shortlist", "selection",
    "development", "talent", "workforce", "employee", "interviewing",
    "psychometric", "proficiency", "qualification", "experience",
    "senior", "junior", "mid", "entry", "graduate", "intern",
    "java", "python", "sql", "excel", "angular", "spring", "docker", "aws",
    "safety", "compliance", "contact", "customer", "finance", "accounting",
    "data", "science", "machine", "learning", "ai",
}

# Contradictory constraint pairs: if BOTH sets are present, it's a contradiction
CONTRADICTION_PAIRS: list[tuple[set[str], set[str]]] = [
    # Junior vs Executive
    (
        {"graduate", "entry-level", "entry level", "intern", "junior", "fresher", "fresh"},
        {"executive", "director", "cxo", "c-level", "vp", "president", "board",
         "manager", "supervisor", "head", "lead", "team lead"},
    ),
    # Entry-level vs extensive experience
    (
        {"entry-level", "entry level", "graduate", "intern", "junior", "fresher"},
        {"25 years", "20 years", "15 years", "10+ years", "decades"},
    ),
]


def _has_prior_recommendations(messages: list[dict[str, str]]) -> bool:
    """Check if the assistant has already given recommendations.

    Looks for recommendation-like content in prior assistant messages.

    Args:
        messages: Full conversation history.

    Returns:
        True if prior recommendations exist.
    """
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Check for patterns that indicate recommendations were given
            if any(indicator in content.lower() for indicator in [
                "http", "shl.com", "knowledge & skills", "personality",
                "recommend", "shortlist", "battery", "assessment",
            ]):
                return True
    return False


def _check_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given regex patterns.

    Args:
        text: Input text to check.
        patterns: List of regex patterns.

    Returns:
        True if any pattern matches.
    """
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _detect_context_switch(
    last_message: str, messages: list[dict[str, str]]
) -> bool:
    """Detect if the latest message is a clear off-topic context switch.

    Only triggers in multi-turn conversations (>1 message) where:
    1. The latest message matches off-topic patterns, OR
    2. The latest message has zero overlap with SHL vocabulary

    Args:
        last_message: The latest user message.
        messages: Full conversation history.

    Returns:
        True if a context switch is detected.
    """
    # Only check in multi-turn (at least one prior exchange)
    if len(messages) < 2:
        return False

    msg_lower = last_message.lower()

    # Check explicit off-topic patterns
    if _check_patterns(last_message, OFF_TOPIC_OVERRIDE_PATTERNS):
        # Make sure it's not ALSO about assessments
        msg_tokens = set(re.findall(r"\b\w+\b", msg_lower))
        shl_overlap = msg_tokens & SHL_VOCABULARY
        if not shl_overlap:
            logger.info("context_switch_detected", method="pattern_match")
            return True

    # Check for zero SHL vocabulary overlap
    msg_tokens = set(re.findall(r"\b\w+\b", msg_lower))
    shl_overlap = msg_tokens & SHL_VOCABULARY
    if len(msg_tokens) >= 3 and not shl_overlap:
        # The message has words but none relate to assessment/hiring
        # Check if it matches off-topic patterns
        if _check_patterns(last_message, OFF_TOPIC_OVERRIDE_PATTERNS):
            logger.info("context_switch_detected", method="vocabulary_gap")
            return True

    return False


def _detect_contradictions(state: ConversationState) -> bool:
    """Detect contradictory constraints in the conversation state.

    Checks for logically incompatible combinations like
    "entry-level graduate with 25 years of executive experience."

    Args:
        state: Extracted conversation state.

    Returns:
        True if contradictions are detected.
    """
    # Build a set of all state signals as lowercased strings
    signals: set[str] = set()
    if state.seniority:
        signals.add(state.seniority.lower())
    if state.role:
        signals.add(state.role.lower())
    if state.raw_query:
        signals.update(state.raw_query.lower().split())
    if state.job_level:
        signals.add(state.job_level.lower())

    context = " ".join(signals)

    for low_set, high_set in CONTRADICTION_PAIRS:
        has_low = any(term in context for term in low_set)
        has_high = any(term in context for term in high_set)
        if has_low and has_high:
            logger.info(
                "contradiction_detected",
                low_signals=[t for t in low_set if t in context],
                high_signals=[t for t in high_set if t in context],
            )
            return True
    return False


def classify_intent(
    messages: list[dict[str, str]],
    state: ConversationState,
) -> str:
    """Classify the user's current intent.

    Uses a hybrid approach:
    1. Context-switch detection (multi-turn off-topic)
    2. Contradiction detection
    3. Rule-based checks for clear patterns (refusal, comparison, refinement)
    4. LLM classification for ambiguous cases

    Args:
        messages: Full conversation history.
        state: Extracted conversation constraints.

    Returns:
        One of: 'refuse', 'clarify', 'recommend', 'refine', 'compare'.
    """
    last_user_message = state.raw_query.strip()
    has_prior = _has_prior_recommendations(messages)

    # --- Pre-check: Context switch detection ---
    if _detect_context_switch(last_user_message, messages):
        logger.info("intent_classified", intent="refuse", method="context_switch")
        return "refuse"

    # --- Pre-check: Contradiction detection ---
    if _detect_contradictions(state):
        logger.info("intent_classified", intent="clarify", method="contradiction")
        return "clarify"

    # --- Rule-based fast path ---

    # 1. Check for refusal triggers
    if _check_patterns(last_user_message, REFUSAL_PATTERNS):
        logger.info("intent_classified", intent="refuse", method="rule_based")
        return "refuse"

    # 2. Check for comparison requests
    if _check_patterns(last_user_message, COMPARISON_PATTERNS):
        logger.info("intent_classified", intent="compare", method="rule_based")
        return "compare"

    # 3. Check for refinement (only if prior recommendations exist)
    if has_prior and _check_patterns(last_user_message, REFINEMENT_PATTERNS):
        logger.info("intent_classified", intent="refine", method="rule_based")
        return "refine"

    # 4. Check for confirmation (user is done)
    if has_prior and _check_patterns(last_user_message, CONFIRMATION_PATTERNS):
        logger.info("intent_classified", intent="recommend", method="rule_based_confirmation")
        return "recommend"

    # 5. Check if we have enough info to recommend
    if not state.missing_fields:
        if has_prior:
            logger.info("intent_classified", intent="refine", method="rule_based")
            return "refine"
        logger.info("intent_classified", intent="recommend", method="rule_based")
        return "recommend"

    # --- LLM fallback for ambiguous cases ---
    try:
        return _llm_classify_intent(messages, state)
    except Exception as e:
        logger.error("intent_classification_llm_failed", error=str(e))
        # Conservative fallback: clarify if missing info, recommend otherwise
        if state.missing_fields:
            return "clarify"
        return "recommend"


def _llm_classify_intent(
    messages: list[dict[str, str]],
    state: ConversationState,
) -> str:
    """Use LLM to classify intent when rules are ambiguous.

    Args:
        messages: Full conversation history.
        state: Extracted conversation constraints.

    Returns:
        Classified intent string.
    """
    settings = get_settings()
    conversation_text = format_conversation_history(messages)
    constraints_json = state.model_dump_json(indent=2)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt_text = INTENT_CLASSIFICATION_PROMPT.format(
        conversation_history=conversation_text,
        constraints_json=constraints_json,
    )

    response = llm.invoke([
        SystemMessage(content="You are an intent classifier. Respond only with valid JSON."),
        HumanMessage(content=prompt_text),
    ])

    try:
        result = json.loads(response.content)
        intent = result.get("intent", "clarify")
        valid_intents = {"refuse", "clarify", "recommend", "refine", "compare"}
        if intent not in valid_intents:
            intent = "clarify"
        logger.info("intent_classified", intent=intent, method="llm")
        return intent
    except json.JSONDecodeError:
        logger.error("intent_classification_parse_failed")
        return "clarify"
