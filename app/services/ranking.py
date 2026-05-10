"""Deterministic ranking pipeline.

Scores and ranks filtered candidate assessments based on
multiple weighted signals including domain heuristics, exact
skill matching with aliases, and dynamic confidence thresholding.
This is entirely deterministic — no LLM involvement in scoring.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.catalog import CatalogEntry
from app.models.conversation import ConversationState
from app.utils.text import get_skill_aliases

logger = get_logger(__name__)

# Scoring weights — skill matching is strongest, semantic is reduced
WEIGHTS = {
    "semantic_similarity": 0.20,
    "skill_keyword_match": 0.30,
    "domain_boost": 0.15,
    "key_category_relevance": 0.15,
    "job_level_match": 0.10,
    "language_match": 0.10,
}

# Confidence threshold: keep only results within this fraction of the best score
CONFIDENCE_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Domain heuristic tables
# ---------------------------------------------------------------------------
# Maps domain signal keywords to assessment name substrings that should be boosted
DOMAIN_BOOSTS: list[tuple[set[str], list[str]]] = [
    # Safety / Industrial
    (
        {"safety", "plant", "chemical", "hazard", "operator", "compliance",
         "manufacturing", "industrial", "warehouse", "factory", "construction"},
        ["dependability and safety", "workplace health and safety", "dsi",
         "manufacturing", "industrial", "mechanical"],
    ),
    # Leadership / Executive
    (
        {"leadership", "executive", "cxo", "director", "vp", "c-level",
         "stakeholder", "board", "strategic"},
        ["occupational personality", "opq", "leadership",
         "managerial", "manager", "executive"],
    ),
    # Contact Centre / Customer Service
    (
        {"contact center", "contact centre", "customer service", "inbound",
         "outbound", "call center", "call centre", "helpdesk", "help desk"},
        ["contact center", "customer serv", "spoken english",
         "svar", "call simulation"],
    ),
    # Finance / Accounting
    (
        {"finance", "accounting", "financial", "banking", "audit", "tax",
         "bookkeeping", "cpa", "actuarial"},
        ["financial accounting", "accounting", "basic statistics",
         "banking", "finance"],
    ),
    # Sales / Business Development
    (
        {"sales", "business development", "revenue", "account management",
         "bdm", "b2b", "b2c"},
        ["sales", "business development", "negotiation",
         "customer relationship"],
    ),
    # Data / Analytics
    (
        {"data", "analytics", "statistics", "data science", "bi",
         "business intelligence", "dashboard"},
        ["data science", "basic statistics", "analytics",
         "data entry", "data analysis"],
    ),
    # Administrative / Clerical
    (
        {"admin", "clerical", "secretary", "office", "administrative",
         "receptionist", "assistant"},
        ["ms excel", "ms word", "proofreading", "data entry",
         "basic computer", "typing", "microsoft"],
    ),
]


def rank_assessments(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[CatalogEntry]:
    """Score and rank candidate assessments with dynamic count.

    Applies a multi-signal scoring function to each candidate,
    then applies confidence thresholding for dynamic recommendation count.

    Args:
        candidates: Filtered (CatalogEntry, retrieval_score) tuples.
        state: Conversation constraints for scoring context.

    Returns:
        Sorted list of CatalogEntry objects, best first.
    """
    settings = get_settings()

    if not candidates:
        return []

    scored: list[tuple[CatalogEntry, float]] = []
    for entry, retrieval_score in candidates:
        total_score = _compute_score(entry, retrieval_score, state)
        scored.append((entry, total_score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Handle explicit additions: boost to top
    scored = _boost_explicit_additions(scored, state)

    # Deduplicate by entity_id
    seen_ids: set[str] = set()
    unique: list[tuple[CatalogEntry, float]] = []
    for entry, score in scored:
        if entry.entity_id not in seen_ids:
            seen_ids.add(entry.entity_id)
            unique.append((entry, score))

    # --- Dynamic recommendation count via confidence thresholding ---
    if unique:
        max_score = unique[0][1]
        threshold = max_score * CONFIDENCE_THRESHOLD
        confident = [(e, s) for e, s in unique if s >= threshold]
        # Clamp: at least 1, at most max_recommendations
        max_recs = min(settings.max_recommendations, len(confident))
        result_tuples = confident[:max_recs]
    else:
        result_tuples = []

    # Ensure at least 1 result if we have candidates
    if not result_tuples and unique:
        result_tuples = [unique[0]]

    result = [entry for entry, _ in result_tuples]

    logger.info(
        "ranking_complete",
        candidates_scored=len(candidates),
        results_returned=len(result),
        top_result=result[0].name if result else "none",
    )
    return result


def _compute_score(
    entry: CatalogEntry,
    retrieval_score: float,
    state: ConversationState,
) -> float:
    """Compute weighted score for a single assessment.

    Args:
        entry: The catalog entry to score.
        retrieval_score: Blended semantic+lexical score from retrieval.
        state: Conversation state for contextual scoring.

    Returns:
        Total weighted score between 0 and 1.
    """
    scores: dict[str, float] = {}

    # 1. Semantic similarity (from retrieval — already blended with lexical)
    scores["semantic_similarity"] = min(retrieval_score, 1.0)

    # 2. Job level match
    scores["job_level_match"] = _score_job_level(entry, state)

    # 3. Key category relevance
    scores["key_category_relevance"] = _score_key_category(entry, state)

    # 4. Skill keyword match (alias-aware, name-weighted)
    scores["skill_keyword_match"] = _score_skill_match(entry, state)

    # 5. Language match
    scores["language_match"] = _score_language(entry, state)

    # 6. Domain heuristic boost
    scores["domain_boost"] = _score_domain_boost(entry, state)

    # Weighted sum
    total = sum(
        scores[signal] * weight
        for signal, weight in WEIGHTS.items()
    )
    return total


def _score_job_level(entry: CatalogEntry, state: ConversationState) -> float:
    """Score how well the entry's job levels match the target.

    Args:
        entry: Catalog entry.
        state: Conversation state.

    Returns:
        Score between 0 and 1.
    """
    target_levels = state.mapped_job_levels
    if not target_levels:
        return 0.5  # Neutral if no preference
    if not entry.job_levels:
        return 0.3  # Slightly penalize entries with no level info

    matching = sum(1 for level in target_levels if level in entry.job_levels)
    return min(matching / len(target_levels), 1.0) if target_levels else 0.5


def _score_key_category(entry: CatalogEntry, state: ConversationState) -> float:
    """Score how well the entry's categories match user needs.

    Args:
        entry: Catalog entry.
        state: Conversation state.

    Returns:
        Score between 0 and 1.
    """
    needed: list[str] = []
    if state.personality_needed:
        needed.append("Personality & Behavior")
    if state.cognitive_needed:
        needed.append("Ability & Aptitude")
    if state.situational_judgment_needed:
        needed.append("Biodata & Situational Judgment")
    if state.simulation_needed:
        needed.append("Simulations")
    if state.technical_skills:
        needed.append("Knowledge & Skills")

    if not needed:
        return 0.5  # Neutral when no specific needs expressed

    matching = sum(1 for key in needed if key in entry.keys)
    return matching / len(needed) if needed else 0.5


def _score_skill_match(entry: CatalogEntry, state: ConversationState) -> float:
    """Score how well the entry matches required technical skills.

    Uses the alias system for broader matching and gives 3x weight
    to matches in the assessment NAME vs. description.

    Args:
        entry: Catalog entry.
        state: Conversation state.

    Returns:
        Score between 0 and 1.
    """
    if not state.technical_skills:
        return 0.5  # Neutral when no skills specified

    name_lower = entry.name.lower()
    desc_lower = entry.description.lower()

    total_weight = 0.0
    max_weight = 0.0

    for skill in state.technical_skills:
        aliases = get_skill_aliases(skill)
        max_weight += 3.0  # Maximum possible per skill (name match)

        # Check name first (3x weight)
        name_matched = any(alias in name_lower for alias in aliases)
        if name_matched:
            total_weight += 3.0
            continue

        # Check description (1x weight)
        desc_matched = any(alias in desc_lower for alias in aliases)
        if desc_matched:
            total_weight += 1.0

    return min(total_weight / max_weight, 1.0) if max_weight > 0 else 0.5


def _score_language(entry: CatalogEntry, state: ConversationState) -> float:
    """Score language availability.

    Args:
        entry: Catalog entry.
        state: Conversation state.

    Returns:
        Score between 0 and 1.
    """
    if not state.language_preference:
        return 0.5  # Neutral
    if not entry.languages:
        return 0.3  # Unknown availability

    lang = state.language_preference.lower()
    if any(lang in l.lower() for l in entry.languages):
        return 1.0
    return 0.0


def _score_domain_boost(entry: CatalogEntry, state: ConversationState) -> float:
    """Score domain-specific relevance using heuristic tables.

    Checks both the conversation domain AND the raw query/role
    against known domain signal keywords.

    Args:
        entry: Catalog entry.
        state: Conversation state.

    Returns:
        Score between 0 and 1.
    """
    # Build the context string from all relevant state fields
    context_parts = []
    if state.domain:
        context_parts.append(state.domain.lower())
    if state.role:
        context_parts.append(state.role.lower())
    if state.raw_query:
        context_parts.append(state.raw_query.lower())
    context = " ".join(context_parts)

    if not context:
        return 0.5  # Neutral

    entry_name_lower = entry.name.lower()
    entry_desc_lower = entry.description.lower()
    entry_text = f"{entry_name_lower} {entry_desc_lower}"

    best_score = 0.0

    for signal_keywords, boost_names in DOMAIN_BOOSTS:
        # Check if any domain signal keyword is in the context
        if any(kw in context for kw in signal_keywords):
            # Check if this entry matches any boosted assessment name
            for boost_name in boost_names:
                if boost_name in entry_text:
                    best_score = max(best_score, 1.0)
                    break

    return best_score if best_score > 0 else 0.5


def _boost_explicit_additions(
    scored: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Boost scores for assessments the user explicitly asked to add.

    Args:
        scored: Current scored list.
        state: Conversation state with explicit additions.

    Returns:
        Re-scored list with boosted entries.
    """
    if not state.explicit_additions:
        return scored

    additions_lower = {a.lower() for a in state.explicit_additions}
    boosted: list[tuple[CatalogEntry, float]] = []
    for entry, score in scored:
        name_lower = entry.name.lower()
        desc_lower = entry.description.lower()
        if any(
            add in name_lower or add in desc_lower
            for add in additions_lower
        ):
            boosted.append((entry, score + 1.0))  # Boost to top
        else:
            boosted.append((entry, score))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted
