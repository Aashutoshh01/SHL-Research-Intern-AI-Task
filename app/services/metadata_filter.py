"""Post-retrieval metadata filtering service.

Applies deterministic filters to the candidate set returned by
vector retrieval. Filters are based on structured constraints
extracted from the conversation.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.models.catalog import CatalogEntry
from app.models.conversation import ConversationState
from app.utils.text import parse_duration_minutes

logger = get_logger(__name__)


def apply_metadata_filters(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Apply all metadata filters to the candidate set.

    Filters are applied in order of strictness. Each filter
    removes entries that don't match the user's constraints.
    Falls back gracefully if filtering removes too many results.

    Args:
        candidates: List of (CatalogEntry, score) from retrieval.
        state: Extracted conversation constraints.

    Returns:
        Filtered list of (CatalogEntry, score) tuples.
    """
    if not candidates:
        return candidates

    filtered = candidates

    # Apply each filter, keeping fallback if too aggressive
    filtered = _apply_with_fallback(
        filtered, lambda c: _filter_by_job_level(c, state), "job_level"
    )
    filtered = _apply_with_fallback(
        filtered, lambda c: _filter_by_language(c, state), "language"
    )
    filtered = _apply_with_fallback(
        filtered, lambda c: _filter_by_duration(c, state), "duration"
    )
    filtered = _apply_with_fallback(
        filtered, lambda c: _filter_by_key_category(c, state), "key_category"
    )
    filtered = _apply_explicit_removals(filtered, state)

    logger.info(
        "metadata_filtering_complete",
        input_count=len(candidates),
        output_count=len(filtered),
    )
    return filtered


def _apply_with_fallback(
    candidates: list[tuple[CatalogEntry, float]],
    filter_fn: callable,
    filter_name: str,
    min_results: int = 3,
) -> list[tuple[CatalogEntry, float]]:
    """Apply a filter with fallback if results are too few.

    If filtering would reduce results below min_results,
    skip that filter to preserve recommendation quality.

    Args:
        candidates: Current candidate list.
        filter_fn: Filter function to apply.
        filter_name: Name for logging.
        min_results: Minimum acceptable results after filtering.

    Returns:
        Filtered or original list.
    """
    result = filter_fn(candidates)
    if len(result) >= min_results:
        logger.debug(
            "filter_applied",
            filter=filter_name,
            before=len(candidates),
            after=len(result),
        )
        return result
    logger.debug(
        "filter_skipped_too_aggressive",
        filter=filter_name,
        would_remain=len(result),
    )
    return candidates


def _filter_by_job_level(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Filter candidates by matching job levels.

    Args:
        candidates: Current candidates.
        state: Conversation state with job level info.

    Returns:
        Filtered candidates matching the requested job levels.
    """
    target_levels = state.mapped_job_levels
    if not target_levels:
        return candidates
    return [
        (entry, score)
        for entry, score in candidates
        if any(level in entry.job_levels for level in target_levels)
    ]


def _filter_by_language(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Filter candidates by language availability.

    Args:
        candidates: Current candidates.
        state: Conversation state with language preference.

    Returns:
        Filtered candidates available in the requested language.
    """
    if not state.language_preference:
        return candidates
    lang = state.language_preference.lower()
    return [
        (entry, score)
        for entry, score in candidates
        if not entry.languages  # No language restriction
        or any(lang in l.lower() for l in entry.languages)
    ]


def _filter_by_duration(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Filter candidates by duration constraint.

    Args:
        candidates: Current candidates.
        state: Conversation state with duration constraint.

    Returns:
        Filtered candidates within the duration limit.
    """
    if not state.duration_constraint:
        return candidates

    constraint = state.duration_constraint.lower()
    max_minutes: int | None = None

    if "quick" in constraint or "short" in constraint or "fast" in constraint:
        max_minutes = 15
    elif "under" in constraint or "less than" in constraint:
        mins = parse_duration_minutes(constraint)
        if mins:
            max_minutes = mins

    if max_minutes is None:
        return candidates

    return [
        (entry, score)
        for entry, score in candidates
        if _within_duration(entry.duration, max_minutes)
    ]


def _within_duration(duration_str: str, max_minutes: int) -> bool:
    """Check if an assessment's duration is within the limit.

    Args:
        duration_str: Duration string from catalog.
        max_minutes: Maximum acceptable minutes.

    Returns:
        True if within limit or if duration is variable/unknown.
    """
    mins = parse_duration_minutes(duration_str)
    if mins is None:
        return True  # Don't filter out variable/unknown durations
    return mins <= max_minutes


def _filter_by_key_category(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Filter candidates by assessment key category.

    Maps the user's needs (personality, cognitive, etc.) to
    catalog key categories.

    Args:
        candidates: Current candidates.
        state: Conversation state with category needs.

    Returns:
        Filtered candidates matching requested categories.
    """
    needed_keys: list[str] = []
    if state.personality_needed:
        needed_keys.append("Personality & Behavior")
    if state.cognitive_needed:
        needed_keys.append("Ability & Aptitude")
    if state.situational_judgment_needed:
        needed_keys.append("Biodata & Situational Judgment")
    if state.simulation_needed:
        needed_keys.append("Simulations")
    if state.communication_needed:
        needed_keys.append("Simulations")  # SVAR tests are under Simulations

    if not needed_keys:
        return candidates

    # Always include Knowledge & Skills so domain-specific knowledge tests
    # (e.g., Financial Accounting, Workplace Health) aren't filtered out
    # when cognitive/personality flags are also set.
    needed_keys.append("Knowledge & Skills")

    return [
        (entry, score)
        for entry, score in candidates
        if any(k in entry.keys for k in needed_keys)
    ]


def _apply_explicit_removals(
    candidates: list[tuple[CatalogEntry, float]],
    state: ConversationState,
) -> list[tuple[CatalogEntry, float]]:
    """Remove assessments the user explicitly asked to drop.

    Args:
        candidates: Current candidates.
        state: Conversation state with removal requests.

    Returns:
        Candidates with explicitly removed entries dropped.
    """
    if not state.explicit_removals:
        return candidates
    removals = {r.lower() for r in state.explicit_removals}
    return [
        (entry, score)
        for entry, score in candidates
        if not any(
            removal in entry.name.lower() or removal in entry.description.lower()
            for removal in removals
        )
    ]
