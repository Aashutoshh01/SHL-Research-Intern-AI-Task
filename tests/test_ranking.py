"""Tests for the deterministic ranking pipeline."""

from __future__ import annotations

from app.models.catalog import CatalogEntry
from app.models.conversation import ConversationState
from app.services.ranking import rank_assessments, _compute_score


def _make_entry(
    name: str = "Test Assessment",
    keys: list[str] | None = None,
    job_levels: list[str] | None = None,
    description: str = "A test assessment.",
    entity_id: str = "1",
    link: str = "https://www.shl.com/test",
) -> CatalogEntry:
    """Create a test catalog entry."""
    return CatalogEntry(
        entity_id=entity_id,
        name=name,
        link=link,
        job_levels=job_levels or ["Mid-Professional"],
        languages=["English (USA)"],
        duration="10 minutes",
        description=description,
        keys=keys or ["Knowledge & Skills"],
    )


def test_ranking_returns_sorted_results() -> None:
    """Ranking should return results sorted by score."""
    entries = [
        (_make_entry("Java Test", description="Advanced Java programming", entity_id="1"), 0.9),
        (_make_entry("Python Test", description="Python basics", entity_id="2"), 0.5),
        (_make_entry("Cooking Test", description="Culinary skills", entity_id="3"), 0.1),
    ]
    state = ConversationState(
        role="software engineer",
        technical_skills=["Java"],
    )

    result = rank_assessments(entries, state)
    assert len(result) > 0
    # Java test should rank higher due to skill match
    assert result[0].name == "Java Test"


def test_ranking_deduplicates() -> None:
    """Ranking should remove duplicate entries."""
    entries = [
        (_make_entry("Same Test", entity_id="1"), 0.9),
        (_make_entry("Same Test", entity_id="1"), 0.8),
    ]
    state = ConversationState(role="engineer")
    result = rank_assessments(entries, state)
    assert len(result) == 1


def test_ranking_respects_max_recommendations() -> None:
    """Ranking should not exceed max recommendations."""
    entries = [
        (_make_entry(f"Test {i}", entity_id=str(i)), 0.5)
        for i in range(20)
    ]
    state = ConversationState(role="engineer")
    result = rank_assessments(entries, state)
    assert len(result) <= 10


def test_ranking_empty_input() -> None:
    """Ranking should handle empty input gracefully."""
    state = ConversationState(role="engineer")
    result = rank_assessments([], state)
    assert result == []


def test_score_computation() -> None:
    """Score should be between 0 and 1."""
    entry = _make_entry(
        "Core Java",
        keys=["Knowledge & Skills"],
        job_levels=["Mid-Professional"],
        description="Java programming assessment",
    )
    state = ConversationState(
        role="java developer",
        seniority="mid-level",
        technical_skills=["Java"],
    )
    score = _compute_score(entry, 0.8, state)
    assert 0.0 <= score <= 1.0


def test_catalog_entry_test_type_codes() -> None:
    """CatalogEntry should derive correct test type codes."""
    entry = _make_entry(keys=["Knowledge & Skills", "Simulations"])
    assert entry.test_type_codes == "K,S"

    entry2 = _make_entry(keys=["Personality & Behavior"])
    assert entry2.test_type_codes == "P"


def test_catalog_entry_to_recommendation_dict() -> None:
    """to_recommendation_dict should produce the API schema format."""
    entry = _make_entry(name="Core Java", keys=["Knowledge & Skills"])
    rec = entry.to_recommendation_dict()
    assert rec["name"] == "Core Java"
    assert rec["url"] == "https://www.shl.com/test"
    assert rec["test_type"] == "K"
