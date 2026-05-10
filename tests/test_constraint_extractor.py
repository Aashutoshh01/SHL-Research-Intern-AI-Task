"""Tests for the constraint extractor."""

from __future__ import annotations

from app.models.conversation import ConversationState, SENIORITY_TO_JOB_LEVEL


def test_conversation_state_defaults() -> None:
    """ConversationState should have sensible defaults."""
    state = ConversationState()
    assert state.role is None
    assert state.seniority is None
    assert state.technical_skills == []
    assert state.personality_needed is False
    assert state.confirmed is False


def test_conversation_state_search_query() -> None:
    """Search query should combine role, seniority, and skills."""
    state = ConversationState(
        role="software engineer",
        seniority="senior",
        technical_skills=["Java", "Spring"],
    )
    query = state.search_query
    assert "software engineer" in query
    assert "senior" in query
    assert "Java" in query
    assert "Spring" in query


def test_conversation_state_missing_fields() -> None:
    """Missing fields should identify what's needed."""
    state = ConversationState()
    missing = state.missing_fields
    assert len(missing) >= 1  # Should flag missing role/skills


def test_conversation_state_no_missing_when_complete() -> None:
    """No missing fields when role and seniority are set."""
    state = ConversationState(
        role="software engineer",
        seniority="senior",
    )
    assert state.missing_fields == []


def test_seniority_to_job_level_mapping() -> None:
    """Seniority mapping should produce valid SHL levels."""
    state = ConversationState(seniority="graduate")
    levels = state.mapped_job_levels
    assert "Graduate" in levels

    state2 = ConversationState(seniority="executive")
    levels2 = state2.mapped_job_levels
    assert "Executive" in levels2 or "Director" in levels2


def test_constraints_summary() -> None:
    """Constraints summary should be human-readable."""
    state = ConversationState(
        role="data scientist",
        seniority="mid-level",
        technical_skills=["Python", "ML"],
        personality_needed=True,
    )
    summary = state.constraints_summary
    assert "data scientist" in summary
    assert "mid-level" in summary
    assert "Python" in summary
    assert "personality" in summary.lower()
