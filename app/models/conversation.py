"""Conversation state model — the heart of the system.

This structured state is derived from the full conversation history
on every request. It captures what the user needs and drives all
downstream retrieval, filtering, and ranking decisions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# Mapping from free-text seniority to SHL catalog job levels
SENIORITY_TO_JOB_LEVEL: dict[str, list[str]] = {
    "entry-level": ["Entry-Level"],
    "entry level": ["Entry-Level"],
    "junior": ["Entry-Level"],
    "graduate": ["Graduate"],
    "fresh graduate": ["Graduate"],
    "recent graduate": ["Graduate"],
    "mid": ["Mid-Professional"],
    "mid-level": ["Mid-Professional"],
    "mid-professional": ["Mid-Professional"],
    "senior": ["Professional Individual Contributor", "Mid-Professional"],
    "senior ic": ["Professional Individual Contributor"],
    "lead": ["Professional Individual Contributor", "Manager"],
    "tech lead": ["Professional Individual Contributor", "Manager"],
    "manager": ["Manager", "Front Line Manager"],
    "front line manager": ["Front Line Manager"],
    "supervisor": ["Supervisor", "Front Line Manager"],
    "director": ["Director"],
    "executive": ["Executive", "Director"],
    "cxo": ["Executive", "Director"],
    "c-level": ["Executive", "Director"],
    "vp": ["Executive", "Director"],
    "general": ["General Population"],
}


GENERIC_ROLES = {
    "engineer", "developer", "manager", "analyst", "designer",
    "consultant", "specialist", "coordinator", "associate", "officer",
    "executive", "director", "lead", "head", "staff", "professional"
}


class ConversationState(BaseModel):
    """Structured representation of user requirements extracted from conversation.

    This is the central data structure that flows through the entire pipeline.
    Every field is derived from explicit user statements — never hallucinated.
    """

    role: str | None = Field(
        default=None,
        description="Job role being hired for (e.g., 'software engineer').",
    )
    seniority: str | None = Field(
        default=None,
        description="Seniority level (e.g., 'senior', 'entry-level').",
    )
    job_level: str | None = Field(
        default=None,
        description="Mapped SHL job level.",
    )
    technical_skills: list[str] = Field(
        default_factory=list,
        description="Specific technical skills mentioned.",
    )
    domain: str | None = Field(
        default=None,
        description="Industry or domain (e.g., 'healthcare').",
    )
    personality_needed: bool = Field(
        default=False,
        description="Whether personality assessment is explicitly needed.",
    )
    cognitive_needed: bool = Field(
        default=False,
        description="Whether cognitive/aptitude assessment is explicitly needed.",
    )
    situational_judgment_needed: bool = Field(
        default=False,
        description="Whether SJT is explicitly needed.",
    )
    communication_needed: bool = Field(
        default=False,
        description="Whether communication assessment is needed.",
    )
    simulation_needed: bool = Field(
        default=False,
        description="Whether simulation-based assessment is needed.",
    )
    language_preference: str | None = Field(
        default=None,
        description="Preferred assessment language.",
    )
    duration_constraint: str | None = Field(
        default=None,
        description="Time constraint (e.g., 'quick', 'under 30 minutes').",
    )
    assessment_preferences: list[str] = Field(
        default_factory=list,
        description="Specific assessment types or names requested.",
    )
    use_case: str | None = Field(
        default=None,
        description="Purpose: 'selection', 'development', 'screening', 'audit'.",
    )
    explicit_additions: list[str] = Field(
        default_factory=list,
        description="Topics/assessments user asked to ADD.",
    )
    explicit_removals: list[str] = Field(
        default_factory=list,
        description="Topics/assessments user asked to REMOVE.",
    )
    raw_query: str = Field(
        default="",
        description="Most recent user message verbatim.",
    )
    confirmed: bool = Field(
        default=False,
        description="Whether the user has confirmed the current recommendations.",
    )

    @property
    def mapped_job_levels(self) -> list[str]:
        """Map seniority text to SHL job level values.

        Returns:
            List of matching SHL job level strings.
        """
        if self.job_level:
            return [self.job_level]
        if self.seniority:
            key = self.seniority.lower().strip()
            return SENIORITY_TO_JOB_LEVEL.get(key, [])
        return []

    @property
    def search_query(self) -> str:
        """Build a retrieval-optimized search query from extracted constraints.

        Combines role, skills, domain, seniority, AND retrieval expansions
        based on boolean flags. This ensures FAISS pulls in personality,
        cognitive, and simulation assessments when they are needed.

        Returns:
            Space-separated search query string.
        """
        parts: list[str] = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        if self.domain:
            parts.append(self.domain)
        if self.technical_skills:
            parts.extend(self.technical_skills)

        # --- Retrieval expansion based on intent signals ---
        if self.personality_needed:
            parts.extend(["personality", "behavioral", "OPQ", "opq32r"])
        if self.cognitive_needed:
            parts.extend(["aptitude", "reasoning", "cognitive", "verify"])
        if self.simulation_needed:
            parts.extend(["simulation", "interactive", "exercise"])
        if self.communication_needed:
            parts.extend(["communication", "spoken", "English", "SVAR"])
        if self.situational_judgment_needed:
            parts.extend(["situational", "judgment", "SJT"])

        if self.raw_query and not parts:
            parts.append(self.raw_query)
        return " ".join(parts)

    @property
    def missing_fields(self) -> list[str]:
        """Identify which critical fields are still missing.

        Used to decide whether to clarify or recommend.
        Seniority is only required if the user hasn't provided
        enough other context (role + domain or role + skills).

        Returns:
            List of missing field descriptions.
        """
        missing = []
        
        has_specific_role = (
            self.role is not None and
            self.role.lower().strip() not in GENERIC_ROLES
        )
        has_any_context = (
            self.technical_skills or
            self.domain or
            self.seniority or
            self.personality_needed or
            self.cognitive_needed or
            self.simulation_needed or
            self.communication_needed or
            self.situational_judgment_needed
        )
        
        if not self.role and not self.technical_skills and not self.domain:
            missing.append("role or position being hired for")
        elif self.role and not has_specific_role and not has_any_context:
            missing.append("more specific role details, seniority level, or required skills")

        has_enough_context = (
            (self.role and self.domain) or
            (self.role and self.technical_skills) or
            (self.role and (self.personality_needed or self.cognitive_needed
                            or self.simulation_needed or self.communication_needed)) or
            (self.domain and self.technical_skills) or
            has_specific_role
        )
        if not self.seniority and not self.job_level and not has_enough_context:
            missing.append("seniority or experience level")
        
        return missing

    @property
    def constraints_summary(self) -> str:
        """Human-readable summary of known constraints.

        Used in prompts to give the LLM context about what we know.

        Returns:
            Formatted multi-line string of known constraints.
        """
        lines: list[str] = []
        if self.role:
            lines.append(f"Role: {self.role}")
        if self.seniority:
            lines.append(f"Seniority: {self.seniority}")
        if self.domain:
            lines.append(f"Domain: {self.domain}")
        if self.technical_skills:
            lines.append(f"Technical skills: {', '.join(self.technical_skills)}")
        if self.language_preference:
            lines.append(f"Language: {self.language_preference}")
        if self.duration_constraint:
            lines.append(f"Duration: {self.duration_constraint}")
        if self.use_case:
            lines.append(f"Use case: {self.use_case}")
        if self.personality_needed:
            lines.append("Needs: personality assessment")
        if self.cognitive_needed:
            lines.append("Needs: cognitive/aptitude assessment")
        if self.situational_judgment_needed:
            lines.append("Needs: situational judgment")
        if self.simulation_needed:
            lines.append("Needs: simulation-based assessment")
        if self.communication_needed:
            lines.append("Needs: communication assessment")
        if self.explicit_additions:
            lines.append(f"Add: {', '.join(self.explicit_additions)}")
        if self.explicit_removals:
            lines.append(f"Remove: {', '.join(self.explicit_removals)}")
        return "\n".join(lines) if lines else "No constraints extracted yet."
