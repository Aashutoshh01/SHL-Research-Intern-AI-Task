"""SHL catalog entry data model.

Represents a single assessment product from the SHL catalog.
Provides serialization, test type mapping, and embedding text generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# Mapping from catalog `keys` to short test type codes used in responses
KEYS_TO_TEST_TYPE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


class CatalogEntry(BaseModel):
    """A single SHL assessment product from the catalog.

    All fields map directly to the scraped catalog JSON structure.
    This model is the source of truth — no data is generated outside it.
    """

    entity_id: str
    name: str
    link: str
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str = ""
    remote: str = "yes"
    adaptive: str = "no"
    description: str = ""
    keys: list[str] = Field(default_factory=list)

    @property
    def test_type_codes(self) -> str:
        """Derive short test type codes from catalog keys.

        Returns:
            Comma-separated test type codes (e.g., 'K,S').
        """
        codes = []
        for key in self.keys:
            code = KEYS_TO_TEST_TYPE.get(key)
            if code and code not in codes:
                codes.append(code)
        return ",".join(codes) if codes else "K"

    @property
    def embedding_text(self) -> str:
        """Generate text representation for embedding.

        Combines name, description, keys, and job levels
        into a single searchable text block.

        Returns:
            Concatenated text suitable for vector embedding.
        """
        parts = [
            self.name,
            self.description,
            f"Assessment types: {', '.join(self.keys)}",
            f"Job levels: {', '.join(self.job_levels)}",
        ]
        if self.languages:
            parts.append(f"Languages: {', '.join(self.languages[:5])}")
        if self.duration:
            parts.append(f"Duration: {self.duration}")
        return ". ".join(parts)

    def to_recommendation_dict(self) -> dict[str, str]:
        """Convert to the API response recommendation format.

        Returns:
            Dict with name, url, and test_type fields.
        """
        return {
            "name": self.name,
            "url": self.link,
            "test_type": self.test_type_codes,
        }
