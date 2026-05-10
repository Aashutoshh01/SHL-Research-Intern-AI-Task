"""Pydantic schemas for the /chat API endpoint.

Defines the strict request/response contract as specified in the assignment.
The response schema is NON-NEGOTIABLE.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history.

    The API receives the FULL conversation history every time.
    No server-side session memory is maintained.
    """

    role: Literal["user", "assistant"] = Field(
        description="Who sent this message.",
    )
    content: str = Field(
        description="The message content.",
        min_length=1,
    )


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    Contains the complete conversation history.
    The API is stateless — all context comes from this list.
    """

    messages: list[Message] = Field(
        description="Full conversation history, oldest first.",
        min_length=1,
    )


class Recommendation(BaseModel):
    """A single assessment recommendation.

    All fields MUST come from the SHL catalog — never hallucinated.
    """

    name: str = Field(description="Assessment name from SHL catalog.")
    url: str = Field(description="Product catalog URL from SHL.")
    test_type: str = Field(description="Test type code(s) — e.g., 'K', 'P,C'.")


class ChatResponse(BaseModel):
    """Response body for POST /chat.

    This schema is NON-NEGOTIABLE per assignment requirements.
    """

    reply: str = Field(
        description="Natural language response to the user.",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description=(
            "Assessment recommendations. Empty when clarifying or refusing. "
            "1-10 items when recommending. All URLs from SHL catalog only."
        ),
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True when the conversation is naturally concluded.",
    )
