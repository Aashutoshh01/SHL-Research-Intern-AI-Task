"""Tests for the /chat endpoint schema validation."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_chat_rejects_empty_messages(client: TestClient) -> None:
    """Chat endpoint should reject requests with empty messages list."""
    response = client.post("/chat", json={"messages": []})
    assert response.status_code == 422  # Validation error


def test_chat_rejects_missing_role(client: TestClient) -> None:
    """Chat endpoint should reject messages missing the role field."""
    response = client.post(
        "/chat",
        json={"messages": [{"content": "test"}]},
    )
    assert response.status_code == 422


def test_chat_rejects_invalid_role(client: TestClient) -> None:
    """Chat endpoint should reject messages with invalid role."""
    response = client.post(
        "/chat",
        json={"messages": [{"role": "system", "content": "test"}]},
    )
    assert response.status_code == 422


def test_chat_rejects_empty_content(client: TestClient) -> None:
    """Chat endpoint should reject messages with empty content."""
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": ""}]},
    )
    assert response.status_code == 422


def test_chat_response_schema(client: TestClient) -> None:
    """Chat response must match the strict schema."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "What assessments for Java developers?"}
            ]
        },
    )
    # May succeed or fail depending on API key, but if 200, check schema
    if response.status_code == 200:
        data = response.json()
        assert "reply" in data
        assert "recommendations" in data
        assert "end_of_conversation" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["end_of_conversation"], bool)

        for rec in data["recommendations"]:
            assert "name" in rec
            assert "url" in rec
            assert "test_type" in rec
