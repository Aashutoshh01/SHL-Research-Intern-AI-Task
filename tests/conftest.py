"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client.

    Returns:
        TestClient instance for the TalentRoute AI app.
    """
    return TestClient(app)
