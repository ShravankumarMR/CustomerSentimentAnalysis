"""Tests for API endpoints."""

import pytest
from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()


def test_predict_endpoint(client):
    """Test predict endpoint."""
    response = client.post("/predict?text=This is a test")
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()
