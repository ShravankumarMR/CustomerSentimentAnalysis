"""Test utilities and fixtures."""

import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This product is amazing! I love it so much."


@pytest.fixture
def sample_texts():
    """Sample texts for batch processing."""
    return [
        "This product is amazing! I love it so much.",
        "Terrible quality. Very disappointed.",
        "It's okay, nothing special.",
    ]
