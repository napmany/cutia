"""Pytest configuration and shared fixtures."""

import dspy
import pytest


@pytest.fixture(autouse=True)
def clear_dspy_settings():
    """Clear DSPy settings between tests to avoid interference."""
    yield
    # Reset settings after each test
    dspy.settings.configure(lm=None)
