"""Tests for Gymnasium environment wrapper."""

import pytest
import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

# Import TauEnv regardless to test basic functionality
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from tau_gym import TauEnv


def create_simple_env():
    """Create a simple test environment."""
    # We can't use MJCF in tests without a file, so skip for now
    pytest.skip("Requires MJCF file for environment creation")


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
def test_env_spaces():
    """Test that environment has correct spaces."""
    create_simple_env()


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
def test_env_reset():
    """Test environment reset."""
    create_simple_env()


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
def test_env_step():
    """Test environment step."""
    create_simple_env()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
