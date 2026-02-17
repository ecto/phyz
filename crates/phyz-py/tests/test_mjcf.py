"""Tests for MJCF loader."""

import pytest
import tau
import os


def test_mjcf_loader_missing_file():
    """Test that loading non-existent file raises error."""
    with pytest.raises(IOError):
        tau.MjcfLoader("nonexistent.xml")


def test_mjcf_loader_if_file_exists():
    """Test MJCF loading if ant.xml exists."""
    # Check if ant.xml exists in examples directory
    test_file = "examples/ant.xml"
    if not os.path.exists(test_file):
        pytest.skip(f"Test file {test_file} not found")

    loader = tau.MjcfLoader(test_file)
    model = loader.build_model()

    # Ant should have multiple bodies and DOFs
    assert model.nbodies > 1
    assert model.nq > 1
    assert model.nv > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
