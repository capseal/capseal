"""Pytest configuration and fixtures for CapSeal integration tests."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add both subdirectories to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "BEF-main"))
sys.path.insert(0, str(ROOT / "otherstuff"))


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with .capseal structure."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / ".capseal").mkdir()
    (project / ".capseal" / "models").mkdir()
    (project / ".capseal" / "runs").mkdir()
    return project


@pytest.fixture
def sample_diff():
    """Return a sample unified diff for testing."""
    return """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,5 +1,6 @@
+# Added security check
 def process_input(user_data):
     if not user_data:
         return None
+    sanitized = escape(user_data)
-    return user_data
+    return sanitized
"""


@pytest.fixture
def sample_findings():
    """Return sample Semgrep findings for testing."""
    return [
        {
            "check_id": "python.security.injection",
            "path": "foo.py",
            "start": {"line": 5, "col": 1},
            "end": {"line": 5, "col": 20},
            "extra": {"severity": "high"},
        },
        {
            "check_id": "python.style.unused-import",
            "path": "bar.py",
            "start": {"line": 1, "col": 1},
            "end": {"line": 1, "col": 15},
            "extra": {"severity": "info"},
        },
    ]


@pytest.fixture
def posteriors_file(tmp_project):
    """Create a sample posteriors file for testing."""
    import numpy as np

    n_points = 1024
    alpha = np.ones(n_points, dtype=np.int64)
    beta = np.ones(n_points, dtype=np.int64)

    # Add some "learned" points
    alpha[0] = 5  # High failure rate
    beta[0] = 2
    alpha[100] = 2  # Low failure rate
    beta[100] = 10

    posteriors_path = tmp_project / ".capseal" / "models" / "beta_posteriors.npz"
    np.savez(
        posteriors_path,
        alpha=alpha,
        beta=beta,
        grid_version=np.array("test"),
        run_uuid=np.array("test-fixture"),
    )
    return posteriors_path
