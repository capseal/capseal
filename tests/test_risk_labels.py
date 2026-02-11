from __future__ import annotations

from capseal.risk_labels import generate_label


def test_complex_cross_cutting_security_label() -> None:
    features = {
        "lines_changed": 75,
        "files_touched": 4,
        "modules_crossed": 3,
        "security": 0.9,
        "change_type": "refactor",
        "test_coverage_delta": 0,
    }
    label = generate_label(features)
    assert "complex" in label
    assert "cross-cutting" in label
    assert "security-sensitive" in label


def test_simple_single_file_bugfix_label() -> None:
    features = {
        "lines_changed": 5,
        "files_touched": 1,
        "modules_crossed": 0,
        "security": 0.0,
        "change_type": "fix",
        "test_coverage_delta": 2,
    }
    label = generate_label(features)
    assert label == "simple + single-file + bugfix"


def test_empty_features_fallback() -> None:
    assert generate_label({}) == "unclassified"
