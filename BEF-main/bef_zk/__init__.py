"""Core BEF zero-knowledge building blocks packaged for import."""

__all__ = [
    "air",
    "capsule",
    "fri",
    "shared",
    "stc",
    "zk_geom",
]

# Note: Submodules are imported lazily to avoid circular/missing imports.
# Import specific modules as needed:
#   from bef_zk.capsule.cli import main
#   from bef_zk.shared.scoring import compute_acquisition_score
