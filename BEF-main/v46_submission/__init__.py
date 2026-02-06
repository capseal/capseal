"""v46_submission package

Provides the BEF (BICEP → ENN → FusionAlpha) implementation sourced from the
NeurIPS submission drop. Modules are re-exported for easier reuse in scripts.
"""

from .pipeline import BEF_EEG  # noqa: F401
from .bicep_eeg import EEGSDE, OscillatorySDEVariant, AdaptiveBICEP  # noqa: F401
from .enn import ENNEncoder, MultiScaleENN  # noqa: F401
from .fusion_alpha import FusionAlphaGNN, HierarchicalFusionAlpha  # noqa: F401

__all__ = [
    "BEF_EEG",
    "EEGSDE",
    "OscillatorySDEVariant",
    "AdaptiveBICEP",
    "ENNEncoder",
    "MultiScaleENN",
    "FusionAlphaGNN",
    "HierarchicalFusionAlpha",
]

