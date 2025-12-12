"""Compatibility surface for legacy imports.

This module mirrors the public fields exposed in pmflow.core.pmflow so existing
code that imports ``pmflow.pmflow`` continues to work.
"""

from pmflow.core.pmflow import (
    ParallelPMField,
    MultiScalePMField,
    vectorized_pm_plasticity,
    contrastive_plasticity,
    batch_plasticity_update,
    hybrid_similarity,
)

# Legacy alias: PMField historically pointed at the parallel implementation.
PMField = ParallelPMField
PMFlow = ParallelPMField

__all__ = [
    "PMFlow",
    "PMField",
    "ParallelPMField",
    "MultiScalePMField",
    "vectorized_pm_plasticity",
    "contrastive_plasticity",
    "batch_plasticity_update",
    "hybrid_similarity",
]
