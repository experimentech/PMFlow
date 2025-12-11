"""
PMFlow - Probabilistic Masked Flow for Neural Embeddings

A BioNN (Biological Neural Network)-enhanced embedding system with contrastive learning and semantic retrieval.
"""

__version__ = "0.3.0"

# Main production encoder
from pmflow.encoder import PMFlowEmbeddingEncoder

# Legacy-compatible surface for downstream projects
from pmflow.pmflow import PMFlow, PMField, ParallelPMField, MultiScalePMField, vectorized_pm_plasticity

# For advanced users - low-level components
from pmflow.core.pmflow import VectorizedLateralEI
from pmflow.bnn.bnn import TemporalPipelineBNN

__all__ = [
    "PMFlowEmbeddingEncoder",  # Primary API
    # Legacy/compat
    "PMFlow",
    "PMField",
    "MultiScalePMField",
    "vectorized_pm_plasticity",
    # Advanced/research
    "ParallelPMField",
    "VectorizedLateralEI",
    "TemporalPipelineBNN",
]



