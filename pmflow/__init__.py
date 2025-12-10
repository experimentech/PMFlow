"""
PMFlow - Probabilistic Masked Flow for Neural Embeddings

A BioNN (Biological Neural Network)-enhanced embedding system with contrastive learning and semantic retrieval.
"""

__version__ = "0.3.0"

# Main production encoder
from pmflow.encoder import PMFlowEmbeddingEncoder

# For advanced users - low-level components
from pmflow.core.pmflow import ParallelPMField, VectorizedLateralEI
from pmflow.bnn.bnn import TemporalPipelineBNN

__all__ = [
    "PMFlowEmbeddingEncoder",  # Primary API
    # Advanced/research
    "ParallelPMField",
    "VectorizedLateralEI",
    "TemporalPipelineBNN",
]



