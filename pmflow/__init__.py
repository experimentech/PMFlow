"""
PMFlow - Probabilistic Masked Flow for Neural Embeddings

Core library plus retrieval and contrastive extensions.

v0.3.4: Added Agentic Physics API
  - PMFlowEmbeddingEncoder.trace_trajectory() - trace reasoning paths
  - PMFlowEmbeddingEncoder.inject_intent() - bias toward goals
  - PMFlowEmbeddingEncoder.clear_intent() - reset frame-dragging
  - enable_flow parameter for physics-based reasoning
"""

__version__ = "0.3.4"

# Primary encoder
from pmflow.encoder import PMFlowEmbeddingEncoder

# Core fields
from pmflow.core.pmflow import (
    ParallelPMField,
    MultiScalePMField,
    vectorized_pm_plasticity,
    contrastive_plasticity,
    batch_plasticity_update,
    hybrid_similarity,
    VectorizedLateralEI,
)

# Retrieval extensions
from pmflow.core.retrieval import (
    QueryExpansionPMField,
    SemanticNeighborhoodPMField,
    HierarchicalRetrievalPMField,
    AttentionWeightedRetrieval,
    CompositionalRetrievalPMField,
)

# Contrastive extensions
from pmflow.core.contrastive import (
    ContrastivePMField,
    contrastive_learning_step,
    train_contrastive_pmfield,
    create_contrastive_encoder,
)

# Experimental BioNN components
from pmflow.bnn.bnn import TemporalPipelineBNN

__all__ = [
    # Primary API
    "PMFlowEmbeddingEncoder",
    # Core fields
    "ParallelPMField",
    "MultiScalePMField",
    "vectorized_pm_plasticity",
    "contrastive_plasticity",
    "batch_plasticity_update",
    "hybrid_similarity",
    "VectorizedLateralEI",
    # Retrieval
    "QueryExpansionPMField",
    "SemanticNeighborhoodPMField",
    "HierarchicalRetrievalPMField",
    "AttentionWeightedRetrieval",
    "CompositionalRetrievalPMField",
    # Contrastive
    "ContrastivePMField",
    "contrastive_learning_step",
    "train_contrastive_pmfield",
    "create_contrastive_encoder",
    # Experimental
    "TemporalPipelineBNN",
]



