# PMFlow - Enhanced Pushing Medium gravitational Flow based neural network

A BioNN (Biological Neural Network) enhanced neural embedding system with contrastive learning and semantic retrieval capabilities.

## Features

- **BioNN-Enhanced Embeddings**: Biological Neural Network layers for uncertainty-aware representations
- **Agentic Physics Engine**: Frame-dragging flow fields and trajectory tracking for active reasoning
- **Contrastive Learning**: Optional contrastive training for semantic similarity
- **Flexible Architecture**: Supports various dimensionalities and latent spaces
- **Production Ready**: Simple API, tested in real applications

## Installation

```bash
pip install pmflow
```

Or from source:
```bash
git clone https://github.com/experimentech/PMFlow.git
cd pmflow
pip install -e .
```

## Quick Start

```python
from pmflow import PMFlowEmbeddingEncoder

# Create encoder
encoder = PMFlowEmbeddingEncoder(
    dimension=64,      # Embedding dimension
    latent_dim=32,     # BioNN latent dimension
    vocab_size=10000   # Vocabulary size (optional)
)

# Encode tokens
tokens = ["hello", "world", "example"]
embedding = encoder.encode(tokens)

print(embedding.shape)  # (64,) - sentence-level embedding
```

## Advanced Usage

### With Contrastive Learning

```python
from pmflow import PMFlowEmbeddingEncoder

encoder = PMFlowEmbeddingEncoder(
    dimension=128,
    latent_dim=64,
    use_contrastive=True,
    temperature=0.07
)

# Train with positive/negative pairs
positive_pairs = [
    (["the", "cat"], ["a", "feline"]),
    (["machine", "learning"], ["deep", "learning"])
]

encoder.train_contrastive(positive_pairs, epochs=10)
```

### Semantic Similarity

```python
# Encode queries
query_emb = encoder.encode(["machine", "learning"])
doc_emb = encoder.encode(["deep", "neural", "networks"])

# Compute similarity
similarity = torch.cosine_similarity(query_emb, doc_emb, dim=0)
```

### Agentic Reasoning with Flow Fields (v0.3.2)

PMFlow now supports "Agentic Physics" where thoughts are driven by intent ("Flow") rather than just passive retrieval ("Gravity").

```python
from pmflow.core.pmflow import ParallelPMField

# Enable Flow Field (Frame Dragging)
pm = ParallelPMField(d_latent=64, enable_flow=True)

# Set Intent (Angular Momentum on Goal Concept)
pm.omegas[target_index] = 2.0 

# Get Full Reasoning Trajectory
trajectory = pm(input_vector, return_trajectory=True)
# Shape: (Batch, Steps+1, Dim)

# Measure "Mental Effort" (Path Length)
effort = torch.sum(torch.norm(trajectory[:, 1:] - trajectory[:, :-1], dim=2), dim=1)
```

## Architecture

PMFlow combines:
- **Probabilistic Masking**: Learned attention over input tokens
- **BioNN Layers**: 
- **Flow-based Aggregation**: Smooth, differentiable token pooling
- **Contrastive Objectives**: Optional semantic similarity training

## API Reference

### `PMFlowEmbeddingEncoder`

Main encoder class.

**Parameters:**
- `dimension` (int): Output embedding dimension
- `latent_dim` (int): BioNN latent space dimension
- `vocab_size` (int, optional): Vocabulary size for embedding layer
- `use_contrastive` (bool): Enable contrastive learning
- `temperature` (float): Temperature for contrastive loss

**Methods:**
- `encode(tokens)`: Encode token sequence to embedding
- `train_contrastive(pairs, epochs)`: Train with contrastive pairs
- `save(path)`: Save model weights
- `load(path)`: Load model weights

## Research Components

For researchers, low-level components are available:

```python
from pmflow.core.pmflow import ParallelPMField
from pmflow.bnn.bnn import TemporalPipelineBNN
from pmflow.core.retrieval import QueryExpansionPMField
```

See `docs/` for detailed component documentation.

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

## Citation

If you use PMFlow in your research, please cite:

```bibtex
@software{pmflow2024,
  title={PMFlow: Enhanced Probabilistic Masked Flow for Neural Embeddings},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pmflow}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

## Changelog

### v0.3.2 (2026-01-30)
- **Agentic Physics Upgrade**:
    - Added Frame-Dragging Flow Field (`u_g = Ω × r`) to `ParallelPMField` via `enable_flow=True`.
    - Added `omegas` parameter to control angular momentum (Intent).
    - Added `return_trajectory` option to retrieve the full geodesic path of the reasoning process.
- Implemented efficient O(D) pairwise rotation kernel for high-dimensional flow.

### v0.3.0 (2025-11-25)
- Production release
- BioNN-enhanced embeddings
- Contrastive learning support
- Tested in Lilith conversational AI
