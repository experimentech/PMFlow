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

### Agentic Reasoning with Flow Fields (v0.3.4)

PMFlow supports "Agentic Physics" where thoughts are driven by intent ("Flow") rather than just passive retrieval ("Gravity").

#### High-Level API (Encoder)

```python
from pmflow import PMFlowEmbeddingEncoder

# Create encoder with flow enabled (required for agentic features)
encoder = PMFlowEmbeddingEncoder(
    dimension=96,
    latent_dim=48,
    enable_flow=True  # Enables frame-dragging physics
)

# Trace a reasoning trajectory through concept space
trajectory, metrics = encoder.trace_trajectory(["machine", "learning"])
print(f"Mental effort: {metrics['path_length']:.3f}")
print(f"Efficiency: {metrics['efficiency']:.3f}")  # 1.0 = straight path (confident)

# Inject intent to bias future reasoning toward a goal
encoder.inject_intent(["deep", "learning"], strength=0.5)

# Now trajectories will curve toward the goal concept
trajectory2, metrics2 = encoder.trace_trajectory(["neural", "networks"])

# Clear intent when done
encoder.clear_intent()
```

#### Low-Level API (Field)

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

### Reactive Agentic Execution (v0.3.5)

PMFlow v0.3.5 adds a step-by-step execution API for reactive planning systems.
Instead of tracing full trajectories upfront, you can evolve position step-by-step
and modify the gravitational field based on outcomes.

```python
from pmflow.core.pmflow import ParallelPMField

pm = ParallelPMField(d_latent=64, enable_flow=True)
z = initial_position  # Your starting latent position

for step in range(max_steps):
    # Single physics step
    z_next = pm.step(z)
    
    # Ground to nearest action (your grounding logic)
    indices, dists, attractions = pm.find_nearest_centers(z_next, top_k=3)
    action = choose_action(indices[0])
    
    # Execute action and get result
    result = execute_action(action)
    
    if result.failed:
        # Mark this region as a hazard - trajectories will curve away
        pm.mark_as_hazard(z_next, radius=1.0, repulsion_strength=-0.5)
        # Also adjust specific center gravity
        pm.adjust_gravity(center_idx, mu_delta=-0.3, omega_delta=0.0)
    elif result.success:
        # Mark as attractor - reinforce this path
        pm.mark_as_attractor(z_next, radius=1.0, attraction_strength=0.3)
    
    # Apply external perturbation if needed
    perturbation = encode_result_as_perturbation(result)
    z = pm.inject_perturbation(z_next, perturbation, blend_factor=0.2)
    
    if goal_reached(z):
        break
```

**New Methods in ParallelPMField:**

| Method | Description |
|--------|-------------|
| `step(z)` | Single physics step (vs full trajectory) |
| `adjust_gravity(idx, mu_delta, omega_delta)` | Modify specific center |
| `inject_perturbation(z, perturbation, blend)` | Apply external force |
| `find_nearest_centers(z, top_k)` | Find closest gravitational centers |
| `mark_as_hazard(z, radius, strength)` | Create repulsive region |
| `mark_as_attractor(z, radius, strength)` | Create attractive region |

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
  title={PMFlow: Enhanced Pushing Medium gravitational Flow based neural network},
  author={Tristan Mumford},
  year={2026},
  url={https://github.com/experimentech/PMFlow}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

## Changelog

### v0.3.4 (2026-02-01)
- **Agentic Physics Encoder API**:
    - Added `PMFlowEmbeddingEncoder.trace_trajectory()` for reasoning path tracing
    - Added `PMFlowEmbeddingEncoder.inject_intent()` for goal-directed bias
    - Added `PMFlowEmbeddingEncoder.clear_intent()` to reset frame-dragging
    - Added `PMFlowEmbeddingEncoder.get_nearby_centers()` for debugging
    - Added `enable_flow` parameter to encoder constructor
    - Proper omega initialization in both MultiScale and Parallel fields

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
