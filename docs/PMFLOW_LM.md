# PMFlow Language Model

## Overview

A language model that uses **gravitational particle dynamics** for sequence modeling. Instead of attention mechanisms, tokens evolve through a learned **gravitational field** shaped by semantic attractors (centers).

### Core Physics

Each token's latent representation is a particle that evolves according to equation 7 from the PM physics model:

```
a = -c²∇ln(n)
```

Where the refractive index field is defined by learned gravitational centers:

```
n(r) = 1 + Σᵢ μᵢ/|r - rᵢ|
```

The gradient of this field creates a "semantic attractor landscape" that guides token transitions naturally, without explicit attention weights.

## Architecture

### Core Components

```
Token ID → Embedding → Projection to Latent
          ↓
      Context Mixer (learns to blend with previous state)
          ↓
      PMFlow Gravitational Field (Eq. 7: a = -c²∇ln(n))
          ↓
      Projection to Vocabulary → Next Token Probabilities
          ↓
    Evolved Latent as Context for Next Token
```

### Key Features

1. **Gravitational Field Evolution**
   - Tokens evolve through a field shaped by learned centers (attractors)
   - No explicit attention: routing happens naturally via gravity
   - Differential equations are smooth and differentiable

2. **Frame-Dragging Intent** (optional)
   - Angular momentum (omegas) on centers creates intentional bias
   - Enables goal-directed generation (bias toward certain next tokens)
   - Equation: `u_g(r) = Σᵢ Ωᵢ × (r - rᵢ)`

3. **Context Blending**
   - Learned gating between current token and previous state
   - Provides memory without explicit RNN unrolling
   - Improves gradient flow during training

4. **Sequence Processing**
   - Left-to-right generation (causal)
   - Each position processed independently given context
   - No need for masking—causality is natural

## Usage

### Basic Training

```python
import torch
from pmflow.lm import PMFlowLanguageModel
from pmflow.lm.train import PMFlowLMTrainer, WarmupScheduler

# Create model
model = PMFlowLanguageModel(
    vocab_size=10000,
    embedding_dim=256,
    latent_dim=128,
    n_centers=128,
    steps_per_token=4,
    enable_flow=True
)

# Setup training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = WarmupScheduler(optimizer, warmup_steps=1000, total_steps=100000)
trainer = PMFlowLMTrainer(model, optimizer, device='cuda')

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        metrics = trainer.train_step(batch)
        scheduler.step()
        print(f"Loss: {metrics['loss']:.4f}, PPL: {metrics['perplexity']:.2f}")
```

### Generation

```python
# Autoregressive generation with sampling
prompt_ids = [1, 2, 3]  # Start token IDs
generated = model.generate(
    prompt_ids=prompt_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

### Analyzing the Learned Field

```python
# Get the semantic attractor landscape
centers = model.get_field_centers()  # (n_centers, latent_dim)
masses = model.get_field_masses()    # (n_centers,)
intents = model.get_field_intents()  # (n_centers,) - angular momentum

print(f"Centers: {centers.shape}")
print(f"Masses range: [{masses.min():.4f}, {masses.max():.4f}]")
print(f"Intent (omega) mean: {intents.mean():.4f}")
```

## Advanced Configuration

### Model Parameters

```python
PMFlowLanguageModel(
    vocab_size=10000,           # Vocabulary size
    embedding_dim=256,          # Token embedding dimension
    latent_dim=128,            # Latent space where physics acts
    n_centers=128,             # Number of gravitational attractors
    steps_per_token=4,         # PMFlow evolution steps per token
    dt=0.15,                   # Time step for integration
    beta=1.2,                  # Scaling for gravitational acceleration
    clamp=3.0,                 # Gradient clipping value
    enable_flow=True,          # Enable frame-dragging (intent)
    dropout=0.1                # Dropout rate
)
```

### Training Parameters

```python
trainer = PMFlowLMTrainer(model, optimizer, device='cuda')

# Single step
metrics = trainer.train_step(
    batch_ids=token_ids,
    max_grad_norm=1.0          # Gradient clipping
)

# Checkpoint
trainer.save_checkpoint('ckpt.pt', epoch=5)
trainer.load_checkpoint('ckpt.pt', load_optimizer=True)
```

### Learning Rate Scheduling

```python
scheduler = WarmupScheduler(
    optimizer,
    warmup_steps=1000,         # Linear warmup
    total_steps=100000         # Then cosine decay
)
```

## Physics Interpretation

### Gravitational Centers as Semantic Attractors

- Each center `rᵢ` represents a learned semantic state
- The mass `μᵢ` indicates attraction strength
- Particles (tokens) get pulled toward nearby centers

### Token Evolution

1. Token embedded in high-dim space
2. Projected to latent dimension
3. Blended with previous context
4. Evolves through gravitational field:
   - Gradient `∇ln(n)` points toward stronger attractors
   - Particle follows geodesics through the field
   - Multiple steps allow complex routing

### Frame-Dragging Intent

- Optional angular momentum `Ω` on centers creates vortices
- Non-zero `Ω` biases trajectories in spiral patterns
- Useful for:
  - Goal-directed generation (set intent toward desired tokens)
  - Learned biases toward common transitions
  - Agentic reasoning (external goals modify field)

## Example Results

From `examples/train_pmflow_lm.py`:

```
Configuration:
  Vocab: 1000, Embedding: 128, Latent: 64, Centers: 32
  Epochs: 3, Batch: 4, Seq Len: 32

Results:
  Epoch 1: Train Loss=6.91, Val Loss=6.91, Val PPL=1003.94
  Epoch 2: Train Loss=6.89, Val Loss=6.92, Val PPL=1008.55
  Epoch 3: Train Loss=6.83, Val Loss=6.92, Val PPL=1013.33

Field Analysis:
  Centers range: [-2.73, 2.96]
  Masses range: [0.32, 0.74], Mean: 0.50
  Intents (omegas) range: [-0.035, 0.050], Mean: 0.0034

Generation:
  Prompt:    [838, 361, 270]
  Generated: [838, 361, 270, 587, 612, 347, 589, 312, 755, 918, ...]
```

## Comparison to Transformers

| Aspect | Transformers | PMFlow LM |
|--------|--------------|-----------|
| **Routing** | Learned attention (O(T²)) | Gravitational field (O(D)) |
| **Memory** | Self-attention over full context | RNN-like state blending |
| **Interpretation** | Attention patterns | Field geometry, particle trajectories |
| **Physics** | None | Derived from GR analogy |
| **Efficiency** | Quadratic in sequence length | Linear in field computations |
| **Causality** | Attention masking | Natural (sequential) |

## Advantages

✓ **Interpretable**: Visualize particle trajectories, field geometry  
✓ **Physics-grounded**: Equations come from general relativity analogy  
✓ **Efficient**: O(D) not O(T²) for field computation  
✓ **Novel**: Explores gravitational dynamics in NLP  
✓ **Flexible**: Frame-dragging enables intent/goal-directed generation  

## Limitations & Future Work

- **Current**: Synthetic data experiments and small models
- **Need**: Evaluation on real language tasks (next tokens, perplexity)
- **Challenge**: Scaling to large models and long sequences
- **Research**: Compare to Transformers on standard benchmarks
- **Exploration**: Multi-layer gravitational fields, layer-wise centers

## Running the Example

```bash
cd PMFlow
python examples/train_pmflow_lm.py
```

This trains a small model on synthetic data and demonstrates:
- Model creation and initialization
- Training loop with metrics
- Validation
- Generation
- Field analysis
- Checkpoint saving

## Citation

If you use PMFlow Language Model in research, please cite:

```bibtex
@inproceedings{pmflow-lm2026,
  title={PMFlow Language Model: Sequential Modeling via Gravitational Particle Dynamics},
  author={Mumford, Tristan},
  year={2026}
}
```

## References

- **PMFlow Core**: `pmflow/core/pmflow.py` - ParallelPMField implementation
- **Physics Model**: `all_formulas_fixed.tex` - Complete mathematical formulation
- **Encoder**: `pmflow/encoder.py` - Embedding encoders

## License

MIT License - See LICENSE for details.
