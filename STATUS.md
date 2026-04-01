# PMFlow Status

Last updated: 2026-04-02
Current version: `0.3.5`

PMFlow is an active, installable package that includes both high-level APIs and
low-level research components.

## Current State

- Package import and exports are aligned with current implementation.
- `PMFlowEmbeddingEncoder` is available in-package and exported from `pmflow`.
- `PMFlowLanguageModel` is available under `pmflow.lm` for sequence modeling and generation.
- Agentic physics features (flow/intents, trajectory tracing, runtime field control)
  are implemented and documented.

## Included Components

### High-Level API

- `PMFlowEmbeddingEncoder`
  - Deterministic hashed base encoder + PMFlow latent refinement
  - Multi-scale field support
  - Optional flow mode for trajectory and intent-driven behavior
  - Save/load field state

### Language Modeling

- `PMFlowLanguageModel`
  - Autoregressive generation via gravitational field evolution
  - Context mixing and PMFlow field dynamics
  - Configurable decoding (`temperature`, `top_k`, `top_p`)

### Core Physics

- `ParallelPMField`
- `MultiScalePMField`
- `VectorizedLateralEI`
- Plasticity helpers (`vectorized_pm_plasticity`, `contrastive_plasticity`,
  `batch_plasticity_update`)

### Retrieval Extensions

- `QueryExpansionPMField`
- `SemanticNeighborhoodPMField`
- `HierarchicalRetrievalPMField`
- `AttentionWeightedRetrieval`
- `CompositionalRetrievalPMField`

### Contrastive Extensions

- `ContrastivePMField`
- `contrastive_learning_step`
- `train_contrastive_pmfield`
- `create_contrastive_encoder`

### Experimental BioNN

- `TemporalPipelineBNN`

## Notes on Documentation

- `README.md` reflects the current package direction and capabilities.
- `STRUCTURE.md` is currently empty and does not define current architecture.
- This status document supersedes older notes that described PMFlow as
  research-only or missing the production encoder.

## Validation Quick Check

```bash
python -c "import pmflow; print(pmflow.__version__)"
python -c "from pmflow import PMFlowEmbeddingEncoder; print('ok')"
python -c "from pmflow.lm import PMFlowLanguageModel; print('ok')"
```

## Near-Term Maintenance

1. Keep STATUS/README/API exports synchronized per release.
2. Add/expand tests for LM generation and flow-intent utilities.
3. Populate `STRUCTURE.md` or remove it to avoid stale references.
