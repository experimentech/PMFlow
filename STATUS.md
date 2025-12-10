# PMFlow Package - Current State

## ⚠️ Important Note

This package currently contains the **low-level PMFlow research components**, NOT the production `PMFlowEmbeddingEncoder` used in Lilith.

### What's Here (Research Code)

- **pmflow/core/pmflow.py**: Low-level PMField components
  - `ParallelPMField` 
  - `VectorizedLateralEI`
  - `MultiScalePMField`
  - `AttentionGatedPMField`
  - `EnergyBasedPMField`

- **pmflow/core/retrieval.py**: Retrieval-specific PMFields
  - `QueryExpansionPMField`
  - `SemanticNeighborhoodPMField`
  - `HierarchicalRetrievalPMField`
  - `AttentionWeightedRetrieval`
  - `CompositionalRetrievalPMField`

- **pmflow/core/contrastive.py**: Contrastive learning
  - `ContrastivePMField`

- **pmflow/bnn/bnn.py**: BioNN temporal layers
  - `TemporalPipelineBNN`
  - `MultiGPUPMBNN`
  - `PMBNNAlwaysPlasticV2`

### What's Missing

The production-ready **`PMFlowEmbeddingEncoder`** that Lilith uses is located in:
```
experiments/retrieval_sanity/pipeline/embedding.py
```

This is the actual working encoder with:
- Simple, clean API
- BioNN integration
- Contrastive learning support
- Production-tested

## TODO: Consolidation Needed

To make this a proper pip package, we need to:

1. **Copy the working `PMFlowEmbeddingEncoder`** from experiments to pmflow/
2. **Create a clean, simple API** that exposes:
   ```python
   from pmflow import PMFlowEncoder
   
   encoder = PMFlowEncoder(dimension=64, latent_dim=32)
   embedding = encoder.encode(tokens)
   ```

3. **Decide what to do with research code**:
   - Option A: Keep in `pmflow.research` submodule
   - Option B: Separate repo for research code
   - Option C: Archive it

4. **Write proper tests** for the production API

5. **Update README** with real usage examples

## Quick Test

To verify the package structure:
```bash
cd pmflow-package
python -c "import pmflow; print(pmflow.__version__)"
```

To see what's actually available:
```bash
python -c "import pmflow; print(dir(pmflow))"
```

## Next Steps

See `STRUCTURE.md` for the proposed clean architecture.
