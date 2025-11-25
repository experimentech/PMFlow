"""
Basic tests for PMFlow package components
"""

import pytest
import torch
import torch.nn as nn


def test_imports():
    """Test that core components can be imported"""
    from pmflow.core.pmflow import ParallelPMField, VectorizedLateralEI
    from pmflow.core.retrieval import QueryExpansionPMField
    from pmflow.core.contrastive import ContrastivePMField
    from pmflow.bnn.bnn import TemporalPipelineBNN
    
    assert ParallelPMField is not None
    assert VectorizedLateralEI is not None
    assert QueryExpansionPMField is not None
    assert ContrastivePMField is not None
    assert TemporalPipelineBNN is not None


def test_parallel_pmfield():
    """Test ParallelPMField basic functionality"""
    from pmflow.core.pmflow import ParallelPMField
    
    vocab_size = 100
    dim = 64
    model = ParallelPMField(vocab_size=vocab_size, dim=dim)
    
    # Test forward pass
    tokens = torch.randint(0, vocab_size, (4, 10))  # batch=4, seq_len=10
    output = model(tokens)
    
    assert output.shape == (4, dim), f"Expected shape (4, {dim}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_vectorized_lateral_ei():
    """Test VectorizedLateralEI"""
    from pmflow.core.pmflow import VectorizedLateralEI
    
    dim = 64
    layer = VectorizedLateralEI(dim=dim)
    
    # Test forward pass
    x = torch.randn(4, dim)
    output = layer(x)
    
    assert output.shape == (4, dim)
    assert not torch.isnan(output).any()


def test_query_expansion():
    """Test QueryExpansionPMField"""
    from pmflow.core.retrieval import QueryExpansionPMField
    
    vocab_size = 100
    dim = 64
    model = QueryExpansionPMField(vocab_size=vocab_size, dim=dim, expansion_size=3)
    
    # Test forward pass
    tokens = torch.randint(0, vocab_size, (4, 10))
    output = model(tokens)
    
    # Should return expanded embeddings
    assert output.shape[0] == 4  # batch size
    assert output.shape[-1] == dim  # embedding dimension


def test_contrastive_pmfield():
    """Test ContrastivePMField"""
    from pmflow.core.contrastive import ContrastivePMField
    
    vocab_size = 100
    dim = 64
    model = ContrastivePMField(vocab_size=vocab_size, dim=dim)
    
    # Test forward pass
    tokens = torch.randint(0, vocab_size, (4, 10))
    output = model(tokens)
    
    assert output.shape == (4, dim)
    assert not torch.isnan(output).any()


def test_temporal_pipeline_bnn():
    """Test TemporalPipelineBNN"""
    from pmflow.bnn.bnn import TemporalPipelineBNN
    
    dim = 64
    model = TemporalPipelineBNN(
        input_size=dim,
        hidden_size=128,
        output_size=dim,
        num_stages=3
    )
    
    # Test forward pass
    x = torch.randn(4, dim)
    output = model(x)
    
    assert output.shape == (4, dim)
    assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
