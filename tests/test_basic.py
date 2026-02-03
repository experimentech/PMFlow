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
    
    d_latent = 32
    n_centers = 64
    model = ParallelPMField(d_latent=d_latent, n_centers=n_centers)
    
    # Test forward pass
    z = torch.randn(4, d_latent)  # batch=4
    output = model(z)
    
    assert output.shape == (4, d_latent), f"Expected shape (4, {d_latent}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_parallel_pmfield_trajectory():
    """Test trajectory return from ParallelPMField"""
    from pmflow.core.pmflow import ParallelPMField
    
    d_latent = 16
    steps = 5
    model = ParallelPMField(d_latent=d_latent, n_centers=32, steps=steps)
    
    z = torch.randn(2, d_latent)
    trajectory = model(z, return_trajectory=True)
    
    # Should return (batch, steps+1, d_latent)
    assert trajectory.shape == (2, steps + 1, d_latent)


def test_parallel_pmfield_step():
    """Test single-step evolution API"""
    from pmflow.core.pmflow import ParallelPMField
    
    d_latent = 16
    model = ParallelPMField(d_latent=d_latent, n_centers=32)
    
    z = torch.randn(2, d_latent)
    z_next = model.step(z)
    
    assert z_next.shape == z.shape
    assert not torch.allclose(z, z_next), "Step should change position"


def test_parallel_pmfield_agentic_api():
    """Test agentic execution API methods"""
    from pmflow.core.pmflow import ParallelPMField
    
    d_latent = 16
    model = ParallelPMField(d_latent=d_latent, n_centers=32, enable_flow=True)
    
    z = torch.randn(1, d_latent)
    
    # Test find_nearest_centers
    indices, dists, attractions = model.find_nearest_centers(z, top_k=5)
    assert indices.shape == (1, 5)
    assert dists.shape == (1, 5)
    assert attractions.shape == (1, 5)
    
    # Test adjust_gravity
    original_mu = model.mus[0].item()
    model.adjust_gravity(0, mu_delta=0.5)
    assert model.mus[0].item() == pytest.approx(original_mu + 0.5, rel=1e-5)
    
    # Test mark_as_hazard
    affected = model.mark_as_hazard(z.squeeze(), radius=2.0)
    assert affected >= 0
    
    # Test inject_perturbation
    perturbation = torch.randn(d_latent)
    z_perturbed = model.inject_perturbation(z, perturbation, blend_factor=0.5)
    assert z_perturbed.shape == z.shape


def test_vectorized_lateral_ei():
    """Test VectorizedLateralEI"""
    from pmflow.core.pmflow import VectorizedLateralEI
    
    d_latent = 32
    layer = VectorizedLateralEI()
    
    # Test forward pass (requires z and h inputs)
    z = torch.randn(4, d_latent)
    h = torch.randn(4, d_latent)
    output = layer(z, h)
    
    assert output.shape == (4, d_latent)
    assert not torch.isnan(output).any()


def test_query_expansion():
    """Test QueryExpansionPMField"""
    from pmflow.core.pmflow import ParallelPMField
    from pmflow.core.retrieval import QueryExpansionPMField
    
    d_latent = 32
    pm_field = ParallelPMField(d_latent=d_latent, n_centers=64)
    model = QueryExpansionPMField(pm_field=pm_field, expansion_k=3)
    
    # Test forward pass
    z = torch.randn(4, d_latent)
    expanded, weights = model.expand_query(z)
    
    assert expanded.shape == (4, d_latent)
    assert not torch.isnan(expanded).any()


def test_contrastive_pmfield():
    """Test ContrastivePMField"""
    from pmflow.core.pmflow import ParallelPMField
    from pmflow.core.contrastive import ContrastivePMField
    
    d_latent = 32
    pm_field = ParallelPMField(d_latent=d_latent, n_centers=64)
    model = ContrastivePMField(pm_field=pm_field)
    
    # Test forward pass
    z = torch.randn(4, d_latent)
    output = model(z)
    
    assert output.shape[0] == 4
    assert not torch.isnan(output).any()


def test_temporal_pipeline_bnn():
    """Test TemporalPipelineBNN"""
    from pmflow.bnn.bnn import TemporalPipelineBNN
    
    d_latent = 8
    n_classes = 10
    model = TemporalPipelineBNN(
        d_latent=d_latent,
        channels=64,
        pm_steps=4,
        n_centers=32,
        n_classes=n_classes,
    )
    
    # Test forward pass (expects MNIST-like input)
    x = torch.randn(4, 1, 28, 28)
    logits, (z, h) = model.parallel_temporal_evolution(x, T=3)
    
    assert logits.shape == (4, n_classes)
    assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
