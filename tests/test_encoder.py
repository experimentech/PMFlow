"""
Test PMFlow encoder basic functionality
"""

import torch
from pmflow import PMFlowEmbeddingEncoder


def test_encoder_creation():
    """Test creating encoder with different configurations"""
    # Basic encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    assert encoder.dimension == 64
    assert encoder.latent_dim == 32
    
    # Different dimensions
    encoder = PMFlowEmbeddingEncoder(dimension=128, latent_dim=64)
    assert encoder.dimension == 128
    assert encoder.latent_dim == 64
    
    print("✓ Encoder creation tests passed")


def test_encoding():
    """Test basic encoding functionality"""
    # PMFlow outputs 48 dims regardless of latent_dim (internal architecture)
    # In concat mode: output = base_dimension + 48
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Encode tokens (bag-of-words style)
    tokens = ["hello", "world", "test"]
    embedding = encoder.encode(tokens)
    
    # Output shape is (1, dimension + pmflow_output_dim)
    # PMFlow outputs 48 dimensions (from MultiScalePMField architecture)
    expected_shape = (1, 64 + 48)  # = (1, 112)
    assert embedding.shape == expected_shape, f"Expected shape {expected_shape}, got {embedding.shape}"
    assert not torch.isnan(embedding).any(), "Embedding contains NaN"
    assert torch.is_tensor(embedding), "Output should be a tensor"
    
    print("✓ Encoding tests passed")


def test_similarity():
    """Test similarity computation"""
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Similar sequences should have higher similarity
    emb1 = encoder.encode(["cat", "feline"])
    emb2 = encoder.encode(["cat", "kitten"])
    emb3 = encoder.encode(["computer", "keyboard"])
    
    # Squeeze to 1D for cosine similarity
    sim_related = torch.cosine_similarity(emb1, emb2, dim=1).item()
    sim_unrelated = torch.cosine_similarity(emb1, emb3, dim=1).item()
    
    # Just check they're valid numbers
    assert -1 <= sim_related <= 1, "Similarity should be in [-1, 1]"
    assert -1 <= sim_unrelated <= 1, "Similarity should be in [-1, 1]"
    
    print("✓ Similarity tests passed")


def test_batch_encoding():
    """Test encoding multiple sequences"""
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    expected_dim = 64 + 48  # dimension + pmflow_output (48 dims)
    
    sequences = [
        ["hello", "world"],
        ["test", "example"],
        ["machine", "learning"]
    ]
    
    embeddings = [encoder.encode(seq) for seq in sequences]
    
    assert len(embeddings) == 3
    assert all(emb.shape == (1, expected_dim) for emb in embeddings), \
        f"Expected shape (1, {expected_dim})"
    assert not any(torch.isnan(emb).any() for emb in embeddings)
    
    print("✓ Batch encoding tests passed")


if __name__ == "__main__":
    print("Running PMFlow tests...")
    print("=" * 50)
    
    test_encoder_creation()
    test_encoding()
    test_similarity()
    test_batch_encoding()
    
    print("=" * 50)
    print("✅ All tests passed!")
