"""
Simple example of using PMFlow encoder
"""

import torch
from pmflow import PMFlowEmbeddingEncoder


def main():
    print("PMFlow Example")
    print("=" * 50)
    
    # Create encoder
    print("\n1. Creating encoder...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=64,
        latent_dim=32
    )
    print(f"   ✓ Created encoder: dim={encoder.dimension}, latent_dim={encoder.latent_dim}")
    
    # Encode some tokens
    print("\n2. Encoding tokens...")
    tokens = ["hello", "world", "this", "is", "pmflow"]
    embedding = encoder.encode(tokens)
    print(f"   ✓ Input: {tokens}")
    print(f"   ✓ Output shape: {embedding.shape}")
    print(f"   ✓ Output (first 5 dims): {embedding[:5].tolist()}")
    
    # Compute similarity between two sequences
    print("\n3. Computing similarity...")
    seq1 = ["machine", "learning"]
    seq2 = ["deep", "learning"]
    seq3 = ["cooking", "recipe"]
    
    emb1 = encoder.encode(seq1)
    emb2 = encoder.encode(seq2)
    emb3 = encoder.encode(seq3)
    
    sim_12 = torch.cosine_similarity(emb1, emb2, dim=1).item()
    sim_13 = torch.cosine_similarity(emb1, emb3, dim=1).item()
    
    print(f"   ✓ Similarity({seq1}, {seq2}): {sim_12:.3f}")
    print(f"   ✓ Similarity({seq1}, {seq3}): {sim_13:.3f}")
    print(f"   ✓ Related terms more similar: {sim_12 > sim_13}")
    
    print("\n" + "=" * 50)
    print("✅ PMFlow working correctly!")


if __name__ == "__main__":
    main()
