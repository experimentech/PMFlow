"""
Simple example of training a PMFlow Language Model.

This demonstrates:
- Creating a PMFlow LM
- Basic training loop
- Generation
- Visualization of field structure
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pmflow.lm import PMFlowLanguageModel
from pmflow.lm.train import PMFlowLMTrainer, WarmupScheduler


def main():
    print("=" * 70)
    print("PMFlow Language Model - Simple Example")
    print("=" * 70)
    
    # Configuration
    vocab_size = 1000
    embedding_dim = 128
    latent_dim = 64
    n_centers = 32
    batch_size = 4
    seq_len = 32
    num_epochs = 3
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n1. Creating model...")
    print(f"   Device: {device}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Centers: {n_centers}")
    
    model = PMFlowLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        n_centers=n_centers,
        steps_per_token=4,
        enable_flow=True,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    # Setup training
    print(f"\n2. Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=num_epochs * 100
    )
    
    trainer = PMFlowLMTrainer(model, optimizer, device)
    print(f"   ✓ Optimizer: AdamW")
    print(f"   ✓ Scheduler: Warmup + Cosine Decay")
    
    # Generate synthetic data
    print(f"\n3. Generating synthetic data...")
    def generate_batch():
        """Generate random token sequences."""
        return torch.randint(0, vocab_size, (batch_size, seq_len))
    
    train_batches = [generate_batch() for _ in range(100)]
    val_batches = [generate_batch() for _ in range(10)]
    print(f"   ✓ Training batches: {len(train_batches)}")
    print(f"   ✓ Validation batches: {len(val_batches)}")
    
    # Training loop
    print(f"\n4. Training ({num_epochs} epochs)...")
    print("-" * 70)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ppl = 0.0
        
        for batch_idx, batch in enumerate(train_batches):
            metrics = trainer.train_step(batch)
            epoch_loss += metrics['loss']
            epoch_ppl += metrics['perplexity']
            scheduler.step()
            
            if (batch_idx + 1) % 25 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_ppl = epoch_ppl / (batch_idx + 1)
                lr = metrics['learning_rate']
                grad_norm = metrics['grad_norm']
                print(f"Epoch {epoch+1}, Batch {batch_idx+1:3d}: "
                      f"Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}, "
                      f"LR={lr:.2e}, GradNorm={grad_norm:.4f}")
        
        # Validation
        print(f"\n   Validating...")
        val_loss = 0.0
        val_ppl = 0.0
        for val_batch in val_batches:
            metrics = trainer.eval_step(val_batch)
            val_loss += metrics['loss']
            val_ppl += metrics['perplexity']
        
        val_loss /= len(val_batches)
        val_ppl /= len(val_batches)
        
        print(f"   Epoch {epoch+1} - "
              f"Train Loss: {epoch_loss/len(train_batches):.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val PPL: {val_ppl:.2f}")
        print()
    
    print("-" * 70)
    print(f"\n5. Analyzing learned field...")
    
    centers = model.get_field_centers()
    masses = model.get_field_masses()
    intents = model.get_field_intents()
    
    print(f"   Centers shape: {centers.shape}")
    print(f"   Centers range: [{centers.min():.4f}, {centers.max():.4f}]")
    print(f"   Masses shape: {masses.shape}")
    print(f"   Masses range: [{masses.min():.4f}, {masses.max():.4f}]")
    print(f"   Masses mean: {masses.mean():.4f}")
    
    if intents is not None:
        print(f"   Intents (omegas) shape: {intents.shape}")
        print(f"   Intents range: [{intents.min():.4f}, {intents.max():.4f}]")
        print(f"   Intents mean: {intents.mean():.4f}")
    
    print(f"\n6. Generation example...")
    
    # Simple prompt: sequence of random tokens
    prompt = [torch.randint(0, vocab_size, (1,)).item() for _ in range(3)]
    print(f"   Prompt: {prompt}")
    
    generated = model.generate(
        prompt_ids=prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50
    )
    print(f"   Generated: {generated}")
    
    print(f"\n7. Saving checkpoint...")
    checkpoint_path = "/tmp/pmflow_lm_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=num_epochs)
    print(f"   ✓ Saved to {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
