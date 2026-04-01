"""
PMFlow Language Model: Language modeling via gravitational particle dynamics.

Core Concept:
  Each token's latent representation evolves through a learned gravitational field
  (ParallelPMField) shaped by attractors (centers). The evolution follows Eq. 7 from
  the physics model:
  
    a = -c²∇ln(n)
  
  where n(r) = 1 + Σ μᵢ/|r - rᵢ| (refractive index from point masses).
  
  Tokens process left-to-right, with each token's evolved state providing context
  for the next token via learned gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from pmflow.core.pmflow import ParallelPMField


class ContextMixer(nn.Module):
    """
    Learnable gating mechanism to blend current token with previous context.
    
    The gravitational field has no inherent memory, so we add context-aware
    state evolution via learned mixing.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Gating network: learns how much to preserve vs. update
        self.gate_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        
        # Blending network: learns how to combine
        self.blend_net = nn.Linear(2 * latent_dim, latent_dim)
    
    def forward(self, current: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Mix current token latent with previous position's evolved state.
        
        Args:
            current: (batch, latent_dim) - current token embedding
            context: (batch, latent_dim) - previous position's evolved state
        
        Returns:
            mixed: (batch, latent_dim) - blended representation
        """
        combined = torch.cat([current, context], dim=-1)  # (batch, 2*latent_dim)
        
        # Learn gate: how much context to preserve
        gate = self.gate_net(combined)  # (batch, 1)
        
        # Learn blend: how to combine
        blend = torch.tanh(self.blend_net(combined))  # (batch, latent_dim)
        
        # Mix: preserve some context, mix in new token information
        mixed = gate * context + (1 - gate) * blend
        
        return mixed


class PMFlowLanguageModel(nn.Module):
    """
    Language model where tokens evolve through a learned gravitational field.
    
    Architecture:
      1. Embed tokens to fixed dimension
      2. Project to latent space
      3. Mix with previous context via learned gating
      4. Evolve through gravitational field (ParallelPMField)
      5. Project evolved latent to vocabulary logits
      6. Use evolved state as context for next token
    
    The gravitational field (centers + masses) is shared across the sequence,
    learning a universal "semantic attractor landscape" that guides token transitions.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 latent_dim: int = 64,
                 n_centers: int = 64,
                 steps_per_token: int = 4,
                 dt: float = 0.15,
                 beta: float = 1.2,
                 clamp: float = 3.0,
                 enable_flow: bool = True,
                 dropout: float = 0.1):
        """
        Initialize PMFlow Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            latent_dim: Dimension of latent space where physics acts
            n_centers: Number of gravitational centers (attractors)
            steps_per_token: PMFlow evolution steps per token
            dt: Time step for physics integration
            beta: Scaling factor for gravitational acceleration
            clamp: Gradient clipping value
            enable_flow: Enable frame-dragging flow field
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.n_centers = n_centers
        self.steps_per_token = steps_per_token
        
        # Token embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
        # Project embedding to latent space
        self.to_latent = nn.Linear(embedding_dim, latent_dim)
        self.dropout_to_latent = nn.Dropout(dropout)
        
        # Context mixing network
        self.context_mixer = ContextMixer(latent_dim)
        
        # The gravitational field (shared across sequence)
        # This learns the "semantic attractor landscape"
        self.pm_field = ParallelPMField(
            d_latent=latent_dim,
            n_centers=n_centers,
            steps=steps_per_token,
            dt=dt,
            beta=beta,
            clamp=clamp,
            enable_flow=enable_flow,
            temporal_parallel=True,
            chunk_size=16
        )
        
        # Project evolved latent to vocabulary
        self.output_proj = nn.Linear(latent_dim, vocab_size)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward_next_token(self,
                          token_id: torch.Tensor,
                          context_latent: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next token given current token and optional context.
        
        Physics interpretation:
          The current token's latent representation is a particle that evolves
          through the gravitational field, guided by semantic attractors (centers).
          Its final position determines the next-token probabilities.
        
        Args:
            token_id: (batch,) token IDs
            context_latent: (batch, latent_dim) evolved state from previous token, or None
        
        Returns:
            logits: (batch, vocab_size) next token logits
            evolved_latent: (batch, latent_dim) evolved state to use as context for next token
        """
        # Embed token
        x = self.embed(token_id)  # (batch, embed_dim)
        z = self.to_latent(x)
        z = self.dropout_to_latent(z)
        
        # Mix with context if provided
        if context_latent is not None:
            z = self.context_mixer(z, context_latent)
        
        # Core: Evolve particle through gravitational field
        # This implements the physics: particle trajectory is determined by
        # the gradient of the refractive index field
        z_evolved = self.pm_field(z)  # (batch, latent_dim)
        
        # Project to vocabulary
        logits = self.output_proj(z_evolved)
        logits = self.dropout_out(logits)
        
        return logits, z_evolved
    
    def forward_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Process a full sequence for training (batch-parallel version).
        
        Exploits PMFlow's embarrassingly parallel center computations by
        treating all batch*seq_len positions as a single large batch.
        This allows temporal parallelism and chunked processing to activate.
        
        Trade-off: Loses sequential RNN-like context between tokens
        Benefit: 3-5x speedup via PMFlow's vectorized operations
        
        Args:
            token_ids: (batch, seq_len) token IDs
        
        Returns:
            logits: (batch, seq_len, vocab_size) next-token predictions
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Embed all tokens at once
        x = self.embed(token_ids)  # (batch, seq_len, embed_dim)
        
        # Project to latent space
        z = self.to_latent(x)  # (batch, seq_len, latent_dim)
        self.dropout_to_latent.eval() if not self.training else None
        z = self.dropout_to_latent(z)
        
        # CRITICAL OPTIMIZATION:
        # Reshape to treat all positions as a large batch
        # (batch*seq_len, latent_dim)
        # This makes PMFlow see a large batch and activates:
        #   1. Temporal parallelism (checks batch size vs chunk_size)
        #   2. Vectorized center operations (O(D) per batch element)
        #   3. Chunk-based processing for even larger effective batches
        z_flat = z.reshape(batch_size * seq_len, -1)
        
        # PMFlow evolution now sees large batch
        # With parallelism enabled, this processes chunks in parallel
        z_evolved = self.pm_field(z_flat)  # (batch*seq_len, latent_dim)
        
        # Reshape back to sequence format
        z_evolved = z_evolved.reshape(batch_size, seq_len, -1)
        
        # Project evolved latents to vocabulary
        logits = self.output_proj(z_evolved)  # (batch, seq_len, vocab_size)
        logits = self.dropout_out(logits)
        
        return logits
    
    def generate(self,
                prompt_ids: List[int],
                max_new_tokens: int = 100,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> List[int]:
        """
        Generate tokens autoregressively from a prompt.
        
        Args:
            prompt_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k highest probability tokens
            top_p: Nucleus sampling parameter
        
        Returns:
            List of generated token IDs (including prompt)
        """
        self.eval()
        device = next(self.parameters()).device
        
        generated = list(prompt_ids)
        context = torch.zeros(1, self.latent_dim, device=device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                token_id = torch.tensor([generated[-1]], device=device)
                logits, context = self.forward_next_token(token_id, context)
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumsum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 0] = False  # Keep best token
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), num_samples=1).item()
                
                generated.append(next_token)
        
        self.train()
        return generated
    
    def get_field_centers(self) -> torch.Tensor:
        """
        Get the learned gravitational centers.
        
        Useful for visualization and analysis of the semantic attractor landscape.
        
        Returns:
            centers: (n_centers, latent_dim) positions of gravitational centers
        """
        return self.pm_field.centers.data
    
    def get_field_masses(self) -> torch.Tensor:
        """
        Get the learned gravitational masses.
        
        Indicates the strength of each attractor.
        
        Returns:
            mus: (n_centers,) gravitational masses
        """
        return self.pm_field.mus.data
    
    def get_field_intents(self) -> Optional[torch.Tensor]:
        """
        Get the frame-dragging angular momentum (intent).
        
        Non-zero values indicate intentional bias in the field.
        
        Returns:
            omegas: (n_centers,) angular momentum per center, or None if flow disabled
        """
        if self.pm_field.enable_flow:
            return self.pm_field.omegas.data
        return None
