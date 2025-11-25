"""
Contrastive-Friendly PMFlow Extension

This module extends PMFlow with contrastive learning capabilities while
respecting the physical semantics of the Pushing-Medium framework.

Key insight: The PMFlow forward pass includes normalization that dampens
center changes. Solution: Add a learnable post-processing layer that can
shape the embedding space without disrupting PMFlow's gravitational dynamics.

Architecture:
1. Input → PMFlow (gravitational dynamics) → Raw embedding
2. Raw embedding → Learnable projection → Output embedding
3. Contrastive learning updates BOTH centers AND projection

This allows:
- PMFlow centers to organize semantic basins
- Projection to fine-tune similarities for contrastive objectives
- Physical semantics preserved (gravity wells still meaningful)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .pmflow import ParallelPMField, MultiScalePMField


class ContrastivePMField(nn.Module):
    """
    PMField with learnable output projection for contrastive learning.
    
    This wraps a standard PMField and adds a lightweight projection layer
    that can be trained via contrastive objectives. The PMField centers
    still organize semantic structure, but the projection can fine-tune
    similarities without fighting against normalization.
    
    Think of it as:
    - PMField centers = coarse semantic organization (mountains)
    - Projection weights = fine-tuning layer (water flow patterns)
    """
    
    def __init__(self, 
                 pm_field: ParallelPMField,
                 output_dim: Optional[int] = None,
                 projection_type: str = "linear"):
        """
        Args:
            pm_field: Underlying PMField (can be ParallelPMField or MultiScalePMField)
            output_dim: Output dimension (default: same as input)
            projection_type: "linear", "residual", or "identity"
                - linear: Simple learned projection
                - residual: Learned delta added to PMFlow output
                - identity: No projection (for testing)
        """
        super().__init__()
        self.pm_field = pm_field
        self.projection_type = projection_type
        
        # Determine input/output dimensions
        if hasattr(pm_field, 'fine_field'):
            # MultiScalePMField - output is concatenated
            sample_input = torch.zeros(1, pm_field.fine_field.centers.shape[1])
            with torch.no_grad():
                sample_output = pm_field(sample_input)
                if isinstance(sample_output, tuple):
                    input_dim = sample_output[2].shape[1]  # Combined dimension
                else:
                    input_dim = sample_output.shape[1]
        else:
            # Standard PMField
            input_dim = pm_field.centers.shape[1]
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        
        # Create projection layer
        if projection_type == "linear":
            self.projection = nn.Linear(input_dim, self.output_dim, bias=False)
            # Initialize to near-identity
            nn.init.eye_(self.projection.weight[:min(input_dim, self.output_dim), 
                                                 :min(input_dim, self.output_dim)])
            if self.output_dim > input_dim:
                nn.init.normal_(self.projection.weight[input_dim:], std=0.01)
            elif self.output_dim < input_dim:
                nn.init.normal_(self.projection.weight[:, self.output_dim:], std=0.01)
                
        elif projection_type == "residual":
            # Residual projection: output = pmflow_out + W @ pmflow_out
            self.projection = nn.Linear(input_dim, input_dim, bias=False)
            # Initialize to zero (start as identity)
            nn.init.zeros_(self.projection.weight)
            
        elif projection_type == "identity":
            self.projection = None
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PMFlow + projection.
        
        Args:
            z: Input latent tensor
            
        Returns:
            projected: PMFlow output projected to contrastive space
        """
        # PMFlow gravitational dynamics
        pm_output = self.pm_field(z)
        
        # Handle MultiScalePMField
        if isinstance(pm_output, tuple) and len(pm_output) == 3:
            pm_output = pm_output[2]  # Use combined multi-scale output
        
        # Apply projection
        if self.projection is None:
            return pm_output
        elif self.projection_type == "residual":
            return pm_output + self.projection(pm_output)
        else:  # linear
            return self.projection(pm_output)
    
    def get_pm_centers(self):
        """Access PMField centers for plasticity updates."""
        if hasattr(self.pm_field, 'fine_field'):
            # MultiScalePMField
            return {
                'fine': self.pm_field.fine_field.centers,
                'coarse': self.pm_field.coarse_field.centers
            }
        else:
            return self.pm_field.centers
    
    def get_pm_mus(self):
        """Access PMField gravitational strengths."""
        if hasattr(self.pm_field, 'fine_field'):
            return {
                'fine': self.pm_field.fine_field.mus,
                'coarse': self.pm_field.coarse_field.mus
            }
        else:
            return self.pm_field.mus


def contrastive_learning_step(
    model: ContrastivePMField,
    similar_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    dissimilar_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    center_lr: float = 1e-4,
    projection_lr: float = 1e-3,
    margin: float = 0.2,
    temperature: float = 0.1
) -> dict:
    """
    Single contrastive learning step for ContrastivePMField.
    
    Updates both PMField centers (coarse semantic organization) and
    projection weights (fine-grained similarity tuning).
    
    Args:
        model: ContrastivePMField to update
        similar_pairs: List of (z1, z2) that should be similar
        dissimilar_pairs: List of (z1, z2) that should be dissimilar
        center_lr: Learning rate for PMField centers
        projection_lr: Learning rate for projection layer
        margin: Margin for contrastive loss
        temperature: Temperature for similarity scaling
        
    Returns:
        metrics: Dict with loss components and similarities
    """
    # Separate optimizers for centers and projection
    centers = model.get_pm_centers()
    mus = model.get_pm_mus()
    
    # Handle MultiScalePMField
    if isinstance(centers, dict):
        center_params = list(centers.values()) + list(mus.values())
    else:
        center_params = [centers, mus]
    
    center_optimizer = torch.optim.SGD(center_params, lr=center_lr)
    
    if model.projection is not None:
        proj_optimizer = torch.optim.Adam(model.projection.parameters(), lr=projection_lr)
    else:
        proj_optimizer = None
    
    # Compute embeddings
    similar_loss = 0.0
    dissimilar_loss = 0.0
    
    # Similar pairs - minimize distance
    if similar_pairs:
        for z1, z2 in similar_pairs:
            emb1 = model(z1.unsqueeze(0))
            emb2 = model(z2.unsqueeze(0))
            
            # Cosine similarity (closer to 1 is better)
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            
            # Loss: want similarity close to 1
            similar_loss += (1.0 - sim).mean()
    
    # Dissimilar pairs - maximize distance
    if dissimilar_pairs:
        for z1, z2 in dissimilar_pairs:
            emb1 = model(z1.unsqueeze(0))
            emb2 = model(z2.unsqueeze(0))
            
            # Cosine similarity
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            
            # Loss: want similarity less than (1 - margin)
            target = 1.0 - margin
            dissimilar_loss += torch.relu(sim - target).mean()
    
    # Total loss
    total_loss = similar_loss + dissimilar_loss
    
    # Backward pass
    if total_loss.requires_grad:
        center_optimizer.zero_grad()
        if proj_optimizer is not None:
            proj_optimizer.zero_grad()
        
        total_loss.backward()
        
        center_optimizer.step()
        if proj_optimizer is not None:
            proj_optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        avg_similar_sim = 0.0
        if similar_pairs:
            for z1, z2 in similar_pairs:
                emb1 = model(z1.unsqueeze(0))
                emb2 = model(z2.unsqueeze(0))
                avg_similar_sim += F.cosine_similarity(emb1, emb2, dim=1).item()
            avg_similar_sim /= len(similar_pairs)
        
        avg_dissimilar_sim = 0.0
        if dissimilar_pairs:
            for z1, z2 in dissimilar_pairs:
                emb1 = model(z1.unsqueeze(0))
                emb2 = model(z2.unsqueeze(0))
                avg_dissimilar_sim += F.cosine_similarity(emb1, emb2, dim=1).item()
            avg_dissimilar_sim /= len(dissimilar_pairs)
    
    return {
        'total_loss': total_loss.item() if total_loss.requires_grad else 0.0,
        'similar_loss': similar_loss.item() if isinstance(similar_loss, torch.Tensor) else similar_loss,
        'dissimilar_loss': dissimilar_loss.item() if isinstance(dissimilar_loss, torch.Tensor) else dissimilar_loss,
        'avg_similar_sim': avg_similar_sim,
        'avg_dissimilar_sim': avg_dissimilar_sim,
        'separation': avg_similar_sim - avg_dissimilar_sim
    }


def train_contrastive_pmfield(
    model: ContrastivePMField,
    similar_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    dissimilar_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 100,
    center_lr: float = 1e-4,
    projection_lr: float = 1e-3,
    margin: float = 0.2,
    temperature: float = 0.1,
    verbose: bool = True
) -> List[dict]:
    """
    Train ContrastivePMField via contrastive learning.
    
    This function runs multiple epochs of contrastive updates, tracking
    progress and returning metrics history.
    
    Args:
        model: ContrastivePMField to train
        similar_pairs: List of similar (z1, z2) latent pairs
        dissimilar_pairs: List of dissimilar (z1, z2) latent pairs
        epochs: Number of training epochs
        center_lr: Learning rate for PMField centers
        projection_lr: Learning rate for projection layer
        margin: Contrastive margin
        temperature: Similarity temperature
        verbose: Whether to print progress
        
    Returns:
        history: List of metric dicts per epoch
    """
    history = []
    
    for epoch in range(epochs):
        metrics = contrastive_learning_step(
            model, similar_pairs, dissimilar_pairs,
            center_lr=center_lr,
            projection_lr=projection_lr,
            margin=margin,
            temperature=temperature
        )
        
        history.append(metrics)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d}: "
                  f"Loss={metrics['total_loss']:.4f}, "
                  f"Similar={metrics['avg_similar_sim']:.3f}, "
                  f"Dissimilar={metrics['avg_dissimilar_sim']:.3f}, "
                  f"Sep={metrics['separation']:.3f}")
    
    return history


# Example usage function
def create_contrastive_encoder(base_encoder, projection_type="residual"):
    """
    Create a contrastive-trainable version of PMFlowEmbeddingEncoder.
    
    This wraps the existing PMField with ContrastivePMField, enabling
    contrastive learning without disrupting the encoder's architecture.
    
    Args:
        base_encoder: PMFlowEmbeddingEncoder instance
        projection_type: "linear", "residual", or "identity"
        
    Returns:
        contrastive_field: ContrastivePMField wrapping the base encoder's PMField
    """
    # Extract the PMField from the encoder
    pm_field = base_encoder.pm_field
    
    # Wrap it in ContrastivePMField
    contrastive_field = ContrastivePMField(
        pm_field=pm_field,
        projection_type=projection_type
    )
    
    return contrastive_field
