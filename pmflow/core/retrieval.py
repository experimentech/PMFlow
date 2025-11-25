"""
PMFlow Extensions for Semantic Retrieval

Extensions that enhance retrieval capabilities while preserving PMFlow's
embarrassingly parallel nature. All operations are stateless and vectorized.

Key principles:
1. No inter-sample dependencies (embarrassingly parallel preserved)
2. Vectorized operations (batch processing efficient)
3. No persistent state changes during forward pass
4. All extensions are optional add-ons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from .pmflow import ParallelPMField, MultiScalePMField


class QueryExpansionPMField(nn.Module):
    """
    Query expansion through gravitational attraction.
    
    Given a query embedding, compute which PMField centers are most attracted
    to it, then use those centers to expand the query into related concepts.
    
    This is embarrassingly parallel - each query processed independently.
    """
    
    def __init__(self, pm_field: ParallelPMField, expansion_k: int = 5):
        """
        Args:
            pm_field: Underlying PMField
            expansion_k: Number of nearest centers to use for expansion
        """
        super().__init__()
        self.pm_field = pm_field
        self.expansion_k = expansion_k
    
    def compute_center_attractions(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational attraction to each center.
        
        Returns attraction weights (high = strong pull toward that center)
        Embarrassingly parallel - no inter-query dependencies.
        """
        # Get centers
        if hasattr(self.pm_field, 'fine_field'):
            # MultiScalePMField - use fine field for precision
            centers = self.pm_field.fine_field.centers
            mus = self.pm_field.fine_field.mus
        else:
            centers = self.pm_field.centers
            mus = self.pm_field.mus
        
        # Compute distances: (B, N)
        dists = torch.cdist(z, centers)  # (B, N)
        
        # Gravitational attraction: μ / r²
        attractions = mus.unsqueeze(0) / (dists ** 2 + 1e-6)  # (B, N)
        
        return attractions
    
    def expand_query(self, z: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand query to include related center embeddings.
        
        Args:
            z: Query latent (B, D)
            top_k: Number of centers to include (default: expansion_k)
            
        Returns:
            expanded_z: Weighted combination of query + related centers (B, D)
            weights: Attraction weights for each center (B, N)
        """
        k = top_k or self.expansion_k
        
        # Compute attractions
        attractions = self.compute_center_attractions(z)  # (B, N)
        
        # Get top-k centers
        top_k_values, top_k_indices = torch.topk(attractions, k, dim=1)  # (B, k)
        
        # Normalize weights
        weights = F.softmax(top_k_values, dim=1)  # (B, k)
        
        # Get center embeddings
        if hasattr(self.pm_field, 'fine_field'):
            centers = self.pm_field.fine_field.centers
        else:
            centers = self.pm_field.centers
        
        # Gather top-k centers
        batch_size = z.shape[0]
        expanded_centers = torch.stack([
            centers[top_k_indices[i]] for i in range(batch_size)
        ])  # (B, k, D)
        
        # Weighted combination
        expanded_z = torch.sum(
            expanded_centers * weights.unsqueeze(2),
            dim=1
        )  # (B, D)
        
        # Blend with original query
        expanded_z = 0.7 * z + 0.3 * expanded_z
        
        return expanded_z, attractions
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with query expansion.
        
        Returns:
            output: PMFlow output from expanded query
            attractions: Center attraction weights
        """
        expanded_z, attractions = self.expand_query(z)
        output = self.pm_field(expanded_z)
        return output, attractions


class SemanticNeighborhoodPMField(nn.Module):
    """
    Find semantic neighbors based on PMFlow energy landscape.
    
    Two concepts are neighbors if they:
    1. Are attracted to similar PMField centers
    2. Experience similar gravitational fields
    
    Embarrassingly parallel - can process entire batches independently.
    """
    
    def __init__(self, pm_field: ParallelPMField):
        super().__init__()
        self.pm_field = pm_field
    
    def compute_field_signature(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational field signature at position z.
        
        Signature captures which centers influence this position.
        Embarrassingly parallel across batch dimension.
        """
        if hasattr(self.pm_field, 'fine_field'):
            centers = self.pm_field.fine_field.centers
            mus = self.pm_field.fine_field.mus
        else:
            centers = self.pm_field.centers
            mus = self.pm_field.mus
        
        # Distance to each center
        dists = torch.cdist(z, centers)  # (B, N)
        
        # Gravitational contribution from each center
        contributions = mus.unsqueeze(0) / (dists + 1e-6)  # (B, N)
        
        # Normalize to get signature
        signature = F.normalize(contributions, p=2, dim=1)  # (B, N)
        
        return signature
    
    def find_neighbors(
        self, 
        query_z: torch.Tensor, 
        candidate_z: torch.Tensor,
        threshold: float = 0.85
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find semantic neighbors using field signature similarity.
        
        Args:
            query_z: Query latent (1, D)
            candidate_z: Candidate latents (N, D)
            threshold: Similarity threshold for neighbors
            
        Returns:
            neighbor_indices: Indices of neighbors
            similarities: Similarity scores
        """
        # Compute signatures
        query_sig = self.compute_field_signature(query_z)  # (1, N_centers)
        candidate_sigs = self.compute_field_signature(candidate_z)  # (N, N_centers)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_sig.expand(candidate_z.shape[0], -1),
            candidate_sigs,
            dim=1
        )  # (N,)
        
        # Filter by threshold
        neighbor_mask = similarities >= threshold
        neighbor_indices = torch.where(neighbor_mask)[0]
        
        return neighbor_indices, similarities[neighbor_indices]
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both output and field signature.
        
        Returns:
            output: PMFlow output
            signature: Field signature for neighbor finding
        """
        output = self.pm_field(z)
        signature = self.compute_field_signature(z)
        return output, signature


class HierarchicalRetrievalPMField(nn.Module):
    """
    Multi-scale retrieval using PMFlow's hierarchical structure.
    
    Coarse field: Find broad categories
    Fine field: Find specific instances
    
    Embarrassingly parallel - independent processing per query.
    """
    
    def __init__(self, pm_field: MultiScalePMField):
        """
        Args:
            pm_field: Must be MultiScalePMField for hierarchical structure
        """
        super().__init__()
        if not isinstance(pm_field, MultiScalePMField):
            raise ValueError("HierarchicalRetrievalPMField requires MultiScalePMField")
        self.pm_field = pm_field
    
    def retrieve_hierarchical(
        self,
        query_z: torch.Tensor,
        candidate_z: torch.Tensor,
        coarse_threshold: float = 0.70,
        fine_threshold: float = 0.85
    ) -> Dict[str, torch.Tensor]:
        """
        Two-stage hierarchical retrieval.
        
        Stage 1: Coarse filtering (broad categories)
        Stage 2: Fine matching (specific concepts)
        
        Args:
            query_z: Query latent (1, D)
            candidate_z: Candidate latents (N, D)
            coarse_threshold: Category-level threshold
            fine_threshold: Instance-level threshold
            
        Returns:
            results: Dict with category_matches, instance_matches, scores
        """
        # Forward through multi-scale field
        q_fine, q_coarse, q_combined = self.pm_field(query_z)
        c_fine, c_coarse, c_combined = self.pm_field(candidate_z)
        
        # Stage 1: Coarse filtering (categories)
        coarse_sims = F.cosine_similarity(
            q_coarse.expand(candidate_z.shape[0], -1),
            c_coarse,
            dim=1
        )
        category_mask = coarse_sims >= coarse_threshold
        category_matches = torch.where(category_mask)[0]
        
        # Stage 2: Fine matching (only on category matches)
        if len(category_matches) > 0:
            fine_candidates = c_fine[category_matches]
            fine_sims = F.cosine_similarity(
                q_fine.expand(fine_candidates.shape[0], -1),
                fine_candidates,
                dim=1
            )
            fine_mask = fine_sims >= fine_threshold
            instance_matches = category_matches[fine_mask]
            instance_scores = fine_sims[fine_mask]
        else:
            instance_matches = torch.tensor([], dtype=torch.long)
            instance_scores = torch.tensor([])
        
        return {
            'category_matches': category_matches,
            'category_scores': coarse_sims[category_matches],
            'instance_matches': instance_matches,
            'instance_scores': instance_scores,
            'coarse_embeddings': c_coarse,
            'fine_embeddings': c_fine
        }
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through multi-scale field."""
        return self.pm_field(z)


class AttentionWeightedRetrieval(nn.Module):
    """
    Use PMFlow gravitational field to compute retrieval attention weights.
    
    Concept: Nearby points in the gravitational field should have higher
    retrieval relevance. Attention weights based on field geometry.
    
    Embarrassingly parallel - vectorized attention computation.
    """
    
    def __init__(self, pm_field: ParallelPMField, temperature: float = 0.1):
        super().__init__()
        self.pm_field = pm_field
        self.temperature = temperature
    
    def compute_field_attention(
        self,
        query_z: torch.Tensor,
        candidate_z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights based on gravitational field similarity.
        
        Args:
            query_z: Query latent (1, D)
            candidate_z: Candidate latents (N, D)
            
        Returns:
            attention: Normalized attention weights (N,)
        """
        # Get centers
        if hasattr(self.pm_field, 'fine_field'):
            centers = self.pm_field.fine_field.centers
            mus = self.pm_field.fine_field.mus
        else:
            centers = self.pm_field.centers
            mus = self.pm_field.mus
        
        # Compute gravitational potential at each point
        # U(r) = Σ μᵢ / |r - rᵢ|
        
        def potential(z):
            dists = torch.cdist(z, centers)  # (B, N_centers)
            pot = torch.sum(mus.unsqueeze(0) / (dists + 1e-6), dim=1)  # (B,)
            return pot
        
        query_potential = potential(query_z)  # (1,)
        candidate_potentials = potential(candidate_z)  # (N,)
        
        # Attention based on potential similarity
        # High attention = similar gravitational potential
        potential_diffs = torch.abs(
            query_potential - candidate_potentials
        )  # (N,)
        
        # Convert to attention (lower diff = higher attention)
        attention_logits = -potential_diffs / self.temperature
        attention = F.softmax(attention_logits, dim=0)
        
        return attention
    
    def retrieve_weighted(
        self,
        query_z: torch.Tensor,
        candidate_z: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k candidates weighted by field attention.
        
        Args:
            query_z: Query latent (1, D)
            candidate_z: Candidate latents (N, D)
            top_k: Number of candidates to return
            
        Returns:
            indices: Top-k candidate indices
            weights: Attention weights for top-k
        """
        attention = self.compute_field_attention(query_z, candidate_z)
        top_k_weights, top_k_indices = torch.topk(attention, min(top_k, len(attention)))
        return top_k_indices, top_k_weights
    
    def forward(
        self,
        query_z: torch.Tensor,
        candidate_z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention-weighted retrieval.
        
        Returns:
            query_output: PMFlow output for query
            attention: Attention weights for candidates
        """
        query_output = self.pm_field(query_z)
        attention = self.compute_field_attention(query_z, candidate_z)
        return query_output, attention


# Utility functions for combining extensions

def create_enhanced_retrieval_encoder(
    pm_field: MultiScalePMField,
    enable_expansion: bool = True,
    enable_neighbors: bool = True,
    enable_hierarchical: bool = True,
    enable_attention: bool = True
) -> Dict[str, nn.Module]:
    """
    Create a suite of retrieval-enhanced PMFlow modules.
    
    All modules share the same underlying PMField, preserving
    embarrassingly parallel semantics while adding retrieval capabilities.
    
    Args:
        pm_field: Base MultiScalePMField
        enable_*: Flags to enable specific extensions
        
    Returns:
        modules: Dict of enhanced retrieval modules
    """
    modules = {'base': pm_field}
    
    if enable_expansion:
        modules['expansion'] = QueryExpansionPMField(pm_field, expansion_k=5)
    
    if enable_neighbors:
        modules['neighbors'] = SemanticNeighborhoodPMField(pm_field)
    
    if enable_hierarchical:
        modules['hierarchical'] = HierarchicalRetrievalPMField(pm_field)
    
    if enable_attention:
        modules['attention'] = AttentionWeightedRetrieval(pm_field, temperature=0.1)
    
    return modules


# Example usage for compositional architecture

class CompositionalRetrievalPMField(nn.Module):
    """
    All-in-one retrieval-enhanced PMField for compositional architecture.
    
    Combines:
    - Query expansion (find related concepts)
    - Hierarchical retrieval (category → instance)
    - Attention weighting (relevance scoring)
    - Neighborhood finding (semantic clustering)
    
    All embarrassingly parallel - no inter-sample dependencies.
    """
    
    def __init__(self, pm_field: MultiScalePMField):
        super().__init__()
        self.pm_field = pm_field
        self.expansion = QueryExpansionPMField(pm_field)
        self.hierarchical = HierarchicalRetrievalPMField(pm_field)
        self.attention = AttentionWeightedRetrieval(pm_field)
        self.neighbors = SemanticNeighborhoodPMField(pm_field)
    
    def retrieve_concepts(
        self,
        query_z: torch.Tensor,
        concept_z: torch.Tensor,
        expand_query: bool = True,
        use_hierarchical: bool = True,
        min_similarity: float = 0.40
    ) -> List[Tuple[int, float]]:
        """
        Comprehensive concept retrieval using all enhancement techniques.
        
        Args:
            query_z: Query latent (1, D)
            concept_z: Concept latents (N, D)
            expand_query: Whether to expand query with related centers
            use_hierarchical: Whether to use hierarchical filtering
            min_similarity: Minimum similarity threshold
            
        Returns:
            results: List of (concept_idx, score) tuples
        """
        # Optional: Expand query
        if expand_query:
            query_z, _ = self.expansion.expand_query(query_z)
        
        # Hierarchical retrieval
        if use_hierarchical:
            h_results = self.hierarchical.retrieve_hierarchical(
                query_z, concept_z,
                coarse_threshold=min_similarity * 0.8,
                fine_threshold=min_similarity
            )
            candidates = h_results['instance_matches']
            scores = h_results['instance_scores']
        else:
            # Simple similarity
            query_emb = self.pm_field(query_z)[2]  # Combined embedding
            concept_embs = self.pm_field(concept_z)[2]
            sims = F.cosine_similarity(
                query_emb.expand(concept_embs.shape[0], -1),
                concept_embs,
                dim=1
            )
            mask = sims >= min_similarity
            candidates = torch.where(mask)[0]
            scores = sims[mask]
        
        # Sort by score
        if len(candidates) > 0:
            sorted_indices = torch.argsort(scores, descending=True)
            results = [
                (candidates[i].item(), scores[sorted_indices[i]].item())
                for i in range(len(sorted_indices))
            ]
        else:
            results = []
        
        return results
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard forward through multi-scale field."""
        return self.pm_field(z)
