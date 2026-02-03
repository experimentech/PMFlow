import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class ParallelPMField(nn.Module):
    """
    Vectorized PMField implementation with temporal parallelism.
    
    This implements the core Pushing-Medium gravitational equations:
    - Refractive index: n(r) = 1 + Σμᵢ/|r-rᵢ|
    - Gradient: ∇ln(n) = ∇n/n = -Σ(μᵢ/|r-rᵢ|³)(r-rᵢ)
    - Flow acceleration: a = -c²∇ln(n)
    
    Enhanced with vectorized center operations and batch processing.
    """
    
    def __init__(self, d_latent=8, n_centers=64, steps=4, dt=0.15, beta=1.2, 
                 clamp=3.0, temporal_parallel=True, chunk_size=16, enable_flow=False):
        super().__init__()
        # Better initialization for gravitational centers
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent) * 0.8)  # Slightly wider spread
        # Initialize mus with more variation for better specialization
        self.mus = nn.Parameter(torch.ones(n_centers) * 0.5 + torch.randn(n_centers) * 0.1)
        
        # Initialize omegas for frame-dragging flow (Equation 2)
        # We initialize close to zero so flow is learned/activated intentionally
        self.omegas = nn.Parameter(torch.randn(n_centers) * 0.01)
        self.enable_flow = enable_flow
        
        self.steps = steps
        self.dt = dt
        self.beta = beta
        self.clamp = clamp
        self.temporal_parallel = temporal_parallel
        self.chunk_size = chunk_size
        
        # Cache for vectorized operations
        self.register_buffer('_eye', torch.eye(d_latent))
        self.register_buffer('_eps', torch.tensor(1e-4))
        
    def vectorized_grad_ln_n(self, z: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of ∇ln(n) for all centers simultaneously.
        
        Implements: ∇ln(n) = -Σ(μᵢ/|r-rᵢ|³)(r-rᵢ) / n_total
        where n_total = 1 + Σμᵢ/|r-rᵢ|
        """
        B, D = z.shape
        N = self.centers.shape[0]
        
        # Vectorized distance computation: (B, N, D)
        rvec = z.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, N, D)
        r2 = torch.sum(rvec * rvec, dim=2) + self._eps  # (B, N)
        r = torch.sqrt(r2)  # (B, N)
        
        # Vectorized refractive index: n = 1 + Σμᵢ/rᵢ
        n_contributions = self.mus.unsqueeze(0) / r  # (B, N)
        n_total = 1.0 + torch.sum(n_contributions, dim=1, keepdim=True)  # (B, 1)
        
        # Vectorized gradient computation
        r3 = r2 * r  # (B, N)
        grad_prefactor = -self.mus.unsqueeze(0).unsqueeze(2) / r3.unsqueeze(2)  # (B, N, 1)
        grad_contributions = grad_prefactor * rvec  # (B, N, D)
        grad_ln_n = torch.sum(grad_contributions, dim=1) / n_total  # (B, D)
        
        return grad_ln_n
    
    def vectorized_flow_field(self, z: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of frame-dragging flow u_g(r).
        
        Implements: u_g(r) = Σ Ωᵢ × (r - rᵢ)
        
        For high-dimensional efficiency (O(D)), we approximate the cross product
        using a fixed skew-symmetric permutation (swapping dimension pairs).
        This creates a rotational 'swirl' orthogonal to the radial vector without
        needing full O(D²) rotation matrices.
        """
        B, D = z.shape
        N = self.centers.shape[0]
        
        # Radial vector: r - r_i
        # (B, N, D)
        rvec = z.unsqueeze(1) - self.centers.unsqueeze(0)
        
        # Efficient "Cross Product" in D-dimensions:
        # We rotate the radial vector 90 degrees in pairwise planes.
        # (x0, x1, x2, x3...) -> (-x1, x0, -x3, x2...)
        rvec_rotated = torch.zeros_like(rvec)
        
        # Determine the limit for pairwise swapping (handle odd dimensions)
        limit = (D // 2) * 2
        
        # Swap pairs up to the limit: 
        # e.g. indices [0, 2, 4] take values from [1, 3, 5] (negated)
        # indices [1, 3, 5] take values from [0, 2, 4]
        rvec_rotated[:, :, 0:limit:2] = -rvec[:, :, 1:limit:2]
        rvec_rotated[:, :, 1:limit:2] =  rvec[:, :, 0:limit:2]
        
        # Handle odd dimension case if D is odd (leave last dim as 0, handled by zeros_like)
        
        # Application of Angular Momentum (Omegas)
        # Scale the rotation by the strength of the vortex (Omega)
        # We also weight by distance inverse to localize the swirl? 
        # The raw Equation 2 in the paper is linear (Rigid rotation), 
        # but for cognitive stability we usually want localized effects.
        # We'll use a mild distance decay (1/r) to preventing unbounded flow at infinity
        # while keeping the "frame dragging" spirit.
        
        r2 = torch.sum(rvec * rvec, dim=2) + self._eps
        r_inv = 1.0 / torch.sqrt(r2) # (B, N)
        
        # Flow contributions: Ωᵢ * Rotated(r_vec) * Weight
        # We broadcast omegas: (N,) -> (1, N, 1)
        flow_contributions = self.omegas.view(1, N, 1) * rvec_rotated * r_inv.unsqueeze(2)
        
        # Sum all vortices
        u_g = torch.sum(flow_contributions, dim=1) # (B, D)
        
        return u_g

    def temporal_pipeline_step(self, z: torch.Tensor) -> torch.Tensor:
        """Single temporal step with vectorized operations."""
        # Gradient forces (Gravity/Refraction)
        grad = self.vectorized_grad_ln_n(z)
        velocity = self.beta * grad
        
        # Flow forces (Frame Dragging/Intent)
        if self.enable_flow:
             flow = self.vectorized_flow_field(z)
             velocity = velocity + flow
            
        z_new = z + self.dt * velocity
        return torch.clamp(z_new, -self.clamp, self.clamp)
    
    def parallel_temporal_evolution(self, z: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        """
        Parallel temporal evolution using pipeline overlapping.
        
        For embarrassingly parallel computation, each PMFlow center
        acts independently like gravitational point masses.
        """
        trajectory = []
        if return_trajectory:
            trajectory.append(z)

        if not self.temporal_parallel or self.steps <= 2:
            # Standard sequential evolution
            for _ in range(self.steps):
                z = self.temporal_pipeline_step(z)
                if return_trajectory:
                    trajectory.append(z)
            
            if return_trajectory:
                 return torch.stack(trajectory, dim=1) # (B, Steps+1, D)
            return z
        
        # Pipeline parallel evolution
        B = z.shape[0]
        if B <= self.chunk_size:
            # Small batch - use standard evolution
            for _ in range(self.steps):
                z = self.temporal_pipeline_step(z)
                if return_trajectory:
                    trajectory.append(z)
            
            if return_trajectory:
                 return torch.stack(trajectory, dim=1)
            return z
        
        # Large batch - use chunked parallel processing
        chunks = torch.chunk(z, math.ceil(B / self.chunk_size), dim=0)
        evolved_chunks = []
        traj_chunks = []
        
        for chunk in chunks:
            z_chunk = chunk
            chunk_history = []
            if return_trajectory:
                chunk_history.append(z_chunk)
                
            for _ in range(self.steps):
                z_chunk = self.temporal_pipeline_step(z_chunk)
                if return_trajectory:
                    chunk_history.append(z_chunk)
            
            evolved_chunks.append(z_chunk)
            if return_trajectory:
                traj_chunks.append(torch.stack(chunk_history, dim=1))
        
        if return_trajectory:
            return torch.cat(traj_chunks, dim=0)
        return torch.cat(evolved_chunks, dim=0)
    
    def forward(self, z: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        """Forward pass with temporal parallelism."""
        return self.parallel_temporal_evolution(z, return_trajectory=return_trajectory)
    
    # ========================================================================
    # Agentic Execution API (v0.3.5+)
    # ========================================================================
    
    def step(self, z: torch.Tensor) -> torch.Tensor:
        """
        Single physics step through the gravitational field.
        
        This is the public API for step-by-step trajectory execution,
        enabling reactive replanning in agentic systems.
        
        Args:
            z: Current position in latent space (B, D)
            
        Returns:
            z_new: Position after one physics step (B, D)
        """
        return self.temporal_pipeline_step(z)
    
    def adjust_gravity(
        self, 
        center_idx: int, 
        mu_delta: float = 0.0, 
        omega_delta: float = 0.0,
        mu_scale: float = 1.0,
        omega_scale: float = 1.0,
    ) -> None:
        """
        Dynamically adjust gravitational parameters for a specific center.
        
        This enables reactive adaptation during agentic execution:
        - Increase μ to make a center more attractive (goal achieved)
        - Decrease μ to make it repulsive (dead end encountered)
        - Adjust ω to modify frame-dragging flow (redirect trajectory)
        
        Args:
            center_idx: Index of the center to modify
            mu_delta: Amount to add to gravitational strength (can be negative)
            omega_delta: Amount to add to spin/rotation (can be negative)
            mu_scale: Multiplicative factor for mu (applied after delta)
            omega_scale: Multiplicative factor for omega (applied after delta)
        """
        with torch.no_grad():
            if 0 <= center_idx < self.mus.shape[0]:
                self.mus[center_idx] = (self.mus[center_idx] + mu_delta) * mu_scale
                if hasattr(self, 'omegas'):
                    self.omegas[center_idx] = (self.omegas[center_idx] + omega_delta) * omega_scale
    
    def inject_perturbation(
        self, 
        z: torch.Tensor, 
        perturbation: torch.Tensor,
        blend_factor: float = 0.5,
    ) -> torch.Tensor:
        """
        Inject an external perturbation into the latent position.
        
        This enables incorporating outcome-based feedback into the trajectory
        without re-encoding from text. The physics then naturally adapts.
        
        Args:
            z: Current position (B, D)
            perturbation: External force/displacement to apply (B, D) or (D,)
            blend_factor: How strongly to apply perturbation (0=ignore, 1=full)
            
        Returns:
            z_perturbed: Position with perturbation applied
        """
        if perturbation.dim() == 1:
            perturbation = perturbation.unsqueeze(0)
        
        z_perturbed = z + blend_factor * perturbation
        return torch.clamp(z_perturbed, -self.clamp, self.clamp)
    
    def find_nearest_centers(
        self, 
        z: torch.Tensor, 
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the nearest gravitational centers to a position.
        
        Useful for grounding trajectory points to action nodes.
        
        Args:
            z: Position in latent space (B, D)
            top_k: Number of nearest centers to return
            
        Returns:
            indices: Center indices (B, top_k)
            distances: Distances to centers (B, top_k)
            attractions: Gravitational attractions μ/r (B, top_k)
        """
        # Compute distances
        dists = torch.cdist(z, self.centers)  # (B, N)
        
        # Get top-k nearest
        top_k_dists, top_k_indices = torch.topk(dists, top_k, dim=1, largest=False)
        
        # Compute attractions for nearest centers
        top_k_mus = self.mus[top_k_indices]  # (B, top_k)
        top_k_attractions = top_k_mus / (top_k_dists + 1e-6)
        
        return top_k_indices, top_k_dists, top_k_attractions
    
    def mark_as_hazard(
        self,
        z: torch.Tensor,
        radius: float = 1.0,
        repulsion_strength: float = -0.5,
    ) -> int:
        """
        Mark a region of latent space as a hazard (dead end).
        
        This reduces the μ of nearby centers, making trajectories
        curve away from this region. Implements the "obstacle avoidance"
        aspect of Pushing-Medium physics.
        
        Args:
            z: Center of hazard region (1, D) or (D,)
            radius: Radius of effect
            repulsion_strength: How much to reduce μ (negative = repulsive)
            
        Returns:
            Number of centers affected
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        with torch.no_grad():
            # Find centers within radius
            dists = torch.cdist(z, self.centers).squeeze(0)  # (N,)
            affected_mask = dists < radius
            
            # Reduce mu for affected centers (proportional to proximity)
            proximity = 1.0 - (dists[affected_mask] / radius)
            self.mus[affected_mask] += repulsion_strength * proximity
            
            # Also add counter-rotation to push away
            if hasattr(self, 'omegas') and self.enable_flow:
                self.omegas[affected_mask] *= -1.0  # Reverse spin
            
            return int(affected_mask.sum().item())
    
    def mark_as_attractor(
        self,
        z: torch.Tensor,
        radius: float = 1.0,
        attraction_strength: float = 0.5,
    ) -> int:
        """
        Mark a region of latent space as attractive (successful outcome).
        
        This increases the μ of nearby centers, making trajectories
        curve toward this region.
        
        Args:
            z: Center of attractive region (1, D) or (D,)
            radius: Radius of effect
            attraction_strength: How much to increase μ
            
        Returns:
            Number of centers affected
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        with torch.no_grad():
            # Find centers within radius
            dists = torch.cdist(z, self.centers).squeeze(0)  # (N,)
            affected_mask = dists < radius
            
            # Increase mu for affected centers
            proximity = 1.0 - (dists[affected_mask] / radius)
            self.mus[affected_mask] += attraction_strength * proximity
            
            return int(affected_mask.sum().item())

class VectorizedLateralEI(nn.Module):
    """
    Vectorized lateral excitation-inhibition with memory optimization.
    
    Implements biological cortical column lateral interactions with
    efficient memory usage for large batch processing.
    """
    
    def __init__(self, sigma_e=0.6, sigma_i=1.2, k_e=0.8, k_i=1.0, 
                 gain=0.05, chunk_ei=True, chunk_size=32):
        super().__init__()
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.k_e = k_e
        self.k_i = k_i
        self.gain = gain
        self.chunk_ei = chunk_ei
        self.chunk_size = chunk_size
        
        # Pre-compute constants
        self.register_buffer('_sigma_e2', torch.tensor(2 * sigma_e ** 2))
        self.register_buffer('_sigma_i2', torch.tensor(2 * sigma_i ** 2))
        
    def compute_ei_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """Compute excitation-inhibition kernel efficiently."""
        B = z.shape[0]
        
        if not self.chunk_ei or B <= self.chunk_size:
            # Standard computation for small batches
            dist2 = torch.cdist(z, z).pow(2)
            Ke = self.k_e * torch.exp(-dist2 / self._sigma_e2)
            Ki = self.k_i * torch.exp(-dist2 / self._sigma_i2)
            K = Ke - Ki
            return K / (K.sum(1, keepdim=True) + 1e-6)
        
        # Chunked computation for memory efficiency
        K = torch.zeros(B, B, device=z.device, dtype=z.dtype)
        chunk_size = self.chunk_size
        
        for i in range(0, B, chunk_size):
            i_end = min(i + chunk_size, B)
            z_i = z[i:i_end]
            
            for j in range(0, B, chunk_size):
                j_end = min(j + chunk_size, B)
                z_j = z[j:j_end]
                
                dist2 = torch.cdist(z_i, z_j).pow(2)
                Ke = self.k_e * torch.exp(-dist2 / self._sigma_e2)
                Ki = self.k_i * torch.exp(-dist2 / self._sigma_i2)
                K[i:i_end, j:j_end] = Ke - Ki
        
        # Normalize rows
        K = K / (K.sum(1, keepdim=True) + 1e-6)
        return K
    
    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward pass with vectorized EI computation."""
        with torch.no_grad():
            K = self.compute_ei_kernel(z)
        return self.gain * (K @ h)

class AdaptiveScheduler:
    """
    Adaptive scheduling for temporal parallelism.
    
    Automatically adjusts chunk sizes and parallel strategies
    based on hardware capabilities and batch size.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_fraction = 0.8  # Use 80% of available memory
        self.min_chunk_size = 8
        self.max_chunk_size = 128
        
    def get_optimal_chunk_size(self, batch_size: int, feature_dim: int) -> int:
        """Determine optimal chunk size based on memory constraints."""
        if self.device.type == 'cuda':
            # Estimate memory usage
            memory_per_sample = feature_dim * 4  # float32
            available_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory *= self.memory_fraction
            
            max_samples = int(available_memory / memory_per_sample)
            chunk_size = min(max_samples, batch_size, self.max_chunk_size)
            chunk_size = max(chunk_size, self.min_chunk_size)
        else:
            # CPU - use smaller chunks
            chunk_size = min(batch_size, 32)
        
        return chunk_size
    
    def should_use_temporal_parallel(self, batch_size: int, steps: int) -> bool:
        """Decide whether to use temporal parallelism."""
        return batch_size >= 16 and steps >= 3

@torch.no_grad()
def vectorized_pm_plasticity(pmfield: ParallelPMField, z_batch: torch.Tensor, 
                           h_batch: torch.Tensor, mu_lr=1e-3, c_lr=1e-3):
    """
    Vectorized plasticity update implementing local Hebbian-style learning.
    
    This implements the biological neural adaptation based on activity patterns
    and gravitational center dynamics.
    """
    s2 = 0.8 ** 2
    C = pmfield.centers  # (N, D)
    B, D = z_batch.shape
    N = C.shape[0]
    
    # Vectorized distance computation
    z_expanded = z_batch.unsqueeze(1)  # (B, 1, D)
    C_expanded = C.unsqueeze(0)  # (1, N, D)
    dist2 = torch.sum((C_expanded - z_expanded) ** 2, dim=2)  # (B, N)
    
    # Vectorized weight computation
    W = torch.exp(-dist2 / (2 * s2))  # (B, N)
    
    # Vectorized activity-based updates
    hpow = torch.sum(h_batch * h_batch, dim=1, keepdim=True)  # (B, 1)
    drive = torch.mean(W * hpow, dim=0)  # (N,)
    
    # Update mus (gravitational strengths)
    pmfield.mus.add_(mu_lr * (drive - 0.1 * pmfield.mus))
    
    # Update centers (gravitational positions)
    W_sum = torch.sum(W, dim=0, keepdim=True).T + 1e-6  # (N, 1)
    weighted_z = torch.sum(W.T.unsqueeze(2) * z_batch.unsqueeze(0), dim=1)  # (N, D)
    target = weighted_z / W_sum  # (N, D)
    pmfield.centers.add_(c_lr * (target - C))


# ============================================================================
# Enhanced Features for Lilith Neuro-Symbolic AI (v0.3.0)
# ============================================================================

class MultiScalePMField(nn.Module):
    """
    Multi-scale PMFlow field for hierarchical concept learning.
    
    Combines multiple PMFields at different resolutions to capture
    both fine-grained details and coarse-grained semantic structure,
    matching hierarchical concept taxonomies.
    
    Example use case:
        - Fine scale: hospital vs library (specific locations)
        - Coarse scale: indoor vs outdoor (categories)
    """
    
    def __init__(self, d_latent=64, n_centers_fine=128, n_centers_coarse=32,
                 steps_fine=5, steps_coarse=3, dt=0.15, beta=1.2, clamp=3.0, enable_flow=False):
        super().__init__()
        
        # Fine-grained field for specific concepts
        self.fine_field = ParallelPMField(
            d_latent=d_latent,
            n_centers=n_centers_fine,
            steps=steps_fine,
            dt=dt,
            beta=beta,
            clamp=clamp,
            enable_flow=enable_flow
        )
        
        # Coarse-grained field for categories
        coarse_dim = d_latent // 2
        self.coarse_field = ParallelPMField(
            d_latent=coarse_dim,
            n_centers=n_centers_coarse,
            steps=steps_coarse,
            dt=dt * 1.5,  # Larger steps for coarser dynamics
            beta=beta,
            clamp=clamp,
            enable_flow=enable_flow
        )
        
        # Learnable pooling for coarse level
        self.coarse_projection = nn.Linear(d_latent, coarse_dim, bias=False)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-scale fields.
        
        Returns:
            fine_emb: Fine-grained embedding
            coarse_emb: Coarse-grained embedding
            combined: Concatenated multi-scale embedding
        """
        # Fine-scale processing
        fine_emb = self.fine_field(z)
        
        # Coarse-scale processing (pooled input)
        z_coarse = self.coarse_projection(z)
        coarse_emb = self.coarse_field(z_coarse)
        
        # Combine scales
        combined = torch.cat([fine_emb, coarse_emb], dim=-1)
        
        return fine_emb, coarse_emb, combined


class AttentionGatedPMField(nn.Module):
    """
    PMField with attention-based gating for selective context integration.
    
    Uses gradient magnitude from PMFlow as natural attention weights,
    allowing the model to focus on relevant context while ignoring noise.
    
    Solves the problem of context flow being unclear (isolated vs coordinated
    stage dimensions differing).
    """
    
    def __init__(self, d_latent=64, n_centers=64, steps=5, dt=0.15, 
                 beta=1.2, clamp=3.0, attention_mode='gradient'):
        super().__init__()
        
        self.pm_field = ParallelPMField(
            d_latent=d_latent,
            n_centers=n_centers,
            steps=steps,
            dt=dt,
            beta=beta,
            clamp=clamp
        )
        
        self.attention_mode = attention_mode
        
        # Learnable attention parameters
        self.attention_scale = nn.Parameter(torch.tensor(1.0))
        self.context_gate = nn.Linear(d_latent, d_latent)
        
    def compute_attention_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Compute attention weights based on PMFlow gradient field."""
        if self.attention_mode == 'gradient':
            # Use gradient magnitude as attention signal
            grad = self.pm_field.vectorized_grad_ln_n(z)
            attention = torch.norm(grad, dim=-1, keepdim=True)
            attention = torch.sigmoid(self.attention_scale * attention)
        elif self.attention_mode == 'learned':
            # Fully learnable attention
            attention = torch.sigmoid(self.context_gate(z).mean(dim=-1, keepdim=True))
        else:
            # Uniform attention (no gating)
            attention = torch.ones(z.shape[0], 1, device=z.device)
        
        return attention
    
    def forward(self, z: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with optional context integration.
        
        Args:
            z: Primary input
            context: Optional context from upstream stages
            
        Returns:
            output: Processed embedding
            attention: Attention weights used
        """
        if context is not None and context.shape[-1] == z.shape[-1]:
            # Compute attention weights
            attention = self.compute_attention_weights(z)
            
            # Blend input with context based on attention
            z_blended = attention * z + (1 - attention) * context
        else:
            z_blended = z
            attention = torch.ones(z.shape[0], 1, device=z.device)
        
        # Process through PMField
        output = self.pm_field(z_blended)
        
        return output, attention


class EnergyBasedPMField(ParallelPMField):
    """
    Enhanced PMField with energy computation for semantic similarity.
    
    Extends ParallelPMField to compute refractive index energy,
    enabling energy-based retrieval in addition to cosine similarity.
    
    Energy landscape captures semantic "gravity wells" - similar concepts
    have similar energy profiles.
    """
    
    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute refractive index energy: E = ∫n(r)dr ≈ Σμᵢ/|r-rᵢ|
        
        Returns:
            energy: Scalar energy per sample (B,)
        """
        B, D = z.shape
        N = self.centers.shape[0]
        
        # Vectorized distance computation
        rvec = z.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, N, D)
        r = torch.sqrt(torch.sum(rvec * rvec, dim=2) + self._eps)  # (B, N)
        
        # Energy contributions from each center
        energy_contributions = self.mus.unsqueeze(0) / r  # (B, N)
        total_energy = torch.sum(energy_contributions, dim=1)  # (B,)
        
        return total_energy
    
    def energy_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute energy-based similarity between two embeddings.
        
        Similar documents have similar energy profiles in the PMFlow landscape.
        
        Returns:
            similarity: Energy similarity scores (B,)
        """
        energy1 = self.compute_energy(z1)
        energy2 = self.compute_energy(z2)
        
        # Similarity inversely proportional to energy difference
        energy_diff = torch.abs(energy1 - energy2)
        similarity = 1.0 / (1.0 + energy_diff)
        
        return similarity


@torch.no_grad()
def contrastive_plasticity(pmfield: ParallelPMField,
                          similar_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                          dissimilar_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                          mu_lr: float = 1e-3,
                          c_lr: float = 1e-3,
                          margin: float = 1.0):
    """
    Contrastive plasticity update for improved concept separation.
    
    Updates PMField parameters to:
    - Pull similar embeddings closer together
    - Push dissimilar embeddings further apart
    
    Combines with standard vectorized_pm_plasticity for dual objectives:
    - Task performance (retrieval accuracy)
    - Semantic structure (contrastive separation)
    
    Args:
        pmfield: PMField to update
        similar_pairs: List of (emb1, emb2) that should be close
        dissimilar_pairs: List of (emb1, emb2) that should be far
        mu_lr: Learning rate for gravitational strengths
        c_lr: Learning rate for center positions
        margin: Minimum distance for dissimilar pairs
    """
    s2 = 0.8 ** 2
    C = pmfield.centers
    N, D = C.shape
    
    # Process similar pairs - pull together
    if similar_pairs:
        similar_targets = []
        similar_weights = []
        
        for emb1, emb2 in similar_pairs:
            # Target is midpoint between similar embeddings
            target = (emb1 + emb2) / 2.0
            similar_targets.append(target.squeeze())
            
            # Compute weights for each center
            dist2 = torch.sum((C - target) ** 2, dim=1)
            weight = torch.exp(-dist2 / (2 * s2))
            similar_weights.append(weight)
        
        if similar_targets:
            similar_targets = torch.stack(similar_targets)  # (P, D)
            similar_weights = torch.stack(similar_weights)  # (P, N)
            
            # Update centers toward similar targets
            W_sum = torch.sum(similar_weights, dim=0, keepdim=True).T + 1e-6  # (N, 1)
            weighted_targets = torch.sum(
                similar_weights.T.unsqueeze(2) * similar_targets.unsqueeze(0),
                dim=1
            )  # (N, D)
            target_pull = weighted_targets / W_sum
            
            # Pull centers toward similar targets
            C.add_(c_lr * 0.5 * (target_pull - C))
            
            # Strengthen mus for active centers
            mu_drive = torch.mean(similar_weights, dim=0)
            pmfield.mus.add_(mu_lr * mu_drive)
    
    # Process dissimilar pairs - push apart
    if dissimilar_pairs:
        for emb1, emb2 in dissimilar_pairs:
            # Compute current distance
            current_dist = torch.norm(emb1 - emb2)
            
            if current_dist < margin:
                # Too close - need to push apart
                push_vector = emb1 - emb2
                push_vector = push_vector / (torch.norm(push_vector) + 1e-6)
                
                # Find centers to push
                dist1_to_centers = torch.sum((C - emb1) ** 2, dim=1)
                dist2_to_centers = torch.sum((C - emb2) ** 2, dim=1)
                
                weight1 = torch.exp(-dist1_to_centers / (2 * s2))
                weight2 = torch.exp(-dist2_to_centers / (2 * s2))
                
                # Push centers away from each other
                push_strength = (margin - current_dist.item()) / margin
                C.add_(c_lr * 0.1 * push_strength * 
                      (weight1.unsqueeze(1) * push_vector - 
                       weight2.unsqueeze(1) * push_vector))


def batch_plasticity_update(pmfield: ParallelPMField,
                           examples: List[torch.Tensor],
                           mu_lr: float = 5e-4,
                           c_lr: float = 5e-4,
                           batch_size: int = 32):
    """
    Efficient batch plasticity update for large-scale training.
    
    Processes examples in mini-batches for memory efficiency while
    leveraging vectorized_pm_plasticity for speed.
    
    Args:
        pmfield: PMField to update
        examples: List of input tensors to learn from
        mu_lr: Learning rate for gravitational strengths
        c_lr: Learning rate for center positions
        batch_size: Mini-batch size
    """
    n_examples = len(examples)
    
    for i in range(0, n_examples, batch_size):
        batch = examples[i:i + batch_size]
        
        # Stack batch
        z_batch = torch.stack([ex.squeeze() for ex in batch])
        
        # Forward pass
        h_batch = pmfield(z_batch)
        
        # Plasticity update
        vectorized_pm_plasticity(pmfield, z_batch, h_batch, mu_lr=mu_lr, c_lr=c_lr)


def hybrid_similarity(query_emb: torch.Tensor,
                     doc_emb: torch.Tensor,
                     pmfield: EnergyBasedPMField,
                     cosine_weight: float = 0.7,
                     energy_weight: float = 0.3) -> torch.Tensor:
    """
    Hybrid similarity combining cosine and energy-based metrics.
    
    Cosine similarity: Good for vector alignment
    Energy similarity: Good for semantic "gravity well" matching
    
    Args:
        query_emb: Query embedding
        doc_emb: Document embedding  
        pmfield: EnergyBasedPMField for energy computation
        cosine_weight: Weight for cosine similarity
        energy_weight: Weight for energy similarity
        
    Returns:
        hybrid_sim: Combined similarity score
    """
    # Cosine similarity
    cosine_sim = F.cosine_similarity(query_emb, doc_emb, dim=-1)
    
    # Energy similarity
    energy_sim = pmfield.energy_similarity(query_emb, doc_emb)
    
    # Weighted combination
    hybrid_sim = cosine_weight * cosine_sim + energy_weight * energy_sim
    
    return hybrid_sim