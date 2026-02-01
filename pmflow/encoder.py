"""Embedding encoders for the language-to-symbol pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


class HashedEmbeddingEncoder:
    """Convert tokens into a fixed-width embedding via hashing.

    This keeps dependencies minimal while providing deterministic vectors that play
    nicely with the existing SQLiteVectorStore.
    """

    def __init__(self, *, dimension: int = 64) -> None:
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension

    def encode(self, tokens: Iterable[str]) -> torch.Tensor:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            index = self._hash_token(token)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return torch.from_numpy(vector).unsqueeze(0)


class PMFlowEmbeddingEncoder:
    """Blend hashed bag-of-words features with a PMFlow latent field.

    The PMFlow field injects a touch of learned-like structure without requiring
    any training loop. We initialise it deterministically so embeddings are
    stable across runs. Optional ``target_pm_dim`` trims/pads PM outputs to a
    deterministic width so downstream callers can preallocate fixed-size
    vectors even when multi-scale fields change the latent width.
    
    Agentic Physics (v0.3.4+):
        When ``enable_flow=True``, the PMFlow field uses frame-dragging physics
        to enable intent-driven reasoning. This allows:
        - Trajectory tracing through concept space
        - Intent injection via omega-spin modulation
        - Non-obvious concept path discovery
    """

    def __init__(
        self,
        *,
        dimension: int = 96,
        latent_dim: int = 48,
        seed: int = 13,
        combine_mode: str = "concat",
        device: Optional[torch.device] = None,
        base_encoder: Optional[HashedEmbeddingEncoder] = None,
        target_pm_dim: Optional[int] = None,
        enable_flow: bool = False,
    ) -> None:
        if combine_mode not in {"concat", "pm-only"}:
            raise ValueError("combine_mode must be 'concat' or 'pm-only'.")
        self.combine_mode = combine_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_encoder = base_encoder or HashedEmbeddingEncoder(dimension=dimension)
        self.dimension = self.base_encoder.dimension
        self.latent_dim = latent_dim
        self.target_pm_dim = target_pm_dim
        self._projection = self._build_projection_matrix(self.dimension, latent_dim, seed).to(self.device)
        self.enable_flow = enable_flow
        self.pm_field = self._init_pm_field(latent_dim, seed, enable_flow)
        self.pm_field.to(self.device)
        self.pm_field.eval()
        self._state_path: Optional[Path] = None

    @staticmethod
    def _build_projection_matrix(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
        return torch.from_numpy(matrix)

    @staticmethod
    def _init_pm_field(latent_dim: int, seed: int, enable_flow: bool = False):
        """Create a deterministic PMFlow field using bundled implementations."""
        from pmflow.core.pmflow import MultiScalePMField, ParallelPMField

        # Prefer multi-scale for hierarchical representations; fallback to parallel field.
        try:
            field = MultiScalePMField(
                d_latent=latent_dim,
                n_centers_fine=128,
                n_centers_coarse=32,
                steps_fine=5,
                steps_coarse=3,
                dt=0.15,
                beta=1.2,
                clamp=3.0,
                enable_flow=enable_flow,
            )
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                centres_fine = torch.randn(
                    field.fine_field.centers.shape,
                    generator=generator,
                    device=field.fine_field.centers.device,
                ) * 0.5
                mus_fine = torch.full(
                    field.fine_field.mus.shape,
                    0.35,
                    device=field.fine_field.mus.device,
                )
                field.fine_field.centers.copy_(centres_fine)
                field.fine_field.mus.copy_(mus_fine)
                # Initialize omegas for frame-dragging (agentic physics)
                if hasattr(field.fine_field, 'omegas'):
                    omegas_fine = torch.randn(
                        field.fine_field.omegas.shape,
                        generator=generator,
                        device=field.fine_field.omegas.device,
                    ) * 0.01
                    field.fine_field.omegas.copy_(omegas_fine)

                centres_coarse = torch.randn(
                    field.coarse_field.centers.shape,
                    generator=generator,
                    device=field.coarse_field.centers.device,
                ) * 0.5
                mus_coarse = torch.full(
                    field.coarse_field.mus.shape,
                    0.35,
                    device=field.coarse_field.mus.device,
                )
                field.coarse_field.centers.copy_(centres_coarse)
                field.coarse_field.mus.copy_(mus_coarse)
                # Initialize omegas for coarse field too
                if hasattr(field.coarse_field, 'omegas'):
                    omegas_coarse = torch.randn(
                        field.coarse_field.omegas.shape,
                        generator=generator,
                        device=field.coarse_field.omegas.device,
                    ) * 0.01
                    field.coarse_field.omegas.copy_(omegas_coarse)
            return field
        except Exception:
            # Fall back to a single-scale field if multi-scale construction fails.
            field = ParallelPMField(d_latent=latent_dim, steps=5, dt=0.08, beta=0.9, clamp=2.5, enable_flow=enable_flow)
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                centres = torch.randn(
                    field.centers.shape,
                    generator=generator,
                    device=field.centers.device,
                ) * 0.5
                mus = torch.full(field.mus.shape, 0.35, device=field.mus.device)
                field.centers.copy_(centres)
                field.mus.copy_(mus)
                # Initialize omegas
                if hasattr(field, 'omegas'):
                    omegas = torch.randn(field.omegas.shape, generator=generator, device=field.omegas.device) * 0.01
                    field.omegas.copy_(omegas)
            return field

    def encode(self, tokens: Iterable[str]) -> torch.Tensor:
        combined, _, _ = self._encode_internal(tokens)
        return combined

    def encode_with_components(self, tokens: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embedding together with PMFlow latent and raw activations.

        The latent corresponds to the input fed into the PMFlow field and the
        raw activation is the unnormalised PMFlow output before concatenation.
        """

        combined, latent, raw_refined = self._encode_internal(tokens)
        return combined, latent, raw_refined

    def attach_state_path(self, path: Optional[Path]) -> None:
        """Opt into persistence of the PMFlow field parameters."""

        self._state_path = path
        if path and path.exists():
            self.load_state(path)

    def save_state(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._state_path
        if path is None:
            return
        
        # Handle MultiScalePMField (has fine_field and coarse_field)
        if hasattr(self.pm_field, 'fine_field') and hasattr(self.pm_field, 'coarse_field'):
            payload = {
                "type": "multiscale",
                "fine_centers": self.pm_field.fine_field.centers.detach().cpu(),
                "fine_mus": self.pm_field.fine_field.mus.detach().cpu(),
                "coarse_centers": self.pm_field.coarse_field.centers.detach().cpu(),
                "coarse_mus": self.pm_field.coarse_field.mus.detach().cpu(),
                "coarse_projection": self.pm_field.coarse_projection.weight.detach().cpu(),
            }
        else:
            # Standard PMField
            payload = {
                "type": "standard",
                "centers": self.pm_field.centers.detach().cpu(),
                "mus": self.pm_field.mus.detach().cpu(),
            }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_state(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._state_path
        if path is None or not path.exists():
            return
        payload = torch.load(path, map_location=self.device)
        
        with torch.no_grad():
            # Handle MultiScalePMField
            if payload.get("type") == "multiscale":
                if hasattr(self.pm_field, 'fine_field') and hasattr(self.pm_field, 'coarse_field'):
                    self.pm_field.fine_field.centers.copy_(payload["fine_centers"].to(self.device))
                    self.pm_field.fine_field.mus.copy_(payload["fine_mus"].to(self.device))
                    self.pm_field.coarse_field.centers.copy_(payload["coarse_centers"].to(self.device))
                    self.pm_field.coarse_field.mus.copy_(payload["coarse_mus"].to(self.device))
                    self.pm_field.coarse_projection.weight.copy_(payload["coarse_projection"].to(self.device))
            # Handle standard PMField (backward compatibility)
            elif "centers" in payload:
                if hasattr(self.pm_field, 'centers'):
                    self.pm_field.centers.copy_(payload["centers"].to(self.device))
                if "mus" in payload and hasattr(self.pm_field, 'mus'):
                    self.pm_field.mus.copy_(payload["mus"].to(self.device))

    def _encode_internal(self, tokens: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base = self.base_encoder.encode(tokens).to(self.device)
            latent = base @ self._projection
            
            # Handle MultiScalePMField which returns (fine, coarse, combined) tuple
            pm_output = self.pm_field(latent)
            if isinstance(pm_output, tuple) and len(pm_output) == 3:
                # MultiScalePMField returns (fine_emb, coarse_emb, combined)
                # Use combined for hierarchical concept representation
                raw_refined: torch.Tensor = pm_output[2]  # Combined multi-scale embedding
            else:
                # Standard PMField returns single tensor
                raw_refined: torch.Tensor = pm_output
            
            refined = self._align_dim(raw_refined)
            refined = F.normalize(refined, p=2, dim=1)
            if self.combine_mode == "concat":
                hashed = F.normalize(base, p=2, dim=1)
                combined = torch.cat([hashed, refined], dim=1)
            else:
                combined = refined
            return combined.cpu(), latent.detach().cpu(), raw_refined.detach().cpu()

    def _align_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Trim or pad PMFlow output to a deterministic width when requested."""

        if self.target_pm_dim is None:
            return tensor

        current = tensor.shape[-1]
        if current == self.target_pm_dim:
            return tensor
        if current > self.target_pm_dim:
            return tensor[..., : self.target_pm_dim]
        pad = self.target_pm_dim - current
        return F.pad(tensor, (0, pad))

    # ========================================================================
    # Agentic Physics API (v0.3.4+)
    # ========================================================================
    
    def trace_trajectory(
        self, 
        tokens: Iterable[str], 
        steps: Optional[int] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Trace a trajectory through concept space for the given input.
        
        This enables "thinking" as a physical process - the trajectory
        represents the reasoning path through the PMFlow latent landscape.
        
        Args:
            tokens: Input tokens to encode and trace
            steps: Override default evolution steps (more steps = deeper thought)
            
        Returns:
            trajectory: Tensor of shape (1, steps+1, D) representing the path
            metrics: Dict with path_length (mental effort), displacement, efficiency
        """
        if not self.enable_flow:
            raise RuntimeError("trace_trajectory requires enable_flow=True")
        
        with torch.no_grad():
            base = self.base_encoder.encode(tokens).to(self.device)
            latent = base @ self._projection
            
            # Get the active field (fine_field for MultiScale)
            if hasattr(self.pm_field, 'fine_field'):
                active_field = self.pm_field.fine_field
            else:
                active_field = self.pm_field
            
            # Override steps if requested
            original_steps = active_field.steps
            if steps is not None:
                active_field.steps = steps
            
            try:
                trajectory = active_field(latent, return_trajectory=True)
            finally:
                active_field.steps = original_steps
            
            # Compute metrics
            start = trajectory[0, 0]
            end = trajectory[0, -1]
            
            diffs = trajectory[0, 1:] - trajectory[0, :-1]
            segment_lengths = torch.norm(diffs, dim=1)
            path_length = segment_lengths.sum().item()
            displacement = torch.norm(end - start).item()
            efficiency = displacement / (path_length + 1e-6)
            
            metrics = {
                "path_length": path_length,  # Mental effort
                "displacement": displacement,  # How far we moved
                "efficiency": min(1.0, efficiency),  # Path straightness (confidence)
                "steps": trajectory.shape[1] - 1,
            }
            
            return trajectory.cpu(), metrics
    
    def inject_intent(
        self, 
        goal_tokens: Iterable[str], 
        strength: float = 0.5,
        decay_radius: float = 2.0
    ) -> int:
        """
        Inject intent (active will) into the PMFlow field.
        
        This modulates the omega (spin) of nearby gravitational centers,
        creating a frame-dragging effect that biases future trajectories
        toward the goal concept. This is "agentic" physics - the field
        actively works toward an outcome.
        
        Args:
            goal_tokens: Tokens representing the goal/intent
            strength: How strongly to activate spin (0.0-1.0)
            decay_radius: Gaussian falloff radius for spin influence
            
        Returns:
            Number of centers affected
        """
        if not self.enable_flow:
            raise RuntimeError("inject_intent requires enable_flow=True")
        
        with torch.no_grad():
            base = self.base_encoder.encode(goal_tokens).to(self.device)
            latent = base @ self._projection
            
            # Get the active field
            if hasattr(self.pm_field, 'fine_field'):
                target_field = self.pm_field.fine_field
            else:
                target_field = self.pm_field
            
            if not hasattr(target_field, 'omegas'):
                return 0
            
            # Calculate proximity to all centers
            dists = torch.cdist(latent, target_field.centers)
            proximity = torch.exp(-dists[0]**2 / (2 * decay_radius**2))
            
            # Activate spin for nearby centers
            current_spin = target_field.omegas.data
            
            # Initialize random direction for zero-spin centers
            zero_mask = (current_spin.abs() < 1e-6)
            if zero_mask.any():
                current_spin[zero_mask] = torch.randn_like(current_spin[zero_mask]) * 0.1
            
            # Add spin proportional to proximity
            prox_norm = proximity / (proximity.sum() + 1e-6)
            target_field.omegas.data += strength * prox_norm * torch.sign(current_spin)
            
            # Count significantly affected centers
            affected = (proximity > 0.1).sum().item()
            return int(affected)
    
    def clear_intent(self) -> None:
        """Reset all omega spins to near-zero (clear active will)."""
        with torch.no_grad():
            if hasattr(self.pm_field, 'fine_field'):
                if hasattr(self.pm_field.fine_field, 'omegas'):
                    self.pm_field.fine_field.omegas.data.mul_(0.01)
                if hasattr(self.pm_field.coarse_field, 'omegas'):
                    self.pm_field.coarse_field.omegas.data.mul_(0.01)
            elif hasattr(self.pm_field, 'omegas'):
                self.pm_field.omegas.data.mul_(0.01)
    
    def get_nearby_centers(
        self, 
        tokens: Iterable[str], 
        topk: int = 5
    ) -> list[tuple[int, float, float, float]]:
        """
        Find gravitational centers closest to the given input.
        
        Useful for debugging and understanding what concepts influence
        the trajectory.
        
        Args:
            tokens: Input to check
            topk: Number of nearest centers to return
            
        Returns:
            List of (center_idx, distance, mu (gravity), omega (spin))
        """
        with torch.no_grad():
            base = self.base_encoder.encode(tokens).to(self.device)
            latent = base @ self._projection
            
            if hasattr(self.pm_field, 'fine_field'):
                target_field = self.pm_field.fine_field
            else:
                target_field = self.pm_field
            
            dists = torch.cdist(latent, target_field.centers)[0]
            nearest = torch.argsort(dists)[:topk]
            
            results = []
            for idx in nearest:
                i = idx.item()
                omega = target_field.omegas[i].item() if hasattr(target_field, 'omegas') else 0.0
                results.append((i, dists[i].item(), target_field.mus[i].item(), omega))
            
            return results