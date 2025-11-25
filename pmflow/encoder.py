"""Embedding encoders for the language-to-symbol pipeline."""

from __future__ import annotations

import hashlib
import importlib
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
    stable across runs.
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
    ) -> None:
        if combine_mode not in {"concat", "pm-only"}:
            raise ValueError("combine_mode must be 'concat' or 'pm-only'.")
        self.combine_mode = combine_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_encoder = base_encoder or HashedEmbeddingEncoder(dimension=dimension)
        self.dimension = self.base_encoder.dimension
        self.latent_dim = latent_dim
        self._projection = self._build_projection_matrix(self.dimension, latent_dim, seed).to(self.device)
        self.pm_field = self._init_pm_field(latent_dim, seed)
        self.pm_field.to(self.device)
        self.pm_field.eval()
        self._state_path: Optional[Path] = None

    @staticmethod
    def _build_projection_matrix(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
        return torch.from_numpy(matrix)

    @staticmethod
    def _init_pm_field(latent_dim: int, seed: int):
        try:
            module = importlib.import_module("pmflow_bnn_enhanced.pmflow")
        except ModuleNotFoundError:
            # Fallback to original pmflow_bnn if enhanced version not available
            try:
                module = importlib.import_module("pmflow_bnn.pmflow")
            except ModuleNotFoundError as exc:  # pragma: no cover - handled by caller fallback
                raise RuntimeError("pmflow_bnn is required for PMFlow embeddings") from exc

        # Try to use MultiScalePMField for hierarchical concept learning
        PMFieldCls = getattr(module, "MultiScalePMField", None)
        if PMFieldCls is not None:
            # Use multi-scale field for better hierarchical concept representation
            field = PMFieldCls(
                d_latent=latent_dim, 
                n_centers_fine=128,  # Fine-grained specific concepts
                n_centers_coarse=32,  # Coarse-grained categories
                steps_fine=5, 
                steps_coarse=3,
                dt=0.15, 
                beta=1.2, 
                clamp=3.0
            )
            # MultiScalePMField manages its own internal fields - no manual init needed
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                # Initialize fine field
                centres_fine = torch.randn(field.fine_field.centers.shape, generator=generator, device=field.fine_field.centers.device) * 0.5
                mus_fine = torch.full(field.fine_field.mus.shape, 0.35, device=field.fine_field.mus.device)
                field.fine_field.centers.copy_(centres_fine)
                field.fine_field.mus.copy_(mus_fine)
                
                # Initialize coarse field
                centres_coarse = torch.randn(field.coarse_field.centers.shape, generator=generator, device=field.coarse_field.centers.device) * 0.5
                mus_coarse = torch.full(field.coarse_field.mus.shape, 0.35, device=field.coarse_field.mus.device)
                field.coarse_field.centers.copy_(centres_coarse)
                field.coarse_field.mus.copy_(mus_coarse)
            return field
        
        # Fallback to standard PMField if MultiScale not available
        PMFieldCls = getattr(module, "PMField", None)
        if PMFieldCls is None:
            PMFieldCls = getattr(module, "ParallelPMField", None)
        if PMFieldCls is None:
            raise RuntimeError("pmflow module missing PMField/ParallelPMField implementation")

        field = PMFieldCls(d_latent=latent_dim, steps=5, dt=0.08, beta=0.9, clamp=2.5)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            centres = torch.randn(field.centers.shape, generator=generator, device=field.centers.device) * 0.5
            mus = torch.full(field.mus.shape, 0.35, device=field.mus.device)
            field.centers.copy_(centres)
            field.mus.copy_(mus)
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
            
            refined = F.normalize(raw_refined, p=2, dim=1)
            if self.combine_mode == "concat":
                hashed = F.normalize(base, p=2, dim=1)
                combined = torch.cat([hashed, refined], dim=1)
            else:
                combined = refined
            return combined.cpu(), latent.detach().cpu(), raw_refined.detach().cpu()