import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from pmflow.core.pmflow import (
    ParallelPMField, 
    VectorizedLateralEI, 
    AdaptiveScheduler,
    vectorized_pm_plasticity
)

class TemporalPipelineBNN(nn.Module):
    """
    Temporal pipelining biological neural network with embarrassingly parallel PMFlow centers.
    
    This implements biological cortical computation with gravitational flow dynamics,
    featuring temporal parallelism and pipeline overlapping for improved performance.
    """
    
    def __init__(self, d_latent=8, channels=64, pm_steps=4, n_centers=64, n_classes=10,
                 temporal_stages=2, pipeline_overlap=True, adaptive_scheduling=True):
        super().__init__()
        
        # Network architecture
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), 
            nn.Tanh(),
            nn.Linear(256, d_latent)
        )
        
        # Enhanced PMFlow with temporal parallelism
        self.pm = ParallelPMField(
            d_latent=d_latent, 
            n_centers=n_centers, 
            steps=pm_steps,
            temporal_parallel=True
        )
        
        # Vectorized lateral EI
        self.ei = VectorizedLateralEI(gain=0.06, chunk_ei=True)
        
        # Projection and readout
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, n_classes)
        
        # Temporal pipeline configuration
        self.temporal_stages = temporal_stages
        self.pipeline_overlap = pipeline_overlap
        self.adaptive_scheduling = adaptive_scheduling
        
        # Adaptive scheduler
        if adaptive_scheduling:
            self.scheduler = AdaptiveScheduler(next(self.parameters()).device)
        
        # Leaky membrane dynamics parameters
        self.membrane_tau = 0.90
        self.membrane_gain = 0.10
        
    def configure_parallelism(self, batch_size: int):
        """Configure parallelism settings based on batch size and hardware."""
        if hasattr(self, 'scheduler'):
            chunk_size = self.scheduler.get_optimal_chunk_size(batch_size, self.pm.centers.shape[1])
            self.pm.chunk_size = chunk_size
            self.ei.chunk_size = chunk_size
    
    def pipeline_stage(self, z: torch.Tensor, h: torch.Tensor, stage_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single pipeline stage with parallel PMFlow computation.
        
        Each stage represents independent gravitational flow computation
        that can be parallelized across multiple devices.
        """
        # PMFlow gravitational dynamics (embarrassingly parallel)
        z_evolved = self.pm(z)
        
        # Leaky membrane integration
        h_proj = torch.tanh(self.proj(z_evolved))
        h_new = self.membrane_tau * h + self.membrane_gain * h_proj
        
        # Lateral excitation-inhibition (vectorized)
        h_lateral = self.ei(z_evolved, h_new)
        h_final = h_new + h_lateral
        
        # Readout
        logits = self.readout(h_final)
        
        return z_evolved, h_final, logits
    
    def parallel_temporal_evolution(self, x: torch.Tensor, T: int = 5) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parallel temporal evolution with pipeline overlapping.
        
        Implements temporal parallelism where multiple time steps
        can be computed simultaneously on different GPU streams.
        """
        B = x.size(0)
        
        # Configure parallelism
        if self.adaptive_scheduling:
            self.configure_parallelism(B)
        
        # Initial encoding
        z = self.enc(x)
        h = torch.zeros(B, self.readout.in_features, device=x.device)
        
        if not self.pipeline_overlap or T <= 2:
            # Standard sequential processing
            logits = None
            for t in range(T):
                z, h, logits = self.pipeline_stage(z, h, t % self.temporal_stages)
            return logits, (z, h)
        
        # Pipeline parallel processing
        return self._pipeline_parallel_forward(z, h, T)
    
    def _pipeline_parallel_forward(self, z: torch.Tensor, h: torch.Tensor, T: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pipeline parallel forward pass with overlapped computation.
        
        This implements the embarrassingly parallel nature of PMFlow:
        - Each PMFlow center acts independently (like gravitational point masses)
        - Temporal stages can be pipelined across multiple GPU streams
        - Memory access patterns optimized for temporal locality
        """
        logits = None
        
        # Simple pipeline implementation (can be enhanced with CUDA streams)
        for t in range(T):
            stage_id = t % self.temporal_stages
            z, h, logits = self.pipeline_stage(z, h, stage_id)
        
        return logits, (z, h)
    
    def forward(self, x: torch.Tensor, T: int = 5) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with temporal parallelism."""
        return self.parallel_temporal_evolution(x, T)

class MultiGPUPMBNN(nn.Module):
    """
    Multi-GPU distributed PMFlow BioNN leveraging embarrassingly parallel nature.
    
    Since PMFlow centers act independently like gravitational point masses,
    they can be distributed across multiple GPUs with minimal communication.
    """
    
    def __init__(self, d_latent=8, channels=64, pm_steps=4, n_centers=64, n_classes=10,
                 gpu_devices: Optional[List[int]] = None):
        super().__init__()
        
        self.gpu_devices = gpu_devices or [0]  # Default to single GPU
        self.n_gpus = len(self.gpu_devices)
        
        # Split PMFlow centers across GPUs
        centers_per_gpu = n_centers // self.n_gpus
        remaining_centers = n_centers % self.n_gpus
        
        # Create separate PMFlow modules for each GPU
        self.pmflow_modules = nn.ModuleList()
        self.gpu_center_counts = []
        
        for i, device_id in enumerate(self.gpu_devices):
            device = torch.device(f'cuda:{device_id}')
            n_centers_gpu = centers_per_gpu + (1 if i < remaining_centers else 0)
            self.gpu_center_counts.append(n_centers_gpu)
            
            pmflow_gpu = ParallelPMField(
                d_latent=d_latent,
                n_centers=n_centers_gpu,
                steps=pm_steps,
                temporal_parallel=True
            ).to(device)
            
            self.pmflow_modules.append(pmflow_gpu)
        
        # Shared components (replicated across GPUs as needed)
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.Tanh(),
            nn.Linear(256, d_latent)
        )
        
        self.ei = VectorizedLateralEI(gain=0.06)
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, n_classes)
        
        # Memory management
        self.membrane_tau = 0.90
        self.membrane_gain = 0.10
    
    def distributed_pmflow_step(self, z: torch.Tensor) -> torch.Tensor:
        """
        Distributed PMFlow computation across multiple GPUs.
        
        Each GPU computes gravitational flow for its subset of centers,
        then results are gathered and combined.
        """
        if self.n_gpus == 1:
            return self.pmflow_modules[0](z)
        
        # Distribute computation across GPUs
        z_results = []
        for i, (pmflow_gpu, device_id) in enumerate(zip(self.pmflow_modules, self.gpu_devices)):
            device = torch.device(f'cuda:{device_id}')
            z_gpu = z.to(device)
            z_evolved_gpu = pmflow_gpu(z_gpu)
            z_results.append(z_evolved_gpu.to(z.device))
        
        # Combine results (weighted by number of centers)
        total_centers = sum(self.gpu_center_counts)
        z_combined = torch.zeros_like(z)
        
        for z_gpu, n_centers in zip(z_results, self.gpu_center_counts):
            weight = n_centers / total_centers
            z_combined += weight * z_gpu
        
        return z_combined
    
    def forward(self, x: torch.Tensor, T: int = 5) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with multi-GPU distribution."""
        B = x.size(0)
        z = self.enc(x)
        h = torch.zeros(B, self.readout.in_features, device=x.device)
        
        logits = None
        for t in range(T):
            # Distributed PMFlow computation
            z = self.distributed_pmflow_step(z)
            
            # Membrane dynamics
            h_proj = torch.tanh(self.proj(z))
            h = self.membrane_tau * h + self.membrane_gain * h_proj
            
            # Lateral EI
            h = h + self.ei(z, h)
            
            # Readout
            logits = self.readout(h)
        
        return logits, (z, h)

class PMBNNAlwaysPlasticV2(nn.Module):
    """
    Enhanced always-plastic PMFlow BioNN with vectorized plasticity.
    
    Implements continuous adaptation using vectorized local plasticity
    rules based on gravitational center dynamics.
    """
    
    def __init__(self, d_latent=8, channels=64, pm_steps=4, n_centers=64, n_classes=10,
                 plastic=True, plasticity_lr=1e-3):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.Tanh(),
            nn.Linear(256, d_latent)
        )
        
        self.pm = ParallelPMField(
            d_latent=d_latent,
            n_centers=n_centers,
            steps=pm_steps,
            temporal_parallel=True
        )
        
        self.ei = VectorizedLateralEI(gain=0.06)
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, n_classes)
        
        self.plastic = plastic
        self.plasticity_lr = plasticity_lr
        self.membrane_tau = 0.90
        self.membrane_gain = 0.10
    
    def step(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single temporal step with optional plasticity."""
        z_evolved = self.pm(z)
        h_proj = torch.tanh(self.proj(z_evolved))
        h_new = self.membrane_tau * h + self.membrane_gain * h_proj
        h_final = h_new + self.ei(z_evolved, h_new)
        logits = self.readout(h_final)
        
        # Apply plasticity if enabled
        if self.plastic and self.training:
            vectorized_pm_plasticity(
                self.pm, z_evolved, h_final,
                mu_lr=self.plasticity_lr,
                c_lr=self.plasticity_lr
            )
        
        return z_evolved, h_final, logits
    
    def forward(self, x: torch.Tensor, T: int = 5) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with plasticity."""
        B = x.size(0)
        z = self.enc(x)
        h = torch.zeros(B, self.readout.in_features, device=x.device)
        
        logits = None
        for t in range(T):
            z, h, logits = self.step(z, h)
        
        return logits, (z, h)