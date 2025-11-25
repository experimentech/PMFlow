import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .bnn import TemporalPipelineBNN, MultiGPUPMBNN, PMBNNAlwaysPlasticV2
from .pmflow import ParallelPMField, VectorizedLateralEI

def get_model_v2(model_type: str, **kwargs) -> nn.Module:
    """
    Enhanced model factory for v0.2.0 with temporal parallelism support.
    
    Available models:
    - 'temporal_pipeline': TemporalPipelineBNN with pipeline overlapping
    - 'multi_gpu': MultiGPUPMBNN for distributed computation
    - 'always_plastic_v2': Enhanced always-plastic version
    - 'standard_v2': Standard PMBNN with v0.2.0 enhancements
    """
    
    default_params = {
        'd_latent': 12,     # Increased from 8
        'channels': 96,     # Increased from 64
        'pm_steps': 4,
        'n_centers': 64,
        'n_classes': 10,
        'dt': 0.15,         # Increased from default
        'beta': 1.2         # Increased from default
    }
    
    # Merge defaults with provided kwargs
    params = {**default_params, **kwargs}
    
    # Validate parameters
    if params['n_centers'] <= 0:
        raise ValueError(f"n_centers must be positive, got {params['n_centers']}")
    if params['pm_steps'] <= 0:
        raise ValueError(f"pm_steps must be positive, got {params['pm_steps']}")
    if params['d_latent'] <= 0:
        raise ValueError(f"d_latent must be positive, got {params['d_latent']}")
    if params['channels'] <= 0:
        raise ValueError(f"channels must be positive, got {params['channels']}")
    
    if model_type == 'temporal_pipeline':
        return TemporalPipelineBNN(
            d_latent=params['d_latent'],
            channels=params['channels'],
            pm_steps=params['pm_steps'],
            n_centers=params['n_centers'],
            n_classes=params['n_classes'],
            temporal_stages=params.get('temporal_stages', 2),
            pipeline_overlap=params.get('pipeline_overlap', True),
            adaptive_scheduling=params.get('adaptive_scheduling', True)
        )
    
    elif model_type == 'multi_gpu':
        gpu_devices = params.get('gpu_devices', None)
        if gpu_devices is None and torch.cuda.device_count() > 1:
            gpu_devices = list(range(torch.cuda.device_count()))
        
        return MultiGPUPMBNN(
            d_latent=params['d_latent'],
            channels=params['channels'],
            pm_steps=params['pm_steps'],
            n_centers=params['n_centers'],
            n_classes=params['n_classes'],
            gpu_devices=gpu_devices
        )
    
    elif model_type == 'always_plastic_v2':
        return PMBNNAlwaysPlasticV2(
            d_latent=params['d_latent'],
            channels=params['channels'],
            pm_steps=params['pm_steps'],
            n_centers=params['n_centers'],
            n_classes=params['n_classes'],
            plastic=params.get('plastic', True),
            plasticity_lr=params.get('plasticity_lr', 1e-3)
        )
    
    elif model_type == 'standard_v2':
        # Standard model using v0.2.0 components
        return TemporalPipelineBNN(
            d_latent=params['d_latent'],
            channels=params['channels'],
            pm_steps=params['pm_steps'],
            n_centers=params['n_centers'],
            n_classes=params['n_classes'],
            temporal_stages=1,  # No pipeline overlapping
            pipeline_overlap=False,
            adaptive_scheduling=True
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_performance_config(hardware_profile: str = 'auto') -> Dict[str, Any]:
    """
    Get optimized configuration based on hardware profile.
    
    Args:
        hardware_profile: 'jetson_nano', 'single_gpu', 'multi_gpu', 'cpu', or 'auto'
    
    Returns:
        Configuration dictionary with optimized parameters
    """
    
    if hardware_profile == 'auto':
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                hardware_profile = 'multi_gpu'
            else:
                # Check if it's Jetson Nano (approximate detection)
                device_name = torch.cuda.get_device_name(0).lower()
                if 'tegra' in device_name or 'jetson' in device_name:
                    hardware_profile = 'jetson_nano'
                else:
                    hardware_profile = 'single_gpu'
        else:
            hardware_profile = 'cpu'
    
    configs = {
        'jetson_nano': {
            'model_type': 'temporal_pipeline',
            'n_centers': 32,  # Reduced for memory constraints
            'chunk_size': 16,
            'temporal_stages': 2,
            'pipeline_overlap': True,
            'adaptive_scheduling': True,
            'pm_steps': 3  # Reduced computational load
        },
        
        'single_gpu': {
            'model_type': 'temporal_pipeline',
            'n_centers': 64,
            'chunk_size': 32,
            'temporal_stages': 3,
            'pipeline_overlap': True,
            'adaptive_scheduling': True,
            'pm_steps': 4
        },
        
        'multi_gpu': {
            'model_type': 'multi_gpu',
            'n_centers': 128,  # More centers for distributed computation
            'chunk_size': 64,
            'temporal_stages': 4,
            'pipeline_overlap': True,
            'adaptive_scheduling': True,
            'pm_steps': 5
        },
        
        'cpu': {
            'model_type': 'temporal_pipeline',
            'n_centers': 24,        # Optimized for CPU efficiency
            'chunk_size': 8,        # Smaller chunks
            'temporal_stages': 2,   # Enable some parallelism
            'pipeline_overlap': True,
            'adaptive_scheduling': True,
            'pm_steps': 3,          # Reduced computational load
            'dt': 0.2,              # Larger time steps
            'beta': 1.0             # Standard coupling
        }
    }
    
    return configs.get(hardware_profile, configs['single_gpu'])

def benchmark_temporal_parallelism(model: nn.Module, batch_sizes: List[int] = [8, 16, 32, 64], 
                                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 num_trials: int = 5) -> Dict[str, Any]:
    """
    Benchmark temporal parallelism performance.
    
    Returns timing and memory usage statistics for different batch sizes.
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    results = {
        'batch_sizes': batch_sizes,
        'forward_times': [],
        'memory_usage': [],
        'throughput': []
    }
    
    for batch_size in batch_sizes:
        # Create dummy input
        x = torch.randn(batch_size, 28 * 28, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(x)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_trials
        throughput = batch_size / avg_time
        
        results['forward_times'].append(avg_time)
        results['throughput'].append(throughput)
        
        # Memory usage (CUDA only)
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            results['memory_usage'].append(memory_used)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            results['memory_usage'].append(0)
    
    return results

def validate_embarrassingly_parallel_scaling(model: nn.Module, max_batch_size: int = 128,
                                           device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, Any]:
    """
    Validate that the model exhibits embarrassingly parallel scaling characteristics.
    
    Tests whether increasing batch size maintains linear throughput scaling
    (characteristic of embarrassingly parallel algorithms).
    """
    batch_sizes = [2**i for i in range(1, int(torch.log2(torch.tensor(max_batch_size)).item()) + 1)]
    
    benchmark_results = benchmark_temporal_parallelism(model, batch_sizes, device)
    
    # Analyze scaling characteristics
    throughputs = benchmark_results['throughput']
    scaling_efficiency = []
    
    baseline_throughput = throughputs[0] * batch_sizes[0]  # samples/sec for batch_size=1 equivalent
    
    for i, (batch_size, throughput) in enumerate(zip(batch_sizes, throughputs)):
        expected_throughput = baseline_throughput
        actual_throughput = throughput
        efficiency = actual_throughput / expected_throughput
        scaling_efficiency.append(efficiency)
    
    results = {
        **benchmark_results,
        'scaling_efficiency': scaling_efficiency,
        'is_embarrassingly_parallel': all(eff > 0.8 for eff in scaling_efficiency),  # 80% efficiency threshold
        'average_efficiency': sum(scaling_efficiency) / len(scaling_efficiency)
    }
    
    return results