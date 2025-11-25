import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
import platform
import subprocess
import json

def get_hardware_info() -> Dict[str, Any]:
    """
    Get comprehensive hardware information for optimization.
    
    Returns:
        Dictionary containing CPU, GPU, and memory information
    """
    info = {
        'platform': platform.platform(),
        'cpu_count': torch.get_num_threads(),
        'cpu_arch': platform.machine(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': None,
        'gpu_count': 0,
        'gpu_devices': [],
        'total_memory': 0,
        'is_jetson': False
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'id': i,
                'name': device_props.name,
                'memory_total': device_props.total_memory,
                'memory_reserved': torch.cuda.memory_reserved(i),
                'memory_allocated': torch.cuda.memory_allocated(i),
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multiprocessor_count': device_props.multi_processor_count
            }
            info['gpu_devices'].append(gpu_info)
            info['total_memory'] += device_props.total_memory
            
            # Detect Jetson devices
            if 'tegra' in device_props.name.lower() or 'jetson' in device_props.name.lower():
                info['is_jetson'] = True
    
    return info

def optimize_for_device(device: torch.device, model: nn.Module, batch_size: int) -> Dict[str, Any]:
    """
    Get device-specific optimization settings.
    
    Args:
        device: Target device
        model: Model to optimize
        batch_size: Expected batch size
    
    Returns:
        Optimization configuration dictionary
    """
    config = {
        'mixed_precision': False,
        'gradient_checkpointing': False,
        'compile_model': False,
        'memory_efficient': False,
        'chunk_size': 32,
        'temporal_parallel': True,
        'pipeline_overlap': True
    }
    
    if device.type == 'cuda':
        hw_info = get_hardware_info()
        
        # Enable optimizations based on hardware
        if hw_info['is_jetson']:
            # Jetson-specific optimizations
            config.update({
                'mixed_precision': True,  # FP16 for memory efficiency
                'memory_efficient': True,
                'chunk_size': 16,
                'gradient_checkpointing': True,
                'temporal_parallel': True,
                'pipeline_overlap': False  # Reduced complexity for Jetson
            })
        else:
            # Standard GPU optimizations
            total_memory = hw_info['total_memory']
            if total_memory > 8 * 1024**3:  # > 8GB
                config.update({
                    'mixed_precision': True,
                    'chunk_size': 64,
                    'temporal_parallel': True,
                    'pipeline_overlap': True,
                    'compile_model': True  # PyTorch 2.0+ compilation
                })
            else:
                config.update({
                    'mixed_precision': True,
                    'memory_efficient': True,
                    'chunk_size': 32,
                    'gradient_checkpointing': True
                })
    
    elif device.type == 'cpu':
        # CPU optimizations
        config.update({
            'mixed_precision': False,
            'chunk_size': 8,
            'temporal_parallel': False,
            'pipeline_overlap': False,
            'memory_efficient': True
        })
    
    return config

def create_training_config(model_type: str, hardware_profile: str = 'auto', 
                         batch_size: int = 32, learning_rate: float = 1e-3) -> Dict[str, Any]:
    """
    Create optimized training configuration for v0.2.0 models.
    
    Args:
        model_type: Type of model ('temporal_pipeline', 'multi_gpu', etc.)
        hardware_profile: Hardware profile ('jetson_nano', 'single_gpu', 'multi_gpu', 'cpu', 'auto')
        batch_size: Training batch size
        learning_rate: Base learning rate
    
    Returns:
        Complete training configuration
    """
    from .factory import get_performance_config
    
    # Get hardware-optimized model configuration
    perf_config = get_performance_config(hardware_profile)
    
    # Base training configuration
    config = {
        'model': {
            'type': model_type,
            **perf_config
        },
        'training': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': 20,
            'weight_decay': 1e-4,
            'scheduler': 'cosine_annealing',
            'warmup_epochs': 2
        },
        'optimization': {
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'grad_clip_norm': 1.0
        },
        'validation': {
            'frequency': 1,  # Every epoch
            'metrics': ['accuracy', 'loss'],
            'early_stopping_patience': 5
        },
        'logging': {
            'log_frequency': 100,  # Every 100 batches
            'save_checkpoints': True,
            'checkpoint_frequency': 5  # Every 5 epochs
        }
    }
    
    # Hardware-specific adjustments
    if hardware_profile == 'jetson_nano':
        config['training'].update({
            'batch_size': min(batch_size, 16),  # Memory constraint
            'epochs': 15,  # Reduced for faster training
            'grad_accumulation_steps': 2  # Simulate larger batches
        })
        config['optimization']['grad_clip_norm'] = 0.5  # More conservative
    
    elif hardware_profile == 'multi_gpu':
        config['training'].update({
            'batch_size': batch_size * 2,  # Can handle larger batches
            'learning_rate': learning_rate * 1.5,  # Scale LR with batch size
        })
        config['model']['gpu_devices'] = list(range(torch.cuda.device_count()))
    
    elif hardware_profile == 'cpu':
        config['training'].update({
            'batch_size': min(batch_size, 8),  # CPU limitation
            'epochs': 10,  # Faster training
        })
        config['optimization']['optimizer'] = 'sgd'  # Often better on CPU
    
    return config

def setup_distributed_training(model: nn.Module, gpu_devices: Optional[list] = None) -> nn.Module:
    """
    Setup distributed training for multi-GPU models.
    
    Args:
        model: Model to distribute
        gpu_devices: List of GPU device IDs to use
    
    Returns:
        Distributed model wrapper
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return model
    
    if gpu_devices is None:
        gpu_devices = list(range(torch.cuda.device_count()))
    
    # Use DataParallel for simple multi-GPU (can be upgraded to DistributedDataParallel)
    if len(gpu_devices) > 1:
        model = nn.DataParallel(model, device_ids=gpu_devices)
    
    return model

def profile_model_performance(model: nn.Module, input_shape: Tuple[int, ...], 
                            device: torch.device, num_warmup: int = 10, 
                            num_trials: int = 100) -> Dict[str, float]:
    """
    Profile model performance with detailed metrics.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_trials: Number of timing trials
    
    Returns:
        Performance metrics dictionary
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    # Time forward passes
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_trials
    throughput = input_shape[0] / avg_time  # samples per second
    
    # Memory usage
    memory_used = 0
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        torch.cuda.reset_peak_memory_stats(device)
    
    return {
        'avg_forward_time': avg_time,
        'throughput': throughput,
        'memory_usage_mb': memory_used,
        'total_time': total_time,
        'trials': num_trials
    }

def validate_temporal_parallelism_improvement(model_v1: nn.Module, model_v2: nn.Module,
                                            input_shape: Tuple[int, ...], 
                                            device: torch.device) -> Dict[str, Any]:
    """
    Compare v0.1.0 vs v0.2.0 performance improvements.
    
    Args:
        model_v1: v0.1.0 model
        model_v2: v0.2.0 model with temporal parallelism
        input_shape: Input shape for testing
        device: Device to run on
    
    Returns:
        Comparison results
    """
    print("Profiling v0.1.0 model...")
    v1_metrics = profile_model_performance(model_v1, input_shape, device)
    
    print("Profiling v0.2.0 model...")
    v2_metrics = profile_model_performance(model_v2, input_shape, device)
    
    # Calculate improvements
    speedup = v1_metrics['avg_forward_time'] / v2_metrics['avg_forward_time']
    throughput_improvement = v2_metrics['throughput'] / v1_metrics['throughput']
    memory_reduction = (v1_metrics['memory_usage_mb'] - v2_metrics['memory_usage_mb']) / v1_metrics['memory_usage_mb']
    
    results = {
        'v1_metrics': v1_metrics,
        'v2_metrics': v2_metrics,
        'improvements': {
            'speedup': speedup,
            'throughput_improvement': throughput_improvement,
            'memory_reduction_percent': memory_reduction * 100
        },
        'summary': {
            'faster': speedup > 1.0,
            'memory_efficient': memory_reduction > 0,
            'overall_better': speedup > 1.0 and memory_reduction >= 0
        }
    }
    
    return results