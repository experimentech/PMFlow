"""
Training utilities for PMFlow Language Model.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import math


class PMFlowLMTrainer:
    """
    Trainer for PMFlow Language Models.
    
    Handles:
    - Forward/backward passes
    - Loss computation
    - Gradient clipping
    - Checkpoint saving/loading
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: PMFlowLanguageModel instance
            optimizer: PyTorch optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.global_step = 0
        self.total_tokens = 0
    
    def train_step(self, batch_ids: torch.Tensor, 
                   max_grad_norm: float = 1.0) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch_ids: (batch_size, seq_len) token IDs
            max_grad_norm: Maximum gradient norm for clipping
        
        Returns:
            Dictionary with loss and perplexity metrics
        """
        batch_ids = batch_ids.to(self.device)
        
        # Forward pass
        logits = self.model.forward_sequence(batch_ids)  # (batch, seq_len, vocab)
        
        # Prepare targets: shift labels by 1 (predict token_{t+1} from token_t)
        targets = batch_ids[:, 1:].contiguous()  # (batch, seq_len-1)
        logits_pred = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab)
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits_pred.shape
        logits_flat = logits_pred.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len)
        
        # Compute loss
        loss = self.loss_fn(logits_flat, targets_flat)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update counters
        self.global_step += 1
        self.total_tokens += batch_size * seq_len
        
        # Compute perplexity
        perplexity = math.exp(loss.item())
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
    
    def eval_step(self, batch_ids: torch.Tensor) -> Dict[str, float]:
        """
        Single evaluation step (no gradients, no optimizer update).
        
        Args:
            batch_ids: (batch_size, seq_len) token IDs
        
        Returns:
            Dictionary with loss and perplexity metrics
        """
        self.model.eval()
        batch_ids = batch_ids.to(self.device)
        
        with torch.no_grad():
            logits = self.model.forward_sequence(batch_ids)
            
            targets = batch_ids[:, 1:].contiguous()
            logits_pred = logits[:, :-1, :].contiguous()
            
            batch_size, seq_len, vocab_size = logits_pred.shape
            logits_flat = logits_pred.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            
            loss = self.loss_fn(logits_flat, targets_flat)
            perplexity = math.exp(loss.item())
        
        self.model.train()
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
        }
    
    def save_checkpoint(self, path: str, epoch: int = 0):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'total_tokens': self.total_tokens,
            'epoch': epoch,
        }, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.total_tokens = checkpoint.get('total_tokens', 0)


class WarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with linear warmup.
    
    Linearly increases LR from 0 to base_lr over warmup_steps,
    then decays following a schedule.
    """
    
    def __init__(self, optimizer, warmup_steps: int, 
                 total_steps: int, last_epoch: int = -1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps to linearly increase LR
            total_steps: Total number of training steps
            last_epoch: Use for resuming training
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        super().__init__(optimizer, lr_lambda, last_epoch)
