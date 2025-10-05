#!/usr/bin/env python3
"""
Enhanced TracIn Implementation - Complete training data attribution

Extends basic TracIn with:
- Cross-influence analysis between training and test samples
- Checkpoint interpolation for finer-grained attribution
- Gradient accumulation over training trajectories
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from tqdm import tqdm
import sys

logger = logging.getLogger(__name__)


class TracInEnhanced:
    """
    Enhanced TracIn implementation with cross-influence and trajectory analysis.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize enhanced TracIn.
        
        Args:
            device: Torch device to use
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_cache = {}
        
    def compute_cross_influence(
        self,
        model: nn.Module,
        train_batch: Dict[str, torch.Tensor],
        test_batch: Dict[str, torch.Tensor],
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute how training samples influence test samples.
        
        This measures how much each training example contributed to the
        model's predictions on test examples.
        
        Args:
            model: Model to analyze
            train_batch: Training data batch
            test_batch: Test data batch
            loss_fn: Loss function (uses CrossEntropyLoss if None)
            
        Returns:
            Dictionary with cross-influence scores
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            
        model.eval()
        
        # Get batch sizes
        train_size = train_batch['input_ids'].shape[0]
        test_size = test_batch['input_ids'].shape[0]
        
        # Initialize influence matrix
        influence_matrix = torch.zeros(test_size, train_size, device=self.device)
        
        logger.info(f"Computing cross-influence: {train_size} train x {test_size} test samples")
        
        # Compute gradients for test samples
        test_gradients = []
        for i in range(test_size):
            # Get single test sample
            test_input = {
                'input_ids': test_batch['input_ids'][i:i+1],
                'attention_mask': test_batch['attention_mask'][i:i+1]
            }
            
            # Forward pass
            model.zero_grad()
            outputs = model(**test_input)
            
            # Compute loss (assuming language modeling)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = test_input['input_ids'][..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.mean()
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            test_grad = self._collect_gradients(model)
            test_gradients.append(test_grad)
            
        # Compute gradients for training samples and calculate influence
        for j in tqdm(range(train_size), desc="Computing training gradients", leave=False, file=sys.stderr):
            # Get single training sample
            train_input = {
                'input_ids': train_batch['input_ids'][j:j+1],
                'attention_mask': train_batch['attention_mask'][j:j+1]
            }
            
            # Forward pass
            model.zero_grad()
            outputs = model(**train_input)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = train_input['input_ids'][..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.mean()
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            train_grad = self._collect_gradients(model)
            
            # Compute influence on each test sample
            for i in range(test_size):
                # TracIn: gradient dot product
                influence = self._gradient_dot_product(test_gradients[i], train_grad)
                influence_matrix[i, j] = influence
                
        # Compute summary statistics
        results = {
            'influence_matrix': influence_matrix,
            'mean_influence': influence_matrix.mean().item(),
            'max_influence': influence_matrix.max().item(),
            'min_influence': influence_matrix.min().item(),
            'most_influential_train_per_test': influence_matrix.argmax(dim=1),  # For each test, most influential train
            'most_influenced_test_per_train': influence_matrix.argmax(dim=0),  # For each train, most influenced test
            'total_influence_per_train': influence_matrix.sum(dim=0),  # Total influence of each training sample
            'total_influence_per_test': influence_matrix.sum(dim=1)   # Total influence on each test sample
        }
        
        return results
        
    def checkpoint_interpolation(
        self,
        checkpoint1_path: Union[str, Path],
        checkpoint2_path: Union[str, Path],
        model_class,
        alpha: float = 0.5,
        n_interpolations: int = 5
    ) -> List[nn.Module]:
        """
        Interpolate between checkpoints for fine-grained TracIn analysis.
        
        Args:
            checkpoint1_path: Path to first checkpoint
            checkpoint2_path: Path to second checkpoint
            model_class: Class or function to create model
            alpha: Interpolation factor (0 = checkpoint1, 1 = checkpoint2)
            n_interpolations: Number of interpolated models to create
            
        Returns:
            List of interpolated models
        """
        logger.info(f"Interpolating between checkpoints with {n_interpolations} steps")
        
        # Load checkpoints
        checkpoint1 = torch.load(checkpoint1_path, map_location=self.device)
        checkpoint2 = torch.load(checkpoint2_path, map_location=self.device)
        
        # Extract state dicts
        if isinstance(checkpoint1, dict) and 'model_state_dict' in checkpoint1:
            state1 = checkpoint1['model_state_dict']
        elif isinstance(checkpoint1, dict) and 'state_dict' in checkpoint1:
            state1 = checkpoint1['state_dict']
        else:
            state1 = checkpoint1
            
        if isinstance(checkpoint2, dict) and 'model_state_dict' in checkpoint2:
            state2 = checkpoint2['model_state_dict']
        elif isinstance(checkpoint2, dict) and 'state_dict' in checkpoint2:
            state2 = checkpoint2['state_dict']
        else:
            state2 = checkpoint2
            
        # Create interpolated models
        interpolated_models = []
        
        for i in range(n_interpolations):
            # Compute interpolation weight
            current_alpha = i / (n_interpolations - 1) if n_interpolations > 1 else alpha
            
            # Create new model
            model = model_class()
            
            # Interpolate parameters
            interpolated_state = {}
            for key in state1.keys():
                if key in state2:
                    # Linear interpolation
                    interpolated_state[key] = (1 - current_alpha) * state1[key] + current_alpha * state2[key]
                else:
                    # Key only in first checkpoint
                    interpolated_state[key] = state1[key]
                    
            # Load interpolated state
            model.load_state_dict(interpolated_state, strict=False)
            model.to(self.device)
            model.eval()
            
            interpolated_models.append(model)
            
        return interpolated_models
        
    def gradient_accumulation_trajectory(
        self,
        checkpoints: List[Union[str, Path]],
        model_class,
        batch: Dict[str, torch.Tensor],
        loss_fn: Optional[nn.Module] = None,
        learning_rate_schedule: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate gradients over training trajectory.
        
        This computes the cumulative influence of training examples
        across multiple checkpoints, weighted by learning rate.
        
        Args:
            checkpoints: List of checkpoint paths
            model_class: Class or function to create model
            batch: Data batch to compute gradients on
            loss_fn: Loss function
            learning_rate_schedule: Learning rates for each checkpoint
            
        Returns:
            Dictionary with accumulated gradients and influence scores
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
            
        if learning_rate_schedule is None:
            # Default: uniform weights
            learning_rate_schedule = [1.0 / len(checkpoints)] * len(checkpoints)
            
        logger.info(f"Accumulating gradients over {len(checkpoints)} checkpoints")
        
        accumulated_gradients = None
        gradient_norms = []
        losses = []
        
        for idx, checkpoint_path in enumerate(tqdm(checkpoints, desc="Processing checkpoints")):
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create and load model
            model = model_class()
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            
            # Forward pass
            model.zero_grad()
            outputs = model(**batch)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['input_ids'][..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Collect and accumulate gradients
            gradients = self._collect_gradients(model)
            
            # Weight by learning rate
            lr_weight = learning_rate_schedule[idx]
            
            if accumulated_gradients is None:
                accumulated_gradients = {k: v * lr_weight for k, v in gradients.items()}
            else:
                for k, v in gradients.items():
                    accumulated_gradients[k] = accumulated_gradients[k] + v * lr_weight
                    
            # Compute gradient norm
            grad_norm = self._compute_gradient_norm(gradients)
            gradient_norms.append(grad_norm)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        # Compute final statistics
        final_norm = self._compute_gradient_norm(accumulated_gradients)
        
        results = {
            'accumulated_gradients': accumulated_gradients,
            'final_gradient_norm': final_norm,
            'checkpoint_gradient_norms': gradient_norms,
            'checkpoint_losses': losses,
            'mean_loss': np.mean(losses),
            'loss_trend': 'decreasing' if losses[-1] < losses[0] else 'increasing',
            'gradient_norm_trend': 'decreasing' if gradient_norms[-1] < gradient_norms[0] else 'increasing'
        }
        
        # Compute influence score based on accumulated gradients
        influence_score = self._compute_influence_score(accumulated_gradients)
        results['trajectory_influence_score'] = influence_score
        
        return results
        
    def compute_self_influence_enhanced(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        n_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Enhanced self-influence computation with more detailed analysis.
        
        Args:
            model: Model to analyze
            batch: Data batch
            n_samples: Number of samples to use (None = all)
            
        Returns:
            Dictionary with self-influence scores and analysis
        """
        model.eval()
        
        batch_size = batch['input_ids'].shape[0]
        if n_samples is not None:
            batch_size = min(batch_size, n_samples)
            
        self_influences = []
        
        for i in tqdm(range(batch_size), desc="Computing self-influence"):
            # Get single sample
            single_input = {
                'input_ids': batch['input_ids'][i:i+1],
                'attention_mask': batch['attention_mask'][i:i+1]
            }
            
            # First forward-backward
            model.zero_grad()
            outputs1 = model(**single_input)
            logits1 = outputs1.logits if hasattr(outputs1, 'logits') else outputs1
            
            # Compute loss
            shift_logits = logits1[..., :-1, :].contiguous()
            shift_labels = single_input['input_ids'][..., 1:].contiguous()
            
            loss1 = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss1.backward()
            
            # Collect gradients
            grad1 = self._collect_gradients(model)
            
            # Second forward-backward (for stability)
            model.zero_grad()
            outputs2 = model(**single_input)
            logits2 = outputs2.logits if hasattr(outputs2, 'logits') else outputs2
            
            shift_logits = logits2[..., :-1, :].contiguous()
            loss2 = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss2.backward()
            
            # Collect gradients
            grad2 = self._collect_gradients(model)
            
            # Compute self-influence (gradient dot product)
            self_inf = self._gradient_dot_product(grad1, grad2)
            self_influences.append(self_inf)
            
        # Compute statistics
        self_influences = torch.tensor(self_influences)
        
        results = {
            'mean_self_influence': self_influences.mean().item(),
            'std_self_influence': self_influences.std().item(),
            'max_self_influence': self_influences.max().item(),
            'min_self_influence': self_influences.min().item(),
            'median_self_influence': self_influences.median().item(),
            'all_self_influences': self_influences.cpu().numpy().tolist(),
            'high_influence_samples': torch.where(self_influences > self_influences.mean() + self_influences.std())[0].tolist(),
            'low_influence_samples': torch.where(self_influences < self_influences.mean() - self_influences.std())[0].tolist()
        }
        
        return results
        
    def _collect_gradients(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Collect gradients from model parameters."""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        return gradients
        
    def _gradient_dot_product(self, grad1: Dict[str, torch.Tensor], grad2: Dict[str, torch.Tensor]) -> float:
        """Compute dot product between two gradient dictionaries."""
        dot_product = 0.0
        for key in grad1:
            if key in grad2:
                dot_product += (grad1[key] * grad2[key]).sum().item()
        return dot_product
        
    def _compute_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of gradients."""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm(2).item() ** 2
        return total_norm ** 0.5
        
    def _compute_influence_score(self, gradients: Dict[str, torch.Tensor]) -> float:
        """
        Compute overall influence score from gradients.
        
        This is a heuristic that combines gradient magnitude and variance.
        """
        # Compute mean magnitude
        magnitudes = [grad.abs().mean().item() for grad in gradients.values()]
        mean_magnitude = np.mean(magnitudes)
        
        # Compute variance across parameters
        variances = [grad.var().item() for grad in gradients.values()]
        mean_variance = np.mean(variances)
        
        # Combine into influence score
        influence_score = mean_magnitude * (1 + mean_variance)
        
        return influence_score
