#!/usr/bin/env python3
"""
Task Vector Analysis - Analyze and manipulate task-specific model adaptations

Implements task arithmetic operations for model editing and composition.
Based on "Editing Models with Task Arithmetic" (Ilharco et al., 2023)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TaskVectorAnalysis:
    """
    Analyze and manipulate task vectors for model editing.
    
    Task vectors represent the difference between fine-tuned and base models,
    capturing task-specific knowledge that can be added, removed, or combined.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize task vector analyzer.
        
        Args:
            device: Torch device to use
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_vectors_cache = {}
        
    def compute_task_vector(
        self,
        base_model: nn.Module,
        finetuned_model: nn.Module,
        normalize: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task vector as difference between fine-tuned and base model.
        
        Args:
            base_model: Pre-trained base model
            finetuned_model: Fine-tuned model
            normalize: Whether to normalize the task vector
            
        Returns:
            Dictionary of parameter differences (task vector)
        """
        logger.info("Computing task vector...")
        
        task_vector = OrderedDict()
        
        base_params = dict(base_model.named_parameters())
        finetuned_params = dict(finetuned_model.named_parameters())
        
        # Compute difference for each parameter
        for name, finetuned_param in finetuned_params.items():
            if name in base_params:
                base_param = base_params[name]
                
                # Ensure parameters have same shape
                if base_param.shape != finetuned_param.shape:
                    logger.warning(f"Shape mismatch for {name}: base {base_param.shape} vs finetuned {finetuned_param.shape}")
                    continue
                    
                # Compute difference
                diff = finetuned_param.data - base_param.data
                
                # Optionally normalize
                if normalize:
                    norm = diff.norm(2)
                    if norm > 0:
                        diff = diff / norm
                        
                task_vector[name] = diff.clone().detach()
            else:
                logger.warning(f"Parameter {name} not found in base model")
                
        # Compute statistics
        stats = self._compute_vector_statistics(task_vector)
        logger.info(f"Task vector computed: {stats['total_parameters']} parameters, norm={stats['total_norm']:.4f}")
        
        return task_vector
        
    def task_arithmetic(
        self,
        vectors: List[Dict[str, torch.Tensor]],
        operation: str = 'add',
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform arithmetic operations on task vectors.
        
        Args:
            vectors: List of task vectors
            operation: 'add', 'subtract', 'average', or 'weighted_sum'
            weights: Weights for weighted operations
            
        Returns:
            Result task vector
        """
        if not vectors:
            raise ValueError("No vectors provided")
            
        if weights is None:
            weights = [1.0] * len(vectors)
            
        if len(weights) != len(vectors):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match vectors ({len(vectors)})")
            
        logger.info(f"Performing task arithmetic: {operation} on {len(vectors)} vectors")
        
        result = OrderedDict()
        
        # Get all parameter names
        all_params = set()
        for vector in vectors:
            all_params.update(vector.keys())
            
        # Perform operation on each parameter
        for param_name in all_params:
            param_vectors = []
            param_weights = []
            
            # Collect parameter from each vector
            for i, vector in enumerate(vectors):
                if param_name in vector:
                    param_vectors.append(vector[param_name])
                    param_weights.append(weights[i])
                    
            if not param_vectors:
                continue
                
            # Perform operation
            if operation == 'add':
                result[param_name] = sum(v * w for v, w in zip(param_vectors, param_weights))
            elif operation == 'subtract':
                if len(param_vectors) != 2:
                    raise ValueError("Subtract requires exactly 2 vectors")
                result[param_name] = param_vectors[0] - param_vectors[1]
            elif operation == 'average':
                result[param_name] = sum(param_vectors) / len(param_vectors)
            elif operation == 'weighted_sum':
                total_weight = sum(param_weights)
                result[param_name] = sum(v * w for v, w in zip(param_vectors, param_weights)) / total_weight
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        return result
        
    def vector_interpolation(
        self,
        vector1: Dict[str, torch.Tensor],
        vector2: Dict[str, torch.Tensor],
        alpha: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two task vectors.
        
        Args:
            vector1: First task vector
            vector2: Second task vector
            alpha: Interpolation factor (0 = vector1, 1 = vector2)
            
        Returns:
            Interpolated task vector
        """
        logger.info(f"Interpolating task vectors with alpha={alpha}")
        
        interpolated = OrderedDict()
        
        # Get common parameters
        common_params = set(vector1.keys()) & set(vector2.keys())
        
        for param_name in common_params:
            # Linear interpolation
            interpolated[param_name] = (1 - alpha) * vector1[param_name] + alpha * vector2[param_name]
            
        # Handle parameters only in one vector
        for param_name in vector1.keys():
            if param_name not in common_params:
                interpolated[param_name] = (1 - alpha) * vector1[param_name]
                
        for param_name in vector2.keys():
            if param_name not in common_params:
                interpolated[param_name] = alpha * vector2[param_name]
                
        return interpolated
        
    def apply_task_vector(
        self,
        base_model: nn.Module,
        task_vector: Dict[str, torch.Tensor],
        scaling_factor: float = 1.0,
        inplace: bool = False
    ) -> nn.Module:
        """
        Apply task vector to a base model.
        
        Args:
            base_model: Base model to modify
            task_vector: Task vector to apply
            scaling_factor: Scale the task vector before applying
            inplace: Modify model in-place (default: False)
            
        Returns:
            Modified model
        """
        logger.info(f"Applying task vector with scaling factor {scaling_factor}")
        
        if not inplace:
            # Create a copy of the model
            import copy
            model = copy.deepcopy(base_model)
        else:
            model = base_model
            
        # Apply task vector to each parameter
        for name, param in model.named_parameters():
            if name in task_vector:
                param.data += scaling_factor * task_vector[name].to(param.device)
                
        return model
        
    def performance_prediction(
        self,
        task_vector: Dict[str, torch.Tensor],
        test_batch: Dict[str, torch.Tensor],
        base_model: nn.Module,
        metric_fn: Optional[callable] = None
    ) -> Dict[str, float]:
        """
        Predict performance of applying task vector without full evaluation.
        
        Uses gradient alignment and magnitude as proxies for performance.
        
        Args:
            task_vector: Task vector to evaluate
            test_batch: Test data batch
            base_model: Base model
            metric_fn: Optional metric function
            
        Returns:
            Dictionary of predicted performance metrics
        """
        logger.info("Predicting task vector performance...")
        
        # Apply task vector with different scaling factors
        scaling_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
        performances = []
        
        for scale in scaling_factors:
            # Apply scaled task vector
            model = self.apply_task_vector(base_model, task_vector, scale, inplace=False)
            model.eval()
            
            with torch.no_grad():
                # Forward pass
                outputs = model(**test_batch)
                
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss.item()
                else:
                    # Compute cross-entropy loss
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = test_batch['input_ids'][..., 1:].contiguous()
                    loss = nn.CrossEntropyLoss()(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ).item()
                    
                performances.append(loss)
                
            del model
            torch.cuda.empty_cache()
            
        # Analyze performance curve
        performances = np.array(performances)
        optimal_scale_idx = np.argmin(performances)
        optimal_scale = scaling_factors[optimal_scale_idx]
        optimal_loss = performances[optimal_scale_idx]
        
        # Compute gradient of performance w.r.t. scaling
        performance_gradient = np.gradient(performances)
        
        # Compute task vector statistics
        vector_stats = self._compute_vector_statistics(task_vector)
        
        predictions = {
            'optimal_scaling_factor': optimal_scale,
            'optimal_loss': optimal_loss,
            'baseline_loss': performances[0],  # Scale = 0
            'improvement': performances[0] - optimal_loss,
            'performance_curve': performances.tolist(),
            'performance_gradient': performance_gradient.tolist(),
            'vector_norm': vector_stats['total_norm'],
            'vector_sparsity': vector_stats['sparsity'],
            'predicted_quality': 'good' if optimal_loss < performances[0] else 'poor'
        }
        
        # Add custom metric if provided
        if metric_fn is not None:
            model = self.apply_task_vector(base_model, task_vector, optimal_scale, inplace=False)
            model.eval()
            with torch.no_grad():
                custom_metric = metric_fn(model, test_batch)
                predictions['custom_metric'] = custom_metric
            del model
            
        return predictions
        
    def multi_task_composition(
        self,
        task_vectors: Dict[str, Dict[str, torch.Tensor]],
        composition_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compose multiple task vectors for multi-task learning.
        
        Args:
            task_vectors: Dictionary mapping task names to task vectors
            composition_weights: Weights for each task (uniform if None)
            
        Returns:
            Composed task vector
        """
        if not task_vectors:
            raise ValueError("No task vectors provided")
            
        task_names = list(task_vectors.keys())
        logger.info(f"Composing {len(task_names)} tasks: {task_names}")
        
        if composition_weights is None:
            # Uniform weights
            composition_weights = {name: 1.0 / len(task_names) for name in task_names}
            
        # Normalize weights
        total_weight = sum(composition_weights.values())
        composition_weights = {k: v / total_weight for k, v in composition_weights.items()}
        
        # Weighted sum of task vectors
        composed = OrderedDict()
        
        # Get all parameter names
        all_params = set()
        for vector in task_vectors.values():
            all_params.update(vector.keys())
            
        for param_name in all_params:
            param_sum = None
            
            for task_name, task_vector in task_vectors.items():
                if param_name in task_vector:
                    weight = composition_weights[task_name]
                    weighted_param = weight * task_vector[param_name]
                    
                    if param_sum is None:
                        param_sum = weighted_param
                    else:
                        param_sum += weighted_param
                        
            if param_sum is not None:
                composed[param_name] = param_sum
                
        # Log composition statistics
        stats = self._compute_vector_statistics(composed)
        logger.info(f"Composed vector: norm={stats['total_norm']:.4f}, sparsity={stats['sparsity']:.3f}")
        
        return composed
        
    def analyze_vector_overlap(
        self,
        vectors: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        Analyze overlap and interference between task vectors.
        
        Args:
            vectors: Dictionary mapping names to task vectors
            
        Returns:
            Analysis of vector relationships
        """
        logger.info(f"Analyzing overlap between {len(vectors)} task vectors")
        
        names = list(vectors.keys())
        n_vectors = len(names)
        
        # Compute pairwise similarities
        similarity_matrix = np.zeros((n_vectors, n_vectors))
        overlap_matrix = np.zeros((n_vectors, n_vectors))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i <= j:
                    # Compute cosine similarity
                    sim = self._cosine_similarity(vectors[name1], vectors[name2])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                    
                    # Compute parameter overlap
                    overlap = self._parameter_overlap(vectors[name1], vectors[name2])
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
                    
        # Identify conflicts (high magnitude, opposite direction)
        conflicts = []
        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                if similarity_matrix[i, j] < -0.5:  # Strong negative correlation
                    conflicts.append({
                        'task1': names[i],
                        'task2': names[j],
                        'similarity': similarity_matrix[i, j],
                        'overlap': overlap_matrix[i, j]
                    })
                    
        # Identify synergies (high positive correlation)
        synergies = []
        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                if similarity_matrix[i, j] > 0.7:  # Strong positive correlation
                    synergies.append({
                        'task1': names[i],
                        'task2': names[j],
                        'similarity': similarity_matrix[i, j],
                        'overlap': overlap_matrix[i, j]
                    })
                    
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'overlap_matrix': overlap_matrix.tolist(),
            'mean_similarity': similarity_matrix[np.triu_indices(n_vectors, k=1)].mean(),
            'mean_overlap': overlap_matrix[np.triu_indices(n_vectors, k=1)].mean(),
            'conflicts': conflicts,
            'synergies': synergies,
            'task_names': names
        }
        
    def _compute_vector_statistics(self, vector: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute statistics of a task vector."""
        total_params = 0
        total_norm = 0.0
        nonzero_params = 0
        
        for param in vector.values():
            total_params += param.numel()
            total_norm += param.norm(2).item() ** 2
            nonzero_params += (param.abs() > 1e-6).sum().item()
            
        total_norm = total_norm ** 0.5
        sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
        
        return {
            'total_parameters': total_params,
            'total_norm': total_norm,
            'sparsity': sparsity,
            'nonzero_parameters': nonzero_params,
            'mean_magnitude': total_norm / (total_params ** 0.5) if total_params > 0 else 0.0
        }
        
    def _cosine_similarity(self, vector1: Dict[str, torch.Tensor], vector2: Dict[str, torch.Tensor]) -> float:
        """Compute cosine similarity between two task vectors."""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        # Use common parameters
        for name in vector1:
            if name in vector2:
                v1 = vector1[name].flatten()
                v2 = vector2[name].flatten()
                
                dot_product += (v1 * v2).sum().item()
                norm1 += v1.norm(2).item() ** 2
                norm2 += v2.norm(2).item() ** 2
                
        norm1 = norm1 ** 0.5
        norm2 = norm2 ** 0.5
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
        
    def _parameter_overlap(self, vector1: Dict[str, torch.Tensor], vector2: Dict[str, torch.Tensor]) -> float:
        """Compute parameter overlap ratio between two task vectors."""
        params1 = set(vector1.keys())
        params2 = set(vector2.keys())
        
        intersection = params1 & params2
        union = params1 | params2
        
        if union:
            return len(intersection) / len(union)
        return 0.0
