"""
ExtendedModularityMetrics: Complete implementation of extended modularity metrics.
REFACTORED VERSION using FisherCollector for efficient group-level Fisher computation.

Includes all bug fixes and new metrics with optimized Fisher handling.

CRITICAL FIXES APPLIED:
1. Fisher key mismatch fixed - gradients now use same task name as Fisher
2. Added missing _compute_cka_per_layer method
3. Fixed identical metrics (effective rank vs participation ratio)
4. Fixed generator device bug
5. Fixed SAM sharpness state corruption
6. Updated all SVD calls to use torch.linalg.svd
7. Fixed device mismatches in function_space_distance
8. Added empty list crash protection
9. Added memory-efficient processing
10. Added safe vocab_size access
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import uuid
import time
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict

# Import base classes
from fisher.core.fisher_collector import FisherCollector
from fisher.core.fisher_compatibility import FisherCompatibilityMixin

# Set up logger
logger = logging.getLogger(__name__)


class ExtendedModularityMetrics(FisherCollector, FisherCompatibilityMixin):
    """
    Complete implementation of extended modularity metrics.
    REFACTORED: Now uses FisherCollector for efficient group-level Fisher computation.

    Key improvements:
    - Inherits from FisherCollector for group-level Fisher storage
    - 100x memory reduction via group-level storage
    - Numerical stability improvements
    - Full backward compatibility
    """

    def __init__(self,
                 seed: Optional[int] = None,
                 use_deterministic: bool = False,
                 # New FisherCollector parameters
                 fisher_reduction: str = 'group',
                 fisher_storage: str = 'cpu_fp16',
                 use_causal_shift: bool = True,
                 # Numerical stability parameters
                 eps_small: float = 1e-6,  # For float32 computations
                 eps_tiny: float = 1e-10,  # For float64 computations
                 # Memory management
                 max_gradient_cache_mb: float = 1000.0,  # Max gradient cache in MB
                 # Configurability
                 default_last_n_layers: int = 6):  # Whether to shift labels for causal LM
        """
        Initialize metrics calculator with FisherCollector backend.

        Args:
            seed: Random seed for reproducibility
            use_deterministic: Use deterministic algorithms
            fisher_reduction: Fisher reduction mode ('param', 'group')
            fisher_storage: Fisher storage strategy
        """
        # Initialize FisherCollector base
        super().__init__(
            reduction=fisher_reduction,
            storage=fisher_storage,
            ema_decay=0.99,  # Default EMA decay
            use_ewc=False,    # ModularityMetrics doesn't need EWC by default
            debug=False
        )

        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if use_deterministic:
            torch.use_deterministic_algorithms(True)

        self.use_causal_shift = use_causal_shift

        # Store numerical stability parameters
        self.eps_small = eps_small
        self.eps_tiny = eps_tiny

        # Memory management
        self.max_gradient_cache_mb = max_gradient_cache_mb
        self._gradient_cache_size = 0  # Track current cache size
        self._gradient_cache = OrderedDict()  # LRU gradient cache

        # Configurability
        self.default_last_n_layers = default_last_n_layers

        logger.info(f"ExtendedModularityMetrics initialized with FisherCollector backend: "
                   f"reduction={fisher_reduction}, storage={fisher_storage}, "
                   f"eps_small={eps_small}, max_cache_mb={max_gradient_cache_mb}")

    # ============= HELPER METHODS =============

    def _key_suffix(self, key: str) -> str:
        """Extract parameter suffix from a Fisher key for matching across tasks.

        Args:
            key: Full Fisher key in format 'task|param|group'

        Returns:
            Suffix containing 'param|group' for matching
        """
        parts = key.split('|')
        if len(parts) >= 3:
            # Return param|group part
            return '|'.join(parts[1:])
        elif len(parts) == 2:
            # Old format, just return param part
            return parts[1]
        return key

    def _safe_get_vocab_size(self, model) -> Optional[int]:
        """Safely get vocab size from model config.

        Args:
            model: The model to check

        Returns:
            Vocab size or None if not available
        """
        config = getattr(model, 'config', None)
        if config is not None:
            return getattr(config, 'vocab_size', None)
        return None

    def _detect_model_layers(self, model):
        """Detect model architecture and return list of layers.

        Args:
            model: The model to analyze

        Returns:
            List of layer modules or None if architecture not recognized
        """
        # Try different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Qwen, LLaMA style
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return list(model.transformer.h)
        elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
            # BERT style
            return list(model.bert.encoder.layer)
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            # GPT-NeoX style
            return list(model.gpt_neox.layers)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # Generic encoder style
            return list(model.encoder.layer)
        else:
            return None

    # ============= BACKWARD COMPATIBILITY WRAPPERS =============

    def update_fisher_ema(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        task: str = 'task1'
    ):
        """
        Backward-compatible wrapper for Fisher EMA updates.
        Now uses FisherCollector's efficient group-level storage.
        """
        # Call parent's update_fisher_ema directly to avoid recursion through collect_fisher
        super().update_fisher_ema(model, batch, task)

    def _estimate_fisher_diagonal(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        n_samples: int = 8,
        layers_prefix: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible wrapper for direct Fisher estimation.
        Now uses FisherCollector's one-shot mode.
        """
        # Create unique temporary task to avoid collisions
        temp_task = f'_temp_oneshot_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}'

        # Collect one-shot Fisher with limited samples
        self.compute_oneshot_fisher(
            model, data_batch, temp_task, n_samples=n_samples
        )

        # Get group Fisher
        group_fisher = self.get_group_fisher(temp_task, bias_corrected=False, mode='oneshot')

        # Filter by layer prefix if specified
        if layers_prefix:
            filtered = {}
            for key, value in group_fisher.items():
                parts = key.split('|')
                if len(parts) >= 2:
                    param_name = parts[1]
                    if any(param_name.startswith(prefix) for prefix in layers_prefix):
                        filtered[param_name] = value
            param_fisher = filtered
        else:
            # Expand to per-parameter for compatibility
            param_fisher = self.expand_group_to_param_fisher(group_fisher, model)

        # Clean up temporary task
        self.clear_fisher(temp_task)

        return param_fisher

    def get_bias_corrected_fisher_ema(self, task: str) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible wrapper for bias-corrected Fisher EMA.
        """
        # Get from parent's method
        corrected = self.get_bias_corrected_fisher(task, mode='ema')

        # Convert to old format (param_name -> values)
        old_format = {}
        for key, value in corrected.items():
            parts = key.split('|')
            if len(parts) >= 2:
                param_name = parts[1]
                old_format[param_name] = value

        return old_format

    def _get_ema_fisher_for_task(
        self,
        task: str,
        layers_prefix: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get EMA Fisher filtered by task and optionally by layer prefixes.
        Now uses FisherCollector's efficient storage.
        """
        # Get bias-corrected group Fisher
        group_fisher = self.get_group_fisher(task, bias_corrected=True)

        if not group_fisher:
            logger.warning(f"No EMA Fisher found for task '{task}'")
            return {}

        # Convert to old format and filter
        result = {}
        for key, value in group_fisher.items():
            parts = key.split('|')
            if len(parts) >= 2:
                param_name = parts[1]

                # Filter by layer prefix if specified
                if layers_prefix is None or any(param_name.startswith(pre) for pre in layers_prefix):
                    result[param_name] = value

        return result

    # ============= FISHER-BASED DAMAGE METRICS =============

    def compute_fisher_weighted_damage(
        self,
        model,
        task_A_batch: Dict[str, torch.Tensor],
        task_B_batch: Dict[str, torch.Tensor],
        n_fisher_samples: int = 8,
        damage_type: str = 'asymmetric',  # CHANGED: asymmetric is theoretically justified for ICML
        target_layers: Optional[List[str]] = None,
        fisher_type: str = 'cached',  # 'cached' (use pre-computed) or 'direct' (compute fresh); 'ema' accepted for backward compat
        fisher_mode: str = 'accumulated',  # 'accumulated' (Welford, unbiased) or 'ema' (exponential decay) - only used if fisher_type='cached'
        task_A_name: str = 'math',
        task_B_name: str = 'general',
        use_abs: bool = True
    ) -> Dict[str, Any]:
        """
        Compute Fisher-weighted gradient damage between tasks.
        Optimized to use group-level Fisher for efficiency.

        Measures how much task B's gradients would damage task A's important parameters.

        Args:
            fisher_type: How to obtain Fisher Information
                - 'cached': Use pre-computed Fisher (fast, RECOMMENDED for ICML)
                - 'direct': Compute fresh Fisher on-the-fly (slow, ~2s overhead)
                - 'ema': DEPRECATED alias for 'cached' (backward compatibility only)

            fisher_mode: Which cached Fisher to use (only applies if fisher_type='cached')
                - 'accumulated': Welford-accumulated Fisher (RECOMMENDED for ICML)
                  * Unbiased estimate (equal weight to all samples)
                  * Lower variance (~1% with 768 samples)
                  * Numerically stable (Welford algorithm)
                  * Publication quality ✅

                - 'ema': Exponentially-weighted moving average Fisher
                  * Biased toward recent samples
                  * Higher variance
                  * Use only for online learning / adaptation
            damage_type: Type of damage metric
                - 'asymmetric': grad_B^2 * Fisher_A (RECOMMENDED for ICML - stronger theoretical justification)
                  Based on Taylor expansion: ΔL_A ≈ ½ g_B^T H_A g_B ≈ ½ Σ g_B²·F_A
                  where F_A is diagonal Fisher approximation to Hessian H_A.
                  Reference: Kirkpatrick et al. 2017 (EWC), Zenke et al. 2017 (Continual Learning)

                - 'symmetric': |grad_B| * sqrt(Fisher_A)
                  Can be interpreted as weighted parameter perturbation distance.
                  Less rigorous theoretical basis. Use for sensitivity analysis only.

                - 'l1_weighted': |grad_B| * Fisher_A
                  L1 norm weighted by importance. Not recommended for publication.

        Note: Uses model.eval() for deterministic behavior. Gradients still work correctly.
        """
        model.eval()
        device = next(model.parameters()).device

        # CRITICAL FIX: Check gradient requirements on all parameters
        # Pretrained models often load with requires_grad=False, causing silent failures
        params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())

        if params_with_grad < total_params * 0.9:
            logger.warning(
                f"⚠️  Only {params_with_grad}/{total_params} parameters have requires_grad=True! "
                f"This will compute gradients for only a tiny fraction of parameters. "
                f"Enabling gradients for all parameters for accurate Fisher damage computation."
            )
            # Store original state and enable gradients for all parameters
            original_requires_grad = {}
            for name, param in model.named_parameters():
                original_requires_grad[name] = param.requires_grad
                param.requires_grad = True
        else:
            original_requires_grad = None

        # Move batches to device
        task_A_batch = self._to_model_device(model, task_A_batch)
        task_B_batch = self._to_model_device(model, task_B_batch)

        # Add labels if needed
        vocab_size = self._safe_get_vocab_size(model)
        task_A_batch = self._with_labels(task_A_batch, vocab_size=vocab_size)
        task_B_batch = self._with_labels(task_B_batch, vocab_size=vocab_size)

        # Get Fisher importance for task A
        if fisher_type == 'direct':
            # Compute fresh Fisher diagonal for task A (one-shot mode)
            self.compute_oneshot_fisher(
                model, task_A_batch, task_A_name, n_samples=n_fisher_samples
            )
            fisher_A = self.get_group_fisher(task_A_name, bias_corrected=False)
        else:  # fisher_type == 'cached' or 'ema' (backward compat)
            # Use pre-computed Fisher (mode controlled by fisher_mode parameter)
            # fisher_mode='accumulated': Welford Fisher (unbiased, ICML quality)
            # fisher_mode='ema': EMA Fisher (biased toward recent samples)
            fisher_A = self.get_group_fisher(task_A_name, bias_corrected=True, mode=fisher_mode)

            if not fisher_A:
                # Fallback to alternate mode if requested mode not available
                fallback_mode = 'ema' if fisher_mode == 'accumulated' else 'accumulated'
                logger.warning(f"Fisher mode='{fisher_mode}' not found for '{task_A_name}', trying mode='{fallback_mode}'")
                fisher_A = self.get_group_fisher(task_A_name, bias_corrected=True, mode=fallback_mode)

                if not fisher_A:
                    raise ValueError(
                        f"No Fisher found for task '{task_A_name}'. "
                        f"Call update_fisher_ema() first or use fisher_type='direct'."
                    )
            else:
                # Log successful use of requested Fisher mode
                n_samples = getattr(self, 'n_samples_seen', {}).get(task_A_name, 0)
                mode_desc = 'Welford-accumulated (unbiased)' if fisher_mode == 'accumulated' else 'EMA (exponential decay)'
                logger.info(f"✓ Using {mode_desc} Fisher for '{task_A_name}' (n={n_samples} samples)")

            # ICML VALIDATION: Check Fisher sample size for statistical validity
            task_samples = getattr(self, 'n_samples_seen', {}).get(task_A_name, 0)
            if task_samples < 3:
                logger.warning(
                    f"⚠️  ICML WARNING: Fisher for task '{task_A_name}' computed from only {task_samples} batch(es). "
                    f"Minimum recommended: 3 batches for bias correction reliability. "
                    f"Results may have high variance. Consider using fisher_type='direct' with n_fisher_samples >= 8."
                )

        # Compute gradients for task B
        try:
            model.zero_grad()
            with torch.enable_grad():
                outputs_B = model(**task_B_batch)
                loss_B = outputs_B.loss
                loss_B.backward()

            # Compute gradients with memory management
            # REPRODUCIBILITY FIX: Sort parameters by name for deterministic ordering
            task_B_grads = {}
            gradient_memory_bytes = 0
            max_bytes = self.max_gradient_cache_mb * 1024 * 1024

            # Sort to ensure deterministic gradient collection
            sorted_params = sorted(model.named_parameters(), key=lambda x: x[0])

            for name, param in sorted_params:
                if param.grad is not None:
                    # Store gradient
                    grad = param.grad.detach().clone()

                    # Check memory before storing
                    grad_bytes = grad.element_size() * grad.nelement()
                    if gradient_memory_bytes + grad_bytes > max_bytes:
                        n_remaining = len(sorted_params) - len(task_B_grads)
                        required_mb = (gradient_memory_bytes + grad_bytes * n_remaining) / (1024 * 1024)
                        logger.error(
                            f"⚠️  CRITICAL: Gradient cache limit reached ({self.max_gradient_cache_mb}MB)! "
                            f"Collected only {len(task_B_grads)}/{len(sorted_params)} parameters ({100*len(task_B_grads)/len(sorted_params):.1f}%). "
                            f"Missing {n_remaining} parameters will BIAS results toward parameters with lower alphabetical names. "
                            f"Required cache: {required_mb:.1f}MB. "
                            f"Increase max_gradient_cache_mb to at least {required_mb:.0f} for complete gradient collection."
                        )
                        # Return error for incomplete gradient collection (ICML requirement)
                        return {
                            'error': 'Incomplete gradient collection - cache limit reached',
                            'parameters_collected': len(task_B_grads),
                            'parameters_total': len(sorted_params),
                            'completion_rate': len(task_B_grads) / len(sorted_params),
                            'required_cache_mb': required_mb,
                            'current_cache_mb': self.max_gradient_cache_mb
                        }

                    # Convert to group-level for efficiency
                    group_grad, group_type, _ = self._reduce_to_groups(
                        name, grad, param.shape, model
                    )

                    # CRITICAL FIX: Create key using task_A_name to match Fisher keys
                    # This ensures we can match Fisher and gradient keys
                    key = self._make_key(task_A_name, name, group_type)
                    task_B_grads[key] = group_grad
                    gradient_memory_bytes += grad_bytes

            model.zero_grad()

            # Compute damage metrics using group-level values
            total_damage = 0.0
            layer_damages = {}

            # Get epsilon for numerical stability
            eps = self.eps_small if hasattr(self, 'eps_small') else 1e-8

            for key, fisher_values in fisher_A.items():
                if key in task_B_grads:
                    grad_B = task_B_grads[key]

                    # Ensure same device
                    if grad_B.device != fisher_values.device:
                        grad_B = grad_B.to(fisher_values.device)

                    # Compute damage based on type with numerical stability
                    if damage_type == 'symmetric':
                        # Symmetric damage: |grad_B| * sqrt(Fisher_A)
                        # NUMERICAL FIX: Clamp Fisher before sqrt to prevent precision loss
                        fisher_safe = fisher_values.clamp(min=eps)
                        damage = (grad_B.abs() if use_abs else grad_B) * fisher_safe.sqrt()
                    elif damage_type == 'asymmetric':
                        # Asymmetric: grad_B^2 * Fisher_A
                        # RECOMMENDED for ICML: Stronger theoretical justification
                        # Based on Taylor expansion: ΔL_A ≈ ½ g_B^T H_A g_B ≈ ½ Σ g_B²·F_A
                        damage = grad_B.pow(2) * fisher_values
                    elif damage_type == 'cosine':
                        # Cosine damage doesn't make mathematical sense between grad and Fisher
                        # Using L2 damage weighted by Fisher instead
                        logger.warning("Cosine damage type is deprecated. Using L2 weighted damage instead.")
                        damage = grad_B.pow(2) * fisher_values
                    elif damage_type == 'l1_weighted':
                        # L1 damage weighted by Fisher
                        damage = (grad_B.abs() if use_abs else grad_B) * fisher_values
                    else:
                        # Default to L1 weighted
                        damage = (grad_B.abs() if use_abs else grad_B) * fisher_values

                    # Aggregate damage
                    damage_val = damage.sum().item()
                    total_damage += damage_val

                    # Track per-layer damage
                    param_name = key.split('|')[1] if '|' in key else key
                    layer_name = '.'.join(param_name.split('.')[:-1])
                    if layer_name not in layer_damages:
                        layer_damages[layer_name] = 0
                    layer_damages[layer_name] += damage_val

            # NUMERICAL FIX: Compute total Fisher in torch before converting to Python float
            # This maintains precision for large sums (>2^53 elements)
            total_fisher_tensor = sum(f.sum() for f in fisher_A.values())
            total_fisher = total_fisher_tensor.item() if torch.is_tensor(total_fisher_tensor) else float(total_fisher_tensor)

            # Normalize by total Fisher importance with stability check
            if total_fisher > eps * len(fisher_A):
                normalized_damage = total_damage / total_fisher
            else:
                logger.warning(f"Total Fisher importance very small ({total_fisher:.2e}), setting normalized_damage to 0")
                normalized_damage = 0.0

            return {
                'total_damage': total_damage,
                'normalized_damage': normalized_damage,
                'layer_damages': layer_damages,
                'fisher_norm': total_fisher,
                'n_parameters': len(fisher_A)
            }

        finally:
            # Restore original requires_grad state if we modified it
            if original_requires_grad is not None:
                for name, param in model.named_parameters():
                    param.requires_grad = original_requires_grad[name]

            # CRITICAL FIX: Explicitly delete task_B_grads to free memory
            # For 1.5B models, this dict holds ~2.79GB that may not be freed by GC immediately
            if 'task_B_grads' in locals():
                del task_B_grads

            # Force CUDA cache cleanup to release fragmented memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def compute_fisher_damage_with_asymmetry(
        self,
        model,
        math_batch: Dict[str, torch.Tensor],
        general_batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute bidirectional Fisher-weighted damage between math and general tasks.
        Shows asymmetry in task interference.

        Note: For efficiency, ensure Fisher EMA is pre-computed via update_fisher_ema()
        or use fisher_type='direct' in kwargs for one-shot computation.
        """
        # torch is already imported at module level, no need to reimport

        # ============ DEBUG: Memory diagnostic at function entry ============
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            logger.info("="*80)
            logger.info("🔍 FISHER DAMAGE DEBUG: Function entry")
            logger.info("="*80)
            logger.info(f"GPU Memory at entry: {allocated:.2f} GB / {total:.2f} GB ({100*allocated/total:.1f}%)")
            logger.info(f"Reserved: {reserved:.2f} GB")

            # Count tensors on GPU
            tensor_count = 0
            tensor_memory = 0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        tensor_count += 1
                        tensor_memory += obj.element_size() * obj.nelement()
                except:
                    pass

            logger.info(f"Tracked tensors on GPU: {tensor_count} tensors, {tensor_memory/1e9:.2f} GB")
            logger.info(f"Unaccounted memory: {allocated - tensor_memory/1e9:.2f} GB")

            # Check batch sizes
            math_size = math_batch['input_ids'].shape if 'input_ids' in math_batch else 'unknown'
            gen_size = general_batch['input_ids'].shape if 'input_ids' in general_batch else 'unknown'
            logger.info(f"math_batch shape: {math_size}")
            logger.info(f"general_batch shape: {gen_size}")
            logger.info("="*80)
        # ============ END DEBUG ============

        # Default to EMA Fisher for efficiency
        if 'fisher_type' not in kwargs:
            kwargs['fisher_type'] = 'ema'

        # Check if we need to compute Fisher EMA first
        if kwargs.get('fisher_type') == 'ema':
            # Check if EMA exists for both tasks
            math_fisher = self.get_group_fisher('math', bias_corrected=True)
            general_fisher = self.get_group_fisher('general', bias_corrected=True)

            if not math_fisher or not general_fisher:
                logger.info("Fisher EMA not found, computing fresh Fisher for both tasks")

                # ============ DEBUG: Before Fisher computation ============
                if torch.cuda.is_available():
                    logger.info("🔍 DEBUG: About to compute Fisher for both tasks")
                    logger.info(f"GPU before Fisher: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                # ============ END DEBUG ============

                # Compute Fisher EMA for both tasks if not available
                self.update_fisher_ema(model, math_batch, 'math')

                # ============ DEBUG: After math Fisher ============
                if torch.cuda.is_available():
                    logger.info(f"🔍 DEBUG: After math Fisher: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                # ============ END DEBUG ============

                self.update_fisher_ema(model, general_batch, 'general')

                # ============ DEBUG: After general Fisher ============
                if torch.cuda.is_available():
                    logger.info(f"🔍 DEBUG: After general Fisher: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                # ============ END DEBUG ============

        # ============ DEBUG: Before damage computation ============
        if torch.cuda.is_available():
            logger.info("🔍 DEBUG: About to compute damage math←general")
            logger.info(f"GPU before damage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # ============ END DEBUG ============

        # Compute damage in both directions
        damage_math_from_gen = self.compute_fisher_weighted_damage(
            model=model,
            task_A_batch=math_batch,
            task_B_batch=general_batch,
            task_A_name='math',
            task_B_name='general',
            **kwargs
        )

        # ============ DEBUG: After first damage ============
        if torch.cuda.is_available():
            logger.info(f"🔍 DEBUG: After damage math←general: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # ============ END DEBUG ============

        # Only clear gradients between computations
        model.zero_grad()

        # ============ DEBUG: Before second damage ============
        if torch.cuda.is_available():
            logger.info("🔍 DEBUG: About to compute damage general←math")
            logger.info(f"GPU before damage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # ============ END DEBUG ============

        damage_gen_from_math = self.compute_fisher_weighted_damage(
            model=model,
            task_A_batch=general_batch,
            task_B_batch=math_batch,
            task_A_name='general',
            task_B_name='math',
            **kwargs
        )

        # ============ DEBUG: After second damage ============
        if torch.cuda.is_available():
            logger.info(f"🔍 DEBUG: After damage general←math: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # ============ END DEBUG ============

        # Clear gradients after computation
        model.zero_grad()

        # ============ DEBUG: After final cleanup ============
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"🔍 DEBUG: After cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            logger.info("="*80)
        # ============ END DEBUG ============

        # Compute asymmetry metrics with numerical stability
        # NUMERICAL FIX: Use relative epsilon to handle different damage scales
        damage_math = damage_math_from_gen['normalized_damage']
        damage_gen = damage_gen_from_math['normalized_damage']

        # Relative epsilon based on damage magnitudes
        eps_abs = self.eps_small if hasattr(self, 'eps_small') else 1e-8
        eps_rel = eps_abs * max(abs(damage_math), abs(damage_gen), 1.0)

        # Compute asymmetry ratio with protection against division by zero
        if abs(damage_gen) < eps_rel:
            if abs(damage_math) < eps_rel:
                asymmetry_ratio = 1.0  # Both damages near zero
                logger.info("Both task damages near zero, asymmetry_ratio = 1.0")
            else:
                asymmetry_ratio = float('inf')  # Math damage >> Gen damage
                logger.warning(f"General task damage near zero ({damage_gen:.2e}), asymmetry_ratio = inf")
        else:
            asymmetry_ratio = damage_math / (damage_gen + eps_abs)

        # Add log asymmetry ratio for better interpretability (recommended for paper)
        if asymmetry_ratio > 0:
            log_asymmetry_ratio = np.log(asymmetry_ratio)
        else:
            log_asymmetry_ratio = float('-inf')

        return {
            'damage_math_from_general': damage_math,
            'damage_general_from_math': damage_gen,
            'asymmetry_ratio': asymmetry_ratio,
            'log_asymmetry_ratio': log_asymmetry_ratio,  # Better for visualization
            'more_vulnerable_task': 'math' if asymmetry_ratio > 1 else 'general',
            'damage_math_details': damage_math_from_gen,
            'damage_general_details': damage_gen_from_math
        }

    # ============= UTILITY FUNCTIONS =============

    def _with_labels(self, batch: Dict[str, torch.Tensor], vocab_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Add labels for language modeling, masking padding tokens.
        """
        batch = batch.copy()

        # Validate and clamp token IDs if vocab_size is provided
        if vocab_size is not None and 'input_ids' in batch:
            max_token_id = batch['input_ids'].max().item()
            if max_token_id >= vocab_size:
                logger.warning(f"Token ID {max_token_id} >= vocab_size {vocab_size}. Clamping...")
                batch['input_ids'] = batch['input_ids'].clamp(0, vocab_size - 1)

        # Add labels if not present
        if 'labels' not in batch:
            # For causal LM, labels are input_ids (HuggingFace handles the shift internally)
            batch['labels'] = batch['input_ids'].clone()

            # Mask padding tokens in labels
            if 'attention_mask' in batch:
                batch['labels'] = batch['labels'].masked_fill(
                    batch['attention_mask'] == 0, -100
                )

        return batch

    def _to_model_device(self, model, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to model's device."""
        device = next(model.parameters()).device
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    # ============= OTHER MODULARITY METRICS =============

    def compute_subspace_distance(self, H1, H2, method='cka', k=10):
        """Compute distance between subspaces."""
        H1 = H1.float()
        H2 = H2.float()

        H1 = H1 - H1.mean(0, keepdim=True)
        H2 = H2 - H2.mean(0, keepdim=True)

        if method == 'cka':
            cka = self._linear_cka(H1, H2)
            return 1 - cka

        elif method == 'principal_angles':
            try:
                # FIX: Use torch.linalg.svd instead of deprecated torch.svd
                U1, _, _ = torch.linalg.svd(H1, full_matrices=False)
                U2, _, _ = torch.linalg.svd(H2, full_matrices=False)
                U1 = U1[:, :min(k, U1.shape[1])]
                U2 = U2[:, :min(k, U2.shape[1])]

                M = U1.T @ U2
                _, S, _ = torch.linalg.svd(M, full_matrices=False)
                S = torch.clamp(S, -1, 1)
                angles = torch.acos(S)
                return angles.mean().item()
            except Exception as e:
                logger.warning(f"SVD failed in principal_angles: {e}, falling back to CKA")
                return self.compute_subspace_distance(H1, H2, method='cka', k=k)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _linear_cka(self, X, Y):
        """
        Compute linear CKA between two representations.

        FIXED VERSION: Corrects theoretical formula for ICML submission.

        The correct CKA formula from Kornblith et al. (2019):
        "Similarity of Neural Network Representations Revisited"
        https://arxiv.org/abs/1905.00414

        CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

        Previous implementation had incorrect normalization.
        """
        # Center the data
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        n = X.shape[0]

        # Use double precision for intermediate computation to avoid numerical issues
        X_64 = X.double()
        Y_64 = Y.double()

        # Compute covariance matrices (scaled by n-1 for unbiased estimator)
        XtX = X_64.T @ X_64 / (n - 1)
        YtY = Y_64.T @ Y_64 / (n - 1)
        XtY = X_64.T @ Y_64 / (n - 1)

        # CKA = ||XtY||_F^2 / (||XtX||_F * ||YtY||_F)
        # This is the CORRECT formula
        similarity = torch.trace(XtY @ XtY.T)

        # Compute Frobenius norms (note: not squared under sqrt)
        norm_x = torch.trace(XtX @ XtX)
        norm_y = torch.trace(YtY @ YtY)

        # Use appropriate epsilon for numerical stability
        eps = 1e-5  # Appropriate for float32 after double precision computation

        # CORRECTED: similarity / (sqrt(norm_x) * sqrt(norm_y))
        cka = similarity / (torch.sqrt(norm_x * norm_y) + eps)

        return cka.item()

    def _compute_cka_per_layer(self, model, batch1, batch2, target_layers):
        """Compute CKA for each specified layer between two batches.

        MEMORY-OPTIMIZED: Extracts hidden states immediately and deletes full outputs
        to prevent CUDA OOM on large models (1.5B+ parameters).

        Args:
            model: The model to analyze
            batch1: First batch of inputs
            batch2: Second batch of inputs
            target_layers: List of layer indices to compute CKA for

        Returns:
            Dict mapping layer index to CKA value
        """
        model.eval()

        # Ensure same batch size
        n = min(batch1['input_ids'].size(0), batch2['input_ids'].size(0))
        b1 = self._slice_batch(batch1, torch.arange(n))
        b2 = self._slice_batch(batch2, torch.arange(n))

        # Move to model device
        b1 = self._to_model_device(model, b1)
        b2 = self._to_model_device(model, b2)

        with torch.inference_mode():
            out1 = model(**b1, output_hidden_states=True)
            out2 = model(**b2, output_hidden_states=True)

        # ===== CRITICAL FIX: Extract hidden states immediately =====
        # This allows us to delete the full output objects, freeing ~7GB per call
        hidden_states_1 = []
        hidden_states_2 = []

        for layer_idx in target_layers:
            if layer_idx >= len(out1.hidden_states) or layer_idx >= len(out2.hidden_states):
                logger.warning(f"Skipping layer {layer_idx} - out of bounds")
                hidden_states_1.append(None)
                hidden_states_2.append(None)
            else:
                # Detach and clone to break reference to full output
                # This is necessary to allow garbage collection of out1/out2
                hidden_states_1.append(out1.hidden_states[layer_idx].detach().clone())
                hidden_states_2.append(out2.hidden_states[layer_idx].detach().clone())

        # ===== CRITICAL: Delete output objects immediately =====
        # Frees ~7 GB of GPU memory for 1.5B models (28 layers × 0.25 GB/layer)
        del out1, out2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Now compute CKA on the extracted hidden states
        layer_ckas = {}

        for i, layer_idx in enumerate(target_layers):
            h1 = hidden_states_1[i]
            h2 = hidden_states_2[i]

            if h1 is None or h2 is None:
                continue

            # Get attention masks
            mask1 = b1['attention_mask'].bool().to(h1.device)
            mask2 = b2['attention_mask'].bool().to(h2.device)

            # Pool representations using mean pooling
            denom1 = mask1.sum(1, keepdim=True).clamp_min(1).float()
            denom2 = mask2.sum(1, keepdim=True).clamp_min(1).float()

            pooled1 = (h1 * mask1.unsqueeze(-1)).sum(1) / denom1
            pooled2 = (h2 * mask2.unsqueeze(-1)).sum(1) / denom2

            # Compute CKA for this layer
            layer_ckas[layer_idx] = self._linear_cka(pooled1, pooled2)

            # Free hidden states after CKA computation to minimize peak memory
            del h1, h2, pooled1, pooled2

        # Final cleanup
        del hidden_states_1, hidden_states_2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return layer_ckas

    # ===== NON-FISHER METHODS FROM ORIGINAL =====
    def _take(self, batch: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Random sampling from batch."""
        batch_size = batch['input_ids'].size(0)
        if n >= batch_size:
            return batch

        indices = torch.randperm(batch_size)[:n]
        return self._slice_batch(batch, indices)

    def _slice_batch(self, batch: Dict[str, Any], indices: torch.Tensor) -> Dict[str, Any]:
        """Slice batch by indices, handling different data types."""

        def _slice_any(v, idx):
            if torch.is_tensor(v):
                return v.index_select(0, idx.to(v.device))
            elif isinstance(v, np.ndarray):
                return v[idx.cpu().numpy()]
            elif isinstance(v, list):
                return [v[i.item()] for i in idx]
            else:
                return v

        return {k: _slice_any(v, indices) for k, v in batch.items()}

    def _get_last_n_layer_prefixes(self, model, n: Optional[int] = None) -> List[str]:
        """Get prefixes for last n transformer layers."""
        if n is None:
            n = self.default_last_n_layers if hasattr(self, 'default_last_n_layers') else 6

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            total_layers = len(model.model.layers)
            start_idx = max(0, total_layers - n)
            return [f"model.layers.{i}." for i in range(start_idx, total_layers)]
        return []

    def _robust_svd(self, X: torch.Tensor, compute_uv: bool = True, full_matrices: bool = False):
        """
        Robust SVD computation that handles CUDA convergence issues.

        Args:
            X: Input tensor
            compute_uv: If True, compute U and V matrices (otherwise just S)
            full_matrices: If True, compute full U and V matrices

        Returns:
            If compute_uv is True: (U, S, V)
            If compute_uv is False: S
        """
        import warnings

        # Normalize input to improve numerical stability
        X_norm = X.norm()
        if X_norm > 1e3:
            X = X / (X_norm / 100.0)

        # Add small regularization to improve conditioning
        # Use appropriate epsilon based on dtype
        eps_val = self.eps_tiny if (hasattr(self, 'eps_tiny') and X.dtype == torch.float64) else (self.eps_small if hasattr(self, 'eps_small') else 1e-6)
        eps = eps_val * torch.eye(min(X.shape[-2:]), device=X.device, dtype=X.dtype)
        if X.dim() == 2 and X.shape[0] == X.shape[1]:
            X = X + eps

        try:
            # First try with default settings, suppressing the warning
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*SVD.*failed to converge.*')
                warnings.filterwarnings('ignore', message='.*During SVD computation.*')
                if compute_uv:
                    U, S, V = torch.linalg.svd(X, full_matrices=full_matrices)
                    return U, S, V
                else:
                    S = torch.linalg.svdvals(X)
                    return S
        except:
            # If that fails, try CPU computation
            try:
                X_cpu = X.cpu().to(torch.float64)
                if compute_uv:
                    U, S, V = torch.linalg.svd(X_cpu, full_matrices=full_matrices)
                    device = X.device
                    return U.to(device), S.to(device), V.to(device)
                else:
                    S = torch.linalg.svdvals(X_cpu)
                    return S.to(X.device)
            except:
                # Last resort: return NaN
                if compute_uv:
                    return None, None, None
                else:
                    return torch.tensor([float('nan')], device=X.device)

    # ============= BLOCK-CKA GAP (with return_per_layer support) =============

    def compute_block_cka_gap(self,
                              model,
                              math_batch: Dict[str, torch.Tensor],
                              general_batch: Dict[str, torch.Tensor],
                              target_layers: Optional[List[int]] = None,
                              n_splits: int = 5,
                              return_per_layer: bool = False) -> Dict[str, float]:
        """
        Memory-efficient block-CKA gap computation with CUDA OOM fixes.

        Critical improvements for ICML:
        - Processes splits sequentially with immediate cleanup
        - Deletes hidden states after each CKA computation
        - Uses corrected CKA formula for theoretical accuracy
        - Adds memory monitoring

        Memory requirements (1.5B model):
        - Before: 15.6GB per split × 5 splits = 78GB+ accumulated
        - After: ~4GB peak with proper cleanup

        Args:
            model: Model to analyze
            math_batch: Math task batch
            general_batch: General task batch
            target_layers: Layers to analyze (default: last 6)
            n_splits: Number of random splits for stability
            return_per_layer: Return per-layer gap statistics

        Returns:
            Dictionary with gap statistics
        """
        model.eval()
        device = next(model.parameters()).device

        # Truncate sequences if too long to prevent OOM
        max_seq_length = 512
        def truncate_batch(batch):
            if batch['input_ids'].shape[1] > max_seq_length:
                logger.warning(f"Truncating sequence from {batch['input_ids'].shape[1]} to {max_seq_length} for memory efficiency")
                return {
                    k: v[:, :max_seq_length] if k in ['input_ids', 'attention_mask'] else v
                    for k, v in batch.items()
                }
            return batch

        math_batch = truncate_batch(math_batch)
        general_batch = truncate_batch(general_batch)

        if target_layers is None:
            with torch.inference_mode():
                dummy_batch = self._take(math_batch, 2)
                dummy_batch = self._to_model_device(model, dummy_batch)
                # Filter to only include necessary keys for hidden state extraction
                filtered_dummy = {
                    k: v for k, v in dummy_batch.items()
                    if k in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
                }
                dummy_out = model(**filtered_dummy, output_hidden_states=True)
                hs = dummy_out.hidden_states

                # Skip embedding layer if present (index 0)
                start = 1 if (len(hs) > 2 and hs[0].shape == hs[1].shape) else 0

                # Select last N transformer layers
                n_last = self.default_last_n_layers if hasattr(self, 'default_last_n_layers') else 6
                target_layers = list(range(max(start, len(hs) - n_last), len(hs)))

                # Safety check
                if target_layers and target_layers[-1] >= len(hs):
                    logger.warning(f"Invalid target_layers {target_layers} for hidden_states length {len(hs)}")
                    target_layers = list(range(max(0, len(hs) - 6), len(hs)))

                # CRITICAL: Clean up dummy outputs
                del dummy_out, hs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gaps = []
        within_math_scores = []
        within_gen_scores = []
        between_scores = []

        per_layer_gaps = {f'layer_{i}': [] for i in target_layers} if return_per_layer else None

        # Process each split separately with cleanup
        for split_idx in range(n_splits):
            # Better seed strategy to avoid correlation
            if self.seed is not None:
                split_seed = hash((self.seed, split_idx)) % (2**32)
                gen = torch.Generator().manual_seed(split_seed)
            else:
                gen = None

            math_A, math_B = self._half_split(math_batch, generator=gen)
            gen_A, gen_B = self._half_split(general_batch, generator=gen)

            try:
                if return_per_layer:
                    within_math_pl = self._compute_cka_per_layer(model, math_A, math_B, target_layers)
                    # CRITICAL: Clean up after each computation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    within_gen_pl = self._compute_cka_per_layer(model, gen_A, gen_B, target_layers)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    between_pl = self._compute_cka_per_layer(model, math_A, gen_A, target_layers)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    for layer_idx in target_layers:
                        if layer_idx in within_math_pl and layer_idx in within_gen_pl and layer_idx in between_pl:
                            layer_gap = ((within_math_pl[layer_idx] + within_gen_pl[layer_idx]) / 2
                                       - between_pl[layer_idx])
                            per_layer_gaps[f'layer_{layer_idx}'].append(layer_gap)

                    within_math = np.mean(list(within_math_pl.values()))
                    within_gen = np.mean(list(within_gen_pl.values()))
                    between = np.mean(list(between_pl.values()))
                else:
                    within_math = self._compute_cka(model, math_A, math_B, target_layers)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    within_gen = self._compute_cka(model, gen_A, gen_B, target_layers)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    between = self._compute_cka(model, math_A, gen_A, target_layers)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                gap = (within_math + within_gen) / 2 - between

                gaps.append(gap)
                within_math_scores.append(within_math)
                within_gen_scores.append(within_gen)
                between_scores.append(between)

            finally:
                # Force cleanup after each split
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle NaN values in aggregation
        def safe_mean(values):
            valid = [v for v in values if not np.isnan(v)]
            return float(np.mean(valid)) if valid else float('nan')

        def safe_std(values):
            valid = [v for v in values if not np.isnan(v)]
            return float(np.std(valid)) if valid else float('nan')

        results = {
            'block_cka_gap': safe_mean(gaps),
            'block_cka_gap_std': safe_std(gaps),
            'within_math_cka': safe_mean(within_math_scores),
            'within_gen_cka': safe_mean(within_gen_scores),
            'between_task_cka': safe_mean(between_scores),
            'n_splits': n_splits
        }

        if return_per_layer and per_layer_gaps:
            results['per_layer_gaps'] = {}
            for layer, values in per_layer_gaps.items():
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    results['per_layer_gaps'][layer] = {
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values))
                    }
                else:
                    results['per_layer_gaps'][layer] = {
                        'mean': float('nan'),
                        'std': float('nan')
                    }

        return results

    def _half_split(self, batch: Dict[str, Any], generator=None) -> Tuple[Dict, Dict]:
        """Split batch in half."""
        n = batch['input_ids'].size(0)
        if n < 2:
            raise ValueError(f"Need at least 2 examples for half-split, got {n}")

        half = n // 2
        n_even = 2 * half

        device = batch['input_ids'].device

        # Fix: Handle generator device properly
        if generator is not None:
            # Generator must be on CPU for randperm
            perm = torch.randperm(n_even, generator=generator)
            if device != torch.device('cpu'):
                perm = perm.to(device)
        else:
            perm = torch.randperm(n_even, device=device)

        idxA, idxB = perm[:half], perm[half:n_even]

        A = self._slice_batch(batch, idxA)
        B = self._slice_batch(batch, idxB)

        return A, B

    def _compute_cka(self, model, batch1, batch2, target_layers):
        """Compute average CKA across layers.

        MEMORY-OPTIMIZED: Extracts hidden states immediately and deletes full outputs
        to prevent CUDA OOM on large models (1.5B+ parameters).
        """
        model.eval()

        n = min(batch1['input_ids'].size(0), batch2['input_ids'].size(0))
        b1 = self._slice_batch(batch1, torch.arange(n))
        b2 = self._slice_batch(batch2, torch.arange(n))

        b1 = self._to_model_device(model, b1)
        b2 = self._to_model_device(model, b2)

        with torch.inference_mode():
            out1 = model(**b1, output_hidden_states=True)
            out2 = model(**b2, output_hidden_states=True)

        # ===== CRITICAL FIX: Extract hidden states immediately =====
        # This allows us to delete the full output objects, freeing ~7GB per call
        hidden_states_1 = []
        hidden_states_2 = []

        for layer_idx in target_layers:
            # Safety check: ensure layer_idx is within bounds
            if layer_idx >= len(out1.hidden_states) or layer_idx >= len(out2.hidden_states):
                logger.warning(f"Skipping layer {layer_idx} - out of bounds (max: {min(len(out1.hidden_states), len(out2.hidden_states))-1})")
                hidden_states_1.append(None)
                hidden_states_2.append(None)
            else:
                # Detach and clone to break reference to full output
                hidden_states_1.append(out1.hidden_states[layer_idx].detach().clone())
                hidden_states_2.append(out2.hidden_states[layer_idx].detach().clone())

        # ===== CRITICAL: Delete output objects immediately =====
        # Frees ~7 GB of GPU memory for 1.5B models
        del out1, out2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Now compute CKA on the extracted hidden states
        layer_ckas = []

        for i, layer_idx in enumerate(target_layers):
            h1 = hidden_states_1[i]
            h2 = hidden_states_2[i]

            if h1 is None or h2 is None:
                continue

            mask1 = b1['attention_mask'].bool().to(h1.device)
            mask2 = b2['attention_mask'].bool().to(h2.device)

            denom1 = mask1.sum(1, keepdim=True).clamp_min(1).float()
            denom2 = mask2.sum(1, keepdim=True).clamp_min(1).float()

            pooled1 = (h1 * mask1.unsqueeze(-1)).sum(1) / denom1
            pooled2 = (h2 * mask2.unsqueeze(-1)).sum(1) / denom2

            cka = self._linear_cka(pooled1, pooled2)
            # Only skip NaN values, keep zeros (they're valid CKA values)
            if not np.isnan(cka):
                layer_ckas.append(cka)

            # Free hidden states after CKA computation
            del h1, h2, pooled1, pooled2

        # Final cleanup
        del hidden_states_1, hidden_states_2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return NaN if no valid layers, otherwise mean of valid layers
        if len(layer_ckas) == 0:
            return float('nan')
        return float(np.mean(layer_ckas))

    def compute_linear_cka_per_layer(self, model, math_batch, general_batch, target_layers=None):
        """Compute layer-wise CKA between math and general representations."""
        model.eval()

        n = min(math_batch['input_ids'].size(0), general_batch['input_ids'].size(0))
        mb = self._slice_batch(math_batch, torch.arange(n))
        gb = self._slice_batch(general_batch, torch.arange(n))

        mb = self._to_model_device(model, mb)
        gb = self._to_model_device(model, gb)

        with torch.inference_mode():
            m_out = model(**mb, output_hidden_states=True)
            g_out = model(**gb, output_hidden_states=True)

        if target_layers is None:
            n_layers = len(m_out.hidden_states)
            n_last = self.default_last_n_layers if hasattr(self, 'default_last_n_layers') else 6
            target_layers = list(range(max(0, n_layers - n_last), n_layers))

        layer_ckas = {}

        for layer_idx in target_layers:
            # Safety check: ensure layer_idx is within bounds
            if layer_idx >= len(m_out.hidden_states) or layer_idx >= len(g_out.hidden_states):
                logger.warning(f"Skipping layer {layer_idx} - out of bounds (max: {min(len(m_out.hidden_states), len(g_out.hidden_states))-1})")
                continue
            m_h = m_out.hidden_states[layer_idx]
            g_h = g_out.hidden_states[layer_idx]

            m_mask = mb['attention_mask'].bool().to(m_h.device)
            g_mask = gb['attention_mask'].bool().to(g_h.device)

            m_denom = m_mask.sum(1, keepdim=True).clamp_min(1).float()
            g_denom = g_mask.sum(1, keepdim=True).clamp_min(1).float()

            m_pooled = (m_h * m_mask.unsqueeze(-1)).sum(1) / m_denom
            g_pooled = (g_h * g_mask.unsqueeze(-1)).sum(1) / g_denom

            layer_ckas[f'layer_{layer_idx}'] = self._linear_cka(m_pooled, g_pooled)

        vals = list(layer_ckas.values())

        # FIX: Handle empty values case to prevent crashes
        if not vals:
            return {
                'per_layer_cka': layer_ckas,
                'mean_cka': float('nan'),
                'median_cka': float('nan'),
                'min_cka': float('nan'),
                'max_cka': float('nan'),
                'modularity_score': float('nan')
            }

        return {
            'per_layer_cka': layer_ckas,
            'mean_cka': float(np.mean(vals)),
            'median_cka': float(np.median(vals)),
            'min_cka': float(np.min(vals)),
            'max_cka': float(np.max(vals)),
            'modularity_score': float(1 - np.median(vals))
        }

    # ============= EFFECTIVE RANK =============

    def compute_effective_rank(self, model, test_batch, target_layers=None,
                               n_positions=64, use_double=True):
        """Compute effective rank with improved sampling."""
        model.eval()

        test_batch = self._to_model_device(model, test_batch)

        # Filter batch to only include necessary keys for hidden state extraction
        # Remove 'labels' as they can cause CUDA assertions with some models
        filtered_batch = {
            k: v for k, v in test_batch.items()
            if k in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
        }

        # Validate input_ids are within vocabulary range if possible
        if 'input_ids' in filtered_batch:
            vocab_size = self._safe_get_vocab_size(model)
            if vocab_size is not None:
                max_id = filtered_batch['input_ids'].max().item()
                if max_id >= vocab_size:
                    logger.warning(f"Input contains token ID {max_id} >= vocab_size {vocab_size}")

        try:
            with torch.inference_mode():
                # Clear CUDA cache for large models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                out = model(**filtered_batch, output_hidden_states=True)
                hidden_states = out.hidden_states

                # CRITICAL FIX: Delete output object immediately after extracting hidden states
                # For Qwen2.5-Math-1.5B: frees 2.48 GB (logits: 32×256×151936 tokens × 2 bytes)
                # Without this, memory accumulates across metric calls causing OOM
                del out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            logger.error(f"Batch keys: {list(filtered_batch.keys())}")
            if 'input_ids' in filtered_batch:
                logger.error(f"Input shape: {filtered_batch['input_ids'].shape}")
                logger.error(f"Max token ID: {filtered_batch['input_ids'].max().item()}")
            raise

        if target_layers is None:
            n_layers = len(hidden_states)
            n_last = self.default_last_n_layers if hasattr(self, 'default_last_n_layers') else 6
            target_layers = list(range(max(0, n_layers - n_last), n_layers))

        ranks = {}
        participation_ratios = {}

        for layer_idx in target_layers:
            # Safety check: ensure layer_idx is within bounds
            if layer_idx >= len(hidden_states):
                logger.warning(f"Skipping layer {layer_idx} - out of bounds (max: {len(hidden_states)-1})")
                continue
            acts = hidden_states[layer_idx]
            B, T, D = acts.shape

            if T <= n_positions:
                X = acts.reshape(-1, D)
            else:
                idx = torch.linspace(0, T - 1, steps=n_positions, device=acts.device).long()
                X = acts[:, idx, :].reshape(-1, D)

            if use_double:
                X = X.double()
            else:
                X = X.float()

            X = X - X.mean(0, keepdim=True)

            try:
                # Always compute SVD in float64 for numerical stability
                X_64 = X.to(torch.float64) if use_double else X
                s = self._robust_svd(X_64, compute_uv=False)
                if s is not None and not torch.isnan(s).any():
                    s_sum = s.sum()
                    s_sq_sum = s.pow(2).sum()
                    # Use appropriate epsilon for numerical stability
                    eps = self.eps_tiny if (hasattr(self, 'eps_tiny') and X_64.dtype == torch.float64) else (self.eps_small if hasattr(self, 'eps_small') else 1e-6)
                    erank = (s_sum ** 2) / (s_sq_sum + eps)
                    ranks[f'layer_{layer_idx}'] = float(erank)

                    # FIX: Compute actual participation ratio (different from effective rank)
                    # Participation ratio = effective rank / dimension
                    participation_ratio = erank / D
                    participation_ratios[f'layer_{layer_idx}'] = float(participation_ratio)
                else:
                    ranks[f'layer_{layer_idx}'] = np.nan
                    participation_ratios[f'layer_{layer_idx}'] = np.nan
            except Exception as e:
                # ICML FIX: Log specific exception instead of silent failure
                logger.warning(f"SVD failed for layer {layer_idx}: {type(e).__name__}: {e}")
                ranks[f'layer_{layer_idx}'] = np.nan
                participation_ratios[f'layer_{layer_idx}'] = np.nan
            finally:
                # CRITICAL FIX: Free intermediate tensors after each layer
                # Prevents accumulation across 6-28 layers (25-50 MB per layer)
                del acts, X, X_64
                if 's' in locals():
                    del s

        # CRITICAL FIX: Free hidden_states tuple after processing all layers
        # For 28-layer Qwen2.5-Math-1.5B: frees 730 MB (29 layers × 32×256×1536 × 2 bytes)
        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        valid_ranks = [v for v in ranks.values() if not np.isnan(v)]
        valid_pr = [v for v in participation_ratios.values() if not np.isnan(v)]

        if len(valid_ranks) == 0:
            return {
                'per_layer_ranks': ranks,
                'per_layer_participation': participation_ratios,
                'mean_rank': np.nan,
                'median_rank': np.nan,
                'mean_participation_ratio': np.nan,
                'median_participation_ratio': np.nan,
                'compression_score': np.nan
            }

        return {
            'per_layer_ranks': ranks,
            'per_layer_participation': participation_ratios,
            'mean_rank': float(np.mean(valid_ranks)),
            'median_rank': float(np.median(valid_ranks)),
            'min_rank': float(np.min(valid_ranks)),
            'max_rank': float(np.max(valid_ranks)),
            'mean_participation_ratio': float(np.mean(valid_pr)),
            'median_participation_ratio': float(np.median(valid_pr)),
            'compression_score': float(1.0 / (np.median(valid_ranks) + 1))
        }

    # ============= FULL EFFECTIVE RANK (with n_positions) =============

    def compute_full_effective_rank(self, model, test_batch, n_positions=64, use_double=True, layer_sampling='all'):
        """Compute effective rank from MLP activations.

        Args:
            model: The model to analyze
            test_batch: Input batch for analysis
            n_positions: Number of positions to sample for efficiency
            use_double: Whether to use double precision for SVD
            layer_sampling: Strategy for sampling layers:
                - 'all': Analyze all layers (ICML default - comprehensive)
                - 'auto': Smart sampling based on model size
                - 'sparse': Sample every 3rd layer
                - 'last_6': Only last 6 layers (legacy behavior)
                - int: Sample every n-th layer

        Note: Changed default from 'auto' to 'all' for ICML submission.
        Memory usage is acceptable after fixing memory leaks (~21 GB peak for 28-layer 1.5B model).
        """
        model.eval()

        activations = {}
        handles = []

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # CRITICAL FIX: Use .detach().clone() to break reference to original output
                # .detach() alone shares storage, preventing the full model output from being freed
                # This fix prevents ~70GB memory leak on H100
                activations[name] = output.detach().clone()

            return hook

        # Improved model architecture detection
        all_layers = self._detect_model_layers(model)

        if all_layers is None:
            # Model structure not recognized, return NaN gracefully
            logger.warning(f"Model structure not recognized for effective rank. Model type: {type(model).__name__}")
            return {
                'per_layer_metrics': [],
                'avg_effective_rank': np.nan,
                'avg_participation_ratio': np.nan,
                'std_effective_rank': np.nan,
                'std_participation_ratio': np.nan,
            }

        n_total = len(all_layers)

        # Determine which layers to analyze based on sampling strategy
        if layer_sampling == 'all':
            layers_to_analyze = list(range(n_total))
            logger.info(f"Analyzing all {n_total} layers (memory intensive)")
        elif layer_sampling == 'auto':
            # Smart sampling based on model size
            if n_total <= 12:
                layers_to_analyze = list(range(n_total))
                logger.info(f"Small model ({n_total} layers): analyzing all")
            elif n_total <= 24:
                layers_to_analyze = list(range(0, n_total, 2))
                logger.info(f"Medium model ({n_total} layers): every 2nd layer ({len(layers_to_analyze)} layers)")
            elif n_total <= 48:
                # Strategic sampling for large models
                early = list(range(0, n_total//4, 2))
                middle = list(range(n_total//4, 3*n_total//4, 3))
                late = list(range(3*n_total//4, n_total, 2))
                layers_to_analyze = sorted(set(early + middle + late))
                logger.info(f"Large model ({n_total} layers): strategic sampling ({len(layers_to_analyze)} layers)")
            else:
                # Sample key positions for huge models
                n_samples = min(16, n_total // 3)
                step = max(1, (n_total - 2) // (n_samples - 2)) if n_samples > 2 else 1
                layers_to_analyze = [0] + list(range(step, n_total-1, step)) + [n_total-1]
                layers_to_analyze = sorted(set(layers_to_analyze))
                logger.info(f"Huge model ({n_total} layers): key positions ({len(layers_to_analyze)} layers)")
        elif layer_sampling == 'sparse':
            layers_to_analyze = list(range(0, n_total, 3))
            logger.info(f"Sparse sampling: every 3rd layer ({len(layers_to_analyze)} layers)")
        elif layer_sampling == 'last_6':
            n_last = self.default_last_n_layers if hasattr(self, 'default_last_n_layers') else 6
            start_idx = max(0, n_total - n_last)
            layers_to_analyze = list(range(start_idx, n_total))
            logger.info(f"Legacy mode: last {len(layers_to_analyze)} layers")
        elif isinstance(layer_sampling, int) and layer_sampling > 0:
            layers_to_analyze = list(range(0, n_total, layer_sampling))
            logger.info(f"Custom: every {layer_sampling} layers ({len(layers_to_analyze)} layers)")
        else:
            # Default to auto
            return self.compute_full_effective_rank(model, test_batch, n_positions, use_double, 'auto')

        # Register hooks for selected layers
        for layer_idx in layers_to_analyze:
            layer = all_layers[layer_idx]
            # Try different MLP module names
            mlp_module = None
            if hasattr(layer, 'mlp'):
                mlp_module = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp_module = layer.feed_forward
            elif hasattr(layer, 'intermediate'):
                mlp_module = layer.intermediate
            elif hasattr(layer, 'fc1'):
                # Some models have fc1/fc2 structure
                mlp_module = layer

            if mlp_module is not None:
                handle = mlp_module.register_forward_hook(hook_fn(f'layer_{layer_idx}'))
                handles.append(handle)

        test_batch = self._to_model_device(model, test_batch)

        # Validate and clamp token IDs to prevent CUDA errors
        if 'input_ids' in test_batch:
            vocab_size = self._safe_get_vocab_size(model)
            if vocab_size is not None:
                max_token_id = test_batch['input_ids'].max().item()
                if max_token_id >= vocab_size:
                    logger.warning(f"Token IDs exceed vocab size ({max_token_id} >= {vocab_size}). Clamping.")
                    test_batch['input_ids'] = torch.clamp(test_batch['input_ids'], 0, vocab_size - 1)

        # Filter batch to only include necessary keys for hidden state extraction
        # Remove 'labels' as they can cause CUDA assertions with some models
        filtered_batch = {
            k: v for k, v in test_batch.items()
            if k in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
        }

        # Validate batch dimensions
        if 'attention_mask' in filtered_batch and 'input_ids' in filtered_batch:
            if filtered_batch['attention_mask'].shape != filtered_batch['input_ids'].shape:
                filtered_batch['attention_mask'] = torch.ones_like(filtered_batch['input_ids'])

        try:
            with torch.inference_mode():
                # Clear CUDA cache for large models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # CRITICAL FIX: Store and explicitly delete output to free memory
                # Using _ = model(...) keeps a reference, preventing garbage collection
                outputs = model(**filtered_batch)
                del outputs  # Free immediately after hooks capture activations
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                logger.error(f"CUDA error in compute_full_effective_rank: {e}")
                logger.error("This is likely due to tokenizer/model mismatch or invalid token IDs")
                # Return NaN results instead of crashing
                for handle in handles:
                    handle.remove()
                return {
                    'per_layer_metrics': [],
                    'avg_effective_rank': np.nan,
                    'avg_participation_ratio': np.nan,
                    'std_effective_rank': np.nan,
                    'std_participation_ratio': np.nan,
                }
            else:
                raise

        layer_metrics = []

        for name, acts in activations.items():
            B, T, D = acts.shape

            if T <= n_positions:
                acts_2d = acts.reshape(-1, D)
            else:
                idx = torch.linspace(0, T - 1, steps=n_positions, device=acts.device).long()
                acts_sampled = acts[:, idx, :]
                acts_2d = acts_sampled.reshape(-1, D)

            if use_double:
                acts_2d = acts_2d.double()
            else:
                acts_2d = acts_2d.float()

            acts_2d = acts_2d - acts_2d.mean(0, keepdim=True)

            try:
                S = self._robust_svd(acts_2d, compute_uv=False)
                if S is not None and not torch.isnan(S).any():
                    S_sum = S.sum()
                    S_sq_sum = (S ** 2).sum()
                    # Use appropriate epsilon for numerical stability
                    eps = self.eps_tiny if (hasattr(self, 'eps_tiny') and acts_2d.dtype == torch.float64) else (self.eps_small if hasattr(self, 'eps_small') else 1e-6)
                    eff_rank = (S_sum ** 2) / (S_sq_sum + eps)

                    # FIX: Compute actual participation ratio
                    participation = eff_rank / D

                    layer_metrics.append({
                        'layer': name,
                        'effective_rank': float(eff_rank),
                        'participation_ratio': float(participation),
                        'dimension': D
                    })
                else:
                    layer_metrics.append({
                        'layer': name,
                        'effective_rank': np.nan,
                        'participation_ratio': np.nan,
                        'dimension': D
                    })
            except Exception as e:
                logger.warning(f"SVD failed for {name}: {e}")
                layer_metrics.append({
                    'layer': name,
                    'effective_rank': np.nan,
                    'participation_ratio': np.nan,
                    'dimension': D
                })

        # CRITICAL FIX: Remove hooks and clear activations dict to free memory
        for handle in handles:
            handle.remove()

        # CRITICAL FIX: Explicitly clear activations dictionary
        # For 28-layer model, this frees ~7-15 GB of GPU memory
        activations.clear()
        del activations

        valid_ranks = [m['effective_rank'] for m in layer_metrics
                       if not np.isnan(m['effective_rank'])]
        valid_pr = [m['participation_ratio'] for m in layer_metrics
                    if not np.isnan(m['participation_ratio'])]

        # CRITICAL FIX: Final GPU memory cleanup
        # Ensures fragmented memory is released back to PyTorch allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return {
            'per_layer_metrics': layer_metrics,
            'avg_effective_rank': float(np.mean(valid_ranks)) if valid_ranks else np.nan,
            'avg_participation_ratio': float(np.mean(valid_pr)) if valid_pr else np.nan,
            'std_effective_rank': float(np.std(valid_ranks)) if valid_ranks else np.nan,
            'std_participation_ratio': float(np.std(valid_pr)) if valid_pr else np.nan,
        }

    # ============= GRADIENT CONFLICT (with eval_mode and subsample_ratio) =============

    # Gradient conflict functions moved to GradientAnalysis.py

    def compute_sam_sharpness(self, model, batch, epsilon=0.01):
        """
        SAM-style sharpness metric (Foret et al., 2021) - FIXED VERSION.

        CRITICAL FIXES APPLIED (2025-09-30):
        ⚡ Fixed parameter restoration bug (was re-computing gradients after perturbation - WRONG!)
        ⚡ Added batch validation (catches None/empty batches)
        ⚡ Added loss=None check
        📝 Improved logging for debugging

        Memory requirements for 1.5B parameter model:
        - Base: 6GB (model parameters)
        - Peak: ~12-15GB (with gradient checkpointing)
        - Recommended batch_size: 8 for H100, 4 for A100

        Args:
            model: Model to analyze
            batch: Input batch (must have 'input_ids')
            epsilon: Perturbation radius (ρ in paper), default 0.01

        Returns:
            Sharpness value (relative loss difference) OR dict with 'error' key if batch invalid

        Reproducibility: Results depend on batch size/seq length. Use consistent config for comparison.
        """
        # ============================================================================
        # CRITICAL VALIDATION - Catches instant completion bug
        # ============================================================================
        if batch is None:
            logger.error("compute_sam_sharpness: batch is None!")
            return {'error': 'Batch is None'}

        if not batch or 'input_ids' not in batch:
            logger.error(f"compute_sam_sharpness: invalid batch! Keys: {list(batch.keys()) if batch else 'None'}")
            return {'error': 'Batch invalid or missing input_ids'}

        batch_size = batch['input_ids'].shape[0]
        if batch_size == 0:
            logger.error("compute_sam_sharpness: batch has size 0!")
            return {'error': 'Batch size is 0'}

        if batch_size < 4:
            logger.warning(f"⚠️  SAM: Batch size {batch_size} very small - results will be noisy. Recommended: ≥8")

        # Save original states
        was_training = model.training
        original_grad_enabled = torch.is_grad_enabled()

        # Check and fix gradient coverage
        params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        grad_coverage = params_with_grad / total_params if total_params > 0 else 0

        original_grad_states = {}
        if grad_coverage < 0.9:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in model.parameters())
            logger.warning(
                f"⚠️ SAM: Only {params_with_grad}/{total_params} parameters "
                f"({grad_coverage*100:.2f}%) have requires_grad=True. "
                f"That's {param_count:,}/{total_count:,} values. "
                f"Enabling gradients for ALL parameters."
            )

            # Store original states and enable all gradients
            for name, param in model.named_parameters():
                original_grad_states[name] = param.requires_grad
                param.requires_grad_(True)

        # Use eval mode for deterministic computation (no dropout)
        model.eval()

        # Enable gradient checkpointing if available and not already enabled
        checkpointing_was_enabled = False
        if hasattr(model, 'gradient_checkpointing') and not model.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                checkpointing_was_enabled = True

        batch = self._to_model_device(model, batch)
        device = next(model.parameters()).device

        try:
            # Clear any existing gradients and cache
            model.zero_grad()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Prepare batch with labels
            vocab_size = self._safe_get_vocab_size(model)
            labeled_batch = self._with_labels(batch, vocab_size=vocab_size)

            # ==== PHASE 1: Compute original loss and gradients ====
            with torch.enable_grad():
                outputs = model(**labeled_batch)
                loss_orig = outputs.loss

                # CRITICAL FIX: Check loss is not None BEFORE checking requires_grad
                if loss_orig is None:
                    logger.error(
                        f"compute_sam_sharpness: model returned loss=None! "
                        f"Model type: {type(model).__name__}. "
                        f"This model may not compute loss internally. Ensure labels are present."
                    )
                    return {'error': 'Model returned loss=None', 'model_type': type(model).__name__}

                if not loss_orig.requires_grad:
                    logger.error(
                        f"compute_sam_sharpness: loss.requires_grad=False even after enabling parameter grads! "
                        f"loss={loss_orig.item():.6f}, params_with_grad={sum(1 for p in model.parameters() if p.requires_grad)}"
                    )
                    return {'error': 'Loss does not require gradients', 'loss_value': loss_orig.item()}

                # Backward pass for gradients
                loss_orig.backward()

                # Store original loss value immediately
                loss_orig_val = loss_orig.item()
                logger.debug(f"SAM: Original loss = {loss_orig_val:.6f}")

            # Clear outputs to free memory
            del outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # ==== PHASE 2: Compute gradient norm efficiently ====
            grad_norm_sq = 0.0
            n_params_with_grad = 0

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        # Use in-place operations to avoid creating new tensors
                        grad_norm_sq += (p.grad.data ** 2).sum().item()
                        n_params_with_grad += p.numel()

            grad_norm = np.sqrt(grad_norm_sq)
            logger.debug(f"SAM: Gradient norm = {grad_norm:.6e}, params={n_params_with_grad:,}")

            # Check for degenerate case
            if grad_norm < 1e-8:  # More conservative than 1e-6
                logger.warning(f"Gradient norm too small ({grad_norm:.2e}), returning 0")
                model.zero_grad()
                return 0.0

            # ==== PHASE 3: Apply perturbation (θ' = θ + ε * g / ||g||) ====
            scale = epsilon / grad_norm
            logger.debug(f"SAM: Perturbation scale = {scale:.6e}")

            with torch.no_grad():
                # Apply perturbation in-place
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(p.grad.data, alpha=scale)

            # ⚡ CRITICAL FIX: DO NOT clear gradients here!
            # We need the original gradients to restore parameters correctly.
            # Previous code cleared them and recomputed at perturbed position - WRONG!

            # ==== PHASE 4: Compute perturbed loss L(θ') ====
            with torch.no_grad():
                outputs_pert = model(**labeled_batch)

                # Check perturbed loss is not None
                if outputs_pert.loss is None:
                    logger.error("compute_sam_sharpness: perturbed forward pass returned loss=None!")
                    # Restore parameters using original gradients before returning error
                    for p in model.parameters():
                        if p.grad is not None:
                            p.data.add_(p.grad.data, alpha=-scale)
                    model.zero_grad()
                    return {'error': 'Perturbed loss is None'}

                loss_pert_val = outputs_pert.loss.item()
                logger.debug(f"SAM: Perturbed loss = {loss_pert_val:.6f}")
                del outputs_pert

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # ==== PHASE 5: Restore original parameters (θ = θ' - ε * g / ||g||) ====
            # ⚡ CRITICAL FIX: Use ORIGINAL gradients (not recomputed ones!)
            # The previous code recomputed gradients at θ', which gives WRONG restoration direction.
            # Correct: θ = θ' - (ε/||g||) * g  where g is gradient at original position θ
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        # Restore using SAME gradient that was used for perturbation
                        p.data.add_(p.grad.data, alpha=-scale)

            logger.debug("SAM: Parameters restored to original values")

            # ==== PHASE 6: Compute sharpness with numerical stability ====
            # Use relative difference for better numerical stability
            if abs(loss_orig_val) > 1e-10:
                # Relative sharpness (scale-invariant)
                sharpness = (loss_pert_val - loss_orig_val) / abs(loss_orig_val)
            else:
                # Absolute sharpness for near-zero loss (relative would overflow)
                sharpness = loss_pert_val - loss_orig_val
                logger.warning(f"SAM: Loss magnitude very small ({loss_orig_val:.2e}), using absolute sharpness")

            logger.info(
                f"SAM: Sharpness = {sharpness:.6f} (relative), "
                f"loss_orig={loss_orig_val:.4f}, loss_pert={loss_pert_val:.4f}, "
                f"batch_size={batch_size}"
            )

            return float(sharpness)

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM in SAM computation: {e}")
            if device.type == 'cuda':
                # Log memory status
                allocated = torch.cuda.memory_allocated(device) / 1e9
                reserved = torch.cuda.memory_reserved(device) / 1e9
                total = torch.cuda.get_device_properties(device).total_memory / 1e9
                logger.error(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            raise

        finally:
            # Comprehensive cleanup
            model.zero_grad()
            model.train(was_training)

            # Restore gradient states
            if original_grad_states:
                for name, param in model.named_parameters():
                    if name in original_grad_states:
                        param.requires_grad_(original_grad_states[name])

            # Disable gradient checkpointing if we enabled it
            if checkpointing_was_enabled and hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()

            # Final cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Restore grad enabled state
            torch.set_grad_enabled(original_grad_enabled)

    # ============= WEIGHT/FUNCTION SPACE DISTANCES =============

    def compute_weight_space_distance(self, model1, model2, normalize=True, layer_pattern='layers'):
        """Normalized weight space distance."""
        distances = []

        # Get parameter lists
        params1 = list(model1.named_parameters())
        params2 = list(model2.named_parameters())

        # Check if models have same architecture
        if len(params1) != len(params2):
            # Different architectures - return NaN
            return np.nan

        for (n1, p1), (n2, p2) in zip(params1, params2):
            if n1 == n2 and layer_pattern in n1:
                # Check shape compatibility
                if p1.shape != p2.shape:
                    # Shape mismatch - skip or return NaN
                    continue

                p2_aligned = p2.to(p1.device, dtype=p1.dtype)
                diff = (p1 - p2_aligned).norm(2)

                if normalize:
                    base_norm = p1.norm(2)
                    if base_norm > 1e-12:
                        diff = diff / base_norm

                distances.append(diff.item())

        if len(distances) == 0:
            return np.nan

        return np.sqrt(np.mean(np.square(distances)))

    def compute_function_space_distance(self, model1, model2, probe_batch, metric='jsd'):
        """Function space distance using output distributions."""
        model1.eval()
        model2.eval()

        # FIX: Create separate batches for each model to handle different devices
        probe_batch1 = self._to_model_device(model1, probe_batch)
        probe_batch2 = self._to_model_device(model2, probe_batch)

        with torch.no_grad():
            logits1 = model1(**probe_batch1).logits
            logits2_raw = model2(**probe_batch2).logits
            # Move logits2 to same device as logits1 for computation
            logits2 = logits2_raw.to(logits1.device)

        if metric == 'jsd':
            p = F.softmax(logits1, dim=-1)
            q = F.softmax(logits2, dim=-1)
            m = 0.5 * (p + q)

            kl_pm = F.kl_div(m.log(), p, reduction='none').sum(-1)
            kl_qm = F.kl_div(m.log(), q, reduction='none').sum(-1)
            jsd = 0.5 * (kl_pm + kl_qm)

            mask = probe_batch1.get('attention_mask')
            if mask is not None:
                mask = mask.float()
            else:
                mask = torch.ones_like(jsd)
            return ((jsd * mask).sum() / (mask.sum() + self.eps_small)).item()

        elif metric == 'kl':
            return self._compute_kl_divergence(logits1, logits2,
                                               probe_batch1.get('attention_mask'))

    def _compute_kl_divergence(self, logits_p, logits_q, attention_mask):
        """KL(p||q) averaged over non-pad tokens."""
        p = logits_p.log_softmax(-1)
        q = logits_q.log_softmax(-1)
        kl_tok = (p.exp() * (p - q)).sum(-1)
        if attention_mask is not None:
            m = attention_mask.float()
        else:
            m = torch.ones_like(kl_tok)
        eps = self.eps_small if hasattr(self, 'eps_small') else 1e-6
        kl_avg = (kl_tok * m).sum() / (m.sum() + eps)
        return kl_avg.item()

    # ============= ELASTICITY =============

    def compute_elasticity(self, model_base, model_crisis, model_recovery, probe_batch):
        """Three-point elasticity metric."""
        model_base.eval()
        model_crisis.eval()
        model_recovery.eval()

        probe_batch = self._to_model_device(model_base, probe_batch)

        with torch.no_grad():
            out_base = model_base(**probe_batch).logits
            out_crisis = model_crisis(**probe_batch).logits
            out_recovery = model_recovery(**probe_batch).logits

        kl_crisis = self._compute_kl_divergence(out_crisis, out_base,
                                                probe_batch.get('attention_mask'))
        kl_recovery = self._compute_kl_divergence(out_recovery, out_base,
                                                  probe_batch.get('attention_mask'))

        S_crisis = np.exp(-kl_crisis)
        S_recovery = np.exp(-kl_recovery)

        if (1 - S_crisis) < self.eps_small:
            elasticity = 0.0
        else:
            elasticity = (S_recovery - S_crisis) / (1 - S_crisis)

        return {
            'elasticity': float(elasticity),
            'similarity_crisis_base': float(S_crisis),
            'similarity_recovery_base': float(S_recovery),
            'kl_crisis_base': float(kl_crisis),
            'kl_recovery_base': float(kl_recovery)
        }