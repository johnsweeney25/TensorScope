#!/usr/bin/env python3
"""
Representation Analysis Metrics
================================

Metrics for analyzing geometric and reconstruction properties of neural network
representations. These complement information-theoretic metrics by measuring
linear relationships, similarities, and reconstruction capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import warnings
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepresentationAnalysisMetrics:
    """
    Metrics for analyzing neural network representations focusing on
    geometric properties and linear reconstruction capabilities.

    These metrics answer questions like:
    - How well can layer j be linearly reconstructed from layer i?
    - How similar are representations between layers (CKA)?
    - What is the effective dimensionality of representations?
    """

    def __init__(self):
        """Initialize the representation analysis metrics."""
        self.cache = {}

    def compute_layer_linear_reconstruction(
        self,
        model,
        train_batch: Dict[str, torch.Tensor],
        test_batch: Optional[Dict[str, torch.Tensor]] = None,
        val_batch: Optional[Dict[str, torch.Tensor]] = None,
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_samples: int = 2000,
        max_dim: Optional[int] = None,  # Only reduce if needed
        alpha_grid: Optional[torch.Tensor] = None,
        stratified_sampling: bool = True,
        compute_cka: bool = True,
        compute_effective_rank: bool = False,
        random_state: int = 42,
        show_progress: bool = False
    ) -> Dict[str, float]:
        """
        Measure out-of-sample linear reconstruction accuracy between layer representations.

        This metric measures how well representations in layer j can be linearly
        reconstructed from layer i using ridge regression. It answers: "How much
        information is preserved in a linearly decodable form?"

        This is NOT:
        - Mutual information (which captures all statistical dependencies)
        - Channel capacity (which requires optimization over input distributions)
        - Information flow (which measures causal influence)

        Args:
            model: The model to analyze
            train_batch: Training data for fitting the linear map
            test_batch: Test data for evaluation (if None, splits train_batch)
            val_batch: Validation data for hyperparameter selection
            layer_pairs: List of (source, target) layer pairs to analyze
            attention_mask: Optional attention mask for valid tokens
            max_samples: Maximum number of samples to use (for efficiency)
            max_dim: Maximum dimensionality after reduction (None = no reduction)
            alpha_grid: Ridge regularization values to search over
            stratified_sampling: Whether to use position/sequence balanced sampling
            compute_cka: Whether to also compute CKA similarity
            compute_effective_rank: Whether to compute effective rank of predictions
            random_state: Random seed for reproducibility
            show_progress: Whether to show progress bars

        Returns:
            Dictionary containing:
            - test_r2: Out-of-sample R² (can be negative!)
            - test_mse: Out-of-sample mean squared error
            - train_r2: Training R²
            - cka: CKA similarity score (if requested)
            - alpha_selected: Chosen regularization parameter
            - layer_results: Per-layer-pair results

        Interpretation:
        - R² > 0.9: Strong linear relationship, most information is linearly decodable
        - R² ∈ [0.5, 0.9]: Moderate linear relationship with some nonlinear transformation
        - R² ∈ [0, 0.5]: Weak linear relationship, substantial nonlinearity
        - R² < 0: Linear model is worse than predicting the mean - indicates strong
                  nonlinearity or information loss
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Validate inputs
        if not train_batch or 'input_ids' not in train_batch:
            return {'error': 'Empty or invalid training batch'}

        if train_batch['input_ids'].numel() == 0:
            return {'error': 'Training batch has zero elements'}

        # CRITICAL FIX: Determine target device from model and ensure ALL batch tensors are on it
        # This prevents device mismatch errors when batches have inconsistent device placement
        model_device = next(model.parameters()).device

        def move_batch_to_device(batch: Optional[Dict[str, torch.Tensor]], device) -> Optional[Dict[str, torch.Tensor]]:
            """Ensure all tensors in batch dict are on the specified device."""
            if batch is None:
                return None
            moved_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    moved_batch[key] = value.to(device)
                else:
                    moved_batch[key] = value
            return moved_batch

        # Move all batches to model device for consistency
        train_batch = move_batch_to_device(train_batch, model_device)
        test_batch = move_batch_to_device(test_batch, model_device)
        val_batch = move_batch_to_device(val_batch, model_device)

        # Split data if needed (batches are already on model_device)
        if test_batch is None:
            if show_progress:
                logger.info("No test batch provided, splitting train batch 60/20/20")
            train_data, val_data, test_data = self._split_batch(
                train_batch, ratios=[0.6, 0.2, 0.2], seed=random_state
            )
        else:
            train_data = train_batch
            test_data = test_batch
            if val_batch is None:
                if show_progress:
                    logger.info("No validation batch provided, splitting train batch 80/20")
                train_data, val_data = self._split_batch(
                    train_batch, ratios=[0.8, 0.2], seed=random_state + 1
                )
            else:
                val_data = val_batch

        # Extract hidden states
        model.eval()
        with torch.no_grad():
            train_hidden = self._get_hidden_states(model, train_data)
            val_hidden = self._get_hidden_states(model, val_data)
            test_hidden = self._get_hidden_states(model, test_data)

        # Extract attention mask if provided
        # Note: Masks are already on correct device from batch preprocessing above
        if attention_mask is None and 'attention_mask' in train_data:
            train_mask = train_data['attention_mask']
            val_mask = val_data['attention_mask']
            test_mask = test_data['attention_mask']
        else:
            train_mask = val_mask = test_mask = attention_mask
            # If attention_mask parameter was provided, ensure it's on model_device
            if attention_mask is not None:
                train_mask = val_mask = test_mask = attention_mask.to(model_device)

        # Default layer pairs: adjacent layers
        if layer_pairs is None:
            layer_pairs = [(i, i+1) for i in range(len(train_hidden)-1)]

        # Setup progress bar
        if show_progress:
            layer_pairs_iter = tqdm(
                layer_pairs,
                desc="Computing Linear Reconstruction",
                unit="layer_pair",
                file=sys.stderr
            )
        else:
            layer_pairs_iter = layer_pairs

        # Default alpha grid if not provided
        # Use model_device for consistency (train_hidden will also be on model_device)
        if alpha_grid is None:
            alpha_grid = torch.logspace(-6, 2, steps=9, device=model_device)
        else:
            # Ensure user-provided alpha_grid is on correct device
            alpha_grid = alpha_grid.to(model_device)

        results = {}
        layer_results = []

        with logging_redirect_tqdm():
            for layer_i, layer_j in layer_pairs_iter:
                if show_progress:
                    layer_pairs_iter.set_description(
                        f"Processing layers {layer_i}→{layer_j}"
                    )

                # Process each data split
                X_train, Y_train = self._prepare_layer_data(
                    train_hidden[layer_i], train_hidden[layer_j],
                    train_mask, max_samples, stratified_sampling, random_state
                )

                X_val, Y_val = self._prepare_layer_data(
                    val_hidden[layer_i], val_hidden[layer_j],
                    val_mask, max_samples, stratified_sampling, random_state + 2
                )

                X_test, Y_test = self._prepare_layer_data(
                    test_hidden[layer_i], test_hidden[layer_j],
                    test_mask, max_samples, stratified_sampling, random_state + 3
                )

                # Check dimensions
                N_train, D_in = X_train.shape
                _, D_out = Y_train.shape

                # Warn about underdetermined systems
                if N_train < D_in:
                    logger.warning(
                        f"Layer {layer_i}→{layer_j}: Underdetermined system "
                        f"(N={N_train} < D={D_in}). Results may be unreliable."
                    )

                # Optional dimensionality reduction
                if max_dim is not None and D_in > max_dim and N_train > max_dim:
                    if show_progress:
                        logger.info(f"Reducing dimensions: {D_in}→{max_dim}")
                    X_train, X_val, X_test, reducer = self._reduce_dimensions(
                        X_train, X_val, X_test, max_dim
                    )
                    D_in = max_dim

                # Center data using training statistics
                X_mean = X_train.mean(dim=0, keepdim=True)
                Y_mean = Y_train.mean(dim=0, keepdim=True)

                X_train_c = X_train - X_mean
                Y_train_c = Y_train - Y_mean
                X_val_c = X_val - X_mean  # Use training mean!
                Y_val_c = Y_val - Y_mean
                X_test_c = X_test - X_mean
                Y_test_c = Y_test - Y_mean

                # Select optimal alpha via validation
                best_alpha, best_val_r2 = self._select_alpha(
                    X_train_c, Y_train_c, X_val_c, Y_val_c, alpha_grid
                )

                # Train final model with best alpha
                beta = self._fit_ridge_regression(X_train_c, Y_train_c, best_alpha)

                # Compute predictions
                Y_train_pred = X_train_c @ beta
                Y_test_pred = X_test_c @ beta

                # Compute R² scores (don't clamp!)
                train_r2 = self._compute_r2(Y_train_c, Y_train_pred)
                test_r2 = self._compute_r2(Y_test_c, Y_test_pred)

                # Compute MSE
                train_mse = ((Y_train_c - Y_train_pred) ** 2).mean().item()
                test_mse = ((Y_test_c - Y_test_pred) ** 2).mean().item()

                # Compute CKA if requested
                cka_score = None
                if compute_cka:
                    cka_score = self._compute_cka(X_test_c, Y_test_c)

                # Compute effective rank if requested
                effective_rank = None
                if compute_effective_rank:
                    effective_rank = self._compute_effective_rank(Y_test_pred)

                # Log negative R² warning
                if test_r2 < 0:
                    logger.info(
                        f"Layer {layer_i}→{layer_j}: Negative test R² ({test_r2:.3f}) "
                        f"indicates linear model performs worse than constant prediction. "
                        f"Strong nonlinearity or information loss between layers."
                    )

                # Store layer pair results
                pair_result = {
                    'layer_pair': (layer_i, layer_j),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_mse': float(train_mse),
                    'test_mse': float(test_mse),
                    'alpha_selected': float(best_alpha),
                    'val_r2_at_alpha': float(best_val_r2),
                    'n_train': N_train,
                    'n_test': X_test.shape[0],
                    'input_dim': D_in,
                    'output_dim': D_out,
                    'cka': float(cka_score) if cka_score is not None else None,
                    'effective_rank': float(effective_rank) if effective_rank is not None else None
                }

                layer_results.append(pair_result)

                # Store in flat format for backward compatibility
                pair_name = f'layer_{layer_i}_to_{layer_j}'
                results[f'{pair_name}_test_r2'] = pair_result['test_r2']
                results[f'{pair_name}_train_r2'] = pair_result['train_r2']
                results[f'{pair_name}_test_mse'] = pair_result['test_mse']
                if cka_score is not None:
                    results[f'{pair_name}_cka'] = pair_result['cka']

        # Aggregate metrics
        all_test_r2 = [r['test_r2'] for r in layer_results]
        all_train_r2 = [r['train_r2'] for r in layer_results]
        all_cka = [r['cka'] for r in layer_results if r['cka'] is not None]

        results.update({
            'mean_test_r2': np.mean(all_test_r2) if all_test_r2 else 0.0,
            'mean_train_r2': np.mean(all_train_r2) if all_train_r2 else 0.0,
            'min_test_r2': min(all_test_r2) if all_test_r2 else 0.0,
            'max_test_r2': max(all_test_r2) if all_test_r2 else 0.0,
            'mean_cka': np.mean(all_cka) if all_cka else None,
            'layer_results': layer_results
        })

        return results

    def _split_batch(
        self,
        batch: Dict[str, torch.Tensor],
        ratios: List[float],
        seed: int = 42
    ) -> List[Dict[str, torch.Tensor]]:
        """Split a batch into multiple parts according to ratios."""
        # Get batch size
        batch_size = batch['input_ids'].shape[0]

        # Generate random permutation
        # Get device from batch tensors (handle potential device mismatch)
        device = batch['input_ids'].device if 'input_ids' in batch else 'cpu'

        # Ensure CPU for index generation to avoid device conflicts
        rng = torch.Generator(device='cpu').manual_seed(seed)
        perm = torch.randperm(batch_size, generator=rng, device='cpu')

        # Calculate split points
        split_points = []
        cumsum = 0
        for ratio in ratios[:-1]:
            cumsum += int(batch_size * ratio)
            split_points.append(cumsum)

        # Split indices
        splits = []
        start = 0
        for end in split_points + [batch_size]:
            indices = perm[start:end]
            split_dict = {}
            for key, tensor in batch.items():
                if torch.is_tensor(tensor):
                    # Move indices to the same device as the tensor for indexing
                    tensor_device = tensor.device
                    indices_on_device = indices.to(tensor_device)
                    split_dict[key] = tensor[indices_on_device]
                else:
                    split_dict[key] = tensor
            splits.append(split_dict)
            start = end

        return splits

    def _get_hidden_states(self, model, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Extract hidden states from all layers."""
        # Handle different model architectures
        outputs = model(**batch, output_hidden_states=True)

        if hasattr(outputs, 'hidden_states'):
            return list(outputs.hidden_states)
        else:
            raise ValueError("Model doesn't support output_hidden_states")

    def _prepare_layer_data(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_samples: int,
        stratified: bool,
        seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare layer data with optional stratified sampling."""
        B, L, D = h_i.shape

        # Handle edge case of empty batch
        if B == 0:
            return h_i.reshape(0, D), h_j.reshape(0, D)

        # CRITICAL: Convert to float32 for numerical stability in regression
        # BFloat16 has limited mantissa (7 bits) which can cause precision loss
        # in covariance computations. Float32 ensures stable ridge regression.
        # This conversion is safe: sklearn will convert to float64 anyway,
        # and PyTorch SVD works best with float32.
        original_dtype = h_i.dtype
        if original_dtype in [torch.bfloat16, torch.float16]:
            h_i = h_i.to(torch.float32)
            h_j = h_j.to(torch.float32)

        if stratified and attention_mask is not None:
            # Stratified sampling by position and sequence
            samples_i = []
            samples_j = []

            # Get device from input tensors
            device = h_i.device
            # Use CPU for index generation to avoid conflicts
            rng = torch.Generator(device='cpu').manual_seed(seed)
            samples_per_seq = max_samples // B

            for seq_idx in range(B):
                seq_mask = attention_mask[seq_idx].bool()
                # Ensure mask is on same device as h_i
                seq_mask = seq_mask.to(device)
                valid_positions = torch.where(seq_mask)[0]

                if len(valid_positions) == 0:
                    continue

                # Divide into quartiles
                n_pos = len(valid_positions)
                n_quartiles = min(4, n_pos)
                quartiles = torch.chunk(valid_positions, n_quartiles)

                samples_per_quartile = samples_per_seq // n_quartiles

                for quartile in quartiles:
                    if len(quartile) > samples_per_quartile:
                        # Generate indices on CPU to avoid conflicts
                        indices = torch.randperm(
                            len(quartile), generator=rng, device='cpu'
                        )[:samples_per_quartile]
                        # Move to device for indexing
                        indices = indices.to(quartile.device)
                        selected = quartile[indices]
                    else:
                        selected = quartile

                    samples_i.append(h_i[seq_idx, selected])
                    samples_j.append(h_j[seq_idx, selected])

            if samples_i:
                X = torch.cat(samples_i, dim=0)
                Y = torch.cat(samples_j, dim=0)
            else:
                # Fallback to reshape if no valid samples
                X = h_i.reshape(B * L, -1)
                Y = h_j.reshape(B * L, -1)
        else:
            # Simple flattening
            X = h_i.reshape(B * L, -1)
            Y = h_j.reshape(B * L, -1)

        # Limit total samples
        if X.shape[0] > max_samples:
            device = X.device
            # Generate indices on CPU to avoid conflicts
            rng = torch.Generator(device='cpu').manual_seed(seed)
            indices = torch.randperm(X.shape[0], generator=rng, device='cpu')[:max_samples]
            # Move indices to data device for indexing
            indices = indices.to(device)
            X = X[indices]
            Y = Y[indices]

        return X, Y

    def _reduce_dimensions(
        self,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        X_test: torch.Tensor,
        max_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Reduce dimensionality using PCA fitted on training data."""
        # Center training data
        X_mean = X_train.mean(dim=0, keepdim=True)
        X_train_c = X_train - X_mean

        # Compute SVD on training data
        U, S, V = torch.linalg.svd(X_train_c, full_matrices=False)

        # Keep top max_dim components
        V_reduced = V[:max_dim, :]

        # Project all data
        X_train_proj = X_train_c @ V_reduced.T
        X_val_proj = (X_val - X_mean) @ V_reduced.T
        X_test_proj = (X_test - X_mean) @ V_reduced.T

        # Store reducer info for potential reuse
        reducer = {
            'mean': X_mean,
            'components': V_reduced,
            'singular_values': S[:max_dim]
        }

        return X_train_proj, X_val_proj, X_test_proj, reducer

    def _select_alpha(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        alpha_grid: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Select optimal regularization using sklearn's RidgeCV.

        Uses Generalized Cross-Validation (GCV) for efficiency.
        This is mathematically equivalent to LOO-CV but much faster.

        References:
            - Golub et al. (1979): Generalized Cross-Validation
            - Hastie et al. (2009): Elements of Statistical Learning, Ch 7.10
        """
        # Convert to numpy for sklearn
        X_train_np = X_train.detach().cpu().numpy()
        Y_train_np = Y_train.detach().cpu().numpy()
        X_val_np = X_val.detach().cpu().numpy()
        Y_val_np = Y_val.detach().cpu().numpy()
        alphas = alpha_grid.detach().cpu().numpy()

        try:
            # Use RidgeCV with GCV for optimal alpha selection
            # Note: store_cv_values parameter removed for sklearn compatibility
            model = RidgeCV(
                alphas=alphas,
                fit_intercept=False,
                cv=None  # Uses efficient GCV
            )
            model.fit(X_train_np, Y_train_np)
            best_alpha = model.alpha_

            # Evaluate on held-out validation set
            Y_pred = model.predict(X_val_np)
            best_r2 = r2_score(Y_val_np, Y_pred,
                              multioutput='uniform_average' if Y_val_np.ndim > 1 else 'variance_weighted')

        except Exception as e:
            logger.warning(f"RidgeCV failed: {e}, using middle alpha")
            best_alpha = alphas[len(alphas) // 2]
            best_r2 = -float('inf')

        return float(best_alpha), float(best_r2)

    def _fit_ridge_regression(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha: float,
        method: str = 'sklearn'
    ) -> torch.Tensor:
        """
        Fit ridge regression using scikit-learn's professional implementation.

        For ICML standards, we use sklearn.linear_model.Ridge which:
        1. Has been tested extensively in production
        2. Implements numerically stable SVD solver
        3. Provides consistent results across platforms
        4. Is well-documented and peer-reviewed

        Args:
            X: Input features (N x D_in)
            Y: Target values (N x D_out)
            alpha: Regularization parameter
            method: 'sklearn' (default) or 'torch' for PyTorch implementation

        Returns:
            Regression coefficients (D_in x D_out)

        References:
            - Pedregosa et al. (2011): Scikit-learn: Machine Learning in Python
            - Ridge implementation uses scipy.linalg.lstsq with SVD solver
        """
        device = X.device

        if method == 'sklearn':
            # Convert to numpy for sklearn
            X_np = X.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()

            # Use sklearn's Ridge regression
            # Note: sklearn uses alpha as regularization, same as our convention
            if Y_np.shape[1] == 1:
                # Single output - use standard Ridge
                model = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
                model.fit(X_np, Y_np.ravel())
                beta_np = model.coef_.reshape(-1, 1)
            else:
                # Multiple outputs - sklearn handles this automatically
                model = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
                model.fit(X_np, Y_np)
                beta_np = model.coef_.T  # sklearn returns (D_out x D_in), we want (D_in x D_out)

            # Convert back to torch tensor on original device
            beta = torch.from_numpy(beta_np).to(device).float()

        elif method == 'torch':
            # Keep PyTorch implementation as fallback for GPU efficiency
            # This uses the same SVD approach as sklearn
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            S_ridge = S / (S**2 + alpha)
            beta = Vt.T @ (torch.diag(S_ridge) @ (U.T @ Y))

            # Log condition number for diagnostics
            if len(S) > 0:
                condition_number = (S[0] / S[-1]).item()
                if condition_number > 1e10:
                    logger.warning(f"High condition number: {condition_number:.2e}")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sklearn' or 'torch'.")

        return beta

    def _compute_r2(self, Y_true: torch.Tensor, Y_pred: torch.Tensor) -> float:
        """
        Compute R² score without clamping.

        Negative values are meaningful: indicate model is worse than mean prediction.
        """
        ss_res = ((Y_true - Y_pred) ** 2).sum()
        ss_tot = ((Y_true) ** 2).sum()  # Y_true is already centered

        if ss_tot < 1e-10:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def _compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute Centered Kernel Alignment (CKA) between representations.

        CKA is more robust than R² for comparing representations as it's
        invariant to orthogonal transformations and isotropic scaling.
        """
        def center_kernel(K):
            n = K.shape[0]
            H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
            return H @ K @ H

        # Linear kernels
        K_X = X @ X.T
        K_Y = Y @ Y.T

        # Center
        K_X_c = center_kernel(K_X)
        K_Y_c = center_kernel(K_Y)

        # HSIC values
        hsic_xy = torch.trace(K_X_c @ K_Y_c)
        hsic_xx = torch.trace(K_X_c @ K_X_c)
        hsic_yy = torch.trace(K_Y_c @ K_Y_c)

        # CKA
        if hsic_xx * hsic_yy < 1e-10:
            return 0.0

        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        return cka.item()

    def _compute_effective_rank(self, Y_pred: torch.Tensor) -> float:
        """
        Compute effective rank (participation ratio) of predictions.

        This measures the effective dimensionality of the predicted representations.
        """
        # Center predictions
        Y_centered = Y_pred - Y_pred.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            _, S, _ = torch.linalg.svd(Y_centered, full_matrices=False)

            # Filter near-zero values
            S = S[S > 1e-10]

            if len(S) == 0:
                return 1.0

            # Participation ratio
            S_normalized = S / S.sum()
            effective_rank = 1 / (S_normalized ** 2).sum()

            return effective_rank.item()
        except Exception as e:
            logger.warning(f"Effective rank computation failed: {e}")
            return float('nan')

    def compute_representational_similarity(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        method: str = 'cka',  # 'cka', 'linear_cka', 'rsa'
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute representational similarity between layers.

        This is a cleaner interface for similarity metrics that don't
        involve training a linear map.
        """
        model.eval()
        with torch.no_grad():
            hidden_states = self._get_hidden_states(model, batch)

        if layer_pairs is None:
            layer_pairs = [(i, i+1) for i in range(len(hidden_states)-1)]

        results = {}
        for layer_i, layer_j in layer_pairs:
            # Flatten representations
            h_i = hidden_states[layer_i].reshape(-1, hidden_states[layer_i].shape[-1])
            h_j = hidden_states[layer_j].reshape(-1, hidden_states[layer_j].shape[-1])

            # Compute similarity
            if method == 'cka':
                sim = self._compute_cka(h_i, h_j)
            elif method == 'cosine':
                sim = self._compute_cosine_similarity(h_i, h_j)
            else:
                raise ValueError(f"Unknown similarity method: {method}")

            pair_name = f'layer_{layer_i}_to_{layer_j}'
            results[f'{pair_name}_similarity'] = sim

        # Add aggregated metrics
        all_sims = [v for k, v in results.items() if '_similarity' in k]
        results['mean_similarity'] = np.mean(all_sims) if all_sims else 0.0
        results['min_similarity'] = min(all_sims) if all_sims else 0.0

        return results

    def _compute_cosine_similarity(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute mean cosine similarity between representations."""
        # Normalize
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-10)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-10)

        # Compute similarities
        similarities = (X_norm * Y_norm).sum(dim=1)

        return similarities.mean().item()