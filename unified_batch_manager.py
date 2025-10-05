"""
Unified Batch Creation Manager for ICML 2026
==============================================

A single, consistent batch creation system that ensures:
1. Reproducibility for paper submission
2. Consistent batch sizes across all experiments
3. Clear documentation of batch configurations
4. Task-aware batching when needed

IMPORTANT: Batch sizes are FIXED for reproducibility.
Any changes must be explicitly documented in the paper.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch creation strategies."""
    SEQUENTIAL = "sequential"  # Original order preserved
    SEPARATE_TASKS = "separate_tasks"  # Keep tasks separate
    MIXED_TASKS = "mixed_tasks"  # Mix tasks together


@dataclass
class BatchConfig:
    """
    Immutable batch configuration for reproducibility.

    These values are FIXED for ICML submission.
    Any changes require re-running all experiments.
    """
    # Standard batch sizes for ICML experiments
    fisher_batch_size: int = 256  # DO NOT CHANGE - used in all Fisher experiments
    gradient_batch_size: int = 256  # DO NOT CHANGE - used in gradient analysis
    tracin_batch_size: int = 64  # Fixed from 16 (which was too small)
    attention_batch_size: int = 256
    general_batch_size: int = 256

    # Task handling
    separate_tasks: bool = True  # Keep math/general separate for cross-task analysis

    # Reproducibility
    seed: int = 42  # Fixed seed for batch creation
    log_batch_creation: bool = True  # Log all batch operations for paper

    def __post_init__(self):
        """Warn if non-standard batch sizes are used."""
        if self.fisher_batch_size != 256:
            logger.warning(f"⚠️ Non-standard Fisher batch size: {self.fisher_batch_size} (standard: 256)")
            logger.warning("  This will affect reproducibility of ICML results!")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration for paper appendix."""
        return {
            'fisher_batch_size': self.fisher_batch_size,
            'gradient_batch_size': self.gradient_batch_size,
            'tracin_batch_size': self.tracin_batch_size,
            'attention_batch_size': self.attention_batch_size,
            'general_batch_size': self.general_batch_size,
            'separate_tasks': self.separate_tasks,
            'seed': self.seed
        }


class UnifiedBatchManager:
    """
    Unified batch creation manager ensuring reproducibility for ICML.

    This replaces all scattered batch creation logic with a single,
    consistent, reproducible system.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize with batch configuration.

        Args:
            config: Batch configuration (uses defaults if None)
        """
        self.config = config or BatchConfig()

        # Set random seed for reproducibility
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Track batch creation for logging
        self.batch_creation_log = []

        if self.config.log_batch_creation:
            logger.info("=" * 60)
            logger.info("UNIFIED BATCH MANAGER INITIALIZED")
            logger.info("Configuration for ICML 2026:")
            for key, value in self.config.to_dict().items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 60)

    def create_batches(
        self,
        data: Union[Dict[str, torch.Tensor], torch.Tensor],
        task_name: str,
        batch_type: str = 'general',
        strategy: BatchStrategy = BatchStrategy.SEQUENTIAL
    ) -> List[Dict[str, Any]]:
        """
        Create batches with consistent, reproducible logic.

        Args:
            data: Input data (dict with 'input_ids', 'attention_mask', etc. or tensor)
            task_name: Name of task ('math', 'general', 'combined', etc.)
            batch_type: Type of batch ('fisher', 'gradient', 'tracin', 'attention', 'general')
            strategy: Batching strategy

        Returns:
            List of batches with metadata

        Raises:
            ValueError: If batch_type is unknown
        """
        # Get appropriate batch size
        batch_size = self._get_batch_size(batch_type)

        # Convert single tensor to dict format
        if isinstance(data, torch.Tensor):
            data = {'input_ids': data}

        # Get total number of samples
        first_key = next(iter(data.keys()))
        total_samples = data[first_key].shape[0]

        # Validate we have enough samples
        if total_samples == 0:
            logger.warning(f"No samples provided for {task_name}")
            return []

        # Create batches
        batches = []

        for batch_idx, start_idx in enumerate(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            actual_batch_size = end_idx - start_idx

            # Extract batch data
            batch = {
                key: value[start_idx:end_idx]
                for key, value in data.items()
            }

            # Add labels for language modeling if needed
            if 'input_ids' in batch and 'labels' not in batch:
                batch['labels'] = batch['input_ids'].clone()

            # Add comprehensive metadata
            batch['_metadata'] = {
                'task': task_name,
                'batch_type': batch_type,
                'batch_idx': batch_idx,
                'global_idx': len(self.batch_creation_log),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'batch_size': actual_batch_size,
                'expected_batch_size': batch_size,
                'strategy': strategy.value,
                'config_seed': self.config.seed
            }

            batches.append(batch)

            # Log batch creation
            self._log_batch_creation(batch['_metadata'])

        # Summary logging
        if self.config.log_batch_creation:
            avg_batch_size = sum(b['_metadata']['batch_size'] for b in batches) / len(batches) if batches else 0
            logger.info(
                f"Created {len(batches)} {batch_type} batches for {task_name}: "
                f"{total_samples} samples, batch_size={batch_size}, "
                f"avg_actual={avg_batch_size:.1f}"
            )

            # Warn if last batch is too small (affects statistics)
            if batches and batches[-1]['_metadata']['batch_size'] < batch_size * 0.5:
                logger.warning(
                    f"  ⚠️ Last batch only has {batches[-1]['_metadata']['batch_size']} samples "
                    f"(less than 50% of {batch_size})"
                )

        return batches

    def create_fisher_batches(
        self,
        math_data: Dict[str, torch.Tensor],
        general_data: Dict[str, torch.Tensor]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create Fisher batches maintaining task separation for cross-task analysis.

        Args:
            math_data: Math task data
            general_data: General task data

        Returns:
            Tuple of (math_batches, general_batches)
        """
        if not self.config.separate_tasks:
            # Combine data if separation not needed
            logger.info("Creating mixed Fisher batches (separate_tasks=False)")
            combined_data = {
                key: torch.cat([math_data[key], general_data[key]], dim=0)
                for key in math_data.keys()
            }
            combined_batches = self.create_batches(
                combined_data,
                task_name='combined',
                batch_type='fisher',
                strategy=BatchStrategy.MIXED_TASKS
            )
            # Return same batches for both (compatibility)
            return combined_batches, []

        # Create separate batches for cross-task analysis
        logger.info("Creating separate Fisher batches for cross-task analysis")

        math_batches = self.create_batches(
            math_data,
            task_name='math',
            batch_type='fisher',
            strategy=BatchStrategy.SEPARATE_TASKS
        )

        general_batches = self.create_batches(
            general_data,
            task_name='general',
            batch_type='fisher',
            strategy=BatchStrategy.SEPARATE_TASKS
        )

        # Log summary
        logger.info(
            f"Fisher batch summary: {len(math_batches)} math + "
            f"{len(general_batches)} general = {len(math_batches) + len(general_batches)} total"
        )

        return math_batches, general_batches

    def create_tracin_batches(
        self,
        data: Dict[str, torch.Tensor],
        num_samples: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Create TracIn batches with proper batch size.

        The original code had batch_size=16 which was too small.
        Now uses configured tracin_batch_size (64).

        Args:
            data: Input data
            num_samples: Number of samples to use for TracIn

        Returns:
            List of TracIn batches
        """
        # Subset data if needed
        first_key = next(iter(data.keys()))
        if data[first_key].shape[0] > num_samples:
            subset_data = {
                key: value[:num_samples]
                for key, value in data.items()
            }
        else:
            subset_data = data

        logger.info(f"Creating TracIn batches from {num_samples} samples")

        batches = self.create_batches(
            subset_data,
            task_name='tracin',
            batch_type='tracin',
            strategy=BatchStrategy.SEQUENTIAL
        )

        return batches

    def _get_batch_size(self, batch_type: str) -> int:
        """Get batch size for given batch type."""
        batch_sizes = {
            'fisher': self.config.fisher_batch_size,
            'gradient': self.config.gradient_batch_size,
            'tracin': self.config.tracin_batch_size,
            'attention': self.config.attention_batch_size,
            'general': self.config.general_batch_size
        }

        if batch_type not in batch_sizes:
            logger.warning(f"Unknown batch type: {batch_type}, using general batch size")
            return self.config.general_batch_size

        return batch_sizes[batch_type]

    def _log_batch_creation(self, metadata: Dict[str, Any]):
        """Log batch creation for reproducibility tracking."""
        self.batch_creation_log.append(metadata)

        # Detailed logging every 100 batches
        if len(self.batch_creation_log) % 100 == 0:
            logger.info(f"  ... created {len(self.batch_creation_log)} total batches")

    def get_batch_report(self) -> Dict[str, Any]:
        """
        Get comprehensive batch creation report for paper appendix.

        Returns:
            Dictionary with batch creation statistics
        """
        if not self.batch_creation_log:
            return {'error': 'No batches created yet'}

        report = {
            'config': self.config.to_dict(),
            'total_batches_created': len(self.batch_creation_log),
            'by_task': {},
            'by_type': {},
            'batch_size_distribution': {}
        }

        # Analyze by task and type
        for entry in self.batch_creation_log:
            task = entry['task']
            batch_type = entry['batch_type']
            batch_size = entry['batch_size']

            # Count by task
            if task not in report['by_task']:
                report['by_task'][task] = 0
            report['by_task'][task] += 1

            # Count by type
            if batch_type not in report['by_type']:
                report['by_type'][batch_type] = 0
            report['by_type'][batch_type] += 1

            # Track batch size distribution
            if batch_size not in report['batch_size_distribution']:
                report['batch_size_distribution'][batch_size] = 0
            report['batch_size_distribution'][batch_size] += 1

        return report

    def save_batch_log(self, filepath: str = 'batch_creation_log.json'):
        """
        Save batch creation log for paper reproducibility.

        Args:
            filepath: Where to save the log
        """
        import json

        log_data = {
            'config': self.config.to_dict(),
            'batches': self.batch_creation_log,
            'summary': self.get_batch_report()
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved batch creation log to {filepath}")
        logger.info(f"  Total batches logged: {len(self.batch_creation_log)}")
        logger.info("  Include this file with paper submission for reproducibility")


# Convenience functions for backward compatibility
def create_fisher_batches(math_data, general_data, config=None):
    """Backward compatible Fisher batch creation."""
    manager = UnifiedBatchManager(config)
    return manager.create_fisher_batches(math_data, general_data)


def create_tracin_batches(data, num_samples=500, config=None):
    """Backward compatible TracIn batch creation."""
    manager = UnifiedBatchManager(config)
    return manager.create_tracin_batches(data, num_samples)