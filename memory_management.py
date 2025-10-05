#!/usr/bin/env python3
"""
Memory management utilities for safe model analysis.

This module provides memory monitoring, limits, and efficient processing
strategies to prevent out-of-memory errors in production.
"""

import os
import gc
import warnings
import psutil
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Generator
from dataclasses import dataclass
from contextlib import contextmanager
import functools


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_gb: float = 8.0
    warning_threshold_gb: float = 6.0
    chunk_size_mb: float = 256.0
    enable_gc: bool = True
    aggressive_gc: bool = False
    cuda_empty_cache: bool = True
    monitor_interval_seconds: float = 1.0


# Global memory configuration
_memory_config = MemoryConfig()


def set_memory_config(config: MemoryConfig):
    """Set global memory configuration."""
    global _memory_config
    _memory_config = config


def get_memory_config() -> MemoryConfig:
    """Get current memory configuration."""
    return _memory_config


# ============= Memory Monitoring =============

class MemoryMonitor:
    """Monitor and track memory usage."""

    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """
        Get system memory statistics in GB.

        Returns:
            Dictionary with memory statistics.
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total': mem.total / (1024**3),
            'available': mem.available / (1024**3),
            'used': mem.used / (1024**3),
            'percent': mem.percent,
            'swap_total': swap.total / (1024**3),
            'swap_used': swap.used / (1024**3),
            'swap_percent': swap.percent
        }

    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """
        Get GPU memory statistics in GB.

        Returns:
            Dictionary with GPU memory stats.
        """
        if not torch.cuda.is_available():
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'free': 0.0,
                'total': 0.0
            }

        # Get memory for current device
        device = torch.cuda.current_device()

        return {
            'allocated': torch.cuda.memory_allocated(device) / (1024**3),
            'reserved': torch.cuda.memory_reserved(device) / (1024**3),
            'free': (torch.cuda.get_device_properties(device).total_memory -
                    torch.cuda.memory_allocated(device)) / (1024**3),
            'total': torch.cuda.get_device_properties(device).total_memory / (1024**3),
            'device': device,
            'device_name': torch.cuda.get_device_name(device)
        }

    @staticmethod
    def get_process_memory() -> Dict[str, float]:
        """
        Get current process memory usage.

        Returns:
            Dictionary with process memory stats.
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        return {
            'rss': mem_info.rss / (1024**3),  # Resident Set Size
            'vms': mem_info.vms / (1024**3),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / (1024**3)
        }

    @classmethod
    def check_memory_available(cls, required_gb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB.

        Returns:
            True if sufficient memory available.
        """
        mem = cls.get_system_memory()
        return mem['available'] >= required_gb

    @classmethod
    def assert_memory_available(cls, required_gb: float, operation: str = "operation"):
        """
        Assert that sufficient memory is available.

        Args:
            required_gb: Required memory in GB.
            operation: Name of operation for error message.

        Raises:
            MemoryError: If insufficient memory.
        """
        mem = cls.get_system_memory()
        if mem['available'] < required_gb:
            raise MemoryError(
                f"Insufficient memory for {operation}: "
                f"required {required_gb:.2f}GB, available {mem['available']:.2f}GB"
            )


# ============= Memory Limits =============

class MemoryLimiter:
    """Enforce memory limits for operations."""

    @staticmethod
    def estimate_model_memory(model: torch.nn.Module,
                            include_gradients: bool = True) -> float:
        """
        Estimate memory required for model in GB.

        Args:
            model: PyTorch model.
            include_gradients: Include gradient storage.

        Returns:
            Estimated memory in GB.
        """
        total_params = 0
        total_bytes = 0

        for param in model.parameters():
            total_params += param.numel()
            # Get bytes per element based on dtype
            total_bytes += param.numel() * param.element_size()

        # Double for gradients if training
        if include_gradients:
            total_bytes *= 2

        # Add overhead for optimizer state (momentum, etc.)
        # Roughly 2x parameters for Adam optimizer
        total_bytes *= 1.5

        return total_bytes / (1024**3)

    @staticmethod
    def estimate_batch_memory(batch: Dict[str, torch.Tensor]) -> float:
        """
        Estimate memory required for batch in GB.

        Args:
            batch: Batch dictionary.

        Returns:
            Estimated memory in GB.
        """
        total_bytes = 0

        for key, tensor in batch.items():
            if torch.is_tensor(tensor):
                total_bytes += tensor.numel() * tensor.element_size()

        # Account for intermediate activations (rough estimate)
        total_bytes *= 3

        return total_bytes / (1024**3)

    @classmethod
    def check_operation_memory(cls, model: torch.nn.Module,
                              batch: Optional[Dict[str, torch.Tensor]] = None,
                              operation: str = "operation") -> None:
        """
        Check if operation will fit in memory limits.

        Args:
            model: Model to analyze.
            batch: Optional batch.
            operation: Operation name.

        Raises:
            MemoryError: If operation would exceed limits.
        """
        config = get_memory_config()

        model_memory = cls.estimate_model_memory(model)
        batch_memory = cls.estimate_batch_memory(batch) if batch else 0

        total_required = model_memory + batch_memory
        available = MemoryMonitor.get_system_memory()['available']

        if total_required > config.max_memory_gb:
            raise MemoryError(
                f"{operation} would exceed memory limit: "
                f"required {total_required:.2f}GB > limit {config.max_memory_gb:.2f}GB"
            )

        if total_required > available:
            raise MemoryError(
                f"Insufficient memory for {operation}: "
                f"required {total_required:.2f}GB > available {available:.2f}GB"
            )

        if total_required > config.warning_threshold_gb:
            warnings.warn(
                f"{operation} using significant memory: {total_required:.2f}GB"
            )


# ============= Memory Context Managers =============

@contextmanager
def memory_tracker(name: str = "operation", verbose: bool = True):
    """
    Track memory usage for an operation.

    Args:
        name: Operation name.
        verbose: Print memory usage.

    Yields:
        Memory statistics dictionary.
    """
    # Get initial memory
    initial_system = MemoryMonitor.get_system_memory()
    initial_gpu = MemoryMonitor.get_gpu_memory()
    initial_process = MemoryMonitor.get_process_memory()

    stats = {'name': name}

    try:
        yield stats
    finally:
        # Get final memory
        final_system = MemoryMonitor.get_system_memory()
        final_gpu = MemoryMonitor.get_gpu_memory()
        final_process = MemoryMonitor.get_process_memory()

        # Calculate deltas
        stats['system_delta_gb'] = final_system['used'] - initial_system['used']
        stats['gpu_delta_gb'] = final_gpu['allocated'] - initial_gpu['allocated']
        stats['process_delta_gb'] = final_process['rss'] - initial_process['rss']
        stats['peak_memory_gb'] = final_process['rss']

        if verbose:
            print(f"Memory usage for {name}:")
            print(f"  System: {stats['system_delta_gb']:+.3f}GB")
            print(f"  GPU: {stats['gpu_delta_gb']:+.3f}GB")
            print(f"  Process: {stats['process_delta_gb']:+.3f}GB")


@contextmanager
def memory_limit(max_gb: Optional[float] = None):
    """
    Context manager to enforce memory limits.

    Args:
        max_gb: Maximum memory in GB.

    Raises:
        MemoryError: If limit exceeded.
    """
    config = get_memory_config()
    limit = max_gb or config.max_memory_gb

    # Check initial memory
    initial = MemoryMonitor.get_process_memory()['rss']

    try:
        yield
    finally:
        # Check final memory
        final = MemoryMonitor.get_process_memory()['rss']

        if final > limit:
            raise MemoryError(
                f"Memory limit exceeded: {final:.2f}GB > {limit:.2f}GB"
            )


# ============= Garbage Collection =============

class GarbageCollector:
    """Enhanced garbage collection utilities."""

    @staticmethod
    def collect(aggressive: bool = False):
        """
        Run garbage collection.

        Args:
            aggressive: Run multiple collection cycles.
        """
        config = get_memory_config()

        if not config.enable_gc:
            return

        # Python garbage collection
        if aggressive or config.aggressive_gc:
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()

        # Clear CUDA cache
        if config.cuda_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    @contextmanager
    def auto_gc(threshold_gb: float = 1.0):
        """
        Context manager for automatic garbage collection.

        Args:
            threshold_gb: Memory increase threshold to trigger GC.
        """
        initial = MemoryMonitor.get_process_memory()['rss']

        try:
            yield
        finally:
            final = MemoryMonitor.get_process_memory()['rss']

            if final - initial > threshold_gb:
                GarbageCollector.collect(aggressive=True)


# ============= Chunked Processing =============

class ChunkedProcessor:
    """Process large data in memory-efficient chunks."""

    @staticmethod
    def chunk_tensor(tensor: torch.Tensor,
                    chunk_size: Optional[int] = None,
                    dim: int = 0) -> Generator[torch.Tensor, None, None]:
        """
        Yield tensor chunks along specified dimension.

        Args:
            tensor: Tensor to chunk.
            chunk_size: Size of each chunk.
            dim: Dimension to chunk along.

        Yields:
            Tensor chunks.
        """
        if chunk_size is None:
            config = get_memory_config()
            # Estimate chunk size based on memory limit
            bytes_per_element = tensor.element_size()
            elements_per_mb = (1024 * 1024) / bytes_per_element
            chunk_size = int(config.chunk_size_mb * elements_per_mb / tensor.shape[dim])

        # Yield chunks
        for start in range(0, tensor.shape[dim], chunk_size):
            end = min(start + chunk_size, tensor.shape[dim])
            indices = slice(start, end)
            # Create slice tuple for the specific dimension
            slice_tuple = tuple(
                indices if i == dim else slice(None)
                for i in range(tensor.dim())
            )
            yield tensor[slice_tuple]

    @staticmethod
    def process_in_chunks(data: torch.Tensor,
                         process_fn: callable,
                         chunk_size: Optional[int] = None,
                         dim: int = 0,
                         collect_gc: bool = True) -> torch.Tensor:
        """
        Process tensor in chunks and concatenate results.

        Args:
            data: Input tensor.
            process_fn: Function to apply to each chunk.
            chunk_size: Size of chunks.
            dim: Dimension to chunk along.
            collect_gc: Run GC between chunks.

        Returns:
            Concatenated results.
        """
        results = []

        for i, chunk in enumerate(ChunkedProcessor.chunk_tensor(data, chunk_size, dim)):
            result = process_fn(chunk)
            results.append(result)

            # Garbage collect periodically
            if collect_gc and i % 10 == 0:
                GarbageCollector.collect()

        # Concatenate results
        return torch.cat(results, dim=dim)


# ============= Memory-Efficient Decorators =============

def memory_efficient(max_memory_gb: Optional[float] = None,
                    chunk_inputs: bool = False):
    """
    Decorator to make functions memory-efficient.

    Args:
        max_memory_gb: Maximum memory for operation.
        chunk_inputs: Process inputs in chunks.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before operation
            if max_memory_gb:
                MemoryMonitor.assert_memory_available(
                    max_memory_gb,
                    func.__name__
                )

            # Track memory usage
            with memory_tracker(func.__name__, verbose=False) as stats:
                with GarbageCollector.auto_gc():
                    result = func(*args, **kwargs)

            # Log if significant memory used
            if stats['process_delta_gb'] > 1.0:
                warnings.warn(
                    f"{func.__name__} used {stats['process_delta_gb']:.2f}GB memory"
                )

            return result

        return wrapper
    return decorator


# ============= Testing =============

def test_memory_management():
    """Test memory management utilities."""
    print("Testing memory management...")

    # Test memory monitoring
    system_mem = MemoryMonitor.get_system_memory()
    print(f"System memory: {system_mem['available']:.2f}GB available")

    gpu_mem = MemoryMonitor.get_gpu_memory()
    print(f"GPU memory: {gpu_mem['total']:.2f}GB total")

    # Test memory estimation
    model = torch.nn.Linear(1000, 1000)
    model_memory = MemoryLimiter.estimate_model_memory(model)
    print(f"Model memory: {model_memory:.4f}GB")

    # Test memory tracking
    with memory_tracker("test_operation", verbose=True):
        # Allocate some memory
        data = torch.randn(1000, 1000)
        result = data @ data.T

    # Test chunked processing
    large_tensor = torch.randn(10000, 100)

    def process_chunk(chunk):
        return chunk.mean(dim=1, keepdim=True)

    result = ChunkedProcessor.process_in_chunks(
        large_tensor,
        process_chunk,
        chunk_size=1000
    )
    assert result.shape == (10000, 1)
    print("✓ Chunked processing working")

    # Test garbage collection
    initial_mem = MemoryMonitor.get_process_memory()['rss']
    large_data = [torch.randn(1000, 1000) for _ in range(10)]
    del large_data
    GarbageCollector.collect(aggressive=True)
    final_mem = MemoryMonitor.get_process_memory()['rss']
    print(f"✓ GC freed {initial_mem - final_mem:.3f}GB")

    print("All memory tests passed!")


if __name__ == "__main__":
    test_memory_management()