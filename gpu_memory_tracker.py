"""
GPU Memory Allocation Tracker

Provides detailed tracking of GPU memory allocations to identify OOM sources.
Tracks every CUDA allocation with stack traces and contextual information.
"""

import torch
import traceback
import functools
import time
from contextlib import contextmanager
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GPUMemoryTracker:
    """
    Tracks GPU memory allocations with detailed context and stack traces.
    """

    def __init__(self, enabled: bool = True, log_threshold_mb: float = 10.0):
        """
        Initialize GPU memory tracker.

        Args:
            enabled: Whether tracking is enabled
            log_threshold_mb: Only log allocations larger than this (in MB)
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.log_threshold_bytes = log_threshold_mb * 1024 * 1024

        # Tracking data structures
        self.allocation_history = deque(maxlen=1000)  # Keep last 1000 allocations
        self.peak_memory = 0
        self.peak_location = None
        self.allocation_by_source = defaultdict(list)
        self.current_context = None

        # Memory snapshots
        self.snapshots = {}

        # Start monitoring
        if self.enabled:
            self.reset_peak_memory()

    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if not self.enabled:
            return
        torch.cuda.reset_peak_memory_stats()
        self.peak_memory = torch.cuda.memory_allocated()
        self.peak_location = None

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory statistics in GB."""
        if not self.enabled:
            return {}

        # Force synchronization to get accurate memory readings
        torch.cuda.synchronize()

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        # Get real memory from nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=1
            )
            used_mb, free_mb = map(float, result.stdout.strip().split(','))
            system_used_gb = used_mb / 1024
            system_free_gb = free_mb / 1024
        except:
            # Fallback if nvidia-smi fails
            system_free_gb = (total_memory - reserved) / 1e9
            system_used_gb = reserved / 1e9

        return {
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'free_gb': system_free_gb,
            'system_used_gb': system_used_gb,
            'cache_gb': (reserved - allocated) / 1e9,
            'total_gb': total_memory / 1e9,
        }

    def log_allocation(self, size_bytes: int, operation: str = "unknown",
                      extra_context: Optional[Dict] = None):
        """
        Log a memory allocation event.

        Args:
            size_bytes: Size of allocation in bytes
            operation: Description of the operation
            extra_context: Additional context information
        """
        if not self.enabled or abs(size_bytes) < self.log_threshold_bytes:
            return

        # Get stack trace (skip this function and caller)
        stack = traceback.extract_stack()[:-2]

        # Find most relevant source location
        source_location = None
        for frame in reversed(stack):
            if 'GradientAnalysis.py' in frame.filename or \
               'unified_model_analysis.py' in frame.filename or \
               'InformationTheoryMetrics.py' in frame.filename:
                source_location = f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                break

        if source_location is None and len(stack) > 0:
            frame = stack[-1]
            source_location = f"{frame.filename.split('/')[-1]}:{frame.lineno}"

        # Create allocation record
        allocation = {
            'timestamp': time.time(),
            'size_mb': size_bytes / (1024 * 1024),
            'operation': operation,
            'source': source_location,
            'context': self.current_context,
            'extra': extra_context,
            'memory_state': self.get_memory_info()
        }

        self.allocation_history.append(allocation)

        # Log if significant
        sign = "+" if size_bytes > 0 else ""
        logger.info(f"[GPU ALLOC] {sign}{allocation['size_mb']:.2f} MB at {source_location} "
                   f"({operation})")

        # Log memory state
        mem_info = allocation['memory_state']
        logger.info(f"    Memory: {mem_info['allocated_gb']:.2f}GB allocated, "
                   f"{mem_info['reserved_gb']:.2f}GB reserved, "
                   f"{mem_info['free_gb']:.2f}GB free")

        # Track by source
        if source_location:
            self.allocation_by_source[source_location].append(allocation)

        # Check for new peak
        current_memory = torch.cuda.memory_allocated()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            self.peak_location = source_location
            logger.warning(f"[GPU PEAK] New peak memory: {current_memory/1e9:.2f}GB at {source_location}")

    @contextmanager
    def track_context(self, context_name: str, log_memory: bool = True):
        """
        Context manager to track memory allocations within a specific context.

        Args:
            context_name: Name of the context (e.g., "forward_pass", "backward")
            log_memory: Whether to log memory before/after
        """
        if not self.enabled:
            yield
            return

        old_context = self.current_context
        self.current_context = context_name

        if log_memory:
            before_mem = torch.cuda.memory_allocated()
            logger.info(f"[GPU TRACK] Entering {context_name}: {before_mem/1e9:.2f}GB allocated")

        try:
            yield
        finally:
            if log_memory:
                after_mem = torch.cuda.memory_allocated()
                delta = after_mem - before_mem
                sign = "+" if delta > 0 else ""
                logger.info(f"[GPU TRACK] Exiting {context_name}: {after_mem/1e9:.2f}GB allocated "
                           f"({sign}{delta/1e6:.2f}MB change)")

                # Log allocation if significant
                if abs(delta) > self.log_threshold_bytes:
                    self.log_allocation(delta, f"{context_name}_total")

            self.current_context = old_context

    def track_function(self, func_name: Optional[str] = None):
        """
        Decorator to track memory allocations within a function.

        Args:
            func_name: Optional function name override
        """
        def decorator(func):
            actual_name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.track_context(actual_name):
                    return func(*args, **kwargs)

            return wrapper if self.enabled else func
        return decorator

    def take_snapshot(self, name: str):
        """
        Take a memory snapshot for later comparison.

        Args:
            name: Name for the snapshot
        """
        if not self.enabled:
            return

        snapshot = {
            'timestamp': time.time(),
            'memory_info': self.get_memory_info(),
            'tensors': self._get_tensor_info(),
        }

        self.snapshots[name] = snapshot
        logger.info(f"[GPU SNAPSHOT] '{name}': {snapshot['memory_info']['allocated_gb']:.2f}GB allocated")

    def compare_snapshots(self, before: str, after: str) -> Dict:
        """
        Compare two memory snapshots.

        Args:
            before: Name of before snapshot
            after: Name of after snapshot

        Returns:
            Comparison results
        """
        if not self.enabled or before not in self.snapshots or after not in self.snapshots:
            return {}

        snap_before = self.snapshots[before]
        snap_after = self.snapshots[after]

        delta_gb = (snap_after['memory_info']['allocated_gb'] -
                   snap_before['memory_info']['allocated_gb'])

        result = {
            'delta_gb': delta_gb,
            'before_gb': snap_before['memory_info']['allocated_gb'],
            'after_gb': snap_after['memory_info']['allocated_gb'],
            'time_delta': snap_after['timestamp'] - snap_before['timestamp']
        }

        sign = "+" if delta_gb > 0 else ""
        logger.info(f"[GPU COMPARE] {before} -> {after}: "
                   f"{result['before_gb']:.2f}GB -> {result['after_gb']:.2f}GB "
                   f"({sign}{delta_gb:.2f}GB)")

        return result

    def _get_tensor_info(self) -> List[Dict]:
        """Get information about large tensors in memory."""
        if not self.enabled:
            return []

        tensors = []

        # Note: This is a simplified version. For full tensor tracking,
        # you'd need to hook into PyTorch's memory allocator more deeply
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
                    if size_mb > 10:  # Only track tensors > 10MB
                        tensors.append({
                            'shape': list(obj.shape),
                            'dtype': str(obj.dtype),
                            'size_mb': size_mb,
                            'device': str(obj.device)
                        })
            except:
                pass

        return sorted(tensors, key=lambda x: x['size_mb'], reverse=True)[:10]

    def print_summary(self):
        """Print a summary of memory allocation patterns."""
        if not self.enabled or not self.allocation_history:
            logger.info("No allocation history available")
            return

        logger.info("\n" + "="*60)
        logger.info("GPU MEMORY ALLOCATION SUMMARY")
        logger.info("="*60)

        # Current memory state
        mem_info = self.get_memory_info()
        logger.info(f"\nCurrent Memory State:")
        logger.info(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
        logger.info(f"  Reserved:  {mem_info['reserved_gb']:.2f} GB")
        logger.info(f"  Free:      {mem_info['free_gb']:.2f} GB")
        logger.info(f"  Peak:      {self.peak_memory/1e9:.2f} GB at {self.peak_location}")

        # Top allocation sources
        logger.info(f"\nTop Allocation Sources:")
        source_totals = {}
        for source, allocs in self.allocation_by_source.items():
            total_mb = sum(a['size_mb'] for a in allocs if a['size_mb'] > 0)
            source_totals[source] = total_mb

        for source, total_mb in sorted(source_totals.items(),
                                       key=lambda x: x[1], reverse=True)[:10]:
            count = len(self.allocation_by_source[source])
            logger.info(f"  {source}: {total_mb:.2f} MB total ({count} allocations)")

        # Recent large allocations
        logger.info(f"\nRecent Large Allocations:")
        large_allocs = [a for a in self.allocation_history
                       if abs(a['size_mb']) > 50][-10:]
        for alloc in large_allocs:
            sign = "+" if alloc['size_mb'] > 0 else ""
            logger.info(f"  {sign}{alloc['size_mb']:.2f} MB at {alloc['source']} "
                       f"({alloc['operation']})")

        logger.info("="*60 + "\n")


# Global tracker instance
_global_tracker = None


def get_tracker() -> GPUMemoryTracker:
    """Get or create the global memory tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = GPUMemoryTracker()
    return _global_tracker


def track_memory(operation: str = "unknown"):
    """
    Decorator to track memory usage of a function.

    Usage:
        @track_memory("forward_pass")
        def my_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_tracker()
            with tracker.track_context(operation or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_memory_state(message: str = ""):
    """Log current memory state with optional message."""
    tracker = get_tracker()
    mem_info = tracker.get_memory_info()

    if not mem_info:
        return

    if message:
        logger.info(f"[GPU STATE] {message}")

    # Simple, clear output
    logger.info(f"[GPU STATE] Used: {mem_info['system_used_gb']:.2f}GB, Free: {mem_info['free_gb']:.2f}GB "
               f"(PyTorch: {mem_info['allocated_gb']:.2f}GB active, {mem_info['cache_gb']:.2f}GB cache)")


def log_gpu_memory(message: str = ""):
    """Alias for log_memory_state for backward compatibility."""
    log_memory_state(message)


# Import gc for tensor tracking
try:
    import gc
except ImportError:
    gc = None


def test_memory_tracking():
    """Test that memory tracking is working correctly."""
    if not torch.cuda.is_available():
        print("No GPU available for testing")
        return

    print("\n=== GPU Memory Tracking Test ===")
    tracker = get_tracker()

    # Get initial state
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    initial = tracker.get_memory_info()
    print(f"Initial: {initial['allocated_gb']:.3f}GB allocated, {initial['free_gb']:.3f}GB free")

    # Allocate some memory
    test_tensor = torch.randn(1024, 1024, 512, device='cuda')  # ~2GB
    torch.cuda.synchronize()
    after_alloc = tracker.get_memory_info()
    print(f"After 2GB alloc: {after_alloc['allocated_gb']:.3f}GB allocated, {after_alloc['free_gb']:.3f}GB free")
    print(f"  Delta: +{(after_alloc['allocated_gb'] - initial['allocated_gb']):.3f}GB")

    # Delete and check
    del test_tensor
    torch.cuda.synchronize()
    after_del = tracker.get_memory_info()
    print(f"After delete: {after_del['allocated_gb']:.3f}GB allocated, {after_del['free_gb']:.3f}GB free")
    print(f"  Cache size: {after_del['cache_gb']:.3f}GB")

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_clear = tracker.get_memory_info()
    print(f"After cache clear: {after_clear['allocated_gb']:.3f}GB allocated, {after_clear['free_gb']:.3f}GB free")
    print(f"  Freed: {(after_clear['free_gb'] - after_del['free_gb']):.3f}GB")

    print("=== Test Complete ===\n")