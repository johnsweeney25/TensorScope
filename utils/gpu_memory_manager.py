"""
GPU Memory Manager - Dynamic Eviction for OOM Prevention

Strategy:
1. Never calculate on CPU (too slow)
2. Dynamically evict things from GPU to make space
3. Restore them when done
4. Priority-based eviction (least important first)

For ICML Reproducibility:
- Eviction order is deterministic
- Results are identical regardless of what gets evicted
- Only affects performance, not correctness
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import time

logger = logging.getLogger(__name__)


class EvictionPriority(IntEnum):
    """Priority levels for GPU memory eviction (lower = evict first)."""
    CACHE_OVERLAP = 1       # Cached overlap matrices (can recompute)
    CACHE_NORMS = 2         # Cached norms (cheap to recompute)
    ACTIVATIONS = 3         # Old activations (not needed for current metric)
    GRADIENTS = 4           # Gradients (if not currently computing)
    FISHER_MATRICES = 5     # Fisher information (can recompute)
    MODEL_PARAMS = 6        # Model parameters (expensive to move but doable)
    CRITICAL = 100          # Never evict (currently computing)


@dataclass
class GPUTensor:
    """Represents a tensor that can be evicted from GPU."""
    name: str
    tensor: torch.Tensor
    priority: EvictionPriority
    size_gb: float
    owner: Optional[Any] = None  # Reference to owner object
    cpu_copy: Optional[torch.Tensor] = None
    is_on_gpu: bool = True
    last_accessed: float = field(default_factory=time.time)

    def evict_to_cpu(self):
        """Move tensor to CPU and clear GPU memory."""
        if not self.is_on_gpu:
            return 0  # Already on CPU

        # Create CPU copy
        self.cpu_copy = self.tensor.cpu()

        # Get size before deletion
        size_freed = self.size_gb

        # Clear GPU reference
        del self.tensor
        torch.cuda.empty_cache()

        self.is_on_gpu = False
        logger.debug(f"Evicted {self.name} ({size_freed:.3f}GB) to CPU")
        return size_freed

    def restore_to_gpu(self):
        """Restore tensor from CPU to GPU."""
        if self.is_on_gpu:
            return  # Already on GPU

        if self.cpu_copy is None:
            raise RuntimeError(f"Cannot restore {self.name}: no CPU copy available")

        # Move back to GPU
        self.tensor = self.cpu_copy.cuda()
        self.cpu_copy = None
        self.is_on_gpu = True
        self.last_accessed = time.time()
        logger.debug(f"Restored {self.name} ({self.size_gb:.3f}GB) to GPU")


class GPUMemoryManager:
    """
    Dynamic GPU memory manager with intelligent eviction.

    Key features:
    1. Tracks all major tensors on GPU
    2. Evicts least important tensors when space needed
    3. Restores tensors when space available
    4. Maintains reproducibility (eviction doesn't affect results)

    Usage:
        manager = GPUMemoryManager()

        # Register tensors that can be evicted
        manager.register_tensor("model", model_params, priority=EvictionPriority.MODEL_PARAMS)
        manager.register_tensor("activations", acts, priority=EvictionPriority.ACTIVATIONS)

        # Before a calculation that needs memory
        manager.ensure_free_memory(required_gb=2.5)

        # After calculation
        manager.restore_evicted()
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.tracked_tensors: Dict[str, GPUTensor] = {}
        self.eviction_history: List[Tuple[str, float]] = []  # (name, time)

    def register_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        priority: EvictionPriority,
        owner: Optional[Any] = None
    ):
        """Register a tensor for potential eviction."""
        if tensor.device.type != 'cuda':
            return  # Only track GPU tensors

        size_gb = (tensor.numel() * tensor.element_size()) / 1e9

        self.tracked_tensors[name] = GPUTensor(
            name=name,
            tensor=tensor,
            priority=priority,
            size_gb=size_gb,
            owner=owner
        )
        logger.debug(f"Registered {name} ({size_gb:.3f}GB) with priority {priority.name}")

    def unregister_tensor(self, name: str):
        """Remove tensor from tracking (when no longer needed)."""
        if name in self.tracked_tensors:
            del self.tracked_tensors[name]
            logger.debug(f"Unregistered {name}")

    def get_free_memory_gb(self) -> float:
        """Get current free GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0

        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        return free_bytes / 1e9

    def get_evictable_tensors(self) -> List[GPUTensor]:
        """Get list of tensors that can be evicted, sorted by priority (lowest first)."""
        evictable = [t for t in self.tracked_tensors.values() if t.is_on_gpu]

        # Sort by priority (lower = evict first), then by last accessed (older first)
        evictable.sort(key=lambda t: (t.priority, t.last_accessed))

        return evictable

    def ensure_free_memory(
        self,
        required_gb: float,
        safety_margin: float = 1.5
    ) -> bool:
        """
        Ensure enough free GPU memory by evicting tensors if needed.

        Args:
            required_gb: Memory needed for calculation
            safety_margin: Multiply by this for safety

        Returns:
            True if memory is available (after eviction if needed)
            False if cannot free enough memory
        """
        needed_gb = required_gb * safety_margin
        free_gb = self.get_free_memory_gb()

        if free_gb >= needed_gb:
            logger.info(f"✅ Sufficient GPU memory: {free_gb:.1f}GB free >= {needed_gb:.1f}GB needed")
            return True

        # Need to evict
        deficit_gb = needed_gb - free_gb
        logger.warning(f"⚠️ Need to free {deficit_gb:.1f}GB GPU memory ({free_gb:.1f}GB free, need {needed_gb:.1f}GB)")

        evictable = self.get_evictable_tensors()

        if not evictable:
            logger.error(f"❌ Cannot free memory: no evictable tensors!")
            return False

        # Evict tensors until we have enough space
        freed_gb = 0
        evicted_names = []

        for tensor_info in evictable:
            if free_gb + freed_gb >= needed_gb:
                break  # Have enough space now

            freed_gb += tensor_info.evict_to_cpu()
            evicted_names.append(tensor_info.name)
            self.eviction_history.append((tensor_info.name, time.time()))

        # Check if we freed enough
        final_free_gb = self.get_free_memory_gb()

        if final_free_gb >= needed_gb:
            logger.info(
                f"✅ Freed {freed_gb:.1f}GB by evicting {len(evicted_names)} tensors: {evicted_names}"
            )
            logger.info(f"   GPU memory now: {final_free_gb:.1f}GB free >= {needed_gb:.1f}GB needed")
            return True
        else:
            logger.error(
                f"❌ Still insufficient memory after evicting {len(evicted_names)} tensors: "
                f"{final_free_gb:.1f}GB free < {needed_gb:.1f}GB needed"
            )
            # Restore evicted tensors (failed to make enough space)
            for name in evicted_names:
                if name in self.tracked_tensors:
                    self.tracked_tensors[name].restore_to_gpu()
            return False

    def restore_evicted(self, limit: Optional[int] = None):
        """
        Restore evicted tensors back to GPU.

        Args:
            limit: Max number to restore (None = restore all)
        """
        evicted = [t for t in self.tracked_tensors.values() if not t.is_on_gpu]

        if not evicted:
            return  # Nothing to restore

        # Sort by priority (higher = restore first)
        evicted.sort(key=lambda t: (-t.priority, -t.last_accessed))

        if limit is not None:
            evicted = evicted[:limit]

        for tensor_info in evicted:
            try:
                tensor_info.restore_to_gpu()
            except RuntimeError as e:
                logger.warning(f"Failed to restore {tensor_info.name}: {e}")
                break  # Stop if OOM on restore

    def cleanup(self):
        """Clean up all tracked tensors (called at end of analysis)."""
        for tensor_info in self.tracked_tensors.values():
            if not tensor_info.is_on_gpu and tensor_info.cpu_copy is not None:
                # Restore before cleanup for proper state
                try:
                    tensor_info.restore_to_gpu()
                except:
                    pass  # Ignore errors during cleanup

        self.tracked_tensors.clear()
        self.eviction_history.clear()
        torch.cuda.empty_cache()

    def get_status(self) -> Dict[str, Any]:
        """Get current memory manager status."""
        free_gb = self.get_free_memory_gb()
        total_tracked_gb = sum(t.size_gb for t in self.tracked_tensors.values())
        on_gpu_gb = sum(t.size_gb for t in self.tracked_tensors.values() if t.is_on_gpu)
        on_cpu_gb = total_tracked_gb - on_gpu_gb

        return {
            'free_gpu_gb': free_gb,
            'tracked_tensors': len(self.tracked_tensors),
            'total_tracked_gb': total_tracked_gb,
            'on_gpu_gb': on_gpu_gb,
            'on_cpu_gb': on_cpu_gb,
            'evictions_count': len(self.eviction_history)
        }


# Global memory manager instance
_global_manager: Optional[GPUMemoryManager] = None


def get_memory_manager(device: int = 0) -> GPUMemoryManager:
    """Get or create global memory manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = GPUMemoryManager(device=device)
    return _global_manager


def reset_memory_manager():
    """Reset global memory manager."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.cleanup()
    _global_manager = None