#!/usr/bin/env python3
"""
Thread-safe random state management for concurrent metric computation.

This module provides thread-local random number generators to avoid race conditions
when multiple threads are computing metrics simultaneously.
"""

import threading
import numpy as np
import torch
from typing import Optional, Dict, Any
import hashlib


class ThreadSafeRandom:
    """
    Thread-local random state management for safe concurrent operations.

    Each thread gets its own isolated random state, preventing race conditions
    and ensuring reproducibility within each thread.
    """

    _thread_local = threading.local()
    _global_seed = None
    _lock = threading.Lock()

    @classmethod
    def set_global_seed(cls, seed: Optional[int] = None):
        """Set the global seed that will be used to derive thread-specific seeds."""
        with cls._lock:
            cls._global_seed = seed

    @classmethod
    def get_thread_seed(cls, thread_id: Optional[int] = None) -> int:
        """
        Generate a deterministic seed for the current thread.

        Args:
            thread_id: Optional thread identifier. If None, uses current thread ID.

        Returns:
            A deterministic seed based on global seed and thread ID.
        """
        if thread_id is None:
            thread_id = threading.current_thread().ident

        if cls._global_seed is None:
            # No global seed set, use thread ID directly
            return hash(thread_id) % (2**32)

        # Combine global seed with thread ID for deterministic thread-specific seed
        combined = f"{cls._global_seed}_{thread_id}".encode()
        hash_val = hashlib.md5(combined).hexdigest()
        return int(hash_val[:8], 16)

    @classmethod
    def get_numpy_rng(cls, seed: Optional[int] = None) -> np.random.RandomState:
        """
        Get thread-local NumPy random state.

        Args:
            seed: Optional seed. If None, uses thread-specific seed.

        Returns:
            Thread-local NumPy RandomState instance.
        """
        if not hasattr(cls._thread_local, 'np_rng'):
            thread_seed = seed if seed is not None else cls.get_thread_seed()
            cls._thread_local.np_rng = np.random.RandomState(thread_seed)
        elif seed is not None:
            # Re-seed if explicit seed provided
            cls._thread_local.np_rng.seed(seed)

        return cls._thread_local.np_rng

    @classmethod
    def get_torch_generator(cls, device: str = 'cpu', seed: Optional[int] = None) -> torch.Generator:
        """
        Get thread-local PyTorch generator.

        Args:
            device: Device for the generator ('cpu' or 'cuda').
            seed: Optional seed. If None, uses thread-specific seed.

        Returns:
            Thread-local PyTorch Generator instance.
        """
        attr_name = f'torch_gen_{device}'

        if not hasattr(cls._thread_local, attr_name):
            gen = torch.Generator(device=device)
            thread_seed = seed if seed is not None else cls.get_thread_seed()
            gen.manual_seed(thread_seed)
            setattr(cls._thread_local, attr_name, gen)
        elif seed is not None:
            # Re-seed if explicit seed provided
            gen = getattr(cls._thread_local, attr_name)
            gen.manual_seed(seed)
        else:
            gen = getattr(cls._thread_local, attr_name)

        return gen

    @classmethod
    def reset_thread_state(cls):
        """Reset all random state for the current thread."""
        if hasattr(cls._thread_local, 'np_rng'):
            delattr(cls._thread_local, 'np_rng')

        # Remove all torch generators
        for attr in list(vars(cls._thread_local).keys()):
            if attr.startswith('torch_gen_'):
                delattr(cls._thread_local, attr)

    @classmethod
    def get_state_dict(cls) -> Dict[str, Any]:
        """
        Get current thread's random state as a dictionary.

        Returns:
            Dictionary containing all random states for the current thread.
        """
        state = {'thread_id': threading.current_thread().ident}

        if hasattr(cls._thread_local, 'np_rng'):
            state['numpy_state'] = cls._thread_local.np_rng.get_state()

        for attr in vars(cls._thread_local):
            if attr.startswith('torch_gen_'):
                gen = getattr(cls._thread_local, attr)
                state[attr] = gen.get_state()

        return state

    @classmethod
    def set_state_dict(cls, state: Dict[str, Any]):
        """
        Restore thread's random state from a dictionary.

        Args:
            state: State dictionary from get_state_dict().
        """
        if 'numpy_state' in state:
            if not hasattr(cls._thread_local, 'np_rng'):
                cls._thread_local.np_rng = np.random.RandomState()
            cls._thread_local.np_rng.set_state(state['numpy_state'])

        for key, value in state.items():
            if key.startswith('torch_gen_'):
                device = key.replace('torch_gen_', '')
                gen = torch.Generator(device=device)
                gen.set_state(value)
                setattr(cls._thread_local, key, gen)


class RandomContext:
    """
    Context manager for temporary random state changes.

    Usage:
        with RandomContext(seed=42):
            # Operations with seed 42
            pass
        # Original random state restored
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random context.

        Args:
            seed: Seed to use within the context.
        """
        self.seed = seed
        self.saved_state = None

    def __enter__(self):
        """Save current state and set new seed."""
        self.saved_state = ThreadSafeRandom.get_state_dict()

        if self.seed is not None:
            ThreadSafeRandom.reset_thread_state()
            # This will create new RNGs with the specified seed
            ThreadSafeRandom.get_numpy_rng(seed=self.seed)
            ThreadSafeRandom.get_torch_generator(seed=self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original state."""
        if self.saved_state:
            ThreadSafeRandom.set_state_dict(self.saved_state)


def test_thread_safety():
    """Test that different threads get different random sequences."""
    import concurrent.futures

    ThreadSafeRandom.set_global_seed(42)

    def worker(thread_num):
        """Generate random numbers in a thread."""
        rng = ThreadSafeRandom.get_numpy_rng()
        values = [rng.rand() for _ in range(5)]
        return thread_num, values

    # Run in multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, i) for i in range(3)]
        results = [f.result() for f in futures]

    # Check that each thread got different values
    sequences = [r[1] for r in results]
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            assert sequences[i] != sequences[j], "Threads should have different random sequences"

    print("✓ Thread safety test passed")


if __name__ == "__main__":
    test_thread_safety()