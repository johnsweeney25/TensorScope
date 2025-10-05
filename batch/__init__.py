"""
Batch processing utilities for memory-efficient computation.

This package provides:
- BatchProcessor: Core batch processing with memory management
- BatchConfig: Configuration for batch processing
- ProcessingMode: Processing mode enumerations
- MultiBatchProcessor: Extended processor for handling multiple batches
- create_batch: Utility function for batch creation
"""

# Core processor imports
from .processor import (
    BatchProcessor,
    BatchConfig,
    ProcessingMode,
    create_batch
)

# Integration utilities
from .integration import (
    MultiBatchProcessor,
    create_proper_batch_processor
)

__all__ = [
    # Core components
    'BatchProcessor',
    'BatchConfig',
    'ProcessingMode',
    'create_batch',
    # Integration components
    'MultiBatchProcessor',
    'create_proper_batch_processor'
]

# Version info
__version__ = '1.0.0'