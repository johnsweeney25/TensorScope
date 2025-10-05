"""
Backward compatibility wrapper for batch processing modules.
This file maintains compatibility with existing code that imports from batch_processor.

All functionality has been moved to the batch/ directory.
"""

# Import everything from the batch package for backward compatibility
from batch import (
    BatchProcessor,
    BatchConfig,
    ProcessingMode,
    create_batch,
    MultiBatchProcessor,
    create_proper_batch_processor
)

# Re-export for backward compatibility
__all__ = [
    'BatchProcessor',
    'BatchConfig',
    'ProcessingMode',
    'create_batch',
    'MultiBatchProcessor',
    'create_proper_batch_processor'
]

# Deprecation warning (optional)
import warnings
warnings.warn(
    "Importing from batch_processor is deprecated. "
    "Please import from 'batch' package instead: 'from batch import BatchProcessor'",
    DeprecationWarning,
    stacklevel=2
)