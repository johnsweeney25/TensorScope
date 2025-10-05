"""
Mechanistic Interpretability Analysis Module
============================================

This module contains the core mechanistic interpretability tools with memory-efficient
implementations for ICML submission.

Key components:
- mechanistic_analyzer_core.py: Main analyzer with OOM fixes
- mechanistic_analyzer_unified.py: Unified interface with statistical rigor
- mechanistic_analyzer_improved.py: Enhanced QK-OV analysis

Recent fixes (2025-09-30):
- Fixed CUDA OOM in compute_attention_head_specialization
- Added proper memory cleanup and chunked processing
- Fixed numerical precision issues (epsilon values)
"""

from .mechanistic_analyzer import MechanisticAnalyzer
from .mechanistic_analyzer_unified import UnifiedMechanisticAnalyzer

__all__ = ['MechanisticAnalyzer', 'UnifiedMechanisticAnalyzer']