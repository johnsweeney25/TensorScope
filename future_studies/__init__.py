"""
Future Studies: Advanced Experimental Interventions
====================================================

This module contains experimental techniques for advanced neural network analysis,
including causal interventions, circuit freezing, and mechanistic interpretability.

Modules:
- experimental_interventions: Original head freezing and intervention vectors
- attention_circuit_freezing: QK/OV selective circuit interventions

WARNING: These are experimental techniques that modify model computation.
Always work with model copies and validate results carefully.
"""

from .experimental_interventions import ExperimentalInterventions
from .attention_circuit_freezing import (
    AttentionCircuitFreezer,
    freeze_qk_circuit,
    freeze_ov_circuit,
    CircuitType,
    FreezeType,
    InterventionConfig,
    ModelArchitecture
)

__all__ = [
    # Main classes
    'ExperimentalInterventions',
    'AttentionCircuitFreezer',

    # Convenience functions
    'freeze_qk_circuit',
    'freeze_ov_circuit',

    # Enums and configs
    'CircuitType',
    'FreezeType',
    'InterventionConfig',
    'ModelArchitecture'
]

__version__ = '0.1.0'