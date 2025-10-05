#!/usr/bin/env python3
"""
TensorScope Exception Classes
============================

Proper exception hierarchy for the TensorScope framework.
Replaces silent error dictionary returns with explicit exceptions.

Author: TensorScope Team
Date: January 2025
"""


class TensorScopeError(Exception):
    """Base exception for all TensorScope errors."""
    pass


class MetricComputationError(TensorScopeError):
    """Raised when a metric fails to compute."""

    def __init__(self, metric_name: str, reason: str, details: dict = None):
        self.metric_name = metric_name
        self.reason = reason
        self.details = details or {}
        super().__init__(f"Metric '{metric_name}' failed: {reason}")


class InsufficientDataError(TensorScopeError):
    """Raised when there's not enough data for statistical validity."""

    def __init__(self, metric_name: str, required: int, provided: int):
        self.metric_name = metric_name
        self.required = required
        self.provided = provided
        super().__init__(
            f"Metric '{metric_name}' requires {required} samples for statistical validity, "
            f"but only {provided} were provided"
        )


class ModelCompatibilityError(TensorScopeError):
    """Raised when a model doesn't support required operations."""

    def __init__(self, model_type: str, operation: str, suggestion: str = None):
        self.model_type = model_type
        self.operation = operation
        self.suggestion = suggestion
        msg = f"Model type '{model_type}' doesn't support {operation}"
        if suggestion:
            msg += f". {suggestion}"
        super().__init__(msg)


class MemoryError(TensorScopeError):
    """Raised when GPU memory is insufficient."""

    def __init__(self, metric_name: str, required_gb: float, available_gb: float, suggestion: str = None):
        self.metric_name = metric_name
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.suggestion = suggestion
        msg = (f"Metric '{metric_name}' requires ~{required_gb:.1f}GB GPU memory, "
               f"but only {available_gb:.1f}GB available")
        if suggestion:
            msg += f". {suggestion}"
        super().__init__(msg)


class ConfigurationError(TensorScopeError):
    """Raised when configuration is invalid or incomplete."""

    def __init__(self, parameter: str, issue: str, valid_values: list = None):
        self.parameter = parameter
        self.issue = issue
        self.valid_values = valid_values
        msg = f"Configuration error for '{parameter}': {issue}"
        if valid_values:
            msg += f". Valid values: {valid_values}"
        super().__init__(msg)


class RequirementError(TensorScopeError):
    """Raised when a metric's requirements aren't met."""

    def __init__(self, metric_name: str, missing_requirement: str, how_to_provide: str = None):
        self.metric_name = metric_name
        self.missing_requirement = missing_requirement
        self.how_to_provide = how_to_provide
        msg = f"Metric '{metric_name}' requires {missing_requirement}"
        if how_to_provide:
            msg += f". {how_to_provide}"
        super().__init__(msg)


class NumericalInstabilityError(TensorScopeError):
    """Raised when numerical computation fails (NaN, inf, convergence)."""

    def __init__(self, metric_name: str, issue: str, values: dict = None):
        self.metric_name = metric_name
        self.issue = issue
        self.values = values or {}
        msg = f"Numerical instability in '{metric_name}': {issue}"
        if values:
            msg += f" (values: {values})"
        super().__init__(msg)


class StatisticalValidityError(TensorScopeError):
    """Raised when statistical requirements aren't met."""

    def __init__(self, metric_name: str, issue: str, power: float = None, min_samples: int = None):
        self.metric_name = metric_name
        self.issue = issue
        self.power = power
        self.min_samples = min_samples
        msg = f"Statistical validity issue for '{metric_name}': {issue}"
        if power is not None:
            msg += f" (power={power:.2f}, minimum 0.80 required)"
        if min_samples is not None:
            msg += f" (minimum {min_samples} samples needed)"
        super().__init__(msg)


class CheckpointError(TensorScopeError):
    """Raised when checkpoint operations fail."""

    def __init__(self, operation: str, checkpoint_path: str, reason: str):
        self.operation = operation
        self.checkpoint_path = checkpoint_path
        self.reason = reason
        super().__init__(f"Checkpoint {operation} failed for '{checkpoint_path}': {reason}")


class DataFormatError(TensorScopeError):
    """Raised when input data has incorrect format."""

    def __init__(self, expected_format: str, received_format: str, field: str = None):
        self.expected_format = expected_format
        self.received_format = received_format
        self.field = field
        msg = f"Expected {expected_format} but received {received_format}"
        if field:
            msg = f"Field '{field}': " + msg
        super().__init__(msg)


# Utility functions for error handling

def convert_error_dict_to_exception(error_dict: dict, metric_name: str = "unknown"):
    """
    Convert legacy error dictionaries to proper exceptions.

    This helps transition from the old error handling to the new exception-based system.
    """
    if not isinstance(error_dict, dict) or 'error' not in error_dict:
        return None

    error_msg = error_dict.get('error', 'Unknown error')

    # Pattern matching for different error types
    if 'OOM' in error_msg or 'out of memory' in error_msg.lower():
        # Try to extract memory info
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*GB', error_msg)
        required = float(match.group(1)) if match else None
        return MemoryError(metric_name, required or 0, 0, error_msg)

    elif 'insufficient' in error_msg.lower() and 'sample' in error_msg.lower():
        # Try to extract sample counts
        import re
        numbers = re.findall(r'\d+', error_msg)
        if len(numbers) >= 2:
            return InsufficientDataError(metric_name, int(numbers[0]), int(numbers[1]))
        return InsufficientDataError(metric_name, 0, 0)

    elif 'nan' in error_msg.lower() or 'inf' in error_msg.lower():
        return NumericalInstabilityError(metric_name, error_msg)

    elif 'not supported' in error_msg.lower() or 'doesn\'t support' in error_msg.lower():
        return ModelCompatibilityError("unknown", error_msg)

    elif 'statistical' in error_msg.lower() or 'power' in error_msg.lower():
        return StatisticalValidityError(metric_name, error_msg)

    elif 'require' in error_msg.lower():
        return RequirementError(metric_name, error_msg)

    else:
        # Generic metric computation error
        return MetricComputationError(metric_name, error_msg, error_dict)


def safe_mode_wrapper(func):
    """
    Decorator to provide backward compatibility with error dictionaries.

    Usage:
        @safe_mode_wrapper
        def some_metric_function(...):
            # May raise exceptions

    If safe_mode=True is in kwargs, exceptions are caught and returned as dicts.
    """
    def wrapper(*args, **kwargs):
        safe_mode = kwargs.pop('safe_mode', False)

        if not safe_mode:
            # Normal mode - let exceptions propagate
            return func(*args, **kwargs)

        # Safe mode - catch exceptions and return as dict
        try:
            return func(*args, **kwargs)
        except TensorScopeError as e:
            # Convert our exceptions to error dict
            error_dict = {'error': str(e)}
            if hasattr(e, 'details'):
                error_dict.update(e.details)
            return error_dict
        except Exception as e:
            # Catch any other exceptions too
            return {'error': str(e), 'exception_type': type(e).__name__}

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper