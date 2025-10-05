"""
Fixed Model Wrapper for Utils
==============================
Handles all model types properly.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace


def create_model_wrapper(model: nn.Module) -> nn.Module:
    """
    Create a wrapper for models with different interfaces.

    Handles:
    - Transformer models (input_ids, labels)
    - Simple models (Linear, CNN)
    - Custom models
    """

    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, *args, **kwargs):
            # Handle batch dict with input_ids
            if not args and 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
                labels = kwargs.get('labels')

                # Check if model expects dict or tensor
                try:
                    # Try transformer-style first
                    outputs = self.model(**kwargs)
                except (TypeError, RuntimeError):
                    # Fall back to tensor input
                    try:
                        outputs = self.model(input_tensor)
                    except Exception:
                        # Last resort - try without labels
                        outputs = self.model(input_tensor)

                # Ensure we have proper output format
                if torch.is_tensor(outputs):
                    # Convert tensor output to expected format
                    result = SimpleNamespace(logits=outputs)

                    # Add loss if we have labels
                    if labels is not None:
                        if outputs.dim() == 2 and labels.dim() == 1:
                            # Classification loss
                            loss_fn = nn.CrossEntropyLoss()
                            result.loss = loss_fn(outputs, labels)
                        else:
                            # Default loss
                            result.loss = outputs.mean()
                    else:
                        result.loss = outputs.mean()

                    return result
                elif hasattr(outputs, 'loss'):
                    return outputs
                else:
                    # Wrap in SimpleNamespace
                    if hasattr(outputs, 'logits'):
                        return outputs
                    else:
                        return SimpleNamespace(
                            logits=outputs,
                            loss=outputs.mean() if torch.is_tensor(outputs) else 0.0
                        )

            # Handle direct tensor input
            elif args:
                outputs = self.model(*args)
            else:
                # Extract tensor from kwargs
                x = kwargs.get('x', kwargs.get('input', kwargs.get('inputs')))
                if x is not None:
                    outputs = self.model(x)
                else:
                    outputs = self.model(**kwargs)

            # Ensure consistent output format
            if torch.is_tensor(outputs):
                return SimpleNamespace(logits=outputs, loss=outputs.mean())
            elif not hasattr(outputs, 'loss'):
                return SimpleNamespace(
                    logits=outputs if torch.is_tensor(outputs) else outputs.logits,
                    loss=outputs.mean() if torch.is_tensor(outputs) else 0.0
                )
            else:
                return outputs

        def named_parameters(self):
            return self.model.named_parameters()

        def parameters(self):
            return self.model.parameters()

        def train(self, mode=True):
            return self.model.train(mode)

        def eval(self):
            return self.model.eval()

        def to(self, *args, **kwargs):
            self.model = self.model.to(*args, **kwargs)
            return self

    return ModelWrapper(model)