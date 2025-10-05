#!/usr/bin/env python3
"""
Test script to verify the CUDA fix for compute_fisher_weighted_damage.
Tests with Qwen2.5-Math-1.5B model to ensure no CUDA device-side asserts.
"""

import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from ModularityMetrics import ExtendedModularityMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_batch(tokenizer, text, batch_size=256, device='cuda'):
    """Create a test batch with proper tokenization."""
    # Tokenize with padding and truncation
    encoded = tokenizer(
        [text] * batch_size,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Move to device
    batch = {k: v.to(device) for k, v in encoded.items()}
    return batch


def test_with_real_model():
    """Test with actual Qwen2.5-Math-1.5B model."""
    logger.info("Loading Qwen2.5-Math-1.5B model...")

    try:
        # Load model and tokenizer
        model_name = "Qwen/Qwen2.5-Math-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
        logger.info(f"Model vocab size: {model.config.vocab_size}")
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

        # Create test batches
        logger.info("Creating test batches...")
        task_A_text = "Solve the equation: 2x + 5 = 13. What is the value of x?"
        task_B_text = "Write a Python function to calculate the factorial of a number."

        task_A_batch = create_test_batch(tokenizer, task_A_text, batch_size=64, device=device)
        task_B_batch = create_test_batch(tokenizer, task_B_text, batch_size=64, device=device)

        # Verify batch sizes
        logger.info(f"Task A batch shape: {task_A_batch['input_ids'].shape}")
        logger.info(f"Task B batch shape: {task_B_batch['input_ids'].shape}")

        # Initialize metrics
        metrics = ExtendedModularityMetrics()

        # Test 1: Standard test with smaller batch
        logger.info("\n=== Test 1: Standard compute_fisher_weighted_damage ===")
        try:
            result = metrics.compute_fisher_weighted_damage(
                model,
                task_A_batch,
                task_B_batch,
                n_fisher_samples=4
            )

            if 'error' in result:
                logger.error(f"FAILED: Got error in result: {result['error']}")
                return False
            else:
                logger.info(f"SUCCESS: Damage computed = {result.get('damage_A_from_B', 'N/A'):.6f}")
                logger.info(f"  Total raw damage: {result.get('total_damage_raw', 'N/A'):.6f}")
                logger.info(f"  Parameter count: {result.get('param_count', 'N/A')}")

        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                logger.error(f"FAILED: CUDA error not handled properly: {e}")
                return False
            else:
                raise

        # Test 2: Test with deliberately large token IDs (should be clamped)
        logger.info("\n=== Test 2: Test with out-of-bounds tokens (should be clamped) ===")

        # Create batch with large token IDs
        bad_batch = task_A_batch.copy()
        bad_batch['input_ids'] = bad_batch['input_ids'].clone()
        # Inject some out-of-bounds token IDs
        bad_batch['input_ids'][0, 0] = model.config.vocab_size + 100
        bad_batch['input_ids'][1, 1] = model.config.vocab_size + 50

        try:
            result = metrics.compute_fisher_weighted_damage(
                model,
                bad_batch,
                task_B_batch,
                n_fisher_samples=2
            )

            if 'error' in result:
                logger.info(f"Got expected error handling: {result['error']}")
            else:
                logger.info(f"SUCCESS: Handled out-of-bounds tokens gracefully")
                logger.info(f"  Damage computed = {result.get('damage_A_from_B', 'N/A'):.6f}")

        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                logger.error(f"FAILED: CUDA error not properly handled with bad tokens: {e}")
                return False
            else:
                raise

        # Test 3: Test with asymmetric damage computation
        logger.info("\n=== Test 3: Test asymmetric damage computation ===")
        try:
            result = metrics.compute_fisher_damage_with_asymmetry(
                model,
                task_A_batch,
                task_B_batch,
                n_fisher_samples=2
            )

            if isinstance(result.get('damage_math_from_general'), (int, float)) and not np.isnan(result['damage_math_from_general']):
                logger.info(f"SUCCESS: Asymmetric damage computed")
                logger.info(f"  Math from General: {result['damage_math_from_general']:.6f}")
                logger.info(f"  General from Math: {result['damage_general_from_math']:.6f}")
                logger.info(f"  Asymmetry: {result['damage_asymmetry']:.6f}")
            else:
                logger.warning("Got NaN values but no CUDA errors - acceptable")

        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                logger.error(f"FAILED: CUDA error in asymmetric computation: {e}")
                return False
            else:
                raise

        logger.info("\n" + "="*50)
        logger.info("ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("The CUDA fix is working correctly.")
        return True

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_mock_model():
    """Quick test with a mock model for development."""
    logger.info("Testing with mock model...")

    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size=100):
            super().__init__()
            self.embeddings = torch.nn.Embedding(vocab_size, 64)
            self.linear = torch.nn.Linear(64, vocab_size)
            self.config = type('Config', (), {'vocab_size': vocab_size})()

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            x = self.embeddings(input_ids)
            logits = self.linear(x.mean(1))

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                # Only compute loss on valid labels (not -100)
                valid_mask = labels != -100
                if valid_mask.any():
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]
                    loss = loss_fct(valid_logits, valid_labels)
                else:
                    loss = torch.tensor(0.0, requires_grad=True)

            return type('Output', (), {'loss': loss, 'logits': logits})()

    # Create mock model and batches
    model = MockModel(vocab_size=100).cuda() if torch.cuda.is_available() else MockModel()
    device = next(model.parameters()).device

    batch_A = {
        'input_ids': torch.randint(0, 100, (32, 10)).to(device),
        'attention_mask': torch.ones(32, 10).to(device)
    }
    batch_B = {
        'input_ids': torch.randint(0, 100, (32, 10)).to(device),
        'attention_mask': torch.ones(32, 10).to(device)
    }

    # Add some out-of-bounds tokens
    batch_A['input_ids'][0, 0] = 150  # Out of bounds

    metrics = ExtendedModularityMetrics()

    try:
        result = metrics.compute_fisher_weighted_damage(model, batch_A, batch_B, n_fisher_samples=2)
        if 'error' not in result:
            logger.info(f"Mock test passed: damage = {result.get('damage_A_from_B', 'N/A')}")
            return True
        else:
            logger.warning(f"Mock test got error (expected): {result['error']}")
            return True
    except Exception as e:
        logger.error(f"Mock test failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    # First run quick mock test
    logger.info("Running mock model test first...")
    if not test_with_mock_model():
        logger.error("Mock test failed!")
        sys.exit(1)

    # Then run full test if CUDA is available
    if torch.cuda.is_available():
        logger.info("\nRunning full test with real model...")
        success = test_with_real_model()
        sys.exit(0 if success else 1)
    else:
        logger.info("CUDA not available, skipping real model test")
        sys.exit(0)