#!/usr/bin/env python3
"""
Verify that the Fisher EMA computation fix works correctly.
"""

import torch
import torch.nn as nn
import logging
from fisher_collector import FisherCollector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 100)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        x = x.mean(dim=1)  # Simple pooling
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Create simple labels for testing
            simple_labels = labels[:, 0] if labels.dim() > 1 else labels
            loss = loss_fct(logits, simple_labels)

        class Output:
            pass

        output = Output()
        output.loss = loss
        output.logits = logits
        return output

def main():
    logger.info("Starting Fisher EMA computation verification...")

    # Create model and Fisher collector
    model = TestModel()
    fisher_collector = FisherCollector(reduction='group', storage='cpu_fp16')

    # Create test batches
    batch1 = {
        'input_ids': torch.randint(0, 100, (4, 10)),
        'attention_mask': torch.ones(4, 10),
        'labels': torch.randint(0, 100, (4,))
    }

    batch2 = {
        'input_ids': torch.randint(0, 100, (4, 10)),
        'attention_mask': torch.ones(4, 10),
        'labels': torch.randint(0, 100, (4,))
    }

    # Test without gradients enabled (should warn)
    logger.info("\n=== Testing with frozen parameters (should warn) ===")
    for param in model.parameters():
        param.requires_grad_(False)

    fisher_collector.update_fisher_ema(model, batch1, task='task1')

    # Test with gradients enabled (should work)
    logger.info("\n=== Testing with gradients enabled (should work) ===")
    for param in model.parameters():
        param.requires_grad_(True)

    logger.info("Computing Fisher EMA for task1...")
    fisher_collector.update_fisher_ema(model, batch1, task='task1')

    logger.info("Computing Fisher EMA for task2...")
    fisher_collector.update_fisher_ema(model, batch2, task='task2')

    # Check stored Fisher values
    task1_keys = [k for k in fisher_collector.fisher_ema.keys() if k.startswith('task1|')]
    task2_keys = [k for k in fisher_collector.fisher_ema.keys() if k.startswith('task2|')]

    logger.info(f"\n=== Results ===")
    logger.info(f"Fisher keys for task1: {len(task1_keys)}")
    logger.info(f"Fisher keys for task2: {len(task2_keys)}")

    if task1_keys:
        logger.info("Sample task1 keys:")
        for key in list(task1_keys)[:3]:
            logger.info(f"  - {key}")

    if task2_keys:
        logger.info("Sample task2 keys:")
        for key in list(task2_keys)[:3]:
            logger.info(f"  - {key}")

    if task1_keys and task2_keys:
        logger.info("\n✅ Fisher EMA computation fix verified successfully!")
        return True
    else:
        logger.error("\n❌ Fisher EMA computation still not working!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)