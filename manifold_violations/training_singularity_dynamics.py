#!/usr/bin/env python3
"""
Training Singularity Dynamics - POWERFUL EXTENSION

Track how embedding singularities evolve during model training.
This is a NOVEL contribution beyond Robinson et al., enabling research into
the relationship between embedding geometry and training stability.

Key Research Questions:
1. Do singularities emerge gradually or suddenly?
2. Can singularity count at epoch N predict final performance?
3. Do successful vs failed runs show different patterns?
4. Which tokens become unstable first during fine-tuning?

This could be a KEY DIFFERENTIATOR for ICLR submission.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from .robinson_fiber_bundle_test import RobinsonFiberBundleTest
from .polysemy_detector import PolysemyDetector
from .singularity_mapper import SingularityMapper


@dataclass
class SingularitySnapshot:
    """Singularity state at a specific training step."""
    step: int
    epoch: float
    total_singularities: int
    new_singularities: List[int]  # Token indices that became singular
    resolved_singularities: List[int]  # Tokens that stopped being singular
    severity_distribution: Dict[str, int]  # mild/moderate/severe counts
    polysemy_rate: float
    avg_local_signal_dim: float
    high_risk_tokens: List[int]

    # Training metrics at this step
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    gradient_norm: Optional[float] = None


@dataclass
class SingularityEvolution:
    """Complete evolution of singularities across training."""
    snapshots: List[SingularitySnapshot]
    token_histories: Dict[int, List[bool]]  # token_id -> [is_singular at each step]
    emergence_points: Dict[int, int]  # token_id -> step when became singular

    # Patterns
    total_created: int
    total_resolved: int
    max_singularities: int
    volatility_score: float  # How much churn in singularities

    # Correlations with training
    correlation_with_loss: float
    correlation_with_gradients: float
    early_warning_score: float  # Do singularities predict future problems?


class TrainingSingularityTracker:
    """
    Track embedding singularities throughout training.

    NOVEL CONTRIBUTIONS:
    1. First systematic study of singularity evolution
    2. Early warning system for training instability
    3. Correlation with training dynamics
    4. Token-level stability tracking
    """

    def __init__(
        self,
        singularity_threshold: float = 0.5,
        track_individual_tokens: bool = True,
        save_snapshots: bool = True,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize tracker.

        Args:
            singularity_threshold: Threshold for considering a token singular
            track_individual_tokens: Track per-token evolution
            save_snapshots: Save snapshots to disk
            output_dir: Directory for saving results
        """
        self.singularity_threshold = singularity_threshold
        self.track_individual_tokens = track_individual_tokens
        self.save_snapshots = save_snapshots
        self.output_dir = Path(output_dir) if output_dir else Path("singularity_tracking")

        # Initialize detectors
        self.mapper = SingularityMapper()
        self.robinson_test = RobinsonFiberBundleTest()
        self.polysemy_detector = PolysemyDetector()

        # Tracking state
        self.snapshots = []
        self.token_histories = defaultdict(list)
        self.previous_singular = set()

    def analyze_checkpoint(
        self,
        embeddings: torch.Tensor,
        step: int,
        epoch: float,
        training_metrics: Optional[Dict[str, float]] = None
    ) -> SingularitySnapshot:
        """
        Analyze singularities at a training checkpoint.

        Args:
            embeddings: Token embeddings at this checkpoint
            step: Training step
            epoch: Training epoch
            training_metrics: Optional metrics (loss, accuracy, etc.)

        Returns:
            Snapshot of singularity state
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        n_tokens = len(embeddings)

        # Identify singular tokens
        singular_tokens = set()
        severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        high_risk = []
        local_dims = []

        # Sample tokens for efficiency (can be made exhaustive)
        sample_size = min(1000, n_tokens)
        token_indices = np.random.choice(n_tokens, sample_size, replace=False)

        for idx in token_indices:
            # Get singularity profile
            profile = self.mapper.map_singularity(embeddings, idx)

            # Check if singular
            if profile.severity != 'none':
                singular_tokens.add(idx)
                severity_counts[profile.severity] += 1

                if profile.risk_level in ['high', 'critical']:
                    high_risk.append(idx)

            local_dims.append(profile.local_signal_dimension)

            # Track individual token history
            if self.track_individual_tokens:
                is_singular = profile.severity != 'none'
                self.token_histories[idx].append(is_singular)

        # Compute polysemy rate
        polysemy_analysis = self.polysemy_detector.analyze_vocabulary(
            embeddings, sample_size=min(500, n_tokens), verbose=False
        )

        # Identify changes from previous snapshot
        current_singular = singular_tokens
        new_singularities = list(current_singular - self.previous_singular)
        resolved_singularities = list(self.previous_singular - current_singular)

        # Create snapshot
        snapshot = SingularitySnapshot(
            step=step,
            epoch=epoch,
            total_singularities=len(singular_tokens),
            new_singularities=new_singularities,
            resolved_singularities=resolved_singularities,
            severity_distribution=severity_counts,
            polysemy_rate=polysemy_analysis.polysemy_rate,
            avg_local_signal_dim=np.mean(local_dims),
            high_risk_tokens=high_risk,
            loss=training_metrics.get('loss') if training_metrics else None,
            accuracy=training_metrics.get('accuracy') if training_metrics else None,
            gradient_norm=training_metrics.get('gradient_norm') if training_metrics else None
        )

        # Update state
        self.snapshots.append(snapshot)
        self.previous_singular = current_singular

        # Save if requested
        if self.save_snapshots:
            self._save_snapshot(snapshot)

        return snapshot

    def track_training_run(
        self,
        checkpoints: List[Dict[str, Any]],
        verbose: bool = True
    ) -> SingularityEvolution:
        """
        Track singularities across entire training run.

        Args:
            checkpoints: List of checkpoint dicts with 'embeddings', 'step', 'epoch', 'metrics'
            verbose: Print progress

        Returns:
            Complete evolution analysis
        """
        for i, checkpoint in enumerate(checkpoints):
            if verbose:
                print(f"Analyzing checkpoint {i+1}/{len(checkpoints)} (step {checkpoint['step']})")

            snapshot = self.analyze_checkpoint(
                checkpoint['embeddings'],
                checkpoint['step'],
                checkpoint['epoch'],
                checkpoint.get('metrics')
            )

            if verbose:
                print(f"  Singularities: {snapshot.total_singularities}")
                print(f"  New: {len(snapshot.new_singularities)}, Resolved: {len(snapshot.resolved_singularities)}")

        # Compute evolution statistics
        evolution = self._compute_evolution_statistics()

        if verbose:
            self._print_evolution_summary(evolution)

        return evolution

    def _compute_evolution_statistics(self) -> SingularityEvolution:
        """
        Compute statistics about singularity evolution.
        """
        if not self.snapshots:
            return None

        # Track emergence points
        emergence_points = {}
        for token_id, history in self.token_histories.items():
            for i, is_singular in enumerate(history):
                if is_singular and (i == 0 or not history[i-1]):
                    emergence_points[token_id] = self.snapshots[i].step
                    break

        # Compute volatility (how much churn)
        total_changes = sum(
            len(s.new_singularities) + len(s.resolved_singularities)
            for s in self.snapshots
        )
        volatility = total_changes / (len(self.snapshots) * len(self.token_histories))

        # Compute correlations
        if any(s.loss is not None for s in self.snapshots):
            losses = [s.loss for s in self.snapshots if s.loss is not None]
            singularity_counts = [s.total_singularities for s in self.snapshots if s.loss is not None]

            if len(losses) > 1:
                correlation_with_loss = np.corrcoef(singularity_counts, losses)[0, 1]
            else:
                correlation_with_loss = 0.0
        else:
            correlation_with_loss = 0.0

        # Early warning: do singularities at step N predict problems at N+k?
        early_warning = self._compute_early_warning_score()

        return SingularityEvolution(
            snapshots=self.snapshots,
            token_histories=dict(self.token_histories),
            emergence_points=emergence_points,
            total_created=sum(len(s.new_singularities) for s in self.snapshots),
            total_resolved=sum(len(s.resolved_singularities) for s in self.snapshots),
            max_singularities=max(s.total_singularities for s in self.snapshots),
            volatility_score=volatility,
            correlation_with_loss=correlation_with_loss,
            correlation_with_gradients=0.0,  # TODO: Implement
            early_warning_score=early_warning
        )

    def _compute_early_warning_score(self, lookahead: int = 5) -> float:
        """
        Compute how well singularities predict future problems.
        """
        if len(self.snapshots) < lookahead + 1:
            return 0.0

        # Check if singularity increases predict loss increases
        warnings = []
        for i in range(len(self.snapshots) - lookahead):
            current = self.snapshots[i]
            future = self.snapshots[i + lookahead]

            if current.loss is not None and future.loss is not None:
                sing_increase = future.total_singularities > current.total_singularities
                loss_increase = future.loss > current.loss

                # Good warning if both increase or both decrease
                warnings.append(sing_increase == loss_increase)

        return np.mean(warnings) if warnings else 0.5

    def find_critical_tokens(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find tokens that are most often singular (potentially problematic).

        Returns:
            List of (token_id, singularity_rate) tuples
        """
        singularity_rates = {}

        for token_id, history in self.token_histories.items():
            if history:
                rate = sum(history) / len(history)
                singularity_rates[token_id] = rate

        # Sort by rate
        critical = sorted(
            singularity_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return critical

    def visualize_evolution(
        self,
        save_path: Optional[Path] = None,
        show_loss: bool = True
    ):
        """
        Visualize singularity evolution.
        """
        if not self.snapshots:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        steps = [s.step for s in self.snapshots]

        # Total singularities over time
        ax = axes[0, 0]
        singularities = [s.total_singularities for s in self.snapshots]
        ax.plot(steps, singularities, 'b-', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Total Singularities')
        ax.set_title('Singularity Count Evolution')
        ax.grid(True, alpha=0.3)

        # Loss overlay if available
        if show_loss and any(s.loss is not None for s in self.snapshots):
            ax2 = ax.twinx()
            losses = [s.loss if s.loss is not None else np.nan for s in self.snapshots]
            ax2.plot(steps, losses, 'r--', alpha=0.7, label='Loss')
            ax2.set_ylabel('Loss', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

        # New vs resolved
        ax = axes[0, 1]
        new = [len(s.new_singularities) for s in self.snapshots]
        resolved = [len(s.resolved_singularities) for s in self.snapshots]
        ax.bar(steps, new, alpha=0.7, label='New', color='red')
        ax.bar(steps, resolved, alpha=0.7, label='Resolved', color='green')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Count')
        ax.set_title('Singularity Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Severity distribution
        ax = axes[1, 0]
        mild = [s.severity_distribution['mild'] for s in self.snapshots]
        moderate = [s.severity_distribution['moderate'] for s in self.snapshots]
        severe = [s.severity_distribution['severe'] for s in self.snapshots]

        ax.stackplot(steps, mild, moderate, severe,
                    labels=['Mild', 'Moderate', 'Severe'],
                    colors=['yellow', 'orange', 'red'],
                    alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Count')
        ax.set_title('Severity Distribution')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Average local dimension
        ax = axes[1, 1]
        local_dims = [s.avg_local_signal_dim for s in self.snapshots]
        ax.plot(steps, local_dims, 'g-', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Avg Local Signal Dimension')
        ax.set_title('Semantic Flexibility Evolution')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Embedding Singularity Evolution During Training')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def _save_snapshot(self, snapshot: SingularitySnapshot):
        """Save snapshot to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        filename = self.output_dir / f"snapshot_step_{snapshot.step}.json"

        # Convert to serializable format
        data = {
            'step': snapshot.step,
            'epoch': snapshot.epoch,
            'total_singularities': snapshot.total_singularities,
            'new_singularities': snapshot.new_singularities,
            'resolved_singularities': snapshot.resolved_singularities,
            'severity_distribution': snapshot.severity_distribution,
            'polysemy_rate': snapshot.polysemy_rate,
            'avg_local_signal_dim': snapshot.avg_local_signal_dim,
            'high_risk_tokens': snapshot.high_risk_tokens,
            'loss': snapshot.loss,
            'accuracy': snapshot.accuracy,
            'gradient_norm': snapshot.gradient_norm
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _print_evolution_summary(self, evolution: SingularityEvolution):
        """Print summary of evolution."""
        print("\n" + "="*60)
        print("SINGULARITY EVOLUTION SUMMARY")
        print("="*60)
        print(f"Total snapshots analyzed: {len(evolution.snapshots)}")
        print(f"Tokens tracked: {len(evolution.token_histories)}")
        print(f"\nDynamics:")
        print(f"  Total created: {evolution.total_created}")
        print(f"  Total resolved: {evolution.total_resolved}")
        print(f"  Max singularities: {evolution.max_singularities}")
        print(f"  Volatility score: {evolution.volatility_score:.3f}")
        print(f"\nCorrelations:")
        print(f"  With loss: {evolution.correlation_with_loss:.3f}")
        print(f"  Early warning score: {evolution.early_warning_score:.3f}")

        # Find critical tokens
        critical = self.find_critical_tokens(5)
        if critical:
            print(f"\nMost frequently singular tokens:")
            for token_id, rate in critical:
                print(f"  Token {token_id}: {rate:.1%} of training")


def correlate_with_training_outcomes(
    evolution: SingularityEvolution,
    final_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Correlate singularity patterns with final training outcomes.

    RESEARCH QUESTION: Do singularity patterns predict success?
    """
    correlations = {}

    # Early singularities vs final performance
    early_snapshots = evolution.snapshots[:len(evolution.snapshots)//3]
    early_avg = np.mean([s.total_singularities for s in early_snapshots])

    # More research-oriented correlations would go here
    # This is a framework for the actual experiments

    return {
        'early_singularities': early_avg,
        'max_singularities': evolution.max_singularities,
        'volatility': evolution.volatility_score,
        'final_accuracy': final_metrics.get('accuracy', 0),
        'training_success': final_metrics.get('success', False)
    }


if __name__ == "__main__":
    # Example usage
    print("Training Singularity Dynamics - Example")
    print("="*60)

    # Simulate training checkpoints
    np.random.seed(42)
    vocab_size = 1000
    embed_dim = 128

    checkpoints = []
    base_embeddings = np.random.randn(vocab_size, embed_dim)

    for step in range(0, 1000, 100):
        # Simulate embedding evolution
        embeddings = base_embeddings + np.random.randn(vocab_size, embed_dim) * 0.1

        # Add some singularities over time
        if step > 300:
            embeddings[50:55] *= 2  # Create singularities
        if step > 600:
            embeddings[100:110] *= 3  # More singularities

        checkpoint = {
            'embeddings': embeddings,
            'step': step,
            'epoch': step / 100,
            'metrics': {
                'loss': 2.0 - step/500 + np.random.randn() * 0.1,
                'accuracy': min(0.9, step/1000 + np.random.randn() * 0.05)
            }
        }
        checkpoints.append(checkpoint)

    # Track evolution
    tracker = TrainingSingularityTracker()
    evolution = tracker.track_training_run(checkpoints, verbose=True)

    # Visualize
    tracker.visualize_evolution()

    # Find critical tokens
    critical = tracker.find_critical_tokens(5)
    print(f"\nCritical tokens: {critical}")

    print("\nThis framework enables studying how embedding geometry evolves during training!")
    print("Key research questions to explore:")
    print("1. Do singularities emerge before loss spikes?")
    print("2. Are successful runs characterized by fewer singularities?")
    print("3. Can we predict training collapse from singularity patterns?")
    print("4. Do fine-tuning and pretraining show different patterns?")