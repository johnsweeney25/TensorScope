#!/usr/bin/env python3
"""
Using Intervention Capabilities for Checkpoint Analysis

When you have multiple checkpoints that SHOULD be similar (e.g., different
random seeds, slight variations in training), these intervention tools
become powerful diagnostics.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

from BombshellMetrics import BombshellMetrics
try:
    from mechanistic.mechanistic_analyzer import MechanisticAnalyzer
except ImportError:
    from mechanistic.mechanistic_analyzer_core import MechanisticAnalyzer
from InformationTheoryMetrics import InformationTheoryMetrics
# from main_qwen_analysis import QwenAnalyzer  # Deprecated - use UnifiedModelAnalyzer instead

logger = logging.getLogger(__name__)


class CheckpointInterventionAnalyzer:
    """
    Analyze drift and differences between supposedly similar checkpoints
    using intervention capabilities.
    """

    def __init__(self, bombshell=None, mechanistic=None, information=None):
        """
        Initialize with optional shared metric instances to avoid duplicate initialization.

        Args:
            bombshell: Optional BombshellMetrics instance to use
            mechanistic: Optional MechanisticAnalyzer instance to use
            information: Optional InformationTheoryMetrics instance to use
        """
        self.bombshell = bombshell if bombshell is not None else BombshellMetrics()
        self.mechanistic = mechanistic if mechanistic is not None else MechanisticAnalyzer()
        self.information = information if information is not None else InformationTheoryMetrics()
        # self.analyzer = QwenAnalyzer()  # Deprecated
        self.analyzer = None  # Will use metrics directly
        
    def analyze_checkpoint_drift(
        self,
        checkpoint_paths: List[str],
        reference_idx: int = 0,
        test_batch: Dict[str, torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Find intervention vectors between checkpoints that should be similar.
        
        This reveals:
        - Which components diverged
        - Whether divergence is significant
        - Corrective directions to align models
        """
        logger.info(f"Analyzing drift across {len(checkpoint_paths)} checkpoints")
        
        # Load reference model
        reference_model = self.analyzer.load_model(checkpoint_paths[reference_idx])
        
        if test_batch is None:
            test_batch = self.analyzer.create_test_batch()
            
        results = {
            'reference': checkpoint_paths[reference_idx],
            'drift_analysis': [],
            'intervention_vectors': [],
            'significance_tests': []
        }
        
        for idx, checkpoint_path in enumerate(checkpoint_paths):
            if idx == reference_idx:
                continue
                
            logger.info(f"Comparing checkpoint {idx} to reference")
            
            # Load comparison model
            model = self.analyzer.load_model(checkpoint_path)
            
            # 1. Find intervention vectors (what changed)
            intervention_vectors = self.bombshell.find_intervention_vectors_enhanced(
                model_before=reference_model,
                model_after=model,
                test_samples=test_batch,
                n_samples=32
            )
            
            # 2. Compute null space projection (what SHOULD be preserved)
            null_projection = self.bombshell.compute_null_space_projection(
                model=model,
                protected_samples=test_batch,  # What to preserve
                update_samples=test_batch,     # Same data = find invariants
                fisher_subsample_size=100
            )
            
            # 3. Identify causal importance of differences
            important_components = self.information.compute_causal_necessity(
                model=model,
                test_batch=test_batch,
                components_to_ablate=['attention', 'mlp'],  # Test major components
                n_bootstrap=100
            )
            
            # 4. Validate differences with activation patching
            if intervention_vectors and 'significant_weights' in intervention_vectors:
                # Test if the differences actually matter
                patching_result = self.mechanistic.validate_with_activation_patching(
                    model=model,
                    hypothesis={
                        'description': 'Weight differences affect behavior',
                        'layers': intervention_vectors['significant_weights'][:5]  # Top 5
                    },
                    source_inputs=test_batch,
                    target_inputs=test_batch,
                    metric='loss'
                )
            else:
                patching_result = None
                
            checkpoint_result = {
                'checkpoint': checkpoint_path,
                'index': idx,
                'intervention_vectors': intervention_vectors,
                'null_space': {
                    'protected_dimension': null_projection.get('null_space_dim', 0),
                    'update_dimension': null_projection.get('update_space_dim', 0),
                    'overlap': null_projection.get('subspace_overlap', 0)
                },
                'causal_importance': important_components,
                'activation_patching': patching_result
            }
            
            # Analyze the drift
            if intervention_vectors:
                drift_magnitude = intervention_vectors.get('total_magnitude', 0)
                drift_significance = intervention_vectors.get('significance_score', 0)
                
                checkpoint_result['drift'] = {
                    'magnitude': drift_magnitude,
                    'significance': drift_significance,
                    'is_concerning': drift_significance > 0.1,  # Threshold
                    'top_divergent_layers': intervention_vectors.get('top_layers', [])
                }
                
            results['drift_analysis'].append(checkpoint_result)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        # Aggregate analysis
        results['summary'] = self._summarize_drift(results['drift_analysis'])
        
        return results
        
    def find_consensus_direction(
        self,
        checkpoint_paths: List[str],
        test_batch: Dict[str, torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Find the "consensus" direction among multiple similar checkpoints.
        Useful for ensemble or finding robust updates.
        """
        logger.info("Finding consensus direction among checkpoints")
        
        if test_batch is None:
            test_batch = self.analyzer.create_test_batch()
            
        models = [self.analyzer.load_model(path) for path in checkpoint_paths]
        
        # Compute pairwise intervention vectors
        all_vectors = []
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                vectors = self.bombshell.find_intervention_vectors_enhanced(
                    model_before=models[i],
                    model_after=models[j],
                    test_samples=test_batch
                )
                all_vectors.append(vectors)
                
        # Find consensus (median direction)
        consensus = self._compute_consensus(all_vectors)
        
        # Validate consensus is safe using null space
        safety_check = self.bombshell.compute_null_space_projection(
            model=models[0],  # Use first as reference
            protected_samples=test_batch,
            update_samples=test_batch
        )
        
        return {
            'consensus_direction': consensus,
            'safety_validation': safety_check,
            'num_checkpoints': len(checkpoint_paths),
            'pairwise_comparisons': len(all_vectors)
        }
        
    def diagnose_training_instability(
        self,
        checkpoint_dir: Path,
        pattern: str = "checkpoint-*"
    ) -> Dict[str, Any]:
        """
        Use intervention tools to diagnose why checkpoints diverge.
        """
        checkpoints = sorted(checkpoint_dir.glob(pattern))
        
        if len(checkpoints) < 2:
            return {'error': 'Need at least 2 checkpoints'}
            
        logger.info(f"Diagnosing instability across {len(checkpoints)} checkpoints")
        
        # Analyze sequential drift
        sequential_drift = []
        
        for i in range(len(checkpoints) - 1):
            model1 = self.analyzer.load_model(str(checkpoints[i]))
            model2 = self.analyzer.load_model(str(checkpoints[i + 1]))
            
            # Find what changed
            changes = self.bombshell.find_intervention_vectors_enhanced(
                model_before=model1,
                model_after=model2,
                test_samples=self.analyzer.create_test_batch()
            )
            
            # Find what's important
            importance = self.information.compute_causal_necessity(
                model=model2,
                test_batch=self.analyzer.create_test_batch(),
                components_to_ablate=['layer.0', 'layer.10', 'layer.20'],
                n_bootstrap=50
            )
            
            sequential_drift.append({
                'step': i,
                'from': checkpoints[i].name,
                'to': checkpoints[i + 1].name,
                'changes': changes,
                'importance_shift': importance
            })
            
            del model1, model2
            torch.cuda.empty_cache()
            
        # Find instability patterns
        instability_diagnosis = self._diagnose_patterns(sequential_drift)
        
        return {
            'sequential_drift': sequential_drift,
            'diagnosis': instability_diagnosis,
            'recommendations': self._generate_recommendations(instability_diagnosis)
        }
        
    def _summarize_drift(self, drift_analysis: List[Dict]) -> Dict[str, Any]:
        """Summarize drift patterns across checkpoints."""
        if not drift_analysis:
            return {}
            
        magnitudes = [d['drift']['magnitude'] for d in drift_analysis if 'drift' in d]
        significances = [d['drift']['significance'] for d in drift_analysis if 'drift' in d]
        
        concerning_checkpoints = [
            d['checkpoint'] for d in drift_analysis 
            if d.get('drift', {}).get('is_concerning', False)
        ]
        
        return {
            'mean_drift_magnitude': np.mean(magnitudes) if magnitudes else 0,
            'max_drift_magnitude': np.max(magnitudes) if magnitudes else 0,
            'mean_significance': np.mean(significances) if significances else 0,
            'num_concerning': len(concerning_checkpoints),
            'concerning_checkpoints': concerning_checkpoints,
            'recommendation': self._get_drift_recommendation(magnitudes, significances)
        }
        
    def _get_drift_recommendation(self, magnitudes: List[float], significances: List[float]) -> str:
        """Generate recommendation based on drift analysis."""
        if not magnitudes:
            return "No drift detected"
            
        mean_sig = np.mean(significances)
        
        if mean_sig < 0.05:
            return "Checkpoints are well-aligned, normal variation"
        elif mean_sig < 0.1:
            return "Minor drift detected, consider ensemble averaging"
        elif mean_sig < 0.2:
            return "Moderate drift, investigate training stability"
        else:
            return "Significant drift, check for training bugs or data issues"
            
    def _compute_consensus(self, vectors: List[Dict]) -> Dict[str, Any]:
        """Compute consensus direction from multiple intervention vectors."""
        # Simplified: aggregate significant weights
        all_weights = {}
        
        for v in vectors:
            if 'significant_weights' in v:
                for layer, weight in v['significant_weights'].items():
                    if layer not in all_weights:
                        all_weights[layer] = []
                    all_weights[layer].append(weight)
                    
        # Compute median for each layer
        consensus = {}
        for layer, weights in all_weights.items():
            consensus[layer] = np.median(weights, axis=0)
            
        return consensus
        
    def _diagnose_patterns(self, sequential_drift: List[Dict]) -> Dict[str, Any]:
        """Diagnose instability patterns from sequential drift."""
        if not sequential_drift:
            return {}
            
        # Look for accelerating drift
        drift_magnitudes = [
            d['changes'].get('total_magnitude', 0) 
            for d in sequential_drift if 'changes' in d
        ]
        
        if len(drift_magnitudes) > 1:
            acceleration = np.diff(drift_magnitudes)
            is_accelerating = np.mean(acceleration) > 0
        else:
            is_accelerating = False
            
        return {
            'is_accelerating': is_accelerating,
            'drift_trend': 'increasing' if is_accelerating else 'stable',
            'max_single_step_drift': max(drift_magnitudes) if drift_magnitudes else 0,
            'cumulative_drift': sum(drift_magnitudes) if drift_magnitudes else 0
        }
        
    def _generate_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on diagnosis."""
        recommendations = []
        
        if diagnosis.get('is_accelerating', False):
            recommendations.append("⚠️ Drift is accelerating - reduce learning rate")
            
        if diagnosis.get('max_single_step_drift', 0) > 0.5:
            recommendations.append("⚠️ Large single-step changes - check for gradient spikes")
            
        if diagnosis.get('cumulative_drift', 0) > 2.0:
            recommendations.append("⚠️ High cumulative drift - consider checkpoint ensemble")
            
        if not recommendations:
            recommendations.append("✅ Training appears stable")
            
        return recommendations


# Example usage
if __name__ == "__main__":
    analyzer = CheckpointInterventionAnalyzer()
    
    # Analyze drift between similar checkpoints
    checkpoints = [
        "checkpoint_seed1/model.pt",
        "checkpoint_seed2/model.pt",
        "checkpoint_seed3/model.pt"
    ]
    
    drift_results = analyzer.analyze_checkpoint_drift(checkpoints)
    print(f"Drift summary: {drift_results['summary']}")
    
    # Find consensus direction
    consensus = analyzer.find_consensus_direction(checkpoints)
    print(f"Consensus direction found with safety validation")
    
    # Diagnose training instability
    diagnosis = analyzer.diagnose_training_instability(Path("./checkpoints"))
    print(f"Diagnosis: {diagnosis['diagnosis']}")
    print(f"Recommendations: {diagnosis['recommendations']}")
