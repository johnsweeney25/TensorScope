#!/usr/bin/env python3
"""
Example: Using Unified Model Analysis with Automatic Report Generation
======================================================================

This example shows how to use the unified model analysis framework with
integrated statistical report generation.
"""

from pathlib import Path
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig, ModelSpec

def main():
    """Run a complete analysis with report generation."""

    # Configure the analysis
    config = UnifiedConfig(
        # Analysis settings
        skip_expensive=False,  # Set to True to skip slow metrics
        skip_checkpoint_metrics=False,  # Set to True if analyzing single model
        batch_size=8,

        # Correlation analysis
        correlation_enabled=True,
        min_correlation_samples=5,

        # Intervention analysis (for checkpoint drift)
        intervention_enabled=True,
        max_intervention_models=5,

        # Statistical comparisons
        statistical_tests=True,
        pairwise_comparisons=True,

        # Output settings
        output_dir=Path("./analysis_results"),
        output_format="json",  # Save raw results as JSON
        visualization=True,

        # Report generation (NEW!)
        generate_report=True,  # Automatically generate PDF/LaTeX report
        report_format="pdf",  # Will fall back to LaTeX if pdflatex not available
        report_style="technical"  # Style: technical, neurips, ieee, executive
    )

    # Define models to analyze
    model_specs = [
        # Example 1: Local checkpoint file
        ModelSpec(
            path="/path/to/model/checkpoint.pt",
            name="model_v1",
            group="baseline"
        ),

        # Example 2: HuggingFace model
        ModelSpec(
            path="meta-llama/Llama-2-7b-hf",
            name="llama2_7b",
            group="baseline"
        ),

        # Example 3: Multiple checkpoints for comparison
        ModelSpec(
            path="/path/to/checkpoint_epoch_1.pt",
            name="epoch_1",
            group="training_progression"
        ),
        ModelSpec(
            path="/path/to/checkpoint_epoch_5.pt",
            name="epoch_5",
            group="training_progression"
        ),
        ModelSpec(
            path="/path/to/checkpoint_epoch_10.pt",
            name="epoch_10",
            group="training_progression"
        ),
    ]

    # Create analyzer
    analyzer = UnifiedModelAnalyzer(config)

    # Run comprehensive analysis
    print("🔬 Starting unified model analysis...")
    results = analyzer.analyze_models(model_specs)

    # Results are automatically saved to:
    # 1. analysis_results/analysis_TIMESTAMP.json - Raw analysis data
    # 2. analysis_results/report_TIMESTAMP.tex - LaTeX source
    # 3. analysis_results/report_TIMESTAMP.pdf - PDF report (if pdflatex available)
    # 4. analysis_results/report_TIMESTAMP_analysis.json - Statistical analysis
    # 5. analysis_results/*.png - Visualizations

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    summary = results.summary()
    print(f"\n📊 Summary:")
    print(f"   Models analyzed: {summary['n_models']}")
    print(f"   Metrics computed: {summary['n_metrics']}")
    print(f"   Groups identified: {len(results.group_analyses)}")
    print(f"   Errors encountered: {summary['n_errors']}")

    # Display top findings
    if results.global_correlations:
        print(f"\n🔗 Top Correlations:")
        for corr in list(results.global_correlations.items())[:5]:
            print(f"   - {corr[0]}: {corr[1]:.3f}")

    print(f"\n📁 Results saved to: {config.output_dir}")
    print("\nThe statistical report includes:")
    print("  • Comprehensive descriptive statistics")
    print("  • Distribution analysis with visualizations")
    print("  • Correlation matrices and heatmaps")
    print("  • Group comparisons with hypothesis tests")
    print("  • Effect size calculations")
    print("  • Principal component analysis")
    print("  • Time series analysis (if applicable)")
    print("  • Publication-ready LaTeX formatting")

if __name__ == "__main__":
    main()