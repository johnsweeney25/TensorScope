# Report Generation Documentation

## Overview
The Unified Model Analysis framework provides comprehensive report generation capabilities in both JSON and PDF formats. This documentation covers the report generation pipeline, configuration options, and output formats.

## Table of Contents
1. [JSON Output Generation](#json-output-generation)
2. [PDF Report Generation](#pdf-report-generation)
3. [Report Configuration](#report-configuration)
4. [Lottery Ticket Metrics in Reports](#lottery-ticket-metrics-in-reports)
5. [Troubleshooting](#troubleshooting)

## JSON Output Generation

### Automatic JSON Saving
The framework automatically saves analysis results to JSON format when running model analysis or trajectory analysis.

#### Model Analysis JSON
```python
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig, ModelSpec

config = UnifiedConfig()
config.output_dir = "./results"  # Where to save JSON files
config.output_format = "json"    # Or "both" for JSON + CSV

analyzer = UnifiedModelAnalyzer(config)
results = analyzer.analyze_models(model_specs)
# JSON automatically saved to output_dir
```

#### JSON Structure
```json
{
  "timestamp": "2024-09-29T10:30:00",
  "config": {
    "device": "cuda",
    "batch_size": 32,
    "skip_expensive": false
  },
  "model_results": {
    "model_name": {
      "metrics": {
        "gradient_alignment": 0.85,
        "fisher_trace": 1.23,
        "effective_rank": 15.7
      },
      "errors": {},
      "skipped": [],
      "computation_dtype": "bfloat16"
    }
  },
  "group_analyses": {},
  "pairwise_comparisons": {},
  "global_correlations": {}
}
```

### Manual JSON Saving
```python
import json
from datetime import datetime

# Convert results to dictionary
result_dict = {
    'timestamp': results.timestamp,
    'config': config.__dict__,
    'model_results': {}
}

for model_name, model_result in results.model_results.items():
    result_dict['model_results'][model_name] = {
        'metrics': model_result.metrics,
        'errors': model_result.errors,
        'skipped': model_result.skipped,
        'computation_dtype': model_result.computation_dtype
    }

# Save JSON
json_path = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(json_path, 'w') as f:
    json.dump(result_dict, f, indent=2, default=str)
```

## PDF Report Generation

### Prerequisites
- `matplotlib` and `seaborn` for visualizations
- `reportlab` for PDF generation (optional)
- `pdflatex` for LaTeX to PDF conversion (optional)

```bash
pip install matplotlib seaborn reportlab
# For LaTeX support:
# apt-get install texlive-full  # Linux
# brew install --cask mactex    # macOS
```

### Automatic PDF Generation
```python
config = UnifiedConfig()
config.generate_report = True  # Enable automatic report generation
config.report_style = 'technical'  # Options: technical, neurips, ieee, executive
config.output_dir = "./results"

analyzer = UnifiedModelAnalyzer(config)
results = analyzer.analyze_models(model_specs)
# PDF report automatically generated
```

### Manual PDF Generation
```python
from statistical_report_generator import StatisticalReportGenerator, ReportConfig

# Configure report
report_config = ReportConfig(
    output_dir="./results",
    figure_dir="./results/figures",
    style='technical',  # Report style
    experiment_type='model_comparison'  # Or 'trajectory', 'lottery_ticket_analysis'
)

# Initialize generator
generator = StatisticalReportGenerator(config=report_config)

# Add JSON results
generator.add_results("path/to/results.json")

# Generate report
report_path = generator.generate_report(output_name="model_analysis_report")
```

### Report Styles
- **technical**: Detailed technical report with all metrics
- **neurips**: NeurIPS conference format
- **ieee**: IEEE conference format
- **executive**: Executive summary with key findings

## Report Configuration

### UnifiedConfig Report Settings
```python
config = UnifiedConfig()

# JSON/CSV output
config.output_dir = Path("./results")  # Output directory
config.output_format = "both"  # "json", "csv", or "both"

# Report generation
config.generate_report = True  # Enable PDF report generation
config.report_style = "technical"  # Report template style
config.no_report = False  # Disable automatic report generation

# Visualization settings
config.figure_format = "pdf"  # Figure format: pdf, png, svg
config.dpi = 300  # Figure resolution
```

### ReportConfig Settings
```python
from statistical_report_generator import ReportConfig

report_config = ReportConfig(
    output_dir=Path("./results"),
    figure_dir=Path("./results/figures"),
    style="technical",
    experiment_type="model_comparison",

    # Visualization options
    figure_width=8.5,
    figure_height=11,
    dpi=300,

    # Report options
    include_raw_data=False,  # Include raw data tables
    include_statistics=True,  # Include statistical tests
    include_visualizations=True,  # Include charts/plots

    # LaTeX options
    use_latex=True,  # Generate LaTeX source
    compile_pdf=True,  # Compile to PDF (requires pdflatex)
)
```

## Lottery Ticket Metrics in Reports

### Lottery Ticket JSON Structure
```json
{
  "model_results": {
    "model_name": {
      "metrics": {
        "pruning_robustness": {
          "winning_ticket_score": 1.25,
          "optimal_sparsity": 0.85,
          "critical_sparsity": 0.92,
          "sparsity_curves": {
            "0.1": {"loss": 2.3, "performance_retention": 0.98},
            "0.5": {"loss": 2.5, "performance_retention": 0.92},
            "0.9": {"loss": 3.8, "performance_retention": 0.65}
          }
        },
        "fisher_importance": {
          "top_10_percent_mass": 0.78,
          "gini_coefficient": 0.82,
          "parameter_scores": {
            "fc1.weight": 0.85,
            "fc2.weight": 0.92
          }
        },
        "magnitude_ticket": {
          "overall_sparsity": 0.90,
          "method": "global_ranking",
          "layer_sparsities": {
            "fc1.weight": 0.88,
            "fc2.weight": 0.92
          }
        }
      }
    }
  }
}
```

### Lottery Ticket Visualizations
The report generator automatically creates:
- **Sparsity-Performance Curves**: Shows performance vs. pruning level
- **Layer Sparsity Distribution**: Bar chart of per-layer pruning
- **Fisher Importance Heatmap**: Visualizes parameter importance
- **Winning Ticket Comparison**: Compares multiple pruning methods

## Troubleshooting

### Common Issues and Solutions

#### 1. JSON Not Saving Automatically
```python
# Ensure output_dir exists
config.output_dir = Path("./results")
config.output_dir.mkdir(exist_ok=True, parents=True)

# Check output format
config.output_format = "json"  # or "both"
```

#### 2. PDF Generation Fails
```python
# Check if dependencies are installed
try:
    from statistical_report_generator import StatisticalReportGenerator
    print("Report generator available")
except ImportError:
    print("Install: pip install matplotlib seaborn reportlab")

# If pdflatex not found, LaTeX file is still generated
# You can compile it manually: pdflatex report.tex
```

#### 3. Large Model Memory Issues
```python
# For large models, save intermediate results
config.save_intermediate = True
config.checkpoint_interval = 10  # Save every 10 models
```

#### 4. Custom Metrics Not Appearing
```python
# Ensure metrics are properly registered
analyzer.registry.register(
    name="custom_metric",
    func=compute_custom_metric,
    module="custom",
    signature_type=SignatureType.STANDARD
)
```

### Testing Report Generation
Use the provided test script:
```bash
python test_report_generation.py
```

This will:
- Test JSON generation and validation
- Test PDF/LaTeX generation
- Verify lottery ticket metrics in reports
- Check all required dependencies

## API Reference

### Key Functions

#### `analyzer.analyze_models(specs) -> AnalysisResults`
Analyzes models and saves results to JSON/CSV.

#### `analyzer.analyze_trajectory(checkpoints) -> TrajectoryResults`
Analyzes training trajectory and saves timeline data.

#### `generator.generate_report(output_name) -> Path`
Generates PDF report from JSON results.

#### `generator.add_results(json_path)`
Adds JSON results to report generator.

### Key Classes

#### `UnifiedConfig`
Main configuration for analysis and output.

#### `ReportConfig`
Configuration for report generation.

#### `StatisticalReportGenerator`
Generates PDF reports from JSON results.

#### `AnalysisResults`
Contains all analysis results with save methods.

## Examples

### Complete Example: Model Comparison with Report
```python
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig, ModelSpec
from pathlib import Path

# Configure
config = UnifiedConfig()
config.device = 'cuda'
config.output_dir = Path("./comparison_results")
config.output_dir.mkdir(exist_ok=True)
config.generate_report = True
config.report_style = 'technical'

# Create analyzer
analyzer = UnifiedModelAnalyzer(config)

# Define models
specs = [
    ModelSpec(path="model1.pt", name="baseline", group="control"),
    ModelSpec(path="model2.pt", name="pruned_90", group="lottery"),
]

# Analyze and generate report
results = analyzer.analyze_models(specs)
print(f"Analysis complete. Report saved to {config.output_dir}")
```

### Example: Lottery Ticket Analysis Report
```python
# After running lottery ticket analysis
lottery_results = {
    "pruning_robustness": compute_pruning_robustness(model, batch),
    "fisher_importance": compute_fisher_importance(model, dataloader),
    "magnitude_ticket": compute_layerwise_magnitude_ticket(model, 0.9)
}

# Save to JSON
import json
with open("lottery_analysis.json", "w") as f:
    json.dump({
        "model_results": {
            "pruned_model": {"metrics": lottery_results}
        }
    }, f, indent=2)

# Generate report
from statistical_report_generator import StatisticalReportGenerator, ReportConfig

config = ReportConfig(
    output_dir="./lottery_report",
    experiment_type="lottery_ticket_analysis"
)

generator = StatisticalReportGenerator(config)
generator.add_results("lottery_analysis.json")
report_path = generator.generate_report("lottery_ticket_report")
```

---
*Documentation generated: September 29, 2024*
*Framework version: 1.0.0*