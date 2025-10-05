# TensorScope Documentation

Complete documentation for TensorScope - Cross-Metric Neural Network Analysis.

## Main Documentation

### [← Back to Main README](../README.md)
The main README contains:
- Quick start and simple examples
- Why this matters (field-level impact)
- What this enables
- Statistical rigor and testing
- Installation

### Detailed Guides

**[Research Recipes](RESEARCH_RECIPES.md)** (163 lines)
- Training dynamics & optimization
- Fisher information & curvature
- Mechanistic interpretability
- Representation analysis
- Data influence & attribution
- Pruning & lottery tickets

**[Detailed Examples](EXAMPLES.md)** (113 lines)
- Cross-metric discovery: Fisher × Superposition
- Cross-metric discovery: Geometry × Conflicts
- Engineering: LoRA placement
- Engineering: Data curation via sample conflicts

**[Metrics Reference](METRICS_REFERENCE.md)** (86 lines)
- Complete list of 80+ methods
- Function signatures and locations
- Organized by category

**[API Reference](API_REFERENCE.md)** (27 lines)
- Output formats (JSON, CSV, NumPy)
- Data structures
- Export options

**[Interpretation Guide](INTERPRETATION_GUIDE.md)** (41 lines)
- Fisher mask overlap thresholds
- Fisher uncertainty interpretation
- Curvature agreement guidelines
- Sample conflict severity
- Superposition regime interpretation

## Documentation Structure

```
README.md (653 lines)
├── Hook & Simple Examples
├── Research Questions
├── Why This Matters (CRITICAL)
├── Quick Start
├── What This Enables
├── Statistical Rigor
├── Testing & Validation
├── Composability
└── Links to detailed docs ↓

maindocs/
├── RESEARCH_RECIPES.md    (How to use for research)
├── EXAMPLES.md            (Detailed code examples)
├── METRICS_REFERENCE.md   (All 80+ methods)
├── API_REFERENCE.md       (Output formats)
└── INTERPRETATION_GUIDE.md (How to interpret results)
```

## Why This Structure?

**Main README (653 lines):**
- Fast to read (~10 minutes)
- Covers "why you should care"
- Simple usage examples
- Credibility builders
- Not overwhelming

**Detailed Docs (430 lines total):**
- Separated by audience/use case
- Easy to maintain
- Can be updated independently
- Better for different needs

**Total: 1,083 lines** (same content, better organized)
