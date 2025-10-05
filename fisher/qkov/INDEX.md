# QKOV Module Index

**Complete file index for Section 4.1 implementation**

## Module Structure

```
fisher/qkov/                              # Main module directory
│
├── Core Implementation
│   ├── qkov_interference.py              # Main metric (M^B_{ij,ℓ,h})
│   ├── qkov_statistics.py                # Statistical testing
│   └── __init__.py                       # Public API exports
│
├── Documentation
│   ├── README.md                         # Module overview
│   ├── QUICK_REFERENCE.md               # One-page cheatsheet
│   ├── MIGRATION.md                      # Import path changes
│   ├── INDEX.md                          # This file
│   └── docs/
│       ├── QKOV_ENGINEERING_NOTES.md     # Detailed usage guide
│       └── QKOV_IMPLEMENTATION_SUMMARY.md # Implementation status
│
└── Tests
    └── tests/
        └── test_qkov_interference.py     # Unit tests (TODO)
```

## File Descriptions

### Core Files

**`qkov_interference.py`** (668 lines)
- Main implementation of Section 4.1 formula
- Classes:
  - `QKOVConfig`: Model architecture configuration
  - `QKOVIndexer`: Parameter slicing by (layer, head, block)
  - `QKOVInterferenceMetric`: Core metric computation
  - `BlockHeadSlice`: Slice specification
  - `InterferenceScore`: Result container
- Features:
  - Auto-detection of model architecture
  - Support for fused/split QKV, GQA/MQA
  - Device/dtype safety
  - Numerical stability (ridge regularization)
  - Score caching

**`qkov_statistics.py`** (330 lines)
- Statistical testing framework
- Classes:
  - `QKOVStatistics`: Main testing class
  - `PermutationTestResult`: Permutation test output
  - `ClusterResult`: Cluster-level test output
- Methods:
  - Permutation null testing
  - Benjamini-Hochberg FDR correction
  - Bootstrap confidence intervals
  - Cluster-level corrections

**`__init__.py`** (60 lines)
- Public API exports
- Version info
- Module docstring with usage example

### Documentation Files

**`README.md`** (350 lines)
- Module overview
- Quick start guide
- Architecture support matrix
- API reference
- Examples
- Integration guide

**`QUICK_REFERENCE.md`** (250 lines)
- One-page cheatsheet
- Common usage patterns
- Troubleshooting tips
- Parameter reference
- Performance notes

**`MIGRATION.md`** (70 lines)
- Import path changes
- Backward compatibility notes
- Rollback instructions

**`INDEX.md`** (This file)
- Complete file listing
- Cross-references
- Line counts

**`docs/QKOV_ENGINEERING_NOTES.md`** (600 lines)
- Detailed engineering guide
- 6 critical pitfalls (from intern feedback)
- Tensor shape reference
- Architecture-specific handling
- API usage examples
- Troubleshooting guide
- Performance optimization

**`docs/QKOV_IMPLEMENTATION_SUMMARY.md`** (350 lines)
- Implementation status
- Fixes applied from intern feedback
- Integration with FisherCollector
- Testing status
- Paper consistency verification
- Next steps

## Related Files (Outside Module)

### Examples

**`examples/qkov_interference_example.py`** (250 lines)
- Complete working example
- Mock model and data
- Step-by-step walkthrough
- All API demonstrated

### Legacy Files (Root Directory)

**`qkov_patching_dynamics.py`**
- Older QK-OV pairing implementation
- Activation patching
- Training dynamics analysis
- **Note**: Separate from Section 4.1 metric

**`test_qkov_simple.py`**
- Legacy tests for old implementation

**`test_qkov_statistical_validity.py`**
- Legacy statistical tests

## Import Paths

### Current (After Migration)

```python
from fisher.qkov import (
    QKOVConfig,
    QKOVIndexer,
    QKOVInterferenceMetric,
    QKOVStatistics,
)
```

### Old (Before Migration) ❌

```python
# Don't use these anymore!
from fisher.core.qkov_interference import ...
from fisher.core.qkov_statistics import ...
```

## Quick Navigation

**I want to...**

- **Get started quickly** → `QUICK_REFERENCE.md`
- **Understand the implementation** → `README.md`
- **Fix import errors** → `MIGRATION.md`
- **Deep dive into details** → `docs/QKOV_ENGINEERING_NOTES.md`
- **Check implementation status** → `docs/QKOV_IMPLEMENTATION_SUMMARY.md`
- **See a working example** → `../../../examples/qkov_interference_example.py`
- **Understand the formula** → `qkov_interference.py` (docstring)
- **Run statistical tests** → `qkov_statistics.py`
- **Find all files** → This file (`INDEX.md`)

## Line Counts

| File | Lines | Type |
|------|------:|------|
| `qkov_interference.py` | 668 | Python |
| `qkov_statistics.py` | 330 | Python |
| `__init__.py` | 60 | Python |
| `README.md` | 350 | Markdown |
| `QUICK_REFERENCE.md` | 250 | Markdown |
| `MIGRATION.md` | 70 | Markdown |
| `INDEX.md` | ~150 | Markdown |
| `docs/QKOV_ENGINEERING_NOTES.md` | 600 | Markdown |
| `docs/QKOV_IMPLEMENTATION_SUMMARY.md` | 350 | Markdown |
| **Total** | **~2,828** | |

## Dependencies

### Required

- `torch` - PyTorch
- `numpy` - Numerical operations
- `scipy` - Statistical testing
- `fisher.core.fisher_collector.FisherCollector` - Fisher data

### Optional

- `matplotlib` - Visualization
- `seaborn` - Heatmaps
- `transformers` - Real models (for examples)

## Testing

**Status**: Unit tests pending

**Coverage needed**:
1. Shape validation (fused/split, GQA)
2. O column slicing correctness
3. Head additivity
4. GQA mapping
5. Numerical stability
6. Statistical testing correctness

**Test file**: `tests/test_qkov_interference.py` (TODO)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-02 | Initial release, reorganized into dedicated module |

## Cross-References

### Paper Sections

- **Section 3.2**: Contribution Safety Theorem → See `qkov_interference.py` docstring
- **Section 4.1**: Interference Metric Formula → See `qkov_interference.py` (compute_block_head_score)
- **Section 6**: Statistical Testing → See `qkov_statistics.py`

### Other Modules

- `fisher.core.fisher_collector` - Provides Fisher data
- `fisher.core.cross_task_conflict_detector` - Alternative conflict detection
- `mechanistic.mechanistic_analyzer_core` - Circuit taxonomy (legacy)

## Support

**Documentation**: Start with `README.md`

**Troubleshooting**: See `QUICK_REFERENCE.md` or `docs/QKOV_ENGINEERING_NOTES.md`

**Issues**: See project root for issue tracker

---

**Last Updated**: 2025-10-02
**Module Version**: 1.0.0
**Status**: Complete
