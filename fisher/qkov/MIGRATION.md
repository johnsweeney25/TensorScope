# QKOV Module Migration Guide

**Date**: 2025-10-02

## What Changed

QKOV-related code has been reorganized into a dedicated module for better organization and discoverability.

### Old Structure (Before)

```
fisher/
├── core/
│   ├── qkov_interference.py     ❌ Moved
│   └── qkov_statistics.py       ❌ Moved
└── docs/
    ├── QKOV_ENGINEERING_NOTES.md  ❌ Moved
    └── QKOV_IMPLEMENTATION_SUMMARY.md  ❌ Moved
```

### New Structure (After)

```
fisher/
└── qkov/                         ✅ New dedicated module
    ├── __init__.py               ✅ Public API
    ├── README.md                 ✅ Module documentation
    ├── qkov_interference.py      ✅ Moved here
    ├── qkov_statistics.py        ✅ Moved here
    ├── docs/
    │   ├── QKOV_ENGINEERING_NOTES.md         ✅ Moved here
    │   └── QKOV_IMPLEMENTATION_SUMMARY.md    ✅ Moved here
    └── tests/
        └── test_qkov_interference.py  ⏳ TODO
```

## Migration Steps

### Update Import Statements

**Before:**
```python
from fisher.core.qkov_interference import QKOVConfig, QKOVInterferenceMetric
from fisher.core.qkov_statistics import QKOVStatistics
```

**After:**
```python
from fisher.qkov import QKOVConfig, QKOVInterferenceMetric, QKOVStatistics
```

### Update Documentation References

**Before:**
```
fisher/docs/QKOV_ENGINEERING_NOTES.md
fisher/docs/QKOV_IMPLEMENTATION_SUMMARY.md
```

**After:**
```
fisher/qkov/docs/QKOV_ENGINEERING_NOTES.md
fisher/qkov/docs/QKOV_IMPLEMENTATION_SUMMARY.md
fisher/qkov/README.md  # New: Module overview
```

## Backward Compatibility

**Breaking changes:**
- Import paths have changed
- Documentation paths have changed

**No functional changes:**
- All APIs remain the same
- Formula implementation unchanged
- Statistical testing unchanged

## Updated Examples

See `examples/qkov_interference_example.py` for updated usage.

## Related Files

**Other QKOV files in repository** (not moved):
- `qkov_patching_dynamics.py` - Legacy QK-OV pairing (root directory)
- `test_qkov_simple.py` - Legacy tests (root directory)
- `test_qkov_statistical_validity.py` - Legacy tests (root directory)

These are **older implementations** separate from the Section 4.1 metric.

## Rollback Instructions

If needed, revert with:

```bash
# Restore old locations
mv fisher/qkov/qkov_interference.py fisher/core/
mv fisher/qkov/qkov_statistics.py fisher/core/
mv fisher/qkov/docs/* fisher/docs/
rm -rf fisher/qkov/
```

## Questions?

See `fisher/qkov/README.md` for module documentation.
