# Lottery Tickets Module - Documentation Index

Quick reference for all lottery tickets documentation.

## Main Documentation Files

### 🔥 **[MEMORY_OPTIMIZATION_DOCUMENTATION.md](MEMORY_OPTIMIZATION_DOCUMENTATION.md)**
**ICML 2026 Critical - Must Read**

Critical GPU memory leak fixes that reduced peak memory by 40% for lottery ticket evaluation on large models.

**Key Topics:**
- 4 critical bug fixes (10 GB saved)
- Theoretical correctness proofs
- Numerical precision analysis
- Testing and validation
- Best practices

**When to Read:** Before working with models >1B parameters or if experiencing OOM errors.

---

### 📚 **[EARLY_BIRD_DOCUMENTATION.md](EARLY_BIRD_DOCUMENTATION.md)**
**Production Ready Implementation**

Complete guide to memory-efficient early bird ticket detection for models up to 14B parameters.

**Key Topics:**
- SGD vs AdamW optimizer comparison
- Memory optimization techniques
- API reference
- Troubleshooting guide

**When to Read:** When implementing early bird ticket detection or training lottery tickets.

---

### 🔧 **[TICKET_OVERLAP_FIXES.md](TICKET_OVERLAP_FIXES.md)**
**Edge Case Handling**

Fixes for ticket overlap computation with proper edge case handling and numerical stability.

**Key Topics:**
- Edge case fixes (empty masks, shape mismatches)
- Jaccard/Dice/Overlap coefficients
- Reproducibility improvements

**When to Read:** When comparing lottery tickets across runs or analyzing ticket consistency.

---

## Quick Start Guides

### Memory-Efficient Lottery Ticket Evaluation

```python
import lottery_tickets
import torch

# Create mask (memory-efficient)
mask = lottery_tickets.create_magnitude_mask(
    model=model,
    sparsity=0.9,
    use_histogram=True  # Saves memory
)

# Evaluate ticket quality
quality = lottery_tickets.compute_lottery_ticket_quality(
    model=model.bfloat16(),
    mask=mask,
    dataloader=[batch],
    precision_mode='high'  # For numerical stability
)
```

**See:** [MEMORY_OPTIMIZATION_DOCUMENTATION.md - Quick Start](MEMORY_OPTIMIZATION_DOCUMENTATION.md#quick-start)

---

### Early Bird Ticket Detection

```python
from lottery_tickets.early_bird import compute_early_bird_tickets

results = compute_early_bird_tickets(
    model=model,
    dataloader=dataloader,
    max_epochs=30,
    use_sgd=True  # 100GB memory savings vs AdamW
)
```

**See:** [EARLY_BIRD_DOCUMENTATION.md - Quick Start](EARLY_BIRD_DOCUMENTATION.md#quick-start)

---

## Documentation by Topic

### Memory Optimization
- **Primary:** [MEMORY_OPTIMIZATION_DOCUMENTATION.md](MEMORY_OPTIMIZATION_DOCUMENTATION.md)
- **Testing:** [../tests/README_MEMORY_TESTS.md](../tests/README_MEMORY_TESTS.md)
- **Implementation:** `magnitude_pruning.py`, `evaluation.py`

### Early Bird Tickets
- **Primary:** [EARLY_BIRD_DOCUMENTATION.md](EARLY_BIRD_DOCUMENTATION.md)
- **Implementation:** `early_bird.py`
- **Legacy Docs:** `../../EARLY_BIRD_*.md` files

### Ticket Overlap
- **Primary:** [TICKET_OVERLAP_FIXES.md](TICKET_OVERLAP_FIXES.md)
- **Implementation:** `evaluation.py::compute_ticket_overlap()`

---

## By Use Case

### "I'm getting OOM errors"
→ Read: [MEMORY_OPTIMIZATION_DOCUMENTATION.md](MEMORY_OPTIMIZATION_DOCUMENTATION.md)

### "I want to train lottery tickets on large models"
→ Read: [EARLY_BIRD_DOCUMENTATION.md](EARLY_BIRD_DOCUMENTATION.md)

### "I need to compare tickets across runs"
→ Read: [TICKET_OVERLAP_FIXES.md](TICKET_OVERLAP_FIXES.md)

### "I'm submitting to ICML 2026"
→ Read all documentation, especially Memory Optimization

---

## Testing

```bash
# Memory optimization tests
cd lottery_tickets/tests
python3 run_tests.py --module memory

# All tests
python3 run_tests.py
```

---

## External References

- **Project Docs:** `../../docs/LOTTERY_TICKETS_DOCUMENTATION.md`
- **Main README:** [README.md](README.md)
- **Test Docs:** [../tests/README_MEMORY_TESTS.md](../tests/README_MEMORY_TESTS.md)

---

**Last Updated:** 2025-09-30
**Status:** ✅ ICML 2026 Ready
