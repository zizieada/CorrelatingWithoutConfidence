# correlation_with_confidence

Bootstrap confidence intervals for correlation coefficients between subjective opinion scores and objective metrics. Native Rust core, pandas-friendly Python API.

## Install

```bash
pip install correlation_with_confidence
# for plotting helpers:
pip install "correlation_with_confidence[plot]"
```

## Quick start

```python
import correlation_with_confidence as cwc
import pandas as pd

sub = pd.read_csv("subjective.csv")   # long, wide-raw, or mean/std layout
obj = pd.read_csv("objective.csv")    # one row per scene, one column per metric

result = cwc.analyze(
    subjective=sub,
    objective=obj,
    coefficients=("pearson", "spearman", "kendall"),
    n_bootstrap=14999,
    inner_bootstrap=1499,
    bootstrap_scenes=True,
    seed=42,
)

print(result.summary())
fig1 = result.plot_violins()
fig2 = result.plot_win_probabilities()
```

## Parallelism

The bootstrap loop parallelizes across chunks of iterations within each metric,
so **single-metric workloads also scale** across all available cores. Control via:

```python
# Use N threads for this call only (doesn't affect rayon's global pool)
result = cwc.analyze(sub, obj, n_threads=8, ...)

# Or set the rayon default for the whole process before import:
import os
os.environ["RAYON_NUM_THREADS"] = "16"
import correlation_with_confidence as cwc
```

Output is bit-identical across runs for a fixed `seed`, independent of thread count.

## Build from source

```bash
pip install maturin
maturin develop --release
```

