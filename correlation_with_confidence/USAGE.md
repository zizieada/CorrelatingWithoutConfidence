# USAGE — from the notebook to the package

This document maps the cells in `example_multiple_metrics.ipynb` onto calls in
`correlation_with_confidence`. The package does in one `analyze()` call what
previously required six cells of glue.

## Before (notebook)

```python
import pandas as pd, numpy as np, copy
import matplotlib.pyplot as plt
from additional_functions import correlation_with_confidence as fcn

# 1) load raw votes and group them per scene
df = pd.read_csv('CID2013.csv')
df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
df = df.dropna(subset=['quality'])
scene_votes = df.groupby('scene')['quality'].apply(list).to_dict()

# 2) load all metric CSVs and merge into one wide DataFrame
metric_df = ...  # merge every CID2013_<metric>.csv on 'name'
metrics_names = [m for m in metric_df.columns if m not in ('name', 'scores')]

# 3) build inner bootstrap (distribution of means per scene)
mos_distributions = fcn.create_mos_distributions(scene_votes, n_iterations=1499)

# 4) per-metric outer bootstrap
pearson_distributions, spearman_distributions, kendall_distributions = {}, {}, {}
for metric_name in metrics_names:
    metric = np.array([name_to_value[n] for n in scene_votes.keys()])
    temp = fcn.compute_correlation_distributions(
        mos_distributions, metric,
        n_bootstrap=14999, bootstrap_scenes=True,
        corr_coeffs=['r', 'rho', 'tau'],
    )
    pearson_distributions[metric_name]  = np.abs(temp['r'])
    spearman_distributions[metric_name] = np.abs(temp['rho'])
    kendall_distributions[metric_name]  = np.abs(temp['tau'])

# 5) summary table (cell 11) — hand-rolled
# 6) violin plot  (cell 13) — hand-rolled
# 7) heatmap      (cell 14) — hand-rolled
```

## After (package)

```python
import pandas as pd
import correlation_with_confidence as cwc

sub = pd.read_csv('CID2013_subjective.csv')    # any layout — auto-detected
obj = pd.read_csv('CID2013_objective.csv')     # one row per scene, N metric cols

result = cwc.analyze(
    subjective=sub,
    objective=obj,
    n_bootstrap=14999,
    inner_bootstrap=1499,
    bootstrap_scenes=True,
    seed=42,
)

result.summary()                    # replaces cell 11 (pandas DataFrame, already sorted)
result.plot_violins()               # replaces cell 13
result.plot_win_probabilities()     # replaces cell 14

# Distributions are still accessible as numpy arrays, same shape as before:
pearson_dists = result.distributions['pearson']  # dict {metric: ndarray of length n_bootstrap}
```

## Cell-by-cell equivalence

| Notebook cell | Package call |
|---|---|
| Cell 3 (load + group votes) | handled internally by `cwc.analyze` — pass the raw DataFrame |
| Cell 6 (`create_mos_distributions`) | `inner_bootstrap` kwarg |
| Cell 7 (loop over metrics, `compute_correlation_distributions`) | single `analyze()` call |
| Cell 9/10 (save/load distributions) | `np.savez(...)` or `pd.DataFrame(result.distributions['pearson']).to_csv(...)` |
| Cell 11 (summary table) | `result.summary()` |
| Cell 13 (violins + 95% CI bars) | `result.plot_violins()` |
| Cell 14 (win-probability heatmap) | `result.plot_win_probabilities()` |
| Cell 15 (thresholded heatmap) | `from correlation_with_confidence.plots import plot_win_probabilities_thresholded` |

## Input layouts auto-detected

The `subjective` DataFrame is inspected by column names and duplicate check on the
scene id column. No kwarg needed for the three common cases:

| Layout | Shape | Trigger |
|---|---|---|
| **Long raw** | `name, vote` (scene repeats, one row per vote) | scene column has duplicates |
| **Wide raw** | `name, v1, v2, ..., vN` (NaN allowed) | unique scenes + no mean/std columns |
| **Mean/std** | `name, mos|mean, std|sd|stddev` (optional `n_votes`) | a column name matches `mean\|mos` AND another matches `std\|sd\|stddev` |

Override detection with `scene_col=`, `vote_col=`, `vote_cols=` if needed.

## Reproducibility

Pass `seed=<int>` for bit-identical output across runs. Omit (or `seed=None`) for
a time-based seed. Reproducibility is independent of how many threads are used.

## Performance notes

- The Rust core releases the GIL during bootstrap — safe to call from
  multi-threaded Python code (e.g. Jupyter with `%%time`).
- Parallelism is across metrics (11 metrics on CID2013 → saturates ~8 cores).
  If you only have one metric, the work is serial per call; batch your metrics
  into the wide `objective` DataFrame to get parallelism.
- `n_bootstrap=14999` on CID2013 (474 scenes, 11 metrics, all 3 coefficients)
  takes ~30-90 seconds on a typical laptop. Pure-Python reference takes
  ~5-10 minutes on the same machine.

## Installing from source

```bash
# Rust toolchain (once):  https://rustup.rs/
pip install maturin
cd correlation_with_confidence/
maturin develop --release       # installs into the current venv
# or:
maturin build --release          # builds a wheel in ./target/wheels/
pip install target/wheels/correlation_with_confidence-*.whl
```

The project ships an `abi3-py39` wheel — one binary works for CPython 3.9+.
