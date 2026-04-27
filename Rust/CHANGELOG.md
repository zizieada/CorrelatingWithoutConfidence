# Changelog

## 0.1.0 - initial

### Parallelism
- Outer bootstrap parallelizes over **chunks of iterations within each metric**
  (not just across metrics), so single-metric workloads saturate all available
  cores. Chunk count defaults to `4 × n_threads` for load balance.
- `n_threads=None` (default) uses rayon's global pool (respects the
  `RAYON_NUM_THREADS` env var; defaults to all logical cores).
- `n_threads=N` uses a per-call pool with exactly `N` threads.
- Determinism: chunk `k` is seeded from `metric_seed XOR (k+1) × SplitMix64_C`,
  so output is bit-identical for a given `seed`, independent of thread count.

### Inputs
- Auto-detects three `subjective` DataFrame layouts:
    1. Long raw votes (scene id duplicated, one row per vote).
    2. Wide raw votes (unique scenes, one column per vote, NaN allowed).
    3. Mean/std summary (columns match `mean|mos` + `std|sd|stddev`).
- `objective` is wide: one row per scene, one numeric column per metric.
- NaN cells in the objective matrix are excluded per-metric.

### API surface
- `cwc.analyze(subjective, objective, ...)` → `CorrelationResult`.
- `result.summary()` → tidy `pandas.DataFrame` with mean, median, 2.5/97.5% CI.
- `result.plot_violins()`, `result.plot_win_probabilities()` match the notebook.
- `result.cliffs_delta_matrix(coefficient)` exposes pairwise effect sizes.

### Known limitations
- Kendall uses a naive O(n²) implementation. Fine for IQA datasets
  (hundreds of stimuli); slow above n ≈ 2000. Knight's O(n log n) is a planned
  improvement.
- Built wheel is Linux x86_64 only. For other platforms, build from source with
  `maturin develop --release` (needs Rust toolchain + Python venv).
