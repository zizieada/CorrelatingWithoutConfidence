"""Main entry point: `cwc.analyze(...)`."""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import _core  # Rust extension module
from .io import (
    SubjectiveData,
    align,
    normalize_coefficients,
    parse_objective,
    parse_subjective,
)
from .result import CorrelationResult


def analyze(
    subjective: Union[pd.DataFrame, dict],
    objective: pd.DataFrame,
    *,
    scene_col: Optional[str] = None,
    vote_col: Optional[str] = None,
    vote_cols: Optional[Sequence[str]] = None,
    metric_cols: Optional[Sequence[str]] = None,
    coefficients: Iterable[str] = ("pearson", "spearman", "kendall"),
    n_bootstrap: int = 14999,
    inner_bootstrap: int = 1499,
    bootstrap_scenes: bool = True,
    bounds: Tuple[float, float] = (0.0, 100.0),
    n_votes: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    verbose: bool = False,
) -> CorrelationResult:
    """Compute bootstrap correlation distributions between subjective and objective data.

    Parameters
    ----------
    subjective : pandas.DataFrame or dict
        Subjective scores. Accepts:
            - DataFrame in long layout (duplicated scene ids + vote column),
            - DataFrame in wide layout (one row per scene, N vote columns with NaN allowed),
            - DataFrame with columns matching `mean|mos` + `std|sd|stddev`,
            - dict {scene_name: [vote_1, vote_2, ...]}.
    objective : pandas.DataFrame
        One row per scene, one numeric column per metric, plus a scene id column.
    scene_col : str, optional
        Name of the scene-identifier column in both inputs. Auto-detected
        if omitted (tries `name`, `scene`, `stimulus`, ..., else first column).
    vote_col : str, optional
        Name of the vote column when using the long subjective layout.
        Auto-detected if there is exactly one numeric non-scene column.
    vote_cols : sequence of str, optional
        Explicit list of vote columns for wide layout.
    metric_cols : sequence of str, optional
        Explicit list of objective metric columns. Otherwise: all numeric
        non-scene columns are used.
    coefficients : iterable of str
        Correlation coefficients to compute. Any of pearson/spearman/kendall
        (or r/rho/tau). Defaults to all three.
    n_bootstrap : int
        Number of outer bootstrap iterations (per metric).
    inner_bootstrap : int
        Number of inner bootstrap samples used to build each scene's mean
        distribution. Only relevant when raw votes are provided.
    bootstrap_scenes : bool
        If True, resample scenes with replacement at each outer iteration.
    bounds : (float, float)
        Lower/upper bounds of the rating scale. Used only for the mean/std
        synthesis path (beta + truncated-normal fallback).
    n_votes : int, optional
        Number of votes to synthesize per scene when input is mean/std.
        Defaults to the `n_votes` column if present, else 30.
    seed : int, optional
        Base seed for reproducibility. None = fresh seed from time.
    n_threads : int, optional
        Number of worker threads for the Rust core. None = all available.
    verbose : bool
        If True, print a brief timing breakdown.

    Returns
    -------
    CorrelationResult
        Exposes `.summary()`, `.distributions`, `.plot_violins()`,
        `.plot_win_probabilities()`.
    """
    coeffs = normalize_coefficients(coefficients)
    if seed is None:
        seed = int(time.time_ns() & 0x7FFF_FFFF_FFFF_FFFF)

    t0 = time.perf_counter()

    sub = parse_subjective(subjective, scene_col=scene_col, vote_col=vote_col, vote_cols=vote_cols)
    obj_scenes, metric_names, obj_values = parse_objective(
        objective, scene_col=scene_col, metric_cols=metric_cols
    )
    sub, obj_aligned, common_scenes = align(sub, obj_scenes, obj_values)

    if verbose:
        print(
            f"[cwc] parsed inputs: {len(common_scenes)} overlapping scenes, "
            f"{len(metric_names)} metric(s), {len(coeffs)} coefficient(s); "
            f"took {time.perf_counter() - t0:.3f}s"
        )

    # Build the (n_scenes, n_inner) inner-sample matrix.
    t_inner = time.perf_counter()
    inner_matrix = _build_inner_matrix(
        sub, inner_bootstrap=inner_bootstrap, bounds=bounds, n_votes=n_votes, seed=seed
    )
    if verbose:
        print(
            f"[cwc] built inner matrix {inner_matrix.shape} in "
            f"{time.perf_counter() - t_inner:.3f}s"
        )

    # Run the outer bootstrap.
    t_outer = time.perf_counter()
    raw = _core.correlate(
        inner_matrix,
        obj_aligned.astype(np.float64, copy=False),
        coeffs,
        int(n_bootstrap),
        bool(bootstrap_scenes),
        int(seed),
        int(n_threads) if n_threads and n_threads > 0 else 0,
    )  # shape (n_metrics, n_coeffs, n_bootstrap)
    if verbose:
        print(
            f"[cwc] outer bootstrap ({n_bootstrap} iters x {len(metric_names)} metrics) "
            f"in {time.perf_counter() - t_outer:.3f}s"
        )

    # Repackage into dict[coeff][metric] -> ndarray.
    distributions: dict[str, dict[str, np.ndarray]] = {c: {} for c in coeffs}
    for m_idx, metric in enumerate(metric_names):
        for c_idx, coeff in enumerate(coeffs):
            distributions[coeff][metric] = np.ascontiguousarray(raw[m_idx, c_idx, :])

    return CorrelationResult(
        coefficients=coeffs,
        metric_names=list(metric_names),
        scenes=common_scenes,
        distributions=distributions,
        n_bootstrap=n_bootstrap,
        bootstrap_scenes=bootstrap_scenes,
    )


def _build_inner_matrix(
    sub: SubjectiveData,
    inner_bootstrap: int,
    bounds: Tuple[float, float],
    n_votes: Optional[int],
    seed: int,
) -> np.ndarray:
    """Dispatch to the right Rust kernel and return a (n_scenes, n_inner) matrix."""
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower >= upper:
        raise ValueError("bounds must satisfy lower < upper")

    if sub.votes is not None:
        return _core.bootstrap_inner_means(sub.votes, int(inner_bootstrap), int(seed))

    # Mean/std path: synthesize votes first, then bootstrap their mean distribution.
    assert sub.mean_std is not None
    means, stds = sub.mean_std
    if n_votes is None:
        n_votes = sub.n_votes_hint if sub.n_votes_hint is not None else 30
    synth = _core.synth_votes_mean_std(
        np.ascontiguousarray(means, dtype=np.float64),
        np.ascontiguousarray(stds, dtype=np.float64),
        int(n_votes),
        lower,
        upper,
        int(seed),
    )  # (n_scenes, n_votes)

    # Feed synthesized votes back through the bootstrap kernel.
    votes_list = [row.tolist() for row in synth]
    return _core.bootstrap_inner_means(votes_list, int(inner_bootstrap), int(seed) + 1)
