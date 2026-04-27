"""Container for bootstrap correlation results + convenience methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import _core


@dataclass
class CorrelationResult:
    """Bootstrap correlation distributions + summary/plotting helpers."""

    coefficients: list[str]
    metric_names: list[str]
    scenes: list[str]
    distributions: dict[str, dict[str, np.ndarray]]
    n_bootstrap: int
    bootstrap_scenes: bool
    _cliffs_cache: dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary(
        self,
        ci: float = 0.95,
        sort: bool = True,
    ) -> pd.DataFrame:
        """Return a tidy DataFrame summarising the bootstrap distributions.

        Columns: coefficient, metric, mean, median, ci_low, ci_high, rank.
        Within each coefficient, rows are sorted by mean (desc) when sort=True
        and a per-coefficient rank is assigned.
        """
        if not 0 < ci < 1:
            raise ValueError("ci must be in (0, 1), e.g. 0.95")
        alpha = (1.0 - ci) / 2.0
        lo_q, hi_q = 100 * alpha, 100 * (1 - alpha)

        rows = []
        for coeff in self.coefficients:
            for metric in self.metric_names:
                vals = self.distributions[coeff][metric]
                lo, hi = np.percentile(vals, [lo_q, hi_q])
                rows.append({
                    "coefficient": coeff,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                })
        df = pd.DataFrame(rows)

        if sort:
            df = (
                df.sort_values(["coefficient", "mean"], ascending=[True, False])
                  .reset_index(drop=True)
            )
            df["rank"] = df.groupby("coefficient").cumcount() + 1
        return df

    # ------------------------------------------------------------------
    # Cliff's delta / win-probability matrix
    # ------------------------------------------------------------------

    def cliffs_delta_matrix(
        self,
        coefficient: str,
        sort_by_mean: bool = True,
    ) -> pd.DataFrame:
        """Pairwise Cliff's delta between all metrics for one coefficient.

        Value at [i, j] in [-1, 1]: how much metric i's distribution dominates
        metric j's. Apply `((d + 1) / 2) * 100` to get win probability in percent.
        """
        cache_key = f"{coefficient}::{'sorted' if sort_by_mean else 'unsorted'}"
        if cache_key in self._cliffs_cache:
            return self._cliffs_cache[cache_key]

        coeff = _canonicalize(coefficient, self.coefficients)

        if sort_by_mean:
            means = {m: float(np.mean(self.distributions[coeff][m])) for m in self.metric_names}
            metrics_ordered = sorted(self.metric_names, key=lambda m: -means[m])
        else:
            metrics_ordered = list(self.metric_names)

        dists = [
            np.ascontiguousarray(self.distributions[coeff][m], dtype=np.float64)
            for m in metrics_ordered
        ]
        matrix = _core.cliffs_delta_matrix(dists)
        df = pd.DataFrame(matrix, index=metrics_ordered, columns=metrics_ordered)
        self._cliffs_cache[cache_key] = df
        return df

    def win_probability_matrix(
        self,
        coefficient: str,
        sort_by_mean: bool = True,
    ) -> pd.DataFrame:
        """Return the Cliff's-delta matrix mapped to win probabilities in [0, 100]."""
        delta = self.cliffs_delta_matrix(coefficient, sort_by_mean=sort_by_mean)
        return (delta + 1.0) / 2.0 * 100.0

    # ------------------------------------------------------------------
    # Plot shortcuts (optional dependency on matplotlib)
    # ------------------------------------------------------------------

    def plot_violins(self, **kwargs):
        from .plots import plot_violins
        return plot_violins(self, **kwargs)

    def plot_win_probabilities(self, **kwargs):
        from .plots import plot_win_probabilities
        return plot_win_probabilities(self, **kwargs)


def _canonicalize(name: str, available: list[str]) -> str:
    mapping = {
        "r": "pearson", "pearson": "pearson",
        "rho": "spearman", "rs": "spearman", "spearman": "spearman",
        "tau": "kendall", "kendall": "kendall",
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"unknown coefficient {name!r}")
    canonical = mapping[key]
    if canonical not in available:
        raise ValueError(
            f"coefficient {canonical!r} was not computed; available: {available}"
        )
    return canonical
