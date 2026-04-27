"""Plotting helpers. Importing this module requires matplotlib."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

_MATPLOTLIB_ERROR = (
    "matplotlib is required for plotting. Install with: "
    "pip install 'correlation_with_confidence[plot]' "
    "(or `pip install matplotlib`)."
)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MATPLOTLIB_ERROR) from exc
    return __import__("matplotlib.pyplot", fromlist=["plt"])


def plot_violins(
    result,
    figsize: Optional[Tuple[float, float]] = None,
    stagger_labels: bool = True,
    ci: float = 0.95,
    title_fmt: str = "{coeff} correlation",
    sharey: bool = True,
):
    """Violin plot of correlation distributions, one subplot per coefficient.

    Metrics within each subplot are sorted by mean correlation (descending),
    with red horizontal lines at the given CI bounds.

    Parameters
    ----------
    result : CorrelationResult
    figsize : (w, h), optional. Default scales with number of coefficients.
    stagger_labels : bool. Alternate y-offset on x-tick labels to reduce overlap.
    ci : float in (0, 1). Confidence interval marked by red bars. Default 0.95.
    title_fmt : str. Format string for subplot titles, `{coeff}` replaced.
    sharey : bool. Share y-axis limits across subplots.
    """
    plt = _require_matplotlib()
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = 100 * alpha, 100 * (1 - alpha)

    n_coeffs = len(result.coefficients)
    if figsize is None:
        figsize = (max(10, 6 + 2 * len(result.metric_names)), 6)
    fig, axes = plt.subplots(1, n_coeffs, figsize=figsize, sharey=sharey, squeeze=False)
    axes = axes.flatten()

    for ax, coeff in zip(axes, result.coefficients):
        means = {m: float(np.mean(result.distributions[coeff][m])) for m in result.metric_names}
        labels = sorted(result.metric_names, key=lambda m: -means[m])
        data = [np.abs(result.distributions[coeff][m]) for m in labels]

        ax.violinplot(data, showmeans=False, showmedians=True)
        for i, d in enumerate(data, start=1):
            lo, hi = np.percentile(d, [lo_q, hi_q])
            ax.hlines([lo, hi], i - 0.15, i + 0.15, color="red", linestyle="-", lw=2)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=0)
        if stagger_labels:
            for i, lbl in enumerate(ax.get_xticklabels()):
                pos = lbl.get_position()
                lbl.set_y(pos[1] - (0.01 if i % 2 == 0 else 0.06))

        ax.set_ylim(0, 1)
        ax.set_title(title_fmt.format(coeff=coeff.capitalize()))
        ax.set_ylabel(f"|{coeff}|")
    fig.tight_layout()
    return fig


def plot_win_probabilities(
    result,
    figsize: Optional[Tuple[float, float]] = None,
    annotate: bool = True,
    cmap: str = "coolwarm",
    stagger_labels: bool = True,
    title_fmt: str = "Probability of higher {coeff} correlation (%)",
):
    """Pairwise Cliff's-delta win-probability heatmap, one per coefficient.

    Cell (i, j) shows P(metric_i > metric_j) derived from Cliff's delta:
    ((delta + 1) / 2) * 100.
    """
    plt = _require_matplotlib()

    n_coeffs = len(result.coefficients)
    n_metrics = len(result.metric_names)
    if figsize is None:
        figsize = (max(10, 5 + 1.5 * n_metrics) * n_coeffs / 3, max(8, 1.2 * n_metrics))
    fig, axes = plt.subplots(1, n_coeffs, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, coeff in zip(axes, result.coefficients):
        wp = result.win_probability_matrix(coeff, sort_by_mean=True)
        im = ax.imshow(wp.to_numpy(), cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(np.arange(n_metrics))
        ax.set_yticks(np.arange(n_metrics))
        ax.set_xticklabels(list(wp.columns))
        ax.set_yticklabels(list(wp.index))
        if stagger_labels:
            for i, lbl in enumerate(ax.get_xticklabels()):
                pos = lbl.get_position()
                lbl.set_y(pos[1] - (0.01 if i % 2 == 0 else 0.06))
        if annotate:
            arr = wp.to_numpy()
            for i in range(n_metrics):
                for j in range(n_metrics):
                    ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center",
                            color="black", fontsize=9)
        ax.set_title(title_fmt.format(coeff=coeff.capitalize()))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def plot_win_probabilities_thresholded(
    result,
    threshold: float = 95.0,
    figsize: Optional[Tuple[float, float]] = None,
    stagger_labels: bool = True,
):
    """Ternary (better / same / worse) version of the win-probability heatmap.

    A cell is "same" when its win probability lies in (100-threshold, threshold),
    "better" when >= threshold, "worse" when <= 100-threshold.
    Matches notebook cell 15.
    """
    plt = _require_matplotlib()
    if not 50 < threshold <= 100:
        raise ValueError("threshold must be in (50, 100]")

    n_coeffs = len(result.coefficients)
    n_metrics = len(result.metric_names)
    if figsize is None:
        figsize = (max(10, 5 + 1.2 * n_metrics) * n_coeffs / 3, max(8, 1.0 * n_metrics))
    fig, axes = plt.subplots(1, n_coeffs, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, coeff in zip(axes, result.coefficients):
        wp = result.win_probability_matrix(coeff, sort_by_mean=True).to_numpy().copy()
        better = wp >= threshold
        worse = wp <= 100 - threshold
        same = ~(better | worse)
        wp[better] = 0.0
        wp[same] = 0.5
        wp[worse] = 1.0

        ax.imshow(wp, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks(np.arange(n_metrics))
        ax.set_yticks(np.arange(n_metrics))
        ax.set_xticklabels(result.metric_names)
        ax.set_yticklabels(result.metric_names)
        if stagger_labels:
            for i, lbl in enumerate(ax.get_xticklabels()):
                pos = lbl.get_position()
                lbl.set_y(pos[1] - (0.01 if i % 2 == 0 else 0.04))
        ax.set_title(f"{coeff.capitalize()} (th={threshold:g}%)")
        ax.set_xticks(np.arange(-0.5, n_metrics), minor=True)
        ax.set_yticks(np.arange(-0.5, n_metrics), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.75)

    fig.tight_layout()
    return fig
