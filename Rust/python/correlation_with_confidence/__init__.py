"""Bootstrap confidence intervals for correlation coefficients between
subjective opinion scores and objective metrics.

Core idea (matches the reference Python/Go implementations):
  1. For each scene with raw votes, bootstrap n_inner resampled means.
  2. Outer loop: n_bootstrap times, pick one random inner mean per scene,
     optionally resample scenes with replacement, compute correlation.
  3. Summaries (mean, median, 2.5/97.5% CI) and Cliff's-delta win-probabilities
     are derived from the resulting distributions.

Public entry points:
    cwc.analyze(subjective, objective, ...)  ->  CorrelationResult
    result.summary()                         ->  pandas.DataFrame
    result.plot_violins()                    ->  matplotlib.figure.Figure
    result.plot_win_probabilities()          ->  matplotlib.figure.Figure
"""

from .analyze import analyze
from .result import CorrelationResult
from .plots import plot_violins, plot_win_probabilities

__all__ = [
    "analyze",
    "CorrelationResult",
    "plot_violins",
    "plot_win_probabilities",
]

__version__ = "0.1.0"
