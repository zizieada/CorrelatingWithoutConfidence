"""DataFrame layout detection and normalization.

Subjective data can arrive in three shapes:

  Layout A -- "long" raw votes (one row per vote)
        scene_col  vote_col
        sceneA     81
        sceneA     79
        sceneA     84
        sceneB     62
        ...

  Layout B -- "wide" raw votes (one row per scene, missing allowed)
        scene_col  v1    v2   v3   v4   ...
        sceneA     81    79   84   NaN
        sceneB     62    58   NaN  NaN

  Layout C -- mean/std summary
        scene_col  mean   std   [n_votes]
        sceneA     81.3   3.1   31
        sceneB     61.2   8.4   29

Detection precedence:
  1. If one column name matches `mean|mos` AND another matches `std|sd|stddev`
     -> Layout C.
  2. Else if the scene column has duplicate values -> Layout A.
  3. Else -> Layout B.

Objective data must be "wide": one row per scene, columns = metric values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_SCENE_COL_GUESSES = ("name", "scene", "stimulus", "stim", "image", "id")
_MEAN_PATTERN = re.compile(r"^(mean|mos)$", re.IGNORECASE)
_STD_PATTERN = re.compile(r"^(std|stddev|sd)$", re.IGNORECASE)
_NVOTES_PATTERN = re.compile(r"^(n_?votes?|count)$", re.IGNORECASE)


@dataclass
class SubjectiveData:
    """Normalized subjective input.

    One of `votes` or `mean_std` is set. `scenes` lists scene names in the
    same order as the rows of `votes` or `mean_std`.
    """

    scenes: list[str]
    votes: Optional[list[list[float]]] = None
    mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (means, stds)
    n_votes_hint: Optional[int] = None  # only for mean/std path


def _guess_scene_col(df: pd.DataFrame) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for guess in _SCENE_COL_GUESSES:
        if guess in lowered:
            return lowered[guess]
    return df.columns[0]


def parse_subjective(
    df: Union[pd.DataFrame, dict],
    scene_col: Optional[str] = None,
    vote_col: Optional[str] = None,
    vote_cols: Optional[Sequence[str]] = None,
) -> SubjectiveData:
    """Normalize subjective input into a SubjectiveData record."""
    # Dict shortcut: {scene_name: [votes, ...]}
    if isinstance(df, dict):
        scenes = list(df.keys())
        votes = [list(df[k]) for k in scenes]
        if any(len(v) == 0 for v in votes):
            raise ValueError("one or more scenes have an empty vote list")
        return SubjectiveData(scenes=[str(s) for s in scenes], votes=votes)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"subjective must be a DataFrame or dict, got {type(df).__name__}")

    if scene_col is None:
        scene_col = _guess_scene_col(df)
    if scene_col not in df.columns:
        raise ValueError(f"scene_col {scene_col!r} not found in DataFrame columns")

    other_cols = [c for c in df.columns if c != scene_col]
    mean_col = next((c for c in other_cols if _MEAN_PATTERN.match(c)), None)
    std_col = next((c for c in other_cols if _STD_PATTERN.match(c)), None)

    # Layout C -- mean/std
    if mean_col and std_col:
        sub = df[[scene_col, mean_col, std_col]].dropna()
        n_votes_col = next((c for c in other_cols if _NVOTES_PATTERN.match(c)), None)
        n_votes_hint = None
        if n_votes_col and n_votes_col in df.columns:
            n_votes_hint = int(df[n_votes_col].max())
        return SubjectiveData(
            scenes=sub[scene_col].astype(str).tolist(),
            mean_std=(sub[mean_col].to_numpy(float), sub[std_col].to_numpy(float)),
            n_votes_hint=n_votes_hint,
        )

    # Layout A -- long (scene column has duplicates)
    if df[scene_col].duplicated().any():
        if vote_col is None:
            numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) != 1:
                raise ValueError(
                    f"long layout detected but vote_col is ambiguous; "
                    f"candidates: {numeric_cols}. Pass vote_col=..."
                )
            vote_col = numeric_cols[0]
        grouped = df.groupby(scene_col)[vote_col].apply(
            lambda s: s.dropna().astype(float).tolist()
        )
        scenes, votes = [], []
        for s, v in grouped.items():
            if len(v) > 0:
                scenes.append(str(s))
                votes.append(v)
        return SubjectiveData(scenes=scenes, votes=votes)

    # Layout B -- wide raw votes
    cols_to_use = list(vote_cols) if vote_cols is not None else other_cols
    cols_to_use = [c for c in cols_to_use if c in df.columns]
    if not cols_to_use:
        raise ValueError("wide layout detected but no vote columns found")
    scenes, votes = [], []
    for _, row in df[[scene_col] + cols_to_use].iterrows():
        name = str(row[scene_col])
        v = row[cols_to_use].to_numpy(dtype=float, na_value=np.nan)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        scenes.append(name)
        votes.append(v.tolist())
    return SubjectiveData(scenes=scenes, votes=votes)


def parse_objective(
    df: pd.DataFrame,
    scene_col: Optional[str] = None,
    metric_cols: Optional[Sequence[str]] = None,
) -> Tuple[list[str], list[str], np.ndarray]:
    """Normalize objective input.

    Returns (scene_names, metric_names, values) where `values` has shape
    (n_scenes, n_metrics) with NaN for missing cells.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"objective must be a DataFrame, got {type(df).__name__}")

    if scene_col is None:
        scene_col = _guess_scene_col(df)
    if scene_col not in df.columns:
        raise ValueError(f"scene_col {scene_col!r} not found in DataFrame columns")

    if metric_cols is None:
        metric_cols = [c for c in df.columns if c != scene_col]
        metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        raise ValueError("objective DataFrame has no numeric metric columns")

    scenes = df[scene_col].astype(str).tolist()
    values = df[list(metric_cols)].to_numpy(dtype=float)
    return scenes, list(metric_cols), values


def align(
    subjective: SubjectiveData,
    obj_scenes: list[str],
    obj_values: np.ndarray,
) -> Tuple[SubjectiveData, np.ndarray, list[str]]:
    """Inner-join subjective and objective on scene name.

    Returns (subjective_aligned, objective_values_aligned, common_scenes).
    """
    obj_index = {s: i for i, s in enumerate(obj_scenes)}
    common_scenes = [s for s in subjective.scenes if s in obj_index]
    if not common_scenes:
        raise ValueError(
            "no scenes overlap between subjective and objective inputs "
            "(check spelling/case of scene identifiers)"
        )

    keep_idx = [subjective.scenes.index(s) for s in common_scenes]
    if subjective.votes is not None:
        aligned_votes = [subjective.votes[i] for i in keep_idx]
        aligned = SubjectiveData(scenes=common_scenes, votes=aligned_votes)
    else:
        means, stds = subjective.mean_std  # type: ignore[misc]
        aligned = SubjectiveData(
            scenes=common_scenes,
            mean_std=(means[keep_idx], stds[keep_idx]),
            n_votes_hint=subjective.n_votes_hint,
        )

    obj_rows = [obj_index[s] for s in common_scenes]
    obj_aligned = obj_values[obj_rows, :]
    return aligned, obj_aligned, common_scenes


def normalize_coefficients(coeffs: Iterable[str]) -> list[str]:
    """Canonicalize coefficient names to ('pearson'|'spearman'|'kendall')."""
    mapping = {
        "r": "pearson", "pearson": "pearson",
        "rho": "spearman", "rs": "spearman", "spearman": "spearman",
        "tau": "kendall", "kendall": "kendall",
    }
    out = []
    for c in coeffs:
        key = str(c).lower()
        if key not in mapping:
            raise ValueError(
                f"unknown correlation coefficient {c!r}; "
                f"expected one of pearson/spearman/kendall (or r/rho/tau)"
            )
        canonical = mapping[key]
        if canonical not in out:
            out.append(canonical)
    return out
