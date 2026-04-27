# Correlating Without Confidence

Bootstrap confidence intervals for correlation coefficients between subjective opinion scores and objective image-quality metrics. Companion code for [Zizien & Fliegel (2025)][paper].

The repository contains three implementations of the same method:

- **`correlation_with_confidence/`** — Python package with a Rust core. Recommended for most users. Install via pip and call from a notebook or script.
- **`Python/`** — reference implementation in pure Python (NumPy, SciPy, pandas, matplotlib). Useful for understanding the method.
- **`Go/`** — reference implementation in Go, exposed as a CLI tool. Reads CSV files and writes per-metric distributions.

The whole repository can be cloned, downloaded as a zip file, or accessed via [OSF][osf].

## Quick start (Python package)

```bash
pip install correlation_with_confidence
# for plotting helpers:
pip install "correlation_with_confidence[plot]"
```

```python
import pandas as pd
import correlation_with_confidence as cwc

sub = pd.read_csv("subjective.csv")
obj = pd.read_csv("objective.csv")

result = cwc.analyze(sub, obj, n_bootstrap=14999, seed=42)

print(result.summary())            # tidy DataFrame with mean, median, 95% CI
fig1 = result.plot_violins()
fig2 = result.plot_win_probabilities()
```

See `correlation_with_confidence/USAGE.md` for a cell-by-cell mapping from the original notebook to the new API, and `correlation_with_confidence/README.md` for full documentation.

## Reference implementations

### Python (`Python/`)

The original implementation, used to produce the figures in the paper. Relies on NumPy, SciPy, pandas, and matplotlib. See the notebooks for worked examples.

### Go (`Go/`)

Standalone command-line tool for batch correlation analysis. Reads CSV files and writes per-metric distributions plus a combined CSV. See `Go/README.md` for build and usage instructions.

## Data

The example code uses subjective ratings from the **CID 2013** dataset ([Zenodo][cid-zenodo], [research group page][cid-group]).

Objective metrics were computed using **pyiqa**; the scripts that produced them are in `Python/pyiqa_scripts/`. For more on pyiqa:

```bibtex
@misc{pyiqa,
  title         = {{IQA-PyTorch}: PyTorch Toolbox for Image Quality Assessment},
  author        = {Chaofeng Chen and Jiadi Mo},
  year          = {2022},
  howpublished  = {[Online]. Available: \url{https://github.com/chaofengc/IQA-PyTorch}}
}
```

Pre-computed metric values for CID 2013 are in `metrics_values/`.

## Citation

If you use this code, please cite the paper:

```bibtex
@Article{Zizien_2025,
  author    = {Zizien, Adam and Fliegel, Karel},
  journal   = {IEEE Access},
  title     = {Correlating Without Confidence: The Overlooked Role of Uncertainty when Ranking Objective Measures},
  year      = {2025},
  issn      = {2169-3536},
  pages     = {1--1},
  doi       = {10.1109/access.2025.3544307},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
}
```

[paper]: https://doi.org/10.1109/access.2025.3544307
[osf]: https://osf.io/2x4g8/
[cid-zenodo]: https://zenodo.org/records/2647033 "CID 2013 on Zenodo"
[cid-group]: https://researchportal.helsinki.fi/en/publications/cid2013-a-database-for-evaluating-no-reference-image-quality-asse "CID 2013 research group home page"
