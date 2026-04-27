//! Rust core for correlation_with_confidence.
//!
//! Exposes four functions to Python:
//!   - bootstrap_inner_means(votes_ragged, n_inner, seed)
//!   - synth_votes_mean_std(means, stds, n_votes, lower, upper, seed)
//!   - correlate(inner, objective, coeffs, n_bootstrap, bootstrap_scenes, seed)
//!   - cliffs_delta_matrix(distributions)
//!
//! All hot paths are parallelized with rayon using stride partitioning,
//! making `-seed` bit-reproducible across runs given a fixed thread count.

use ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF, Normal};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn worker_rng(base_seed: u64, worker_id: u64) -> ChaCha8Rng {
    // XOR with a shifted worker id so worker 0 is distinct from base.
    ChaCha8Rng::seed_from_u64(base_seed ^ worker_id.wrapping_add(0x9E37_79B9_7F4A_7C15))
}

/// Pearson correlation via stable one-pass algorithm. Returns 0.0 if
/// either input has zero variance.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let inv_n = 1.0 / n as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..n {
        mx += x[i];
        my += y[i];
    }
    mx *= inv_n;
    my *= inv_n;

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        sxy / denom
    }
}

/// Fractional ranks with ties averaged, written into `out`.
/// `scratch` is reusable buffer of indices.
fn rank_into(values: &[f64], out: &mut [f64], scratch: &mut Vec<usize>) {
    let n = values.len();
    scratch.clear();
    scratch.extend(0..n);
    scratch.sort_unstable_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        let v = values[scratch[i]];
        while j < n && values[scratch[j]] == v {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0; // 1-indexed ranks
        for k in i..j {
            out[scratch[k]] = avg;
        }
        i = j;
    }
}

fn spearman(x: &[f64], y: &[f64], rx: &mut Vec<f64>, ry: &mut Vec<f64>, scratch: &mut Vec<usize>) -> f64 {
    rx.resize(x.len(), 0.0);
    ry.resize(y.len(), 0.0);
    rank_into(x, rx, scratch);
    rank_into(y, ry, scratch);
    pearson(rx, ry)
}

fn kendall_tau_a(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut c: i64 = 0;
    let mut d: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let prod = dx * dy;
            if prod > 0.0 {
                c += 1;
            } else if prod < 0.0 {
                d += 1;
            }
        }
    }
    let total = (n * (n - 1) / 2) as i64;
    if total == 0 {
        0.0
    } else {
        (c - d) as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Coefficient enum
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
enum Coeff {
    Pearson,
    Spearman,
    Kendall,
}

fn parse_coeff(name: &str) -> PyResult<Coeff> {
    match name.to_ascii_lowercase().as_str() {
        "pearson" | "r" => Ok(Coeff::Pearson),
        "spearman" | "rho" | "rs" => Ok(Coeff::Spearman),
        "kendall" | "tau" => Ok(Coeff::Kendall),
        other => Err(PyValueError::new_err(format!(
            "unknown correlation coefficient: {other:?} (expected pearson|spearman|kendall)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// bootstrap_inner_means
// ---------------------------------------------------------------------------

/// For each scene's raw votes, bootstrap `n_inner` mean samples.
///
/// Returns a dense (n_scenes, n_inner) array where row i holds n_inner
/// bootstrapped means of scene i's votes (sampled with replacement, size = len(votes_i)).
#[pyfunction]
#[pyo3(signature = (votes, n_inner, seed))]
fn bootstrap_inner_means<'py>(
    py: Python<'py>,
    votes: Vec<Vec<f64>>,
    n_inner: usize,
    seed: u64,
) -> PyResult<&'py PyArray2<f64>> {
    let n_scenes = votes.len();
    if n_scenes == 0 {
        return Err(PyValueError::new_err("empty votes list"));
    }
    for (i, v) in votes.iter().enumerate() {
        if v.is_empty() {
            return Err(PyValueError::new_err(format!(
                "scene {i} has no votes"
            )));
        }
    }

    let mut out = Array2::<f64>::zeros((n_scenes, n_inner));

    py.allow_threads(|| {
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let mut rng = worker_rng(seed, (i as u64).wrapping_add(1));
                let v = &votes[i];
                let n = v.len();
                let inv_n = 1.0 / n as f64;
                for j in 0..n_inner {
                    let mut sum = 0.0;
                    for _ in 0..n {
                        sum += v[rng.gen_range(0..n)];
                    }
                    row[j] = sum * inv_n;
                }
            });
    });

    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// synth_votes_mean_std (beta with truncated-normal fallback)
// ---------------------------------------------------------------------------

fn beta_samples(
    rng: &mut ChaCha8Rng,
    mean: f64,
    std: f64,
    lower: f64,
    upper: f64,
    n: usize,
    out: &mut [f64],
) -> bool {
    if mean <= 0.0 || std <= 0.0 || mean < lower || mean > upper {
        out.fill(mean);
        return false;
    }
    let rescaled_mean = (mean - lower) / (upper - lower);
    let rescaled_var = (std / (upper - lower)).powi(2);
    if rescaled_var < 1e-10 {
        out.fill(mean);
        return false;
    }
    let temp = (rescaled_mean * (1.0 - rescaled_mean) / rescaled_var) - 1.0;
    let alpha = rescaled_mean * temp;
    let beta_p = (1.0 - rescaled_mean) * temp;
    if alpha <= 0.0 || beta_p <= 0.0 {
        out.fill(mean - 0.5);
        return false;
    }
    let dist = match Beta::new(alpha, beta_p) {
        Ok(d) => d,
        Err(_) => {
            out.fill(mean);
            return false;
        }
    };
    for i in 0..n {
        let u = rng.gen::<f64>();
        // statrs Beta doesn't expose a sampler that takes &mut RNG directly without `rand::distributions::Distribution`;
        // use inverse CDF instead (monotonic, reproducible given RNG).
        out[i] = lower + dist.inverse_cdf(u) * (upper - lower);
    }
    true
}

/// Iterative adjustment of (mu, sigma) so that truncating N(mu, sigma)
/// to [lower, upper] yields samples with the requested (mean, std).
fn adjust_truncated_normal(
    target_mean: f64,
    target_std: f64,
    lower: f64,
    upper: f64,
    max_iter: usize,
    eps: f64,
) -> (f64, f64) {
    let norm = Normal::new(0.0, 1.0).unwrap();
    let pdf = |x: f64| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = |x: f64| norm.cdf(x);

    let mut mu = target_mean;
    let mut sigma = target_std;

    for _ in 0..max_iter {
        if sigma < eps {
            break;
        }
        let a = (lower - mu) / sigma;
        let b = (upper - mu) / sigma;
        let pdf_a = pdf(a);
        let pdf_b = pdf(b);
        let cdf_a = cdf(a);
        let cdf_b = cdf(b);
        let alpha = pdf_a - pdf_b;
        let beta = cdf_b - cdf_a;
        if beta.abs() < eps {
            break;
        }
        let trunc_mean = mu + (alpha / beta) * sigma;
        let var_term = (1.0 + ((a * pdf_a - b * pdf_b) / beta) - (alpha / beta).powi(2)).max(0.0);
        let trunc_std = (sigma * sigma * var_term).sqrt();
        mu += target_mean - trunc_mean;
        if trunc_std < eps {
            break;
        }
        sigma *= target_std / trunc_std;
    }
    (mu, sigma)
}

fn trunc_normal_samples(
    rng: &mut ChaCha8Rng,
    mean: f64,
    std: f64,
    lower: f64,
    upper: f64,
    out: &mut [f64],
) {
    let dist = match Normal::new(mean, std.max(1e-12)) {
        Ok(d) => d,
        Err(_) => {
            out.fill(mean);
            return;
        }
    };
    let cdf_lower = dist.cdf(lower);
    let cdf_upper = dist.cdf(upper);
    let span = cdf_upper - cdf_lower;
    if span.abs() < 1e-12 {
        out.fill(mean);
        return;
    }
    for v in out.iter_mut() {
        let u = cdf_lower + rng.gen::<f64>() * span;
        *v = dist.inverse_cdf(u);
    }
}

fn sample_stats(s: &[f64]) -> (f64, f64) {
    let n = s.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean: f64 = s.iter().sum::<f64>() / n;
    let var: f64 = s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn get_samples(
    rng: &mut ChaCha8Rng,
    mean: f64,
    std: f64,
    lower: f64,
    upper: f64,
    out: &mut [f64],
) {
    if std <= 0.0 {
        out.fill(mean);
        return;
    }
    let _ = beta_samples(rng, mean, std, lower, upper, out.len(), out);
    let (sm, ss) = sample_stats(out);
    if (sm - mean).abs() > 0.2 || (ss - std).abs() > 0.1 {
        let (adj_m, adj_s) = adjust_truncated_normal(mean, std, lower, upper, 100, 1e-8);
        let mut tn = vec![0.0; out.len()];
        trunc_normal_samples(rng, adj_m, adj_s, lower, upper, &mut tn);
        let (tm, ts) = sample_stats(&tn);
        let beta_err = (sm - mean).abs() + (ss - std).abs();
        let tn_err = (tm - mean).abs() + (ts - std).abs();
        if tn_err < beta_err {
            out.copy_from_slice(&tn);
        }
    }
}

/// Synthesize `n_votes` votes per scene from per-scene (mean, std).
/// Returns shape (n_scenes, n_votes). Uses beta with truncated-normal fallback.
#[pyfunction]
#[pyo3(signature = (means, stds, n_votes, lower, upper, seed))]
fn synth_votes_mean_std<'py>(
    py: Python<'py>,
    means: PyReadonlyArray1<f64>,
    stds: PyReadonlyArray1<f64>,
    n_votes: usize,
    lower: f64,
    upper: f64,
    seed: u64,
) -> PyResult<&'py PyArray2<f64>> {
    if means.len() != stds.len() {
        return Err(PyValueError::new_err("means and stds have different lengths"));
    }
    if lower >= upper {
        return Err(PyValueError::new_err("lower must be < upper"));
    }
    let means_v = means.as_slice()?.to_vec();
    let stds_v = stds.as_slice()?.to_vec();
    let n_scenes = means_v.len();

    let mut out = Array2::<f64>::zeros((n_scenes, n_votes));

    py.allow_threads(|| {
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let mut rng = worker_rng(seed, (i as u64).wrapping_add(1));
                let row_slice = row.as_slice_mut().expect("contiguous");
                get_samples(&mut rng, means_v[i], stds_v[i], lower, upper, row_slice);
            });
    });

    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// correlate: main bootstrap loop over coefficients
// ---------------------------------------------------------------------------

/// Bootstrap correlation coefficients.
///
/// inner: shape (n_scenes, n_inner). Row i is the bootstrap-sample distribution of
///        scene i's mean.
/// objective: shape (n_scenes, n_metrics). NaN cells cause that scene to be excluded
///        for that metric only.
/// coeffs: list of coefficient names ("pearson"/"spearman"/"kendall").
/// n_threads: number of worker threads (0 = use rayon's global pool).
/// Returns (n_metrics, n_coeffs, n_bootstrap) — absolute values of correlations.
#[pyfunction]
#[pyo3(signature = (inner, objective, coeffs, n_bootstrap, bootstrap_scenes, seed, n_threads=0))]
#[allow(clippy::too_many_arguments)]
fn correlate<'py>(
    py: Python<'py>,
    inner: PyReadonlyArray2<f64>,
    objective: PyReadonlyArray2<f64>,
    coeffs: Vec<String>,
    n_bootstrap: usize,
    bootstrap_scenes: bool,
    seed: u64,
    n_threads: usize,
) -> PyResult<&'py PyArray3<f64>> {
    let inner_view = inner.as_array();
    let obj_view = objective.as_array();
    if inner_view.nrows() != obj_view.nrows() {
        return Err(PyValueError::new_err(format!(
            "inner and objective have different n_scenes: {} vs {}",
            inner_view.nrows(),
            obj_view.nrows()
        )));
    }
    if n_bootstrap == 0 {
        return Err(PyValueError::new_err("n_bootstrap must be positive"));
    }
    let coeff_enums: Vec<Coeff> = coeffs
        .iter()
        .map(|s| parse_coeff(s))
        .collect::<PyResult<_>>()?;

    let n_metrics = obj_view.ncols();
    let n_coeffs = coeff_enums.len();
    let mut out = Array3::<f64>::zeros((n_metrics, n_coeffs, n_bootstrap));

    // Ensure `inner` and `objective` are contiguous so we can take raw slices
    // for hot-loop indexing (avoids ndarray 2D index arithmetic overhead).
    let inner_owned = inner_view.to_owned();
    let obj_owned = obj_view.to_owned();
    let inner_cols = inner_owned.ncols();
    let obj_cols = obj_owned.ncols();
    let inner_flat: &[f64] = inner_owned.as_slice().expect("contiguous");
    let obj_flat: &[f64] = obj_owned.as_slice().expect("contiguous");

    // Decide chunk count: aim for ~4x thread count chunks per metric for load balance.
    let effective_threads = if n_threads > 0 {
        n_threads
    } else {
        rayon::current_num_threads().max(1)
    };
    let target_chunks = effective_threads.saturating_mul(4).max(1);
    let chunk_size = ((n_bootstrap + target_chunks - 1) / target_chunks).max(32);

    // Build the per-call thread pool once (if requested) and run inside it.
    let pool_opt = if n_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .ok()
    } else {
        None
    };

    py.allow_threads(|| {
        let out_ref = &mut out;
        let coeff_ref = &coeff_enums;
        let inner_flat_ref: &[f64] = inner_flat;
        let obj_flat_ref: &[f64] = obj_flat;
        let nrows = obj_owned.nrows();

        let mut run = || {
            for metric_idx in 0..n_metrics {
                // Collect valid rows for this metric (NaN-aware).
                let mut valid_rows: Vec<usize> = Vec::with_capacity(nrows);
                for i in 0..nrows {
                    let v = obj_flat_ref[i * obj_cols + metric_idx];
                    if v.is_finite() {
                        valid_rows.push(i);
                    }
                }
                if valid_rows.len() < 3 {
                    continue;
                }

                let metric_seed = seed
                    .wrapping_add((metric_idx as u64).wrapping_mul(0xA24B_AED4_963E_E407));

                // Chunks along the bootstrap axis for this metric's slab.
                let mut metric_slab = out_ref.index_axis_mut(Axis(0), metric_idx);
                let chunks: Vec<ndarray::ArrayViewMut2<f64>> = metric_slab
                    .axis_chunks_iter_mut(Axis(1), chunk_size)
                    .collect();

                chunks
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(chunk_id, mut chunk)| {
                        let start_iter = chunk_id * chunk_size;
                        run_chunk(
                            coeff_ref,
                            inner_flat_ref,
                            inner_cols,
                            obj_flat_ref,
                            obj_cols,
                            metric_idx,
                            &valid_rows,
                            bootstrap_scenes,
                            metric_seed,
                            chunk_id as u64,
                            start_iter,
                            &mut chunk,
                        );
                    });
            }
        };

        match pool_opt {
            Some(p) => p.install(run),
            None => run(),
        }
    });

    Ok(out.into_pyarray(py))
}

/// Run one chunk of bootstrap iterations for a single metric.
/// Writes absolute correlation values into `chunk`, which has shape
/// (n_coeffs, chunk_len).
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    coeffs: &[Coeff],
    inner_flat: &[f64],
    inner_cols: usize,
    obj_flat: &[f64],
    obj_cols: usize,
    metric_idx: usize,
    valid_rows: &[usize],
    bootstrap_scenes: bool,
    metric_seed: u64,
    chunk_id: u64,
    start_iter: usize,
    chunk: &mut ndarray::ArrayViewMut2<f64>,
) {
    let n_valid = valid_rows.len();
    let chunk_len = chunk.ncols();

    // Seed unique to (metric, chunk). Mixing constant chosen from splitmix64.
    let chunk_seed = metric_seed
        ^ chunk_id
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed);

    let mut boot_subj = vec![0.0_f64; n_valid];
    let mut obj_vec = vec![0.0_f64; n_valid];
    let mut rank_x: Vec<f64> = Vec::with_capacity(n_valid);
    let mut rank_y: Vec<f64> = Vec::with_capacity(n_valid);
    let mut rank_scratch: Vec<usize> = Vec::with_capacity(n_valid);

    // obj_vec is constant across iterations when scenes are not resampled.
    if !bootstrap_scenes {
        for (k, &row) in valid_rows.iter().enumerate() {
            obj_vec[k] = obj_flat[row * obj_cols + metric_idx];
        }
    }

    let _ = start_iter; // chunk-relative indexing only; absolute iter isn't needed here

    for j in 0..chunk_len {
        if bootstrap_scenes {
            for k in 0..n_valid {
                let scene = valid_rows[rng.gen_range(0..n_valid)];
                let col = rng.gen_range(0..inner_cols);
                boot_subj[k] = inner_flat[scene * inner_cols + col];
                obj_vec[k] = obj_flat[scene * obj_cols + metric_idx];
            }
        } else {
            for k in 0..n_valid {
                let scene = valid_rows[k];
                let col = rng.gen_range(0..inner_cols);
                boot_subj[k] = inner_flat[scene * inner_cols + col];
            }
        }

        for (c_idx, &c) in coeffs.iter().enumerate() {
            let v = match c {
                Coeff::Pearson => pearson(&boot_subj, &obj_vec),
                Coeff::Spearman => {
                    spearman(&boot_subj, &obj_vec, &mut rank_x, &mut rank_y, &mut rank_scratch)
                }
                Coeff::Kendall => kendall_tau_a(&boot_subj, &obj_vec),
            };
            chunk[[c_idx, j]] = v.abs();
        }
    }
}

// ---------------------------------------------------------------------------
// cliffs_delta_matrix
// ---------------------------------------------------------------------------

/// Efficient Cliff's delta between two sorted sequences using merge-based counting.
/// Returns (delta in [-1, 1]).
fn cliffs_delta_sorted(a_sorted: &[f64], b_sorted: &[f64]) -> f64 {
    let m = a_sorted.len();
    let n = b_sorted.len();
    if m == 0 || n == 0 {
        return 0.0;
    }
    // Count pairs where a > b and where a < b.
    // For each a, find how many b < a (gives "a>b" count) and how many b > a (gives "a<b" count).
    let mut gt: u64 = 0;
    let mut lt: u64 = 0;
    for &a in a_sorted {
        // number of b_sorted strictly less than a
        let lt_b = b_sorted.partition_point(|&x| x < a) as u64;
        // number of b_sorted strictly greater than a
        let le_b = b_sorted.partition_point(|&x| x <= a) as u64;
        let gt_b = n as u64 - le_b;
        gt += lt_b;
        lt += gt_b;
    }
    let total = (m as u64) * (n as u64);
    (gt as f64 - lt as f64) / total as f64
}

/// Pairwise Cliff's delta matrix across a list of 1-D distributions.
/// Output[i, j] = Cliff's delta of distribution i vs distribution j.
/// Diagonal is 0. Matrix is antisymmetric.
#[pyfunction]
fn cliffs_delta_matrix<'py>(
    py: Python<'py>,
    distributions: Vec<PyReadonlyArray1<f64>>,
) -> PyResult<&'py PyArray2<f64>> {
    let n = distributions.len();
    if n == 0 {
        return Err(PyValueError::new_err("empty distributions"));
    }

    // Pre-sort each distribution once (O(k log k)) to reuse during O(k log k) pairwise counting.
    let sorted: Vec<Vec<f64>> = distributions
        .iter()
        .map(|arr| {
            let mut v = arr.as_slice().unwrap_or(&[]).to_vec();
            v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            v
        })
        .collect();

    let mut out = Array2::<f64>::zeros((n, n));

    py.allow_threads(|| {
        // Parallelize over pairs (i, j) with i < j.
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        let deltas: Vec<((usize, usize), f64)> = pairs
            .par_iter()
            .map(|&(i, j)| ((i, j), cliffs_delta_sorted(&sorted[i], &sorted[j])))
            .collect();

        for ((i, j), d) in deltas {
            out[[i, j]] = d;
            out[[j, i]] = -d;
        }
    });

    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap_inner_means, m)?)?;
    m.add_function(wrap_pyfunction!(synth_votes_mean_std, m)?)?;
    m.add_function(wrap_pyfunction!(correlate, m)?)?;
    m.add_function(wrap_pyfunction!(cliffs_delta_matrix, m)?)?;
    Ok(())
}
