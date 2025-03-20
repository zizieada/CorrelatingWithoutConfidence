import numpy as np
import scipy.stats as stats
import multiprocessing as mp
from functools import partial
import pandas as pd
import warnings
from typing import Dict, List, Union, Tuple, Optional, TypeVar, Sequence
from pathlib import Path
from itertools import combinations

T = TypeVar("T", int, float)


def create_mos_distributions(
    scene_votes: Union[Dict[int, List[float]], List[Tuple[float, float, int]]],
    n_iterations: int = 10000,
    n_processes: Optional[int] = None,
    lower: float = 0,
    upper: float = 100,
    resample: bool = False,
) -> np.ndarray:
    """
    Create distributions of Mean Opinion Scores (MOS) through bootstrapping.

    This function supports two input types:
    1. Dictionary of raw vote scores
    2. List of (mean, std, n_votes) tuples

    Parameters:
    -----------
    scene_votes : Union[Dict[int, List[float]], List[Tuple[float, float, int]]]
        Either a dictionary of vote scores or a list of (mean, std, n_votes) tuples
    n_iterations : int, optional
        Number of bootstrap iterations (default: 10000)
    n_processes : Optional[int], optional
        Number of parallel processes to use (default: all available cores)
    lower : float, optional
        Lower bound for score distribution (default: 0)
    upper : float, optional
        Upper bound for score distribution (default: 100)
    resample : bool, optional
        Whether to resample distributions to a consistent length (default: False)

    Returns:
    --------
    np.ndarray
        Bootstrapped distribution of Mean Opinion Scores
    """

    # for raw scores
    if isinstance(scene_votes, dict):
        # dictionary with n_iterations samples from a distribution (create a distribution of means)
        bootstrapped_distributions = parallel_bootstrap(
            scene_votes=scene_votes, n_iterations=n_iterations, n_processes=n_processes
        )

        # leverage pandas to convert dictionary to numpy array
        mos_distributions = pd.DataFrame(bootstrapped_distributions).to_numpy().T

    # for mean/std values
    elif isinstance(scene_votes, list):
        means, stds, n_votes, *_ = scene_votes

        if len(means) != len(stds):
            raise SystemExit(
                "The length of mean and std values is not the same! Exiting..."
            )

        if not (isinstance(n_votes, list) or isinstance(n_votes, np.ndarray)):
            n_votes = [n_votes] * len(means)

        if len(n_votes) != len(means):
            warnings.warn(
                "The n_votes is not the same length as means/stds. Taking the max of the supplied votes."
            )

            n_votes = [int(round(np.max(n_votes), 0))] * len(means)

        max_votes = int(np.median(n_votes))

        if not resample:
            if any(np.array(n_votes) != max_votes):
                warnings.warn(
                    "Resample is turned off. Will take the max of n_votes for all stimuli."
                )

                n_votes = [int(np.max(n_votes))] * len(means)

        # # estimate the distribution of scores
        # combined_df = pd.DataFrame.from_dict({'mos': means, 'std': stds, 'name':[f'name_{x}' for x in range(len(means))]})
        # par_mos = parallel_processing(combined_df, n_iterations, lower, upper)

        # dist_values = [values for _, values in par_mos]
        # mos_distributions = np.vstack(dist_values)

        # estimate the distribution of means

        # estimate the distribution of scores
        dist_values = []

        for x, y, z in zip(means, stds, n_votes):
            samples = get_samples(x, y, z, lower, upper)

            if resample:
                if z != max_votes:
                    samples = advanced_resample_distribution(
                        samples, max_votes, method="kernel"
                    )

            dist_values.append(samples)

        mos_distributions = np.vstack(dist_values)

        scene_votes = {i: row.tolist() for i, row in enumerate(mos_distributions)}

        # # dictionary with n_iterations samples from a distribution (create a distribution of means)
        bootstrapped_distributions = parallel_bootstrap(
            scene_votes=scene_votes, n_iterations=n_iterations, n_processes=n_processes
        )

        # # leverage pandas to convert dictionary to numpy array
        mos_distributions = pd.DataFrame(bootstrapped_distributions).to_numpy().T

    else:
        raise SystemExit(
            "The supplied type of scene_votes is incorrect! Should be either a dict or a list. Exiting..."
        )

    return mos_distributions


def advanced_resample_distribution(
    original_dist: np.ndarray, new_length: int, method: str = "kernel"
) -> np.ndarray:
    """
    Resample a distribution to a new length while preserving statistical properties.

    Parameters:
    -----------
    original_dist : np.ndarray
        Original distribution to be resampled
    new_length : int
        Desired length of the resampled distribution
    method : str, optional
        Resampling method ('kernel' or 'quantile', default: 'kernel')

    Returns:
    --------
    np.ndarray
        Resampled distribution with similar statistical properties
    """

    # Compute original statistics
    original_mean = np.mean(original_dist)
    original_std = np.std(original_dist)

    if method == "kernel":
        # Kernel Density Estimation approach
        kde = stats.gaussian_kde(original_dist)

        # Generate samples from the estimated density
        resampled_dist = kde.resample(new_length)[0]

        # Normalize to match original moments
        resampled_mean = np.mean(resampled_dist)
        resampled_std = np.std(resampled_dist)

        normalized_dist = (resampled_dist - resampled_mean) * (
            original_std / resampled_std
        ) + original_mean

    elif method == "quantile":
        # Quantile-based matching
        # Sort the original distribution
        sorted_orig = np.sort(original_dist)

        # Create quantile-based interpolation
        quantiles = np.linspace(0, 1, len(sorted_orig))
        new_quantiles = np.linspace(0, 1, new_length)

        # Interpolate at new quantile points
        normalized_dist = np.interp(new_quantiles, quantiles, sorted_orig)

    else:
        raise ValueError("Invalid method. Choose 'kernel' or 'quantile'.")

    # Verify preservation of key statistical moments
    assert np.isclose(np.mean(normalized_dist), original_mean, rtol=1e-2), (
        "Mean not preserved"
    )
    assert np.isclose(np.std(normalized_dist), original_std, rtol=1e-2), (
        "Standard deviation not preserved"
    )

    return normalized_dist


def bootstrap_votes(votes: np.ndarray, n_iterations: int = 10000) -> np.ndarray:
    """
    Generate bootstrapped samples by resampling with replacement.

    Parameters:
    -----------
    votes : np.ndarray
        Original vote samples
    n_iterations : int, optional
        Number of bootstrap iterations (default: 10000)

    Returns:
    --------
    np.ndarray
        Array of bootstrapped mean samples
    """
    # Generates bootstrap samples with mean statistics
    bootstrap_samples = np.random.choice(
        votes, size=(n_iterations, len(votes)), replace=True
    )
    bootstrap_statistics = np.mean(bootstrap_samples, axis=1)

    return bootstrap_statistics


def process_key(
    key_id_votes: Tuple[int, List[float]], n_iterations: int = 10000
) -> Tuple[int, np.ndarray]:
    """
    Process a single key's votes for bootstrapping.

    Parameters:
    -----------
    key_id_votes : Tuple[int, List[float]]
        Tuple of key ID and corresponding vote list
    n_iterations : int, optional
        Number of bootstrap iterations (default: 10000)

    Returns:
    --------
    Tuple[int, np.ndarray]
        Key ID and bootstrapped statistics
    """
    key_id, votes = key_id_votes
    return key_id, bootstrap_votes(votes, n_iterations)


def parallel_bootstrap(
    scene_votes: Dict[int, List[float]],
    n_iterations: int = 10000,
    n_processes: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Perform parallel bootstrapping across multiple scenes.

    Parameters:
    -----------
    scene_votes : Dict[int, List[float]]
        Dictionary of scene votes
    n_iterations : int, optional
        Number of bootstrap iterations (default: 10000)
    n_processes : Optional[int], optional
        Number of parallel processes (default: all available cores)

    Returns:
    --------
    Dict[int, np.ndarray]
        Dictionary of bootstrapped statistics for each scene
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    pool = mp.Pool(processes=n_processes)

    key_id_votes = list(zip(scene_votes.keys(), scene_votes.values()))

    results = pool.map(partial(process_key, n_iterations=n_iterations), key_id_votes)

    pool.close()
    pool.join()

    return dict(results)


# estimate the beta distribution
def beta_distribution(
    mean: float, std: float, lower: float, upper: float, n_samples: int
) -> np.ndarray:
    """
    Generate a beta distribution with specified mean, standard deviation, and bounds.

    Parameters:
    -----------
    mean : float
        Target mean of the distribution
    std : float
        Target standard deviation
    lower : float
        Lower bound of the distribution
    upper : float
        Upper bound of the distribution
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    np.ndarray
        Generated samples from the beta distribution
    """
    # Input validation

    if mean <= 0 or std <= 0:
        return np.full(n_samples, mean)

    if not (lower <= mean <= upper):
        raise ValueError("Mean must be between lower and upper bounds.")

    # Rescale mean and variance for [0, 1] beta distribution
    rescaled_mean = (mean - lower) / (upper - lower)
    rescaled_var = (std / (upper - lower)) ** 2

    # Check for zero or near-zero variance
    if rescaled_var < 1e-10:
        return np.full(n_samples, mean)

    # Calculate alpha and beta parameters
    temp = (rescaled_mean * (1 - rescaled_mean) / rescaled_var) - 1
    alpha = rescaled_mean * temp
    beta_param = (1 - rescaled_mean) * temp

    # Check if alpha and beta are valid
    if alpha <= 0 or beta_param <= 0:
        return np.full(n_samples, mean - 0.5)

    try:
        # Sample from beta distribution
        beta_samples = stats.beta.rvs(alpha, beta_param, size=n_samples)
    except Exception as e:
        raise ValueError(
            f"Invalid parameters. Try adjusting input values! Original error: {e}."
        )

    # Rescale back to the original range [lower, upper]
    return lower + beta_samples * (upper - lower)


# estimate the gaussian distribution
def adjust_truncated_normal_params(
    target_mean: float,
    target_std: float,
    lower: float,
    upper: float,
    max_iterations: int = 100,
    epsilon: float = 1e-8,
) -> Tuple[float, float]:
    """
    Adjust parameters for a truncated normal distribution.

    Parameters:
    -----------
    target_mean : float
        Desired mean of the distribution
    target_std : float
        Desired standard deviation
    lower : float
        Lower bound of the distribution
    upper : float
        Upper bound of the distribution
    max_iterations : int, optional
        Maximum iterations for parameter adjustment (default: 100)
    epsilon : float, optional
        Small value to prevent division by zero (default: 1e-8)

    Returns:
    --------
    Tuple[float, float]
        Adjusted mean and standard deviation
    """
    # Initialize parameters (start with target values)
    mu = target_mean
    sigma = target_std

    # Helper function to calculate the truncated mean and std
    def truncated_normal_moments(mu, sigma, lower, upper):
        if sigma < epsilon:
            return mu, sigma  # Return original values if sigma is too small

        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        alpha = stats.norm.pdf(a) - stats.norm.pdf(b)
        beta = stats.norm.cdf(b) - stats.norm.cdf(a)

        if abs(beta) < epsilon:
            return mu, sigma  # Return original values if beta is too small

        truncated_mean = mu + (alpha / beta) * sigma

        # Use np.clip to ensure the argument to sqrt is non-negative
        variance_term = np.clip(
            1
            + ((a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / beta)
            - (alpha / beta) ** 2,
            0,
            None,
        )
        truncated_var = sigma**2 * variance_term

        return truncated_mean, np.sqrt(truncated_var)

    # Iterate to adjust parameters
    for i in range(max_iterations):
        trunc_mean, trunc_std = truncated_normal_moments(mu, sigma, lower, upper)

        # Adjust mu and sigma to match the target mean and std after truncation
        mu += target_mean - trunc_mean

        # Prevent division by zero or very small numbers
        if trunc_std < epsilon:
            print(
                f"Warning: trunc_std is very small ({trunc_std}) at iteration {i}. Stopping iterations."
            )
            break

        sigma *= target_std / trunc_std

    return mu, sigma


def process_row(
    row: Tuple[float, float, str], n_samples: int, lower: float, upper: float
) -> Tuple[str, np.ndarray]:
    """
    Process a single row of data to generate samples.

    Parameters:
    -----------
    row : Tuple[float, float, str]
        Tuple of (mean, std, filename)
    n_samples : int
        Number of samples to generate
    lower : float
        Lower bound of the distribution
    upper : float
        Upper bound of the distribution

    Returns:
    --------
    Tuple[str, np.ndarray]
        Filename and generated samples
    """
    mean, std, filename = row

    output = get_samples(mean, std, n_samples, lower, upper)

    return (filename, output)


def parallel_processing(
    combined_df: pd.DataFrame, n_samples: int, lower: float, upper: float
) -> List[Tuple[str, np.ndarray]]:
    """
    Perform parallel processing of DataFrame rows.

    Parameters:
    -----------
    combined_df : pd.DataFrame
        DataFrame with MOS, standard deviation, and name columns
    n_samples : int
        Number of samples to generate per row
    lower : float
        Lower bound of the distribution
    upper : float
        Upper bound of the distribution

    Returns:
    --------
    List[Tuple[str, np.ndarray]]
        List of (filename, samples) tuples
    """
    # Parallel processing of DataFrame rows using multiprocessing
    # Create a partial function with fixed arguments
    process_row_partial = partial(
        process_row, n_samples=n_samples, lower=lower, upper=upper
    )

    # Get the number of CPU cores
    num_cores = mp.cpu_count()

    # Create a pool of worker processes
    with mp.Pool(num_cores) as pool:
        # Apply the function to each row of the DataFrame
        results = pool.map(
            process_row_partial,
            combined_df[["mos", "std", "name"]].itertuples(index=False, name=None),
        )

    print(results)

    return results


def get_samples(
    mean: float, std: float, n_samples: int, lower: float, upper: float
) -> np.ndarray:
    """
    Generate samples with sophisticated distribution handling.

    Attempts to generate samples using beta distribution first,
    then falls back to truncated normal if initial samples
    don't match target statistics closely.

    Parameters:
    -----------
    mean : float
        Target mean of the distribution
    std : float
        Target standard deviation
    n_samples : int
        Number of samples to generate
    lower : float
        Lower bound of the distribution
    upper : float
        Upper bound of the distribution

    Returns:
    --------
    np.ndarray
        Generated samples
    """

    if std > 0:
        samples = beta_distribution(mean, std, lower, upper, n_samples)
    else:
        samples = np.full(n_samples, mean)

    if np.abs(samples.mean() - mean) > 0.2 or np.abs(samples.std() - std) > 0.1:
        adjusted_mean, adjusted_std = adjust_truncated_normal_params(
            mean, std, lower, upper
        )
        trunc_samples = stats.truncnorm.rvs(
            (lower - adjusted_mean) / adjusted_std,
            (upper - adjusted_mean) / adjusted_std,
            loc=adjusted_mean,
            scale=adjusted_std,
            size=n_samples,
        )
    else:
        trunc_samples = samples

    if (np.abs(samples.mean() - mean) + np.abs(samples.std() - std)) < (
        np.abs(trunc_samples.mean() - mean) + np.abs(trunc_samples.std() - std)
    ):
        return samples
    else:
        return trunc_samples


def bootstrap_single(subjective_array: np.ndarray) -> np.ndarray:
    """
    Perform bootstrap sampling on a 2D subjective score array.

    Parameters:
    -----------
    subjective_array : np.ndarray
        2D array of subjective scores

    Returns:
    --------
    np.ndarray
        Bootstrapped ratings
    """

    n_x, n_y = subjective_array.shape
    sample = np.random.choice(n_y, size=n_x, replace=True)
    bootstrap_ratings = subjective_array[np.arange(subjective_array.shape[0]), sample]

    return bootstrap_ratings


def process_wrapper_combined(
    args: Tuple[np.ndarray, np.ndarray, bool, List[str]],
) -> np.ndarray:
    """
    Wrapper function for processing combined subjective and objective scores.

    Parameters:
    -----------
    args : Tuple[np.ndarray, np.ndarray, bool, List[str]]
        Tuple containing:
        - Subjective scores
        - Objective scores
        - Flag for scene bootstrapping
        - List of correlation coefficients to compute

    Returns:
    --------
    np.ndarray
        Array of computed correlation coefficients
    """

    subjective_scores, objective_scores, bootstrap_scenes, corr_coeffs = args

    if bootstrap_scenes:
        # random sample (bootstrapping) of scenes
        sample = np.random.randint(
            0, subjective_scores.shape[0], size=subjective_scores.shape[0]
        )

        resampled_subjective = np.array(subjective_scores)[sample]
        resampled_objective = np.array(objective_scores)[sample]
    else:
        resampled_subjective = subjective_scores
        resampled_objective = objective_scores

    # random sample of each distribution of means
    bootstrap_means = bootstrap_single(resampled_subjective)

    # compute correlation coefficients
    corr_outputs = []

    for corr_coeff in corr_coeffs:
        if corr_coeff.lower() == "pearson" or corr_coeff == "r":
            p_corr, _ = stats.pearsonr(bootstrap_means, resampled_objective)
            corr_outputs.append(p_corr)

        if corr_coeff.lower() == "spearman" or corr_coeff == "rho":
            s_corr, _ = stats.spearmanr(bootstrap_means, resampled_objective)
            corr_outputs.append(s_corr)

        if corr_coeff.lower() == "kendall" or corr_coeff == "tau":
            k_corr, _ = stats.kendalltau(bootstrap_means, resampled_objective)
            corr_outputs.append(k_corr)

    return np.array(corr_outputs)


def compute_correlation_distributions(
    subjective_array: np.ndarray,
    objective_scores: np.ndarray,
    n_bootstrap: int = 10000,
    bootstrap_scenes: bool = True,
    corr_coeffs: List[str] = ["pearson", "spearman", "kendall"],
) -> Dict[str, np.ndarray]:
    """
    Compute distributions of correlation coefficients.

    Performs bootstrapping to generate distributions of correlation
    coefficients between subjective and objective scores.

    Parameters:
    -----------
    subjective_array : np.ndarray
        2D array of subjective scores
    objective_scores : np.ndarray
        1D array of objective scores
    n_bootstrap : int, optional
        Number of bootstrap iterations (default: 10000)
    bootstrap_scenes : bool, optional
        Whether to bootstrap scenes (default: True)
    corr_coeffs : List[str], optional
        List of correlation coefficients to compute
        (default: ['pearson', 'spearman', 'kendall'])

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary of correlation coefficient distributions
    """

    # Get the number of CPU cores
    num_cores = mp.cpu_count()

    with mp.Pool(num_cores) as pool:
        # Apply the bootstrap_single function n_bootstrap times
        correlations = pool.map(
            process_wrapper_combined,
            [
                (
                    subjective_array[~np.isnan(objective_scores)],
                    objective_scores[~np.isnan(objective_scores)],
                    bootstrap_scenes,
                    corr_coeffs,
                )
            ]
            * n_bootstrap,
        )

    correlations = np.array(correlations)

    corr_outputs = {}

    if len(correlations[0]) == len(corr_coeffs):
        for idx, corr_coeff in enumerate(corr_coeffs):
            corr_outputs[corr_coeff] = np.abs(correlations[:, idx])

    # output is a dictionary
    # each key is a correlation coefficient
    # each value is a distribution (sample) of coefficient values
    return corr_outputs


def save_df_numpy(df: pd.DataFrame, filename: Union[str, Path]) -> None:
    """Save DataFrame to NumPy .npz format."""
    np.savez_compressed(filename, data=df.values, columns=df.columns, index=df.index)


def load_df_numpy(filename: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from NumPy .npz format."""
    with np.load(filename, allow_pickle=True) as data:
        return pd.DataFrame(data["data"], columns=data["columns"], index=data["index"])


def cliffs_delta(
    lst1: Union[List[T], Sequence[T], np.ndarray],
    lst2: Union[List[T], Sequence[T], np.ndarray],
) -> float:
    """
    Calculate Cliff's Delta using NumPy for efficiency.

    Args:
        lst1: First list/array of values
        lst2: Second list/array of values

    Returns:
        float: The Cliff's Delta value between the two arrays
    """
    # Convert to numpy arrays if they aren't already
    x = np.asarray(lst1)
    y = np.asarray(lst2)

    # Get dimensions
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Vectorized dominance calculation
    dominance: float = 0

    # x > y cases (positive dominance)
    dominance += np.sum(x[:, np.newaxis] > y)

    # x < y cases (negative dominance)
    dominance -= np.sum(x[:, np.newaxis] < y)

    # Calculate delta
    delta: float = dominance / (m * n)

    return delta

def compare_pair(args: Tuple[int, int, List[float], List[float]]) -> Tuple[int, int, float]:
    """
    Compare two distributions using Cliff's delta effect size measure.
    
    Args:
        args: A tuple containing:
            i: Index of the first distribution
            j: Index of the second distribution
            dist_X: The first distribution values
            dist_Y: The second distribution values
            
    Returns:
        A tuple containing the indices i, j and the calculated Cliff's delta value
    """
    i, j, dist_X, dist_Y = args
    delta = cliffs_delta(dist_X, dist_Y)
    return i, j, delta

def compute_stats(corr_dict: Dict[str, List[float]]) -> np.ndarray:
    """
    Compute pairwise Cliff's delta statistics between all metrics in the correlation dictionary.
    
    Args:
        corr_dict: Dictionary mapping metric names to their distribution values
        
    Returns:
        A square matrix of Cliff's delta values between each pair of metrics
    """
    metrics = list(corr_dict.keys())
    n = len(metrics)
    cliffs_values = np.zeros((n, n))
    tasks = [(i, j, corr_dict[metrics[i]], corr_dict[metrics[j]]) for i, j in combinations(range(n), 2)]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(compare_pair, tasks)
    
    for i, j, delta in results:
        cliffs_values[i, j] = delta
        cliffs_values[j, i] = -delta
    
    return cliffs_values