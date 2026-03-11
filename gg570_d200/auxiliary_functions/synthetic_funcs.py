import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import seaborn as sns
from scipy.special import expit
from gg570_d200.auxiliary_functions.overlap_funcs import propensity_scores, overlap_measures
from gg570_d200.auxiliary_functions.forest_riesz_funcs import call_forestriesz, call_forestriesz_cross


def synthetic_data_func(
    scaled_covars: pd.DataFrame,
    overlap_intensity: float,
    synthetic_ate: float,
    return_heterogeneity: bool = False,
) -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]
    | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]
):
    """
    Function that generates synthetic outcomes and treatment processes.

    Parameters:
    scaled_covars: pd.DataFrame
        Scaled covariates used to generate synthetic data.
    overlap_intensity: float
        Strength of treatment selection in the assignment model.
    synthetic_ate: float
        Target (homogeneous) average treatment effect.
    return_heterogeneity: bool
        Whether to also return the heterogeneous effect component.

    Returns:
    tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]
        Synthetic outcome and treatment arrays, with optional heterogeneity array.
    """
    scaled_covars = np.asarray(scaled_covars)
    n, p = scaled_covars.shape
    
    covariate_coefs = np.random.normal(0, 1, (p, 3)) # Col 0 for selection, col 1 for outcome, col 2 for heterogeneity
    
    matrix_mult = scaled_covars @ covariate_coefs

    prob_assign = expit(overlap_intensity * matrix_mult[:, 0]) # treatment assignment probabilities, moderated by the overlap intensity factor.
    synthetic_treat = np.random.binomial(1, prob_assign) # for each observation i, assign treat with prob = prob_assign_i
    
    heterogeneous_component = 0.01 * matrix_mult[:, 2]**2 # add some nonlinearity to the heterogeneity
    heterogeneous_component -= np.mean(heterogeneous_component) # centre the heterogeneous component to have mean zero, so that the ATE is not affected by the heterogeneity
    
    synthetic_y = ((synthetic_ate + heterogeneous_component) * synthetic_treat) + (matrix_mult[:, 1]) + np.random.normal(0, 1, n) # normal errors
    
    if return_heterogeneity:
        return synthetic_y, synthetic_treat, heterogeneous_component
    else:
        return synthetic_y, synthetic_treat


def synthetic_loop(
    df_scaled: pd.DataFrame,
    covariate_cols: list[str],
    iterations: int,
    synthetic_ate: float,
    root: Path,
    cross_fit: bool = True,
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Function that runs repeated synthetic simulations and
    stores overlap and estimator performance metrics.

    Parameters:
    df_scaled: pd.DataFrame
        Input scaled dataframe.
    covariate_cols: list[str]
        Covariate column names.
    iterations: int
        Number of simulation iterations.
    synthetic_ate: float
        True ATE used to generate synthetic outcomes.
    root: Path
        Project root path used to save results.
    cross_fit: bool
        Whether to use cross-fitting.

    Returns:
    dict[str, npt.NDArray[np.float64]]
        Dictionary of arrays with overlap and estimation metrics by iteration.
    """
    iterations_dict = {'overlap_intensity': np.zeros(iterations),
                       'extreme_scores': np.zeros(iterations),
                       'ess': np.zeros(iterations),
                       'dr_estimates': np.zeros(iterations),
                       'dr_in_ci': np.zeros(iterations),
                       'plugin_estimates': np.zeros(iterations),
                       'plugin_in_ci': np.zeros(iterations)}

    # Caching dictionary arrays for efficiency when filling them in the loop.
    # The 'dr' method refers to the 'ForestRiesz' method mentioned in the report.
    # It is the standard implementation of ForestRiesz.

    overlap_arr = iterations_dict['overlap_intensity']
    extreme_arr = iterations_dict['extreme_scores']
    ess_arr = iterations_dict['ess']
    dr_est = iterations_dict['dr_estimates']
    dr_ci = iterations_dict['dr_in_ci']
    plugin_est = iterations_dict['plugin_estimates']
    plugin_ci = iterations_dict['plugin_in_ci']

    df_scaled_covars = df_scaled[covariate_cols]
    df_scaled_synthetic = df_scaled_covars.copy()
    synthetic_y = 'synthetic_y'
    synthetic_treat = 'synthetic_treat'
    prop_scores = 'prop_scores'
    
    methods = ['dr', 'plugin']

    # Simple condition to set print frequency, to be able to simulation progress.
    print_cases = max(1, int(np.ceil(iterations / 10)))

    for i in range(iterations):
        iter_num = i + 1
        if iterations < 9:
            print(iter_num)
        elif iter_num == 1 or iter_num == iterations or iter_num % print_cases == 0:
            print(iter_num)

        overlap_intensity = np.random.uniform(0, 1) # Randomly select an overlap intensity for the synthetic data.
        
        # Calling the synthetic_data_func helper from above to generate
        # synthetic outcomes and treatments, given the overlap intensity and the scaled covariates.
        df_scaled_synthetic[synthetic_y], df_scaled_synthetic[synthetic_treat] = synthetic_data_func(df_scaled_covars, overlap_intensity, synthetic_ate)

        # Using the created synthetic treatment, alongside the scaled covariates,
        # to estimate propensity scores and compute overlap measures for the particular iteration.
        df_scaled_synthetic[prop_scores] = propensity_scores(df_scaled_synthetic, covariate_cols, synthetic_treat)
        extreme_scores, ess = overlap_measures(df_scaled_synthetic, synthetic_treat, prop_scores)

        # Cross-fitting is not enabled for the plugin method,
        # so it is redirected to use call_forestriesz rather than call_forestriesz_cross
        if cross_fit:
            riesz_estimate_dr = call_forestriesz_cross(df_scaled_synthetic, covariate_cols, synthetic_treat, synthetic_y, ['dr'])
            dr_est[i] = riesz_estimate_dr['dr']['est']
            dr_ci[i] = int(riesz_estimate_dr['dr']['low'] <= synthetic_ate <= riesz_estimate_dr['dr']['high'])

            riest_estimate_plugin = call_forestriesz(df_scaled_synthetic, covariate_cols, synthetic_treat, synthetic_y, ['plugin'])
            plugin_est[i] = riest_estimate_plugin['plugin']['est']
            plugin_ci[i] = int(riest_estimate_plugin['plugin']['low'] <= synthetic_ate <= riest_estimate_plugin['plugin']['high'])
        else:
            riesz_estimate = call_forestriesz(df_scaled_synthetic, covariate_cols, synthetic_treat, synthetic_y, methods)
            dr_est[i] = riesz_estimate['dr']['est']
            dr_ci[i] = int(riesz_estimate['dr']['low'] <= synthetic_ate <= riesz_estimate['dr']['high'])
            plugin_est[i] = riesz_estimate['plugin']['est']
            plugin_ci[i] = int(riesz_estimate['plugin']['low'] <= synthetic_ate <= riesz_estimate['plugin']['high'])
        
        overlap_arr[i] = overlap_intensity
        extreme_arr[i] = extreme_scores
        ess_arr[i] = ess

    (root / "results").mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(iterations_dict)
    results_df.to_csv(root / f"results/synthetic_iterations_results.csv", index=False)

    return iterations_dict


def prepare_heatmap(
    iterations_dict: dict[str, npt.NDArray[np.float64]],
    synthetic_ate: float,
    true_extreme_scores_perc: float,
    true_ess_perc: float,
    bins: int = 5,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.object_],
    int | None,
    int | None,
    pd.IntervalIndex,
    pd.IntervalIndex,
]:
    """
    Function that prepares bins, and bin statistics and annotations,
    for heatmap visualization.

    Parameters:
    iterations_dict: dict[str, npt.NDArray[np.float64]]
        Dictionary of simulation arrays from ``synthetic_loop``.
    synthetic_ate: float
        True ATE used in the synthetic simulations.
    true_extreme_scores_perc: float
        Extreme score percentage in the real data.
    true_ess_perc: float
        ESS percentage in the real data.
    bins: int
        Number of quantile bins for overlap measures.

    Returns:
    tuple[npt.NDArray[np.float64], npt.NDArray[np.object_], int | None, int | None, pd.IntervalIndex, pd.IntervalIndex]
        Heatmap numeric matrix, annotation matrix, true-bin indices, and bin categories.
    """

    # Extreme propensity scores and ESS are binned into quantiles
    extreme_scores_bins = pd.qcut(iterations_dict['extreme_scores'], q=bins, duplicates='drop')
    extreme_scores_categories = extreme_scores_bins.categories
    true_extreme_bin = pd.cut([true_extreme_scores_perc], bins=extreme_scores_categories)[0]

    ess_bins = pd.qcut(iterations_dict['ess'], q=bins, duplicates='drop')
    ess_categories = ess_bins.categories
    true_ess_bin = pd.cut([true_ess_perc], bins=ess_categories)[0]

    dr_estimates = np.array(iterations_dict['dr_estimates'])
    dr_cis = np.array(iterations_dict['dr_in_ci'])

    plugin_estimates = np.array(iterations_dict['plugin_estimates'])
    plugin_cis = np.array(iterations_dict['plugin_in_ci'])

    data_matrix = np.zeros((len(ess_categories), len(extreme_scores_categories)))
    str_matrix = np.empty((len(ess_categories), len(extreme_scores_categories)), dtype=object)

    true_bin_i, true_bin_j = None, None

    # Looping through bins, filling in MSE, MAE, and coverage statistics for each method,
    # and storing them in an annotations matrix str_matrix for the heatmap.
    # data_matrix simply stores the difference in MSE between the plugin and DR methods,
    # which is used for the heatmap colors.
    for i, ess_cat in enumerate(ess_categories):
        for j, extreme_scores_cat in enumerate(extreme_scores_categories):
            if ess_cat == true_ess_bin and extreme_scores_cat == true_extreme_bin:
                true_bin_i, true_bin_j = i, j
            
            mask = (extreme_scores_bins == extreme_scores_cat) & (ess_bins == ess_cat)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                dr_subset = dr_estimates[indices]
                plugin_subset = plugin_estimates[indices]
                
                dr_bias = dr_subset.mean() - synthetic_ate
                dr_var = dr_subset.var(ddof=0)
                dr_mse = dr_bias**2 + dr_var
                dr_mae = np.mean(np.abs(dr_subset - synthetic_ate))

                plugin_bias = plugin_subset.mean() - synthetic_ate
                plugin_var = plugin_subset.var(ddof=0)
                plugin_mse = plugin_bias**2 + plugin_var
                plugin_mae = np.mean(np.abs(plugin_subset - synthetic_ate))

                plugin_mse_minus_dr_mse = plugin_mse - dr_mse

                dr_perc_in_ci = dr_cis[indices].mean() * 100
                plugin_perc_in_ci = plugin_cis[indices].mean() * 100
                
                data_matrix[i, j] = plugin_mse_minus_dr_mse
                str_matrix[i, j] = f"{plugin_mse_minus_dr_mse:.2f}\n(n={len(indices)})\n{plugin_mse:.2f}, {dr_mse:.2f}\n{plugin_mae:.2f}, {dr_mae:.2f}\n{plugin_perc_in_ci:.1f}%, {dr_perc_in_ci:.1f}%"
            else:
                data_matrix[i, j] = np.nan
                str_matrix[i, j] = np.nan

    return data_matrix, str_matrix, true_bin_i, true_bin_j, extreme_scores_categories, ess_categories


def plot_heatmap(
    data_matrix: npt.NDArray[np.float64],
    str_matrix: npt.NDArray[np.object_],
    true_bin_i: int | None,
    true_bin_j: int | None,
    extreme_scores_categories: pd.IntervalIndex,
    ess_categories: pd.IntervalIndex,
    synthetic_ate: float,
    root: Path,
) -> None:
    """
    Function that plots and saves the synthetic overlap heatmap.

    Parameters:
    data_matrix: npt.NDArray[np.float64]
        Numeric matrix used for heatmap colors.
    str_matrix: npt.NDArray[np.object_]
        Annotation matrix shown in heatmap cells.
    true_bin_i: int | None
        Row index of the real-data bin in the heatmap.
    true_bin_j: int | None
        Column index of the real-data bin in the heatmap.
    extreme_scores_categories: pd.IntervalIndex
        Bin intervals for extreme propensity score percentages.
    ess_categories: pd.IntervalIndex
        Bin intervals for ESS percentages.
    synthetic_ate: float
        True ATE used as a benchmark.
    root: Path
        Project root path used to save the figure.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Creating the heatmap, where colours are decided by data_matrix and annotations by str_matrix.
    sns.heatmap(data_matrix, annot=str_matrix, annot_kws={'size': 10}, fmt='', cmap='RdYlGn', 
                xticklabels=[f"({interval.left:.3f}, {interval.right:.3f}]" for interval in extreme_scores_categories],
                yticklabels=[f"({interval.left:.3f}, {interval.right:.3f}]" for interval in ess_categories],
                cbar_kws={'label': 'Plugin MSE minus DR MSE in bin'},
                ax=ax)

    if true_bin_i is not None and true_bin_j is not None:
        true_bin_edge = Rectangle((true_bin_j, true_bin_i), 1, 1, fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(true_bin_edge)

    ax.set_xlabel('\nExtreme Scores Percentage', fontweight='bold', fontsize=14)
    ax.set_ylabel('Effective Sample Size Percentage\n', fontweight='bold', fontsize=14)
    ax.set_title(f'Plugin and DR: Per-Bin MSE Comparison (Synthetic ATE: {synthetic_ate})')

    legend_text = "Bin contents:\nMSE difference\nSimulations in bin\nMSE (Plugin, DR)\nMAE (Plugin, DR)\n% in CI (Plugin, DR)"
    plt.figtext(-0.15, -0.25, legend_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white'), horizontalalignment='center')

    plt.tight_layout()
    plt.show()

    (root / "results").mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"results/synthetic_overlap_heatmap.png", dpi=500, bbox_inches='tight')