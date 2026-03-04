import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.special import expit
from gg570_d200.auxiliary_functions.overlap_funcs import propensity_scores, overlap_measures
from gg570_d200.auxiliary_functions.forest_riesz_funcs import call_forestriesz, call_forestriesz_cross


def synthetic_data(scaled_covars, overlap_intensity, synthetic_ate, return_heterogeneity=False):
    scaled_covars = np.asarray(scaled_covars)
    n, p = scaled_covars.shape
    
    covariate_coefs = np.random.normal(0, 1, (p, 3)) # Col 0 for selection, col 1 for outcome, col 2 for heterogeneity
    
    matrix_mult = scaled_covars @ covariate_coefs # Again, col 0 for selection, col 1 for outcome, col 2 for heterogeneity

    prob_assign = expit(overlap_intensity * matrix_mult[:, 0])
    synthetic_treat = np.random.binomial(1, prob_assign) # for each observation i, assign treat with prob = prob_assign_i
    
    heterogeneous_component = 0.01 * matrix_mult[:, 2]**2 # add some nonlinearity to the heterogeneity
    heterogeneous_component -= np.mean(heterogeneous_component) # centre the heterogeneous component to have mean zero, so that the ATE is not affected by the heterogeneity
    
    synthetic_y = ((synthetic_ate + heterogeneous_component) * synthetic_treat) + (matrix_mult[:, 1]) + np.random.normal(0, 1, n) # normal errors
    
    if return_heterogeneity:
        return synthetic_y, synthetic_treat, heterogeneous_component
    else:
        return synthetic_y, synthetic_treat


def synthetic_loop(df_scaled, covariate_cols, iterations, synthetic_ate, root, cross_validate=True):
    iterations_dict = {'overlap_intensity': np.zeros(iterations),
                       'extreme_scores': np.zeros(iterations),
                       'ess': np.zeros(iterations),
                       'dr_estimates': np.zeros(iterations),
                       'dr_in_ci': np.zeros(iterations),
                       'plugin_estimates': np.zeros(iterations),
                       'plugin_in_ci': np.zeros(iterations)}

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

    print_cases = max(1, int(np.ceil(iterations / 10)))

    for i in range(iterations):
        iter_num = i + 1
        if iterations < 9:
            print(iter_num)
        elif iter_num == 1 or iter_num == iterations or iter_num % print_cases == 0:
            print(iter_num)

        overlap_intensity = np.random.uniform(0, 1) # Randomly select an overlap intensity for the synthetic data.
        
        df_scaled_synthetic[synthetic_y], df_scaled_synthetic[synthetic_treat] = synthetic_data(df_scaled_covars, overlap_intensity, synthetic_ate)
        df_scaled_synthetic[prop_scores] = propensity_scores(df_scaled_synthetic, covariate_cols, synthetic_treat)
        #plot_propensity_scores(df_scaled_synthetic, synthetic_treat, prop_scores)
        extreme_scores, ess = overlap_measures(df_scaled_synthetic, synthetic_treat, prop_scores)

        if cross_validate:
            riesz_estimate = call_forestriesz_cross(df_scaled_synthetic, covariate_cols, synthetic_treat, synthetic_y, methods)
        else:
            riesz_estimate = call_forestriesz(df_scaled_synthetic, covariate_cols, synthetic_treat, synthetic_y, methods)

        overlap_arr[i] = overlap_intensity
        extreme_arr[i] = extreme_scores
        ess_arr[i] = ess
        
        dr_est[i] = riesz_estimate['dr']['est']
        dr_ci[i] = int(riesz_estimate['dr']['low'] <= synthetic_ate <= riesz_estimate['dr']['high'])
        
        plugin_est[i] = riesz_estimate['plugin']['est']
        plugin_ci[i] = int(riesz_estimate['plugin']['low'] <= synthetic_ate <= riesz_estimate['plugin']['high'])

    (root / "results").mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(iterations_dict)
    results_df.to_csv(root / f"results/synthetic_iterations_results.csv", index=False)

    return iterations_dict


def prepare_heatmap(iterations_dict, synthetic_ate, true_extreme_scores_perc, true_ess_perc, bins=5):
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
    for i, ess_cat in enumerate(ess_categories):
        for j, extreme_scores_cat in enumerate(extreme_scores_categories):
            if ess_cat == true_ess_bin and extreme_scores_cat == true_extreme_bin:
                true_bin_i, true_bin_j = i, j
            
            mask = (extreme_scores_bins == extreme_scores_cat) & (ess_bins == ess_cat)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # Subset arrays once for efficiency
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


def plot_heatmap(data_matrix, str_matrix, true_bin_i, true_bin_j, extreme_scores_categories, ess_categories, synthetic_ate, root):
    fig, ax = plt.subplots(figsize=(10, 6))
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