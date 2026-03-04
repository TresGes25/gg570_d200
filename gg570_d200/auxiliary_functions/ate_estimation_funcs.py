import numpy as np
from gg570_d200.auxiliary_functions.forest_riesz_funcs import calculate_p_value


def forest_riesz_gate(df, covariate_cols, treatment_col, outcome_col, est, mask):
    group_with_treatment = df.loc[mask, [treatment_col] + covariate_cols].values
    y_group = df.loc[mask, outcome_col].values
    ate_result = est.predict_ate(group_with_treatment, y_group, method='dr')
    p_val = calculate_p_value(ate_result[0], ate_result[1], ate_result[2])
    ate_result = list(ate_result)
    ate_result.append(p_val)

    return(ate_result)


def forest_riesz_gate_cross(df, covariate_cols, treatment_col, outcome_col, est_list, mask):
    group_with_treatment = df.loc[mask, [treatment_col] + covariate_cols].values
    y_group = df.loc[mask, outcome_col].values
    
    means, lows, highs = [], [], []
    for est in est_list:
        ate_result = est.predict_ate(group_with_treatment, y_group, method='dr')
        means.append(ate_result[0])
        lows.append(ate_result[1])
        highs.append(ate_result[2])

    means, lows, highs = np.array(means), np.array(lows), np.array(highs)

    std_errors = (highs - lows) / (2 * 1.96)
    pooled_var = np.mean(std_errors**2) + np.var(means, ddof=1)
    pooled_std_error = np.sqrt(pooled_var)

    mean = np.mean(means)
    low = mean - 1.96*pooled_std_error
    high = mean + 1.96*pooled_std_error
    p_val = calculate_p_value(mean, low, high)

    return([mean, low, high, p_val])


def causal_dml_gate(df, covariate_cols, est, mask):
    group = df.loc[mask, covariate_cols]
    ate = est.ate(X=group)
    low, high = est.ate_interval(X=group, alpha=0.05)
    p_val = calculate_p_value(ate, low, high)

    return([ate, low, high, p_val])