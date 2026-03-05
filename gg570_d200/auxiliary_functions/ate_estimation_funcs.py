import numpy as np
from gg570_d200.auxiliary_functions.forest_riesz_funcs import calculate_p_value


def forest_riesz_gate(df, covariate_cols, treatment_col, outcome_col, est, mask):
    group = df.loc[mask, [treatment_col] + covariate_cols].values
    y_group = df.loc[mask, outcome_col].values
    ate_result = est.predict_ate(group, y_group, method='dr')
    p_val = calculate_p_value(ate_result[0], ate_result[1], ate_result[2])
    ate_result = list(ate_result)
    ate_result.append(p_val)

    return(ate_result)


def forest_riesz_gate_cross(df, covariate_cols, treatment_col, outcome_col, est_list, test_id_list, mask):
    means, lows, highs = [], [], []

    for est, test_id in zip(est_list, test_id_list):
        fold_mask = np.zeros(len(df), dtype=bool)
        fold_mask[test_id] = True
        effective_mask = mask & fold_mask

        group = df.loc[effective_mask, [treatment_col] + covariate_cols].values
        y_group = df.loc[effective_mask, outcome_col].values

        ate_result = est.predict_ate(group, y_group, method='dr')
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


"""
def causal_dml_gate(df, covariate_cols, treatment_col, outcome_col, est_list, test_id_list, mask):
    gates = []
    std_errors = []

    for i, test_id in enumerate(test_id_list):
        fold_mask = np.zeros(len(df), dtype=bool)
        fold_mask[test_id] = True
        effective_mask = mask & fold_mask
        
        X = df.iloc[test_id][covariate_cols].values
        T = df.iloc[test_id][treatment_col].values
        y = df.iloc[test_id][outcome_col].values

        res_y_all = y - est_list.models_y[0][i].predict(X)
        res_t_all = T - est_list.models_t[0][i].predict(X)

        effective_id = effective_mask[test_id]
        res_y = res_y_all[effective_id]
        res_t = res_t_all[effective_id]

        gate = np.mean(res_y * res_t) / np.mean(res_t**2)
        
        influence_func = res_t * (res_y - gate * res_t)
        gate_var = np.var(influence_func, ddof=1) / (len(res_t) * (np.mean(res_t**2))**2)
        gate_std_errors = np.sqrt(gate_var)
        
        gates.append(gate)
        std_errors.append(gate_std_errors)

    gates = np.array(gates)
    std_errors = np.array(std_errors)
    
    pooled_var = np.mean(std_errors**2) + np.var(gates, ddof=1)
    pooled_se = np.sqrt(pooled_var)

    mean = np.mean(gates)
    low = mean - 1.96*pooled_se
    high = mean + 1.96*pooled_se
    p_val = calculate_p_value(mean, low, high)

    return([mean, low, high, p_val])
"""