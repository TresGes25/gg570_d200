from gg570_d200.auxiliary_functions.forest_riesz_funcs import calculate_p_value


def forest_riesz_gate(df, covariate_cols, treatment_col, outcome_col, est, mask):
    group_with_treatment = df.loc[mask, [treatment_col] + covariate_cols].values
    y_group = df.loc[mask, outcome_col].values
    ate_result = est.predict_ate(group_with_treatment, y_group, method='dr')
    p_val = calculate_p_value(ate_result[0], ate_result[1], ate_result[2])
    ate_result = list(ate_result)
    ate_result.append(p_val)

    return(ate_result)


def causal_dml_gate(df, covariate_cols, est, mask):
    group = df.loc[mask, covariate_cols]
    ate = est.ate(X=group)
    low, high = est.ate_interval(X=group, alpha=0.05)
    p_val = calculate_p_value(ate, low, high)

    return([ate, low, high, p_val])