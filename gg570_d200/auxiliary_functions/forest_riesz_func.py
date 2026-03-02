from gg570_d200.external_code.forestriesz import ForestRieszATE


def call_forestriesz(df, covariate_cols, treatment_col, outcome_col, methods):
    covariates = df[covariate_cols].values
    treat = df[treatment_col].values
    outcome = df[outcome_col].values
    
    est = ForestRieszATE(criterion='mse', n_estimators=100, min_samples_leaf=2,
                         min_var_fraction_leaf=0.001, min_var_leaf_on_val=True,
                         min_impurity_decrease = 0.01, max_samples=.8, max_depth=None,
                         warm_start=False, inference=False, subforest_size=1,
                         honest=True, verbose=0, n_jobs=-1, random_state=21)
    
    est.fit(covariates, treat, outcome)

    pred_data = df[[treatment_col] + covariate_cols].values
    
    results = {}
    for method in methods:
        ate = est.predict_ate(pred_data, outcome, method=method)
        # ate = tuple((np.asarray(ate, dtype=float) * y_scaler.scale_[0]).tolist())
        results[method] = {'est' : ate[0],
                           'low' : ate[1],
                           'high' : ate[2]}

    return results