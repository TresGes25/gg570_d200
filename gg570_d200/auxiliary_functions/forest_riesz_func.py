import numpy as np
from sklearn.model_selection import KFold
from gg570_d200.external_code.forestriesz import ForestRieszATE


def call_forestriesz(df, covariate_cols, treatment_col, outcome_col, methods):
    covariates = df[covariate_cols].values
    treat = df[treatment_col].values
    outcome = df[outcome_col].values
    
    est = ForestRieszATE(criterion='mse', n_estimators=100, min_samples_leaf=2,
                         min_var_fraction_leaf=0.001, min_var_leaf_on_val=True,
                         min_impurity_decrease = 0.01, max_samples=.8, max_depth=None,
                         warm_start=False, inference=False, subforest_size=1,
                         honest=True, verbose=0, n_jobs=-2, random_state=21)
    
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


def call_forestriesz_cross(df, covariate_cols, treatment_col, outcome_col, methods):
    covariates = df[covariate_cols].values
    treat = df[treatment_col].values
    outcome = df[outcome_col].values
    
    kf = KFold(n_splits=3, shuffle=True, random_state=21)
    
    fold_results = {method: {'est': [], 'low': [], 'high': []} for method in methods}
    
    for train_idx, test_idx in kf.split(covariates):
        X_train, X_test = covariates[train_idx], covariates[test_idx]
        treat_train, treat_test = treat[train_idx], treat[test_idx]
        y_train, y_test = outcome[train_idx], outcome[test_idx]
        
        est = ForestRieszATE(criterion='mse', n_estimators=100, min_samples_leaf=2,
                             min_var_fraction_leaf=0.001, min_var_leaf_on_val=True,
                             min_impurity_decrease = 0.01, max_samples=.8, max_depth=None,
                             warm_start=False, inference=False, subforest_size=1,
                             honest=True, verbose=0, n_jobs=-2, random_state=21)
        
        est.fit(X_train, treat_train, y_train)
        
        pred_data = np.column_stack([treat_test, X_test])
        
        for method in methods:
            ate = est.predict_ate(pred_data, y_test, method=method)
            fold_results[method]['est'].append(ate[0])
            fold_results[method]['low'].append(ate[1])
            fold_results[method]['high'].append(ate[2])
    
    results = {}
    for method in methods:
        results[method] = {
            'est': np.mean(fold_results[method]['est']),
            'low': np.mean(fold_results[method]['low']),
            'high': np.mean(fold_results[method]['high'])
        }
    
    return results