import numpy as np
import numpy.typing as npt
import pandas as pd
from gg570_d200.auxiliary_functions.forest_riesz_funcs import calculate_p_value
from gg570_d200.external_code.forestriesz import ForestRieszATE


boolean_mask = npt.NDArray[np.bool_] | pd.Series


def forest_riesz_gate(
    df: pd.DataFrame,
    covariate_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    est: ForestRieszATE,
    mask: boolean_mask,
) -> list[float]:
    """
    Function that estimates a GATE from a fitted ForestRieszATE estimator,
    given a boolean mask.

    Parameters:
    df: pd.DataFrame
        Input data.
    covariate_cols: list[str]
        Covariate column names.
    treatment_col: str
        Treatment column name.
    outcome_col: str
        Outcome column name.
    est: ForestRieszATE
        Fitted ForestRieszATE estimator.
    mask: boolean_mask
        Boolean mask for the subgroup.

    Returns:
    list[float]
        [estimate, ci_low, ci_high, p_value].
    """
    group = df.loc[mask, [treatment_col] + covariate_cols].values
    y_group = df.loc[mask, outcome_col].values
    # 'dr' refers to the method simply denoted 'ForestRiesz' in the report,
    # as it is the standard implementation considered by Chernozhukov et al. (2021).
    ate_result = est.predict_ate(group, y_group, method='dr')
    p_val = calculate_p_value(ate_result[0], ate_result[1], ate_result[2])
    ate_result = list(ate_result)
    ate_result.append(p_val)

    return(ate_result)


def forest_riesz_gate_cross(
    df: pd.DataFrame,
    covariate_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    est_list: list[ForestRieszATE],
    test_id_list: list[npt.NDArray[np.intp]],
    mask: boolean_mask,
) -> list[float]:
    """
    Function that estimates a cross-fitted GATE from a
    list of fitted ForestRieszATE estimators (one per fold)
    and their corresponding test indices,
    given a boolean mask.

    Parameters:
    df: pd.DataFrame
        Input data.
    covariate_cols: list[str]
        Covariate column names.
    treatment_col: str
        Treatment column name.
    outcome_col: str
        Outcome column name.
    est_list: list[ForestRieszATE]
        Fold-specific fitted estimators.
    test_id_list: list[npt.NDArray[np.intp]]
        Fold test indices aligned with est_list.
    mask: boolean_mask
        Boolean mask for the subgroup.

    Returns:
    list[float]
        [pooled_estimate, ci_low, ci_high, p_value].
    """
    means, lows, highs = [], [], []

    for est, test_id in zip(est_list, test_id_list):
        fold_mask = np.zeros(len(df), dtype=bool)
        fold_mask[test_id] = True
        effective_mask = mask & fold_mask
        # The effective mask ensures that those in the current fold and
        # satisfying the subgroup/mask condition are selected.

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


def add_std_error_from_ci(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
) -> pd.DataFrame:
    """
    Function that adds standard errors derived from 
    a DF with confidence interval bounds per observation.

    Parameters:
    df: pd.DataFrame
        Input dataframe.
    high_col: str
        Upper CI bound column name.
    low_col: str
        Lower CI bound column name.

    Returns:
    pd.DataFrame
        Copy of ``df`` with added ``std_error`` column.
    """
    df_out = df.copy()
    df_out['std_error'] = (df_out[high_col] - df_out[low_col]) / (2 * 1.96)
    return df_out