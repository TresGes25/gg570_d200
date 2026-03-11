import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression


def propensity_scores(
    df: pd.DataFrame,
    covariate_cols: list[str],
    treatment_col: str,
) -> npt.NDArray[np.float64]:
    """
    Function that estimates propensity scores using logistic regression.

    Parameters:
    df: pd.DataFrame
        Input data.
    covariate_cols: list[str]
        Covariate column names.
    treatment_col: str
        Treatment column name.

    Returns:
    npt.NDArray[np.float64]
        Predicted treatment probabilities (propensity scores)
    """
    log_model = LogisticRegression(solver='liblinear')
    log_model.fit(df[covariate_cols], df[treatment_col])
    
    prop_scores = log_model.predict_proba(df[covariate_cols])[:, 1]
    return prop_scores


def plot_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    prop_score_col: str,
    root: Path,
) -> None:
    """
    Function that plots treated/control propensity score densities
    and saves the overlap figure.

    Parameters:
    df: pd.DataFrame
        Input data.
    treatment_col: str
        Treatment column name.
    prop_score_col: str
        Propensity score column name.
    root: Path
        Project root path used to save the figure.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    treated = df[df[treatment_col] == 1][prop_score_col]
    control = df[df[treatment_col] == 0][prop_score_col]

    plt.hist(treated, bins=30, alpha=0.5, label='Treated (1)', density=True, color='blue')
    plt.hist(control, bins=30, alpha=0.5, label='Control (0)', density=True, color='orange')

    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.title('Propensity Score Densities (by Treatment Status)')

    (root / "results").mkdir(parents=True, exist_ok=True)
    plt.savefig(root / f"results/true_overlap.png", dpi=500, bbox_inches='tight')

    plt.show()


def overlap_measures(
    df: pd.DataFrame,
    treatment_col: str,
    prop_scores_col: str,
) -> tuple[float, float]:
    """
    Function that computes overlap measures from propensity scores.

    Parameters:
    df: pd.DataFrame
        Input data.
    treatment_col: str
        Treatment column name.
    prop_scores_col: str
        Propensity score column name.

    Returns:
    tuple[float, float]
        (percentage of extreme propensity scores,
        effective sample size percentage relative to true sample size).
    """
    extreme_scores_perc = round(100 * (len(df[(df[prop_scores_col] < 0.05) | (df[prop_scores_col] > 0.95)])/len(df)), 2)

    ipw = np.where(df[treatment_col] == 1, 1/df[prop_scores_col], 1/(1 - df[prop_scores_col]))
    ess = (np.sum(ipw)**2) / np.sum(ipw**2)
    # Effective sample size (ESS) as a function of inverse probability weights
    ess_perc = round(100 * (ess/len(df)), 2)

    return extreme_scores_perc, ess_perc