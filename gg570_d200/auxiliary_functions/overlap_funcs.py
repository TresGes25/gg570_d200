import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def propensity_scores(df, covariate_cols, treatment_col):
    log_model = LogisticRegression(solver='liblinear')
    log_model.fit(df[covariate_cols], df[treatment_col])
    
    prop_scores = log_model.predict_proba(df[covariate_cols])[:, 1]
    return prop_scores


def plot_propensity_scores(df, treatment_col, prop_score_col, root):
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


def overlap_measures(df, treatment_col, prop_scores_col):
    extreme_scores_perc = round(100 * (len(df[(df[prop_scores_col] < 0.05) | (df[prop_scores_col] > 0.95)])/len(df)), 2)

    ipw = np.where(df[treatment_col] == 1, 1/df[prop_scores_col], 1/(1 - df[prop_scores_col]))
    ess = (np.sum(ipw)**2) / np.sum(ipw**2)
    ess_perc = round(100 * (ess/len(df)), 2)

    return extreme_scores_perc, ess_perc