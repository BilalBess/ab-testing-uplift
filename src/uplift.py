"""Uplift modelling helpers: who should we email? (the people it CHANGES, not the people who would buy anyway)"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class TLearner:
    """Two models, one trained on treated users and one on control users.
    uplift(x) = P(y=1 | x, treated) - P(y=1 | x, control)."""

    def __init__(self, make_model=None):
        self.make_model = make_model or (lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, random_state=0))

    def fit(self, X, y, t):
        X, y, t = np.asarray(X), np.asarray(y), np.asarray(t)
        self.m1 = self.make_model().fit(X[t == 1], y[t == 1])
        self.m0 = self.make_model().fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        X = np.asarray(X)
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


def qini_curve(y, t, score, n_points=100):
    """Cumulative incremental outcome when we target the top x% by score.
    qini(x) = (mean_y_treated_in_top - mean_y_control_in_top) * n_top.
    Returns fractions (0..1) and incremental outcome (in number of users who respond because of the treatment)."""
    y, t, score = np.asarray(y, float), np.asarray(t), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n = len(y)
    fractions = np.linspace(0, 1, n_points + 1)
    values = [0.0]
    for f in fractions[1:]:
        k = int(round(f * n))
        yt, yc = y[:k][t[:k] == 1], y[:k][t[:k] == 0]
        if len(yt) == 0 or len(yc) == 0:
            values.append(values[-1])
            continue
        values.append((yt.mean() - yc.mean()) * k)
    return fractions, np.array(values)


def auuc(y, t, score, n_points=100):
    """Area between the Qini curve and the 'random targeting' straight line (bigger is better, 0 = random)."""
    f, q = qini_curve(y, t, score, n_points)
    random_line = f * q[-1]
    return float(np.trapezoid(q - random_line, f))


def policy_table(y_value, t, score, fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), cost=0.0):
    """If we email only the top x%, what is the incremental value per targeted user and the total profit?
    y_value can be spend (euros). Incremental value is estimated from the randomised test group."""
    y_value, t, score = np.asarray(y_value, float), np.asarray(t), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    y_value, t = y_value[order], t[order]
    rows = []
    for f in fractions:
        k = int(round(f * len(y_value)))
        yt, yc = y_value[:k][t[:k] == 1], y_value[:k][t[:k] == 0]
        inc = yt.mean() - yc.mean()
        rows.append({"share_targeted": f, "n_targeted": k, "incremental_value_per_user": inc,
                     "profit_per_user": inc - cost, "total_profit": (inc - cost) * k})
    return pd.DataFrame(rows)
