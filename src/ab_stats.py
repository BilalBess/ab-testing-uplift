"""Small, tested statistics helpers for A/B tests (no hidden magic, everything is a short function)."""
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def two_proportion_test(x_t, n_t, x_c, n_c, alpha=0.05):
    """Treatment vs control on a rate. Returns difference, relative lift, Wald CI and two-sided p-value
    (pooled standard error for the test, unpooled for the confidence interval)."""
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = diff / se_test
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_ci = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    q = stats.norm.ppf(1 - alpha / 2)
    return {"p_treat": p_t, "p_control": p_c, "diff": diff, "rel_lift": diff / p_c if p_c else np.nan,
            "ci_low": diff - q * se_ci, "ci_high": diff + q * se_ci, "z": z, "p_value": p_value}


def holm_correction(p_values):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate, less strict than Bonferroni)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def bootstrap_mean_diff(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap CI for mean(a) - mean(b). Useful for spend, which is mostly zeros with a long tail."""
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        diffs[k] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": a.mean() - b.mean(), "ci_low": lo, "ci_high": hi,
            "p_value": min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}


def sample_size_per_group(p_baseline, relative_lift, alpha=0.05, power=0.8):
    """Users needed in EACH group to detect a relative lift on a rate."""
    effect = proportion_effectsize(p_baseline * (1 + relative_lift), p_baseline)
    return int(np.ceil(NormalIndPower().solve_power(effect, alpha=alpha, power=power, ratio=1.0)))


def minimum_detectable_lift(p_baseline, n_per_group, alpha=0.05, power=0.8):
    """Smallest relative lift detectable with n users per group (searched numerically)."""
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sample_size_per_group(p_baseline, mid, alpha, power) > n_per_group:
            lo = mid
        else:
            hi = mid
    return hi


def standardized_mean_difference(a, b):
    """Balance check between two groups on a numeric covariate (|SMD| < 0.1 is usually considered balanced)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def beta_posterior_prob_greater(x_t, n_t, x_c, n_c, n_draws=200_000, seed=0):
    """Bayesian view with a flat Beta(1, 1) prior: P(rate_treat > rate_control) and expected loss if we ship treat."""
    rng = np.random.RandomState(seed)
    t = rng.beta(1 + x_t, 1 + n_t - x_t, n_draws)
    c = rng.beta(1 + x_c, 1 + n_c - x_c, n_draws)
    return {"prob_treat_better": float((t > c).mean()), "expected_loss_if_ship": float(np.maximum(c - t, 0).mean()),
            "diff_mean": float((t - c).mean())}


def false_positive_rate_with_peeking(n_per_group=5000, p=0.1, n_looks=10, n_sims=2000, alpha=0.05, seed=0):
    """A/A test simulation: both groups have the SAME rate. Compare stopping at the first significant look
    ("peeking") with looking once at the end."""
    rng = np.random.RandomState(seed)
    checkpoints = np.linspace(n_per_group / n_looks, n_per_group, n_looks).astype(int)
    peek_hits = final_hits = 0
    for _ in range(n_sims):
        a, b = rng.rand(n_per_group) < p, rng.rand(n_per_group) < p
        significant = []
        for n in checkpoints:
            r = two_proportion_test(b[:n].sum(), n, a[:n].sum(), n)
            significant.append(r["p_value"] < alpha)
        peek_hits += any(significant)
        final_hits += significant[-1]
    return {"peeking": peek_hits / n_sims, "single_look": final_hits / n_sims, "n_looks": n_looks}
