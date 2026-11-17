import numpy as np
import pytest

import ab_stats as ab
import uplift as up


def test_two_proportion_test_known_example():
    r = ab.two_proportion_test(120, 1000, 100, 1000)
    assert r["diff"] == pytest.approx(0.02)
    assert r["rel_lift"] == pytest.approx(0.2)
    assert r["z"] == pytest.approx(1.4291, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.153, abs=2e-3)
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_two_proportion_test_is_symmetric_in_sign():
    a = ab.two_proportion_test(120, 1000, 100, 1000)
    b = ab.two_proportion_test(100, 1000, 120, 1000)
    assert a["p_value"] == pytest.approx(b["p_value"])
    assert a["diff"] == pytest.approx(-b["diff"])


def test_holm_correction():
    adjusted = ab.holm_correction([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert (adjusted >= np.array([0.01, 0.04, 0.03])).all()      # never smaller than the raw p-value
    assert ab.holm_correction([0.9, 0.8]).max() <= 1.0


def test_sample_size_and_minimum_detectable_lift_are_consistent():
    n = ab.sample_size_per_group(0.10, 0.20)                       # 10% -> 12%
    assert 3700 < n < 4000
    assert ab.minimum_detectable_lift(0.10, n) == pytest.approx(0.20, abs=0.01)
    assert ab.sample_size_per_group(0.10, 0.10) > 3 * n            # smaller effects need much more data


def test_standardized_mean_difference():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 5000)
    assert abs(ab.standardized_mean_difference(a, a)) == 0
    assert ab.standardized_mean_difference(a + 1, a) == pytest.approx(1.0, abs=0.05)


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.RandomState(1)
    a = rng.exponential(2.0, 3000) * (rng.rand(3000) < 0.3)        # zero-inflated like spend
    b = rng.exponential(1.0, 3000) * (rng.rand(3000) < 0.3)
    r = ab.bootstrap_mean_diff(a, b, n_boot=500)
    assert r["ci_low"] < 0.3 * (2.0 - 1.0) < r["ci_high"]
    assert r["p_value"] < 0.05


def test_bayesian_probability():
    assert ab.beta_posterior_prob_greater(200, 1000, 100, 1000)["prob_treat_better"] > 0.999
    same = ab.beta_posterior_prob_greater(100, 1000, 100, 1000)["prob_treat_better"]
    assert same == pytest.approx(0.5, abs=0.02)


def test_peeking_inflates_false_positives():
    r = ab.false_positive_rate_with_peeking(n_per_group=2000, n_looks=10, n_sims=400)
    assert r["single_look"] < 0.09
    assert r["peeking"] > r["single_look"] + 0.05


def heterogeneous_data(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 2))
    t = rng.randint(0, 2, n)
    p = 0.10 + 0.25 * t * (x[:, 0] > 0)            # the treatment only works when x0 > 0
    y = (rng.rand(n) < p).astype(int)
    return x, y, t


def test_t_learner_finds_who_responds():
    x, y, t = heterogeneous_data()
    m = up.TLearner().fit(x[:12000], y[:12000], t[:12000])
    u = m.predict_uplift(x[12000:])
    assert u[x[12000:, 0] > 0].mean() > u[x[12000:, 0] <= 0].mean() + 0.1


def test_qini_and_auuc_reward_good_targeting():
    x, y, t = heterogeneous_data(seed=1)
    good = up.auuc(y, t, x[:, 0])                                  # score = the true driver
    random_scores = [up.auuc(y, t, np.random.RandomState(s).rand(len(y))) for s in range(5)]
    assert good > 10 * abs(np.mean(random_scores)) and good > 100
    f, q = up.qini_curve(y, t, x[:, 0])
    assert f[0] == 0 and q[0] == 0 and f[-1] == 1


def test_policy_table_last_row_is_the_overall_effect():
    x, y, t = heterogeneous_data(seed=2)
    pol = up.policy_table(y.astype(float), t, x[:, 0], cost=0.05)
    overall = y[t == 1].mean() - y[t == 0].mean()
    assert pol.iloc[-1]["incremental_value_per_user"] == pytest.approx(overall)
    assert pol.iloc[0]["incremental_value_per_user"] > pol.iloc[-1]["incremental_value_per_user"]
    assert (pol["profit_per_user"] == pol["incremental_value_per_user"] - 0.05).all()
