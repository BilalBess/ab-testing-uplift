"""E-mail marketing A/B test (Hillstrom dataset): analysis + uplift modelling.

Run:  python src/analysis.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

import ab_stats as ab
import uplift as up

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hillstrom.csv"
REP = ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
URL = ("http://www.minethatdata.com/"
       "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
ARMS = {"Mens E-Mail": "Men's e-mail", "Womens E-Mail": "Women's e-mail"}
COST_PER_EMAIL = 0.10   # dollars, assumption for the targeting policy


def load():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_bytes(requests.get(URL, timeout=60).content)
    return pd.read_csv(DATA)


def main():
    df = load()
    ctrl = df[df["segment"] == "No E-Mail"]
    out = {"n_total": len(df), "n_per_arm": df["segment"].value_counts().to_dict()}

    # ------------------------------------------------------------ 1) was the randomisation fine?
    numeric = ["recency", "history", "mens", "womens", "newbie"]
    smd = pd.DataFrame({ARMS[a]: [ab.standardized_mean_difference(df.loc[df.segment == a, c], ctrl[c]) for c in numeric]
                        for a in ARMS}, index=numeric).round(4)
    chi = {c: round(float(stats.chi2_contingency(pd.crosstab(df["segment"], df[c]))[1]), 3) for c in ["zip_code", "channel"]}
    print("SMD vs control:\n", smd, "\nchi-square p-values:", chi)
    out["balance_max_abs_smd"] = float(smd.abs().max().max())
    out["balance_chi2_p"] = chi

    # ------------------------------------------------------------ 2) effect of each e-mail, 3 metrics
    rows = []
    for arm, label in ARMS.items():
        t = df[df["segment"] == arm]
        for metric in ["visit", "conversion"]:
            r = ab.two_proportion_test(t[metric].sum(), len(t), ctrl[metric].sum(), len(ctrl))
            rows.append({"arm": label, "metric": metric, **r})
        b = ab.bootstrap_mean_diff(t["spend"], ctrl["spend"])
        rows.append({"arm": label, "metric": "spend", "p_treat": t["spend"].mean(), "p_control": ctrl["spend"].mean(),
                     "diff": b["diff"], "rel_lift": b["diff"] / ctrl["spend"].mean(), "ci_low": b["ci_low"],
                     "ci_high": b["ci_high"], "p_value": b["p_value"]})
    res = pd.DataFrame(rows)
    res["p_holm"] = ab.holm_correction(res["p_value"].values)       # 6 tests in the family
    res.round(5).to_csv(REP / "ab_results.csv", index=False)
    print(res[["arm", "metric", "p_treat", "p_control", "diff", "rel_lift", "ci_low", "ci_high", "p_value", "p_holm"]]
          .round(4).to_string(index=False))

    # men's vs women's e-mail (is one better than the other?)
    m, w = df[df.segment == "Mens E-Mail"], df[df.segment == "Womens E-Mail"]
    for metric in ["visit", "conversion"]:
        r = ab.two_proportion_test(m[metric].sum(), len(m), w[metric].sum(), len(w))
        out[f"mens_vs_womens_{metric}"] = {k: round(float(v), 5) for k, v in r.items()}
    out["mens_vs_womens_spend"] = {k: round(float(v), 5) for k, v in ab.bootstrap_mean_diff(m.spend, w.spend).items()}

    # ------------------------------------------------------------ 3) Bayesian view of conversion
    bay = {}
    for arm, label in ARMS.items():
        t = df[df.segment == arm]
        bay[label] = ab.beta_posterior_prob_greater(t.conversion.sum(), len(t), ctrl.conversion.sum(), len(ctrl))
    bay["Men's vs women's"] = ab.beta_posterior_prob_greater(m.conversion.sum(), len(m), w.conversion.sum(), len(w))
    out["bayesian_conversion"] = {k: {a: round(b, 4) for a, b in v.items()} for k, v in bay.items()}
    print("Bayesian:", out["bayesian_conversion"])

    # ------------------------------------------------------------ 4) variance reduction (regression adjustment)
    d2 = df[df.segment.isin(["Mens E-Mail", "No E-Mail"])].copy()
    d2["treated"] = (d2.segment == "Mens E-Mail").astype(int)
    simple = smf.ols("spend ~ treated", d2).fit(cov_type="HC1")
    adj = smf.ols("spend ~ treated + recency + history + mens + womens + newbie + C(zip_code) + C(channel)", d2).fit(cov_type="HC1")
    out["spend_men_vs_control"] = {
        "simple_effect": round(float(simple.params["treated"]), 4), "simple_se": round(float(simple.bse["treated"]), 4),
        "adjusted_effect": round(float(adj.params["treated"]), 4), "adjusted_se": round(float(adj.bse["treated"]), 4)}
    print(out["spend_men_vs_control"])

    # ------------------------------------------------------------ 5) power: what could this test detect?
    n_arm = int(len(ctrl))
    p0 = float(ctrl.conversion.mean())
    out["power"] = {
        "baseline_conversion": round(p0, 5), "n_per_arm": n_arm,
        "min_detectable_relative_lift_conversion": round(ab.minimum_detectable_lift(p0, n_arm), 3),
        "n_needed_for_10pct_lift": ab.sample_size_per_group(p0, 0.10),
        "n_needed_for_50pct_lift": ab.sample_size_per_group(p0, 0.50)}
    print(out["power"])

    # ------------------------------------------------------------ 6) subgroup effects: careful!
    sub_rows = []
    for var in ["channel", "zip_code", "newbie", "history_segment"]:
        for val, g in d2.groupby(var):
            r = ab.two_proportion_test(g[g.treated == 1].visit.sum(), (g.treated == 1).sum(),
                                       g[g.treated == 0].visit.sum(), (g.treated == 0).sum())
            sub_rows.append({"variable": var, "value": str(val), "n": len(g), "visit_diff_pp": 100 * r["diff"],
                             "ci_low_pp": 100 * r["ci_low"], "ci_high_pp": 100 * r["ci_high"]})
    sub = pd.DataFrame(sub_rows)
    sub.round(3).to_csv(REP / "subgroup_effects_visit.csv", index=False)
    inter = {}
    for var in ["channel", "zip_code", "newbie", "history_segment"]:
        full = smf.logit(f"visit ~ treated * C({var})", d2).fit(disp=0)
        base = smf.logit(f"visit ~ treated + C({var})", d2).fit(disp=0)
        lr = 2 * (full.llf - base.llf)
        inter[var] = float(stats.chi2.sf(lr, full.df_model - base.df_model))
    out["interaction_tests_p"] = {k: round(v, 4) for k, v in inter.items()}
    out["interaction_tests_p_holm"] = dict(zip(inter, np.round(ab.holm_correction(list(inter.values())), 4).tolist()))
    print(sub.round(2).to_string(index=False))
    print("interaction p (raw):", out["interaction_tests_p"], "\n(holm):", out["interaction_tests_p_holm"])

    # ------------------------------------------------------------ 7) uplift model (men's e-mail vs control)
    feats = pd.get_dummies(d2[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                           columns=["zip_code", "channel"], dtype=float)
    X_tr, X_te, y_tr, y_te, t_tr, t_te, s_tr, s_te = train_test_split(
        feats.values, d2["visit"].values, d2["treated"].values, d2["spend"].values,
        test_size=0.4, stratify=d2["treated"].values, random_state=0)
    tl = up.TLearner().fit(X_tr, y_tr, t_tr)
    score_uplift = tl.predict_uplift(X_te)
    # the classic mistake: target the people who are most likely to visit anyway (model trained on control only)
    resp = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                          random_state=0).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
    score_resp = resp.predict_proba(X_te)[:, 1]
    rng = np.random.RandomState(0)
    score_rand = rng.rand(len(y_te))
    scores = {"Uplift model (T-learner)": score_uplift, "Response model (likely visitors)": score_resp,
              "Random": score_rand}
    a = {k: up.auuc(y_te, t_te, v) for k, v in scores.items()}
    out["auuc_visit"] = {k: round(v, 2) for k, v in a.items()}
    print("AUUC (incremental visits vs random line):", out["auuc_visit"])

    # bootstrap CI of the AUUC advantage of the uplift model over the response model
    gaps = []
    for b in range(200):
        idx = np.random.RandomState(b).choice(len(y_te), len(y_te))
        gaps.append(up.auuc(y_te[idx], t_te[idx], score_uplift[idx]) - up.auuc(y_te[idx], t_te[idx], score_resp[idx]))
    out["auuc_gap_uplift_minus_response_ci95"] = [round(float(np.percentile(gaps, 2.5)), 2),
                                                  round(float(np.percentile(gaps, 97.5)), 2)]
    print("AUUC gap CI:", out["auuc_gap_uplift_minus_response_ci95"])

    # repeat on 10 different train/test splits and add a logistic-regression T-learner, to avoid trusting one lucky split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rep = {"GBM T-learner": [], "Logistic T-learner": [], "Response model": [], "Random": []}
    for seed in range(10):
        Xa, Xb, ya, yb, ta, tb = train_test_split(feats.values, d2["visit"].values, d2["treated"].values,
                                                  test_size=0.4, stratify=d2["treated"].values, random_state=seed)
        g = up.TLearner().fit(Xa, ya, ta).predict_uplift(Xb)
        lg = up.TLearner(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))).fit(Xa, ya, ta).predict_uplift(Xb)
        rp = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                            random_state=0).fit(Xa[ta == 0], ya[ta == 0]).predict_proba(Xb)[:, 1]
        rd = np.random.RandomState(seed).rand(len(yb))
        for k, v in zip(rep, [g, lg, rp, rd]):
            rep[k].append(up.auuc(yb, tb, v))
    out["auuc_10_splits_mean_std"] = {k: [round(float(np.mean(v)), 2), round(float(np.std(v)), 2)] for k, v in rep.items()}
    print("AUUC over 10 splits (mean, std):", out["auuc_10_splits_mean_std"])

    pol = up.policy_table(s_te, t_te, score_uplift, cost=COST_PER_EMAIL)
    pol_all = up.policy_table(s_te, t_te, score_resp, cost=COST_PER_EMAIL)
    pol.round(4).to_csv(REP / "targeting_policy_uplift_model.csv", index=False)
    print(pol.round(3).to_string(index=False))
    best = pol.loc[pol["total_profit"].idxmax()]
    out["policy"] = {"cost_per_email": COST_PER_EMAIL,
                     "best_share_targeted": float(best["share_targeted"]),
                     "profit_best": round(float(best["total_profit"]), 1),
                     "profit_email_everyone": round(float(pol.iloc[-1]["total_profit"]), 1),
                     "profit_best_response_model_at_same_share": round(float(
                         pol_all.loc[pol_all["share_targeted"] == best["share_targeted"], "total_profit"].iloc[0]), 1),
                     "n_test": len(y_te)}

    # ------------------------------------------------------------ 8) peeking simulation
    sim = ab.false_positive_rate_with_peeking(n_per_group=5000, p=0.1, n_looks=10, n_sims=2000)
    out["peeking_simulation"] = sim
    print(sim)

    # ------------------------------------------------------------ figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, ttl, scale in [(axes[0], "visit", "Site visit rate (%)", 100),
                                    (axes[1], "conversion", "Conversion rate (%)", 100),
                                    (axes[2], "spend", "Spend per user ($)", 1)]:
        sub_m = res[res.metric == metric]
        labels = ["No e-mail"] + sub_m["arm"].tolist()
        vals = [sub_m["p_control"].iloc[0] * scale] + [v * scale for v in sub_m["p_treat"]]
        lo = [0] + [(r.diff - r.ci_low) * scale for r in sub_m.itertuples()]
        hi = [0] + [(r.ci_high - r.diff) * scale for r in sub_m.itertuples()]
        ax.bar(labels, vals, yerr=[lo, hi], color=["#a0aec0", "#2b6cb0", "#c05621"], capsize=4)
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Effect of each e-mail (bars = group mean, whiskers = 95% CI of the difference)")
    fig.tight_layout()
    fig.savefig(FIG / "ab_results.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    smd.plot.barh(ax=ax)
    ax.axvline(0.1, color="red", linestyle="--")
    ax.axvline(-0.1, color="red", linestyle="--")
    ax.set_xlabel("Standardised mean difference vs control (red = 0.1 threshold)")
    ax.set_title("Randomisation check")
    fig.tight_layout()
    fig.savefig(FIG / "balance_check.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, s in scores.items():
        f, q = up.qini_curve(y_te, t_te, s)
        ax.plot(f, q, label=f"{name}")
    f, q = up.qini_curve(y_te, t_te, score_rand)
    ax.plot([0, 1], [0, up.qini_curve(y_te, t_te, score_uplift)[1][-1]], "k--", alpha=0.5, label="Random line")
    ax.set_xlabel("Share of users e-mailed (best scores first)")
    ax.set_ylabel("Extra site visits caused by the e-mail")
    ax.set_title("Qini curves (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "qini_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pol["share_targeted"] * 100, pol["total_profit"], marker="o", label="Uplift model")
    ax.plot(pol_all["share_targeted"] * 100, pol_all["total_profit"], marker="o", label="Response model")
    ax.set_xlabel("% of users e-mailed")
    ax.set_ylabel(f"Incremental profit on test set ($, cost {COST_PER_EMAIL:.2f}/e-mail)")
    ax.set_title("Who should we e-mail?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "targeting_profit.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Look once\nat the end", f"Peek {sim['n_looks']} times,\nstop when p<0.05"],
           [sim["single_look"] * 100, sim["peeking"] * 100], color=["#2f855a", "#c53030"])
    ax.axhline(5, color="k", linestyle="--")
    ax.set_ylabel("False positives in A/A tests (%)")
    ax.set_title("Why you should not peek")
    fig.tight_layout()
    fig.savefig(FIG / "peeking.png", dpi=150)
    plt.close(fig)

    n_grid = np.linspace(1000, 40000, 40).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_grid, [ab.minimum_detectable_lift(p0, n) * 100 for n in n_grid])
    ax.axvline(n_arm, color="red", linestyle="--", label="this test")
    ax.set_xlabel("Users per group")
    ax.set_ylabel("Minimum detectable relative lift (%)")
    ax.set_title(f"Power: conversion, baseline {p0:.2%}, 80% power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "power.png", dpi=150)
    plt.close(fig)

    (REP / "summary.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
