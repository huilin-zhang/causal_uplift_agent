"""Run the estimator selector and the selected learners on public uplift datasets.

    python -m scripts.run_real_benchmarks

For each dataset found in data/external/: profile it, let selector.select()
pick estimators and a policy, fit only those learners on 70% of rows, and
report the observed Qini on the other 30% with a bootstrap interval. Real
data has no true effect, so there is no PEHE here.
Code guide: section "Real datasets" (sec:real).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.causal import metrics as M
from app.causal.bootstrap import stratified_weights
from app.causal.learners import make_learner
from app.causal.selector import describe, profile_data, select
from app.core.config import get_settings
from app.data.real_datasets import available, load_criteo, load_hillstrom, load_orange

CRITEO_FULL_ROWS = 13_979_592  # selection uses the size of the table the sample came from


def qini_with_ci(score, good, t, reps: int = 200, seed: int = 0) -> dict[str, float]:
    """Observed Qini on the test rows plus a stratified bootstrap interval (frozen model)."""
    point = M.qini_coefficient(score, good, t)
    w = stratified_weights(t, reps, seed)
    boot = []
    for b in range(reps):
        idx = np.repeat(np.arange(len(t)), w[b])
        boot.append(M.qini_coefficient(score[idx], good[idx], t[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"qini": point, "ci_low": float(lo), "ci_high": float(hi)}


def evaluate(ds, estimators: list[str], seed: int = 0, trees: int = 200) -> dict:
    """Fit each allowed learner on 70%, score Qini on 30%. Learners model bad = 1 - good."""
    tr, te = train_test_split(np.arange(len(ds.t)), train_size=0.7, stratify=ds.t, random_state=seed)
    share = float(ds.t[tr].mean())  # known assignment share of this randomized test
    bad = 1 - ds.good
    rows = {}
    for name in ["churn_risk"] + estimators:
        model = make_learner(name, seed=seed, treat_share=share, forest_trees=trees)
        model.fit(ds.x.iloc[tr], ds.t[tr], bad[tr])
        # churn_risk scores P(bad outcome without treatment): the usual "highest risk first" list.
        score = model.predict(ds.x.iloc[te])
        rows[name] = qini_with_ci(score, ds.good[te], ds.t[te])
    rng = np.random.default_rng(seed)
    rows["random"] = qini_with_ci(rng.random(len(te)), ds.good[te], ds.t[te])
    return {"n_train": len(tr), "n_test": len(te), "treat_share": share, "qini": rows}


def run_one(name: str) -> dict:
    if name == "hillstrom":
        raw = load_hillstrom().raw
        profile = profile_data(raw, "segment", "visit", "rct", control_label="No E-Mail")
        selection = select(profile)
        arms = {}
        for arm in ["Mens E-Mail", "Womens E-Mail"]:  # one learner per arm vs control
            arms[arm] = evaluate(load_hillstrom(arm=arm), selection.estimators)
        result = {"arms": arms}
    else:
        ds = load_orange() if name == "orange_churn" else load_criteo()
        df = pd.DataFrame({"t": ds.t, "y": 1 - ds.good if name == "orange_churn" else ds.good})
        profile = profile_data(df, "t", "y", "rct")
        if name == "criteo_sample":
            profile.n = CRITEO_FULL_ROWS
        selection = select(profile)
        result = evaluate(ds, selection.estimators)
    return {"profile": profile.__dict__, "selection": selection.to_dict(),
            "summary": describe(profile, selection), **result}


def main() -> None:
    names = available()
    if not names:
        print("No datasets in data/external/. See the operation manual for download steps.")
        return
    report = {}
    for name in names:
        print(f"== {name}")
        report[name] = run_one(name)
        print(report[name]["summary"])
        blocks = report[name].get("arms", {"": report[name]})
        for arm, block in blocks.items():
            for model, q in block["qini"].items():
                print(f"  {arm:14s} {model:14s} Qini {q['qini']:+.4f}  [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}]")
    out = Path(get_settings()["paths"]["artifacts"]) / "reports" / "real_benchmarks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
