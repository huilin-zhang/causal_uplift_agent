"""End-to-end pipeline. Each step is a plain function so the same code runs
from the command line, from tests, and from the Airflow DAG.

    python -m pipelines.run_pipeline                # full run
    python -m pipelines.run_pipeline --steps generate,analyze

Steps: generate -> validate -> analyze -> benchmark -> train -> gates ->
register -> score -> agent. Code guide: section "Pipeline" (sec:pipeline).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.causal.benchmark import run_benchmark
from app.causal.experiment import analyze_experiment, observational_check
from app.causal.learners import make_learner
from app.causal.scorer import UpliftScorer, config_fingerprint, data_fingerprint
from app.core.config import get_settings
from app.data.simulate import FEATURES, ORACLE_COLUMNS, feature_matrix, simulate
from app.data.store import Store
from pipelines.gates import evaluate_gates

STEPS = ["generate", "validate", "analyze", "benchmark", "train", "gates", "register", "score", "agent"]
CURRENT_SEED_OFFSET = 10_000  # current subscribers are a new cohort, not the test population


def _reports(cfg) -> Path:
    path = Path(cfg["paths"]["artifacts"]) / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(obj: dict, path: Path) -> Path:
    clean = {k: v for k, v in obj.items() if not str(k).startswith("_")}
    path.write_text(json.dumps(clean, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def step_generate(cfg) -> dict[str, Any]:
    """Simulate the A/B test, an observational variant, and the current subscriber base."""
    store = Store(cfg["paths"]["database"])
    n, seed, sim = cfg["n_subscribers"], cfg["seed"], cfg["sim"]
    exp = simulate(n, seed, sim, "rct")
    obs = simulate(n, seed, sim, "observational")
    cur = simulate(n, seed + CURRENT_SEED_OFFSET, sim, "rct")

    # Tables the agent can query never contain the oracle columns.
    store.write_table("experiment", exp.drop(columns=ORACLE_COLUMNS))
    store.write_table("observational", obs.drop(columns=ORACLE_COLUMNS))
    current = cur[["customer_id"] + FEATURES + ["value", "offer_cost"]]
    store.write_table("current_subscribers", current)
    # Oracle tables exist only for evaluation (and the agent eval), never for decisions.
    store.write_table("oracle_experiment", exp[["customer_id"] + ORACLE_COLUMNS])
    store.write_table("oracle_observational", obs[["customer_id"] + ORACLE_COLUMNS])
    store.write_table("oracle_current", cur[["customer_id"] + ORACLE_COLUMNS])
    return {"status": "success", "rows": {"experiment": len(exp), "observational": len(obs), "current": len(cur)}}


def step_validate(cfg) -> dict[str, Any]:
    """Data contract: required columns, no missing values, binary flags are 0/1."""
    store = Store(cfg["paths"]["database"])
    exp = store.read_table("experiment")
    problems = []
    for col in FEATURES + ["treatment", "churn_30d", "value", "offer_cost"]:
        if col not in exp:
            problems.append(f"missing column {col}")
        elif exp[col].isna().any():
            problems.append(f"nulls in {col}")
    for col in ("treatment", "churn_30d", "plan_annual"):
        if col in exp and not set(exp[col].unique()) <= {0, 1}:
            problems.append(f"{col} is not 0/1")
    if problems:
        raise ValueError("; ".join(problems))
    return {"status": "success", "rows": len(exp)}


def step_analyze(cfg) -> dict[str, Any]:
    """SRM, balance, and the average effect. Stops the pipeline if the test is invalid."""
    store = Store(cfg["paths"]["database"])
    exp = store.read_table("experiment")
    obs = store.read_table("observational").merge(store.read_table("oracle_observational"), on="customer_id")
    result = analyze_experiment(
        exp, feature_matrix(exp), cfg["experiment"]["srm_alpha"],
        cfg["experiment"]["smd_threshold"], cfg["sim"]["treat_share"],
    )
    result["observational_check"] = observational_check(obs, feature_matrix(obs))
    _save_json(result, _reports(cfg) / "experiment.json")
    if not result["valid"]:
        raise RuntimeError("experiment failed SRM or balance checks; stop before modeling")
    return result


def step_benchmark(cfg, log=print) -> dict[str, Any]:
    bench = run_benchmark(cfg["n_subscribers"], cfg["seed"], cfg["sim"], cfg["benchmark"], log=log)
    _save_json(bench, _reports(cfg) / "benchmark.json")
    _save_curves(bench, cfg)
    return bench


def _save_curves(bench: dict, cfg) -> None:
    """Qini curves of draw 0 for the UI."""
    from app.causal.metrics import expected_gain_curve, qini_curve

    d0 = bench["_draw0"]
    store = Store(cfg["paths"]["database"])
    exp = store.read_table("experiment").merge(store.read_table("oracle_experiment"), on="customer_id")
    te = d0["test_index"]
    t, r, tau = exp["treatment"].to_numpy()[te], 1 - exp["churn_30d"].to_numpy()[te], exp["true_cate"].to_numpy()[te]
    rows = []
    for name, score in d0["scores"].items():
        phi, q = qini_curve(score, r, t)
        g = expected_gain_curve(score, tau)
        k = np.round(phi * (len(g) - 1)).astype(int)
        rows += [{"model": name, "share": p, "qini_observed": qq, "gain_true": g[kk]} for p, qq, kk in zip(phi, q, k, strict=True)]
    store.write_table("qini_curves", pd.DataFrame(rows))


def step_train(cfg) -> dict[str, Any]:
    """Fit the pre-specified primary learner and the risk model on the full A/B data."""
    store = Store(cfg["paths"]["database"])
    exp = store.read_table("experiment")
    x, t, y = feature_matrix(exp), exp["treatment"].to_numpy(), exp["churn_30d"].to_numpy()
    b = cfg["benchmark"]
    bench = _load_json(_reports(cfg) / "benchmark.json")
    method = b["primary_model"]
    uplift = make_learner(method, seed=cfg["seed"], forest_trees=b["forest_trees"]).fit(x, t, y)
    risk = make_learner("churn_risk", seed=cfg["seed"]).fit(x, t, y)
    metrics = {
        "qini_true_normalized": bench["table"][method]["qini_true_normalized"]["mean"],
        "pehe_pp": bench["table"][method]["pehe_pp"]["mean"],
        "policy_true": bench["table"][method]["policy_true"]["mean"],
        "risk_policy_true": bench["table"]["churn_risk"]["policy_true"]["mean"],
        "relative_gain": bench["headline"]["relative_gain_ratio_of_totals"]["estimate"],
    }
    scorer = UpliftScorer(uplift, risk, method, list(x.columns), version=f"local-{time.strftime('%Y%m%d%H%M%S')}", metrics=metrics)
    scorer.save(cfg["paths"]["model"])
    return {"status": "success", "method": method, "path": str(cfg["paths"]["model"]),
            "data_hash": data_fingerprint(exp), "metrics": metrics}


def step_gates(cfg) -> dict[str, Any]:
    result = evaluate_gates(_load_json(_reports(cfg) / "experiment.json"), _load_json(_reports(cfg) / "benchmark.json"))
    _save_json(result, _reports(cfg) / "gates.json")
    return result


def step_register(cfg, force_promote: bool = False) -> dict[str, Any]:
    """Log to MLflow and register. Promote to champion only when gates passed
    (force_promote exists for tests and manual recovery only)."""
    from app.services.registry import log_and_register

    gates = _load_json(_reports(cfg) / "gates.json")
    scorer = UpliftScorer.load(cfg["paths"]["model"])
    store = Store(cfg["paths"]["database"])
    params = {
        "method": scorer.method,
        "n_subscribers": cfg["n_subscribers"],
        "seed": cfg["seed"],
        "n_draws": cfg["benchmark"]["n_draws"],
        "forest_trees": cfg["benchmark"]["forest_trees"],
        "config_hash": config_fingerprint({k: v for k, v in cfg.items() if k != "paths"}),
        "data_hash": data_fingerprint(store.read_table("experiment")),
    }
    return log_and_register(
        Path(cfg["paths"]["model"]), params, scorer.metrics,
        _reports(cfg) / "benchmark.json", cfg, promote=gates["passed"] or force_promote,
    )


def force_register(cfg) -> dict[str, Any]:
    return step_register(cfg, force_promote=True)


def step_score(cfg) -> dict[str, Any]:
    """Batch-score the current subscribers with the champion model."""
    from app.services.registry import load_champion

    store = Store(cfg["paths"]["database"])
    scorer = load_champion(cfg)
    scored = scorer.score(store.read_table("current_subscribers"))
    store.write_table("scores", scored[["customer_id", "tau_hat", "churn_risk", "net_value", "model_version"]])
    return {"status": "success", "scored": len(scored), "model_version": scorer.version,
            "positive_net_value": int((scored["net_value"] > 0).sum())}


def step_agent(cfg) -> dict[str, Any]:
    from app.agents.graph import run_agent

    state = run_agent("Plan this month's retention campaign.", budget=cfg["agent"]["default_budget"], settings=cfg)
    report = {k: state.get(k) for k in ("intent", "selection", "policy_summary", "review", "narrative")}
    _save_json(report, _reports(cfg) / "agent_report.json")
    return report


def write_summary(cfg) -> Path:
    """Small JSON with the headline numbers; this one file is kept in git."""
    exp = _load_json(_reports(cfg) / "experiment.json")
    bench = _load_json(_reports(cfg) / "benchmark.json")
    gates_path = _reports(cfg) / "gates.json"
    keep = ["pehe_pp", "qini_observed", "qini_true_normalized", "policy_true", "netpolicy_true_net_value", "netpolicy_n_contacted"]
    summary = {
        "n_subscribers": cfg["n_subscribers"],
        "experiment": {
            "srm_p": exp["srm"]["p_value"],
            "max_abs_smd": exp["balance"]["max_abs_smd"],
            "ate_lin_pp": {k: 100 * exp["ate_lin"][k] for k in ("estimate", "ci_low", "ci_high")},
            "ate_diff_in_means_pp": {k: 100 * exp["ate_diff_in_means"][k] for k in ("estimate", "ci_low", "ci_high")},
            "true_ate_pp": 100 * exp["observational_check"]["true_ate"],
            "observational_pp": {
                "naive": 100 * exp["observational_check"]["naive_diff"],
                "ipw": 100 * exp["observational_check"]["ipw"],
                "aipw": 100 * exp["observational_check"]["aipw"]["estimate"],
                "aipw_se": 100 * exp["observational_check"]["aipw"]["se"],
            },
        },
        "benchmark": {
            "n_draws": bench["n_draws"],
            "primary_model": bench["primary_model"],
            "table_mean": {m: {k: (row[k] or {}).get("mean") for k in keep if k in row} for m, row in bench["table"].items()},
            "headline": bench["headline"],
            "observed_readout_draw0": bench["observed_readout_draw0"],
            "dr_choice_counts": bench["dr_choice_counts"],
        },
        "gates": _load_json(gates_path) if gates_path.exists() else None,
    }
    summary["experiment"].update({
        "n_treat": exp["srm"]["n_treat"],
        "n_control": exp["srm"]["n_control"],
        "churn_control_pct": 100 * exp["ate_diff_in_means"]["churn_control"],
        "churn_treat_pct": 100 * exp["ate_diff_in_means"]["churn_treat"],
        "max_abs_smd_before_weighting": exp["observational_check"]["max_abs_smd_before"],
        "max_abs_smd_after_weighting": exp["observational_check"]["max_abs_smd_after_weighting"],
    })
    summary["campaign"] = _campaign_facts(cfg)
    return _save_json(summary, _reports(cfg) / "benchmark_summary.json")


def _campaign_facts(cfg) -> dict[str, Any] | None:
    """Latest agent offer list scored against the simulation's hidden truth, plus one example customer.
    The true values come from the oracle tables, which decisions never read."""
    store = Store(cfg["paths"]["database"])
    tables = store.tables()
    if not {"pending_offers", "oracle_current", "scores", "oracle_experiment"} <= set(tables):
        return None
    cur = store.read_table("current_subscribers")
    orc = store.read_table("oracle_current")
    offers = store.read_table("pending_offers")
    offers = offers[offers["campaign_id"] == offers["campaign_id"].max()]
    m = offers.merge(orc, on="customer_id")
    true_net_all = cur.merge(orc, on="customer_id").eval("true_cate * value - offer_cost")
    example = cur.merge(store.read_table("scores"), on="customer_id").query("customer_id == 42")
    tau = store.read_table("oracle_experiment")["true_cate"]
    out = {
        "n_offers": int(len(m)),
        "predicted_saved": float(m["tau_hat"].sum()),
        "true_saved": float(m["true_cate"].sum()),
        "predicted_net_value": float(m["net_value"].sum()),
        "true_net_value": float((m["true_cate"] * m["value"] - m["offer_cost"]).sum()),
        "oracle_net_value": float(true_net_all[true_net_all > 0].sum()),
        "true_cate_sd_pp": float(100 * tau.std()),
    }
    if not example.empty:
        r = example.iloc[0]
        out["customer_42"] = {k: float(r[k]) for k in ("churn_risk", "tau_hat", "value", "offer_cost", "net_value", "monthly_fee")}
    return out


def run(steps: list[str], cfg=None) -> dict[str, Any]:
    cfg = cfg or get_settings()
    funcs = {name: globals()[f"step_{name}"] for name in STEPS}
    results = {}
    for name in steps:
        start = time.perf_counter()
        out = funcs[name](cfg)
        results[name] = out
        status = out.get("status") or ("passed" if out.get("passed") else out.get("valid", ""))
        print(f"[{name}] done in {time.perf_counter() - start:.1f}s {status}")
    if (_reports(cfg) / "benchmark.json").exists() and (_reports(cfg) / "experiment.json").exists():
        write_summary(cfg)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=",".join(STEPS))
    args = parser.parse_args()
    run([s.strip() for s in args.steps.split(",") if s.strip()])


if __name__ == "__main__":
    main()
