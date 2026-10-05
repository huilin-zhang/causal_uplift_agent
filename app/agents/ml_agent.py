"""ML sub-agent: check the experiment, choose estimators by data type, and
attach the champion model's scores.

It does not train models during a chat request; training belongs to the
scheduled pipeline. It reads the pipeline's reports and the score table.
Code guide: section "ML agent" (sec:ml-agent).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agents.state import AgentState, Deps
from app.causal.experiment import analyze_experiment
from app.causal.selector import describe, profile_data, select
from app.data.simulate import feature_matrix


def _report(deps: Deps, name: str) -> dict | None:
    path = Path(deps.settings["paths"]["artifacts"]) / "reports" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_ml_agent(state: AgentState, deps: Deps) -> AgentState:
    trace = list(state.get("trace", []))
    exp_df = deps.store.read_table("experiment")

    experiment = _report(deps, "experiment.json")
    if experiment is None:
        experiment = analyze_experiment(exp_df, feature_matrix(exp_df), expected_share=deps.settings["sim"]["treat_share"])
        trace.append("ml_agent: no saved experiment report; analyzed the table directly")

    # The design (randomized or not) comes from the data owner's metadata.
    profile = profile_data(exp_df, "treatment", "churn_30d", design="rct", value_col="value", cost_col="offer_cost")
    selection = select(profile)
    trace.append("ml_agent: " + describe(profile, selection).splitlines()[1])

    model = {"method": None, "version": None}
    if deps.scorer is not None:
        model = {"method": deps.scorer.method, "version": deps.scorer.version, "metrics": deps.scorer.metrics}
        bench = _report(deps, "benchmark.json")
        if bench:
            model["headline"] = bench["headline"]
            model["observed_readout"] = bench["observed_readout_draw0"]
    else:
        trace.append("ml_agent: no trained model found; run the pipeline first")

    return {
        "experiment": {
            "valid": experiment["valid"],
            "srm_p": experiment["srm"]["p_value"],
            "max_abs_smd": experiment["balance"]["max_abs_smd"],
            "ate_lin": experiment["ate_lin"],
            "ate_diff_in_means": experiment["ate_diff_in_means"],
        },
        "profile": profile.__dict__,
        "selection": selection.to_dict(),
        "model": model,
        "trace": trace,
    }
