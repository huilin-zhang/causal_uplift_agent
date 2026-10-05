"""Streamlit dashboard with four tabs: Experiment, Benchmark, Targeting, Agent.

    streamlit run app/ui/streamlit_app.py

Code guide: section "UI" (sec:ui).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `streamlit run` from any folder

from app.agents.graph import run_agent  # noqa: E402
from app.agents.policy_agent import load_scores  # noqa: E402
from app.agents.state import build_deps  # noqa: E402
from app.causal.policy import policy_summary, rank_by_net_value, rank_by_risk  # noqa: E402
from app.core.config import get_settings  # noqa: E402

st.set_page_config(page_title="Causal Uplift Agent", layout="wide")
settings = get_settings()
reports = Path(settings["paths"]["artifacts"]) / "reports"


@st.cache_resource
def deps():
    return build_deps(settings)


def load(name: str) -> dict | None:
    path = reports / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


st.title("Causal Uplift Agent for Customer Retention")
experiment, summary = load("experiment.json"), load("benchmark_summary.json")
if experiment is None or summary is None:
    st.warning("No pipeline results yet. Run `python -m pipelines.run_pipeline` first.")
    st.stop()

tab_exp, tab_bench, tab_target, tab_agent = st.tabs(["Experiment", "Benchmark", "Targeting", "Agent"])

with tab_exp:
    lin, dim = experiment["ate_lin"], experiment["ate_diff_in_means"]
    c = st.columns(4)
    c[0].metric("Subscribers", f"{experiment['n']:,}")
    c[1].metric("SRM p-value", f"{experiment['srm']['p_value']:.3f}", "pass" if experiment["srm"]["passed"] else "FAIL")
    c[2].metric("Max |SMD|", f"{experiment['balance']['max_abs_smd']:.3f}", "pass" if experiment["balance"]["passed"] else "FAIL")
    c[3].metric("Churn reduction (Lin)", f"{100 * lin['estimate']:.2f} pp",
                f"95% CI {100 * lin['ci_low']:.2f} to {100 * lin['ci_high']:.2f}", delta_color="off")
    st.caption(
        f"Difference in means: {100 * dim['estimate']:.2f} pp (95% CI {100 * dim['ci_low']:.2f} to "
        f"{100 * dim['ci_high']:.2f}). Control churn {100 * dim['churn_control']:.2f}%, treated {100 * dim['churn_treat']:.2f}%."
    )
    st.subheader("Covariate balance (SMD)")
    st.bar_chart(pd.Series(experiment["balance"]["smd"], name="SMD"))
    obs = experiment["observational_check"]
    st.subheader("Same customers, non-random assignment")
    st.table(pd.DataFrame({
        "estimate (pp)": [100 * obs["true_ate"], 100 * obs["naive_diff"], 100 * obs["ipw"], 100 * obs["aipw"]["estimate"]],
    }, index=["true effect", "naive difference", "IPW", "AIPW"]).round(2))

with tab_bench:
    b = summary["benchmark"]
    head = b["headline"]
    rel = head["relative_gain_ratio_of_totals"]
    st.markdown(
        f"**{head['description']}** over {b['n_draws']} simulation draws: "
        f"{100 * rel['estimate']:.1f}% more customers retained (95% CI {100 * rel['ci_low']:.1f}% to "
        f"{100 * rel['ci_high']:.1f}%). Uplift ranking won in {100 * head['share_of_draws_uplift_wins']:.0f}% of draws."
    )
    obs_r = b["observed_readout_draw0"]["abs_gain_customers"]
    st.caption(
        f"An A/B readout on one test set could not confirm this: observed difference {obs_r['median']:.0f} customers "
        f"(95% CI {obs_r['ci_low']:.0f} to {obs_r['ci_high']:.0f})."
    )
    st.dataframe(pd.DataFrame(b["table_mean"]).T.round(4), use_container_width=True)
    curves = deps().store.read_table("qini_curves") if "qini_curves" in deps().store.tables() else None
    if curves is not None:
        metric = st.radio("Curve", ["gain_true", "qini_observed"], horizontal=True)
        wide = curves.pivot_table(index="share", columns="model", values=metric)
        st.line_chart(wide)

with tab_target:
    budget = st.slider("Budget (offers)", 100, 10_000, settings["agent"]["default_budget"], step=100)
    scored = load_scores(deps())
    up = rank_by_net_value(scored, budget)
    risk = rank_by_risk(scored, budget)
    c = st.columns(3)
    c[0].metric("Offers sent (uplift)", f"{len(up):,}")
    c[1].metric("Model-expected net value", f"${policy_summary(up)['expected_net_value']:,.0f}")
    c[2].metric("Same budget by churn risk: net value", f"${risk['net_value'].sum():,.0f}")
    st.caption("Uplift ranking stops when net value turns negative; risk ranking spends the whole budget.")
    st.dataframe(up.head(200), use_container_width=True)

with tab_agent:
    question = st.text_input("Ask the agent", "Plan this month's retention campaign.")
    agent_budget = st.number_input("Budget", 100, 50_000, settings["agent"]["default_budget"], step=100)
    if st.button("Run"):
        state = run_agent(question, int(agent_budget), settings, deps())
        st.markdown(state["narrative"].replace("\n", "  \n").replace("$", "\\$"))
        st.json(state.get("review") or {})
        with st.expander("Trace"):
            for line in state["trace"]:
                st.text(line)
