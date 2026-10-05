"""Golden-case evaluation of the agent.

    python -m pipelines.agent_eval

Runs each case in evals/golden_cases.json against a COPY of the database, so
evaluation never changes real offers, contacts, or memory. Exits non-zero
unless every case passes. Code guide: section "Agent evaluation" (sec:eval).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.agents.graph import run_agent
from app.agents.reviewer_agent import review
from app.agents.state import Deps, build_deps
from app.core.config import ROOT, get_settings
from app.data.store import Store
from app.services.memory import MemoryService
from app.services.sql_service import SQLService

CASES = ROOT / "evals" / "golden_cases.json"


def sandbox_deps(settings: dict, folder: Path) -> Deps:
    """Same model and knowledge, but a private copy of the database."""
    db = folder / "uplift.db"
    shutil.copy(settings["paths"]["database"], db)
    deps = build_deps(settings)
    store = Store(db)
    return Deps(settings=settings, store=store, sql=SQLService(store), kb=deps.kb,
                memory=MemoryService(store), scorer=deps.scorer)


def injected_state(kind: str) -> dict:
    """Hand-made states that the reviewer must reject."""
    good_exp = {"srm_p": 0.5, "max_abs_smd": 0.01}
    base = {"experiment": good_exp, "model": {"method": "causal_forest"},
            "selection": {"estimators": ["causal_forest"]}, "budget": 10, "memory": {}}
    if kind == "sleeping_dog_offer":
        base["offers"] = [{"customer_id": 1, "tau_hat": -0.02, "net_value": -5.0}]
    elif kind == "srm_failure":
        base["experiment"] = {"srm_p": 1e-6, "max_abs_smd": 0.01}
        base["offers"] = []
    return base


def check(case: dict, state: dict, deps: Deps) -> list[str]:
    """Return the list of failed expectations (empty = pass)."""
    exp, fails = case["expect"], []
    rev = state.get("review") or {}
    if "intent" in exp and state.get("intent") != exp["intent"]:
        fails.append(f"intent {state.get('intent')} != {exp['intent']}")
    if "review_passed" in exp and rev.get("passed") != exp["review_passed"]:
        fails.append(f"review passed = {rev.get('passed')}")
    offers = pd.DataFrame(state.get("offers") or [])
    if exp.get("offers_valid") and not offers.empty:
        if (offers["tau_hat"] <= 0).any() or (offers["net_value"] <= 0).any():
            fails.append("an offer has tau_hat <= 0 or net value <= 0")
    if exp.get("within_budget") and len(offers) > case["budget"]:
        fails.append(f"{len(offers)} offers > budget {case['budget']}")
    if exp.get("beats_population_true_effect") and not offers.empty:
        oracle = deps.store.read_table("oracle_current")
        chosen = oracle[oracle["customer_id"].isin(offers["customer_id"])]["true_cate"].mean()
        if not chosen > oracle["true_cate"].mean():
            fails.append("chosen customers do not have a higher true effect than average")
    if exp.get("no_recent_contacts") and not offers.empty:
        contacted = set(deps.store.read_table("contacts")["customer_id"])
        if contacted & set(offers["customer_id"]):
            fails.append("offered to customers inside the cooldown window")
    for text in exp.get("narrative_has", []):
        if text not in (state.get("narrative") or ""):
            fails.append(f"narrative missing '{text}'")
    if "citation_has" in exp:
        cites = " ".join(h["citation"] for h in state.get("knowledge") or [])
        if exp["citation_has"] not in cites:
            fails.append(f"no citation from {exp['citation_has']} (got: {cites})")
    if "fixable_issue" in exp and not any(exp["fixable_issue"] in i for i in rev.get("fixable", [])):
        fails.append("expected fixable issue not raised")
    if "blocking_issue" in exp and not any(exp["blocking_issue"] in i for i in rev.get("blocking", [])):
        fails.append("expected blocking issue not raised")
    return fails


def run_eval(settings: dict | None = None) -> dict:
    settings = settings or get_settings()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        deps = sandbox_deps(settings, Path(tmp))
        for case in cases:
            if "inject" in case:
                state = {"review": review(injected_state(case["inject"]), settings)}
            else:
                if case.get("setup") == "approve_previous_campaign":
                    first = run_agent(case["question"], case.get("budget"), settings, deps)
                    deps.store.log_contacts(pd.DataFrame(first["offers"]), first["campaign_id"],
                                            datetime.now(UTC).isoformat())
                state = run_agent(case["question"], case.get("budget"), settings, deps)
            fails = check(case, state, deps)
            results.append({"id": case["id"], "passed": not fails, "failures": fails})
    passed = sum(r["passed"] for r in results)
    return {"pass_rate": passed / len(results), "passed": passed, "total": len(results), "cases": results}


def main() -> None:
    report = run_eval()
    for r in report["cases"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']}  {'; '.join(r['failures'])}")
    print(f"pass rate: {report['pass_rate']:.2f} ({report['passed']}/{report['total']})")
    out = Path(get_settings()["paths"]["artifacts"]) / "reports" / "agent_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if report["pass_rate"] == 1.0 else 1)


if __name__ == "__main__":
    main()
