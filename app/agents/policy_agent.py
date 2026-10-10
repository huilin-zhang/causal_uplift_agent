"""Compatibility entry points; new workflows use Causal Targeting and Planner."""

from app.agents.causal_targeting_agent import load_scores as _load_scores
from app.agents.causal_targeting_agent import run_causal_targeting_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.state import AgentState, Deps


def load_scores(deps: Deps):
    return _load_scores(deps)


def run_policy_agent(state: AgentState, deps: Deps) -> AgentState:
    if "targeting" not in state:
        state = {**state, **run_causal_targeting_agent({**state, "intent": "campaign"}, deps)}
    return run_planner_agent(state, deps)
