# Growth Marketing Agents: implementation and operation

The business workflow and eight role descriptions come from
[Growth_Marketing_Agents_Overview.md](Growth_Marketing_Agents_Overview.md).
Detailed design requirements come from the parent project's `plan/PRD.md`.
The existing causal benchmark remains a **subscription-retention simulator**.
Its results do not establish effectiveness for food delivery or travel.

## Runtime responsibilities

| Role | Module | Current behavior |
|---|---|---|
| Opportunity Discovery | `app/agents/opportunity_discovery_agent.py` | Checks two pre-specified plan segments using an independent randomized cohort, with Bonferroni-adjusted intervals; stores briefs, rejections and exploration requests. |
| Causal Targeting | `app/agents/causal_targeting_agent.py` | Validates the historical experiment, reads version-matched scores or scores with the champion, and supplies unconstrained individual point estimates. No budgets or policy allocation. |
| Planner | `app/agents/planner_agent.py` | Designs supported candidate audiences, then uses an integer optimizer for count, monetary budget, gross ROI and new-user share constraints. Returns the best candidate and an alternative. No effect re-estimation. |
| Marketing Content Agent | `app/agents/marketing_content_agent.py` | Produces short, neutral copy with attributes fixed by the design. Renders numeric placeholders only from approved workflow values. |
| Guardrail | `app/agents/guardrail_agent.py` | Rechecks experiment validity, evidence values, budget, ROI, audience share, cooldown, weekly frequency and copy consistency. At most two allocation revisions. |
| Campaign Evaluation | `app/agents/campaign_evaluation_agent.py` | Measures intention-to-treat churn reduction after the complete outcome window, using the persisted randomized assignments. Records a Welch interval and prediction error; flags serious harm and pauses the local campaign. |
| Report | `app/agents/report.py:run_report` | Requires a completed evaluation and quotes its results. Optional LLM wording uses the existing numeric guard. |
| Chat | `app/agents/chat_agent.py` | Answers workflow, historical experiment, customer and method questions using read-only tools. Does not create offers, contacts or memory entries. |

SQL, retrieval and memory are shared tools, not additional business agents.
`ml_agent.py` and `reviewer_agent.py` retain low-level helpers for compatibility.
The older `policy_agent.py` entry point delegates to targeting and planning.

## Supported local workflow

1. Run the data/model pipeline. It creates a separate `discovery_experiment`
   cohort, which is never used for model fitting. Existing databases without
   this cohort return exploration requests rather than fabricated evidence.
2. Ask to discover opportunities, or explicitly request a campaign plan.
   Discovery proposals are advisory and cannot launch campaigns. A direct plan
   request uses the existing champion's evidence; it does not imply that a
   discovery brief has been approved.
3. Submit human-authored `policy_targets` through `POST /agent/run`. Natural
   language does not silently change monetary or ROI constraints. The older
   `budget` argument is still a **maximum audience count**, not currency.
4. Inspect the plan, alternative, copy and guardrail result. A passing plan is
   stored as `pending_approval`. Approval expires after 48 hours by default.
5. `POST /campaigns/{id}/decision` records explicit approval or rejection.
   Approval randomly reserves a holdout and logs only treated contacts. This
   is a local simulation: no email, push, or external service is invoked.
   Repeated decisions are rejected. Audience edits require a new plan.
6. Submit complete binary `churn_30d` outcomes through
   `POST /campaigns/{id}/evaluate`, with `outcome_window_complete: true`.
   Customer IDs must exactly match the randomized population, including the
   holdout. Each arm needs at least two observations. Outcomes are immutable.
7. Evaluation stores findings for discovery and calibration for planning, and
   preserves assignment/outcome rows for future training. Model retraining
   and automatic policy changes are not performed by the evaluation endpoint.
   Read the report and snapshot through `GET /campaigns/{id}` or Chat.

Example agent request:

```json
{
  "question": "Plan a retention campaign.",
  "policy_targets": {
    "objective": "incremental_retention",
    "max_contacts": 500,
    "total_budget": 1500,
    "min_roi": 1.5,
    "min_new_user_share": 0.3,
    "holdout_fraction": 0.1,
    "max_contacts_per_week": 2
  }
}
```

Here ROI is predicted incremental gross customer value / expected offer cost.
New users have tenure of at most three months. Optimization reserves enough
budget for the whole candidate audience before holdout assignment, so actual
expected spend can be smaller. Audience share constraints apply to that
pre-randomization audience. Numeric estimates and constraints are never
changed by copy generation. An infeasible plan returns to the human.

## Boundaries against the PRD

This update implements the eight-role structure and a runnable local lifecycle;
it is not the complete PRD. The following capabilities still require additional
data and implementation:

- Multi-tier/dose-response models, per-person confidence intervals, joint
  message/channel/timing effects and message-only campaigns. Unknown treatments
  cannot be substituted into the existing causal estimates. The current model
  has one treatment, `retention_discount`, and no individual significance claim.
- Autonomous LLM opportunity tool loops, external context ingestion, rejection
  deduplication across refreshed datasets, and natural-language policy parsing
  with a separate human confirmation checkpoint. Discovery currently checks
  two fixed hypotheses and stores previous findings for inspection.
- Multi-channel copy, LLM semantic brand review, automatically revised copy,
  and prompt optimization. Human review remains required for every draft.
- Automatic budget-response calibration, evaluation-driven retraining,
  message-attribute OPE, and multi-week closed-loop simulations.
- Agent/autonomous execution modes, external channel delivery, authentication,
  and production approval identity. Only assisted local simulation is enabled.
- Sequential monitoring and early-stop inference. The evaluation endpoint is
  a fixed-horizon readout, not a valid repeated-peeking monitor.

The historical README benchmark numbers remain results of the original
retention model run. They are not results of this new campaign lifecycle.
The older PDF/code manuals describe that baseline; use this guide for the
current orchestration and API.

## Checks

```text
python -m ruff check .
python -m pytest -q
python docs/tools/sync_agent_overview.py --check
python docs/tools/build_manuals.py --check
python -m pipelines.agent_eval
```

Use a project-local temporary directory for test artifacts and redirect tool
caches and system temporary files to the configured D: locations on Windows.
