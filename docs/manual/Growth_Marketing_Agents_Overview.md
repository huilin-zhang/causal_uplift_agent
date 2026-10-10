# Growth Marketing Agents with Causal ML

*LLM agents use causal ML to optimize promotions and drive measurable growth.*

## Background

Food delivery platforms use promotions and marketing messages to drive first orders, repeat purchases, reactivation, and cross-category sales. They need to decide whom to target, what to offer, and how to allocate their marketing budget.

This agent system combines causal ML and LLM. Causal ML estimates intervention uplift to guide promotion decisions and evaluates campaign impact. LLMs analyze market opportunities, propose campaigns based on causal results, generate evaluation reports, and answer questions about the workflow. 
## Workflow

```mermaid
flowchart LR
    J["Chat Agent:<br/>Answers questions about any part of the workflow"]
    A([Historical data and randomized experiments]) --> B["Opportunity Discovery Agent:<br/>Finds campaign opportunities"]
    P([Policy targets])
    B <--> C["Causal Targeting Agent:<br/>Estimates audience and intervention uplift"]
    C --> D["Planner Agent:<br/>Builds promotion plans from policy and causal ML"]
    A ~~~ J
    J ~~~ C
    B ~~~ P
    P --> D
    D --> E["Marketing Content Agent:<br/>Writes channel-specific campaign copy"]
    E --> F["Guardrail Agent:<br/>Checks plans, copy, and constraints"]
    F --> H([Launch campaign with a holdout group])
    H --> I["Campaign Evaluation Agent:<br/>Measures lift and monitors risk"]
    I -->|Experiment data and model errors| C
    I -->|Campaign findings| B
    I -->|Budget-response calibration| D
    I --> R["Report Agent:<br/>Generates a report from campaign evaluation"]
    classDef context fill:#f7f7f7,stroke:#b8b8b8,color:#555,stroke-width:1px
    class A,P,H context
```

**How the roles fit together:** The Opportunity Discovery Agent uses historical data and causal estimates to identify campaign opportunities. The Causal Targeting Agent estimates audience and intervention uplift, and the Planner Agent combines these estimates with policy targets to design a promotion experiment. The Marketing Content Agent writes channel-specific copy. Guardrail Agent checks the plan and content, and the campaign launches with a holdout group. The Campaign Evaluation Agent measures lift and monitors risk, and the Report Agent summarizes the results and generates a report. The Chat Agent answers questions about any part of the workflow, including historical data, policy targets, current campaigns, and future plans.


Evaluation data and model errors improve causal targeting, campaign findings guide opportunity discovery, and budget-response calibration refines promotion planning. 

## Agent Overview

| Agent | What it does |
|---|---|
| **Opportunity Discovery Agent** | Scans trends and audience signals, validates hypotheses with causal estimates, and prepares briefs with expected lift, cost, and risks. |
| **Causal Targeting Agent** | Before a campaign, estimates the incremental effect of promotion tiers and message attributes for each audience; recommends who to target and what tier to use. |
| **Planner Agent** | Applies business goals, budget, ROI, and user constraints to the proposed user list and promotion tiers; pairs users with messages without re-estimating effects. |
| **Marketing Content Agent** | Drafts email, push, and other channel copy from an approved plan; campaign values are inserted by the workflow.  |
| **Guardrail Agent** | Checks budget, contact frequency, cooldown periods, prohibited claims, brand tone, and consistency of copy with approved numbers; returns failures for revision. |
| **Campaign Evaluation Agent** | After launch, estimates actual lift, confidence intervals, and prediction error against a holdout; returns data to model training and findings to opportunity discovery. It can pause a campaign for serious negative effects. |
| **Report Agent** | After campaign evaluation, summarizes the results and generates a readable report without creating new effect estimates. |
| **Chat Agent** | Answers read-only questions about any part of the workflow, including historical data, policy targets, opportunities, current activity, planned campaigns, messages, guardrails, and evaluation results. |
