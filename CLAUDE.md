# CLAUDE.md

## Delegation rules (main session + planner subagent)

The `planner` subagent is defined in `.claude/agents/planner.md`.

- The main session writes all routine code, tests, configs, and docs itself.
- At the start of each project stage, call the planner subagent for a design and follow it.
- After any causal-analysis code runs, call planner to review the method and results before moving on.
- If a bug survives two fix attempts, hand it to planner with the error output and the relevant files.
- Do not call planner for routine coding, formatting, or doc writing.
