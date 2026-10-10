# Causal Uplift Agent — Operation Manual

*For Customer Retention · Version 0.1 · 2026*

## About this manual

This manual gets the project running on your computer and shows where to look at the results. It does not explain the methods; the *Code Guide* does that. Do the steps in order the first time. The first full run needs no cloud account, no API key, no Docker, and no LLM.

## What you get after a run

| Output | Where |
| --- | --- |
| Simulated A/B test, current subscribers, scores, offers, memory | `artifacts/uplift.db` (SQLite) |
| Experiment checks (SRM, balance, average effect) | `artifacts/reports/experiment.json` |
| Benchmark of all models over 30 simulations | `artifacts/reports/benchmark.json` |
| Headline numbers (the only report kept in git) | `artifacts/reports/benchmark_summary.json` |
| Quality gate results | `artifacts/reports/gates.json` |
| Agent brief for this month's campaign | `artifacts/reports/agent_report.json` |
| Trained model | `artifacts/models/uplift_scorer.joblib` and MLflow |
| MLflow runs and model registry | `artifacts/mlflow.db`, `artifacts/mlartifacts/` |
| API | <http://localhost:8000> (interactive docs at </docs>) |
| Dashboard | <http://localhost:8501> |

## What you need

- Python 3.11 (3.12 also works). Check with `python --version`.
- Git.
- About 2 GB of free disk space for packages, plus 1 GB if you download the public datasets.
- Optional: Docker Desktop (Section [Docker](#docker)), an Anthropic API key (Section [Optional: LLM-written briefs](#optional-llm-written-briefs)), TeX Live with XeLaTeX and `ctex` to rebuild these manuals.

A laptop CPU is enough. The full pipeline takes about 7 minutes; a reduced run takes about 1 minute.

## Get the code

```bash
git clone https://github.com/huilin-zhang/causal_uplift_agent.git
cd causal_uplift_agent
```

Open the whole folder in your editor (in VS Code: File, Open Folder), not a single file.

## Create a virtual environment

A virtual environment keeps this project's packages separate from other projects.

Windows PowerShell:

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

The prompt now starts with `(.venv)`.

> **Small system drive?**
> The environment and pip's download cache can live on any drive. Create the environment elsewhere (for example `python -m venv D:\venvs\uplift`) and set `PIP_CACHE_DIR` to a folder on the same drive before installing.

## Install the packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

`requirements.txt` pins the exact versions that produced the reported results. The second command installs the project itself so `python -m ...` works from the project folder.

## Optional: public datasets

The main results use simulated data and need nothing extra. Three public datasets show how the agent picks methods for different kinds of data. They are large and have their own licences, so they are not in the repository. Put them here:

- **Orange telecom churn uplift** (Verhelst et al.\ 2023). Download: <https://www.dropbox.com/s/27kyinnh9jcjdcg/churn_uplift_anonymized.csv?dl=0><br> Source page: <https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper><br> Place at `data/external/orange/churn_uplift_anonymized.csv`.
- **Hillstrom e-mail test** (2008). Download: <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv><br> Source page: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html><br> Place in `data/external/hillstrom/` (any .csv name).
- **Criteo uplift v2.1** (311 MB compressed, CC BY-NC-SA 4.0). Download: <https://ailab.criteo.com/criteo-uplift-prediction-dataset/><br> Keep it anywhere and make a sample (below).

The Criteo file has 14 million rows. Make a 0.5% sample (about 70,000 rows, 1.5 MB) once:

```bash
python -m scripts.sample_criteo --source path/to/criteo-uplift-v2.1.csv.gz --rate 0.005
```

To keep the datasets somewhere else, set `UPLIFT_DATA_DIR` to that folder.

## Run the pipeline

```bash
python -m pipelines.run_pipeline
```

You will see one line per step:

```text
[generate] done in 4.7s success
[validate] done in 0.2s success
[analyze] done in 2.1s True
draw 1/30 done in 13.5s
...
[benchmark] done in 330.5s
[train] done in 12.7s success
[gates] done in 0.0s passed
[register] done in 8.1s
[score] done in 1.2s success
[agent] done in 2.1s
```

For a quick run, use fewer simulation draws and trees. PowerShell:

```bash
$env:UPLIFT_DRAWS="3"; $env:UPLIFT_TREES="100"; $env:UPLIFT_BOOTSTRAP="200"
python -m pipelines.run_pipeline
```

macOS or Linux:

```bash
UPLIFT_DRAWS=3 UPLIFT_TREES=100 UPLIFT_BOOTSTRAP=200 python -m pipelines.run_pipeline
```

To run only some steps: `python -m pipelines.run_pipeline --steps analyze,gates`.

> **If the pipeline stops at analyze**
> The analyze step stops everything when the A/B test fails its SRM or balance check. That is on purpose: effects from a broken test are not reported.

## Read the results

Open `artifacts/reports/benchmark_summary.json`. With the default settings (seed 0) you should see the values below, printed from the latest run when this manual was built:

| Field | Value |
| --- | --- |
| `experiment.srm_p` | 0.567 (passes) |
| `experiment.max_abs_smd` | 0.022 (passes) |
| `experiment.ate_lin_pp` | 0.58 pp, CI 0.05 to 1.11 (significant at 5%) |
| `benchmark.headline.median_relative_gain` | 19.1%, CI 9.2% to 33.0% |
| `benchmark.headline.share_of_draws_uplift_wins` | 26 of 30 draws |
| `gates.passed` | true |

Numbers can differ in the last digit on another operating system or library version. The Code Guide, Section ``Results at a glance'', explains each number.

For the public datasets:

```bash
python -m scripts.run_real_benchmarks
```

This prints, for each dataset, which estimators the selector chose and why, and the held-out Qini of each with a 95% interval. It writes `artifacts/reports/real_benchmarks.json`.

## Ask the agent

```bash
python -m scripts.ask_agent "Plan this month's retention campaign." --budget 500 --trace
python -m scripts.ask_agent "Is the A/B test valid? Check SRM and balance."
python -m scripts.ask_agent "Should we send an offer to customer 42?"
python -m scripts.ask_agent "Explain what the Qini coefficient measures."
```

`--trace` prints which sub-agents ran. A campaign request ends with a campaign id that waits for human approval (next section).

## Start the API

```bash
uvicorn app.api.main:app --reload
```

Open <http://localhost:8000/docs>. Every endpoint can be tried from that page.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Status, loaded model version, tables |
| `GET /model` | Model method, metrics, which MLflow versions hold `champion` and `previous_champion` |
| `POST /model/reload` | Load the current champion (after a rollback) |
| `GET /experiment`, `GET /benchmark` | The saved reports |
| `POST /uplift/score` | Score new subscribers; returns effect, risk, net value, model version |
| `POST /policy/recommend` | Plain net-value ranking, no agent |
| `POST /agent/run` | Full agent: question in, brief, review, and campaign id out |
| `POST /campaigns/{id}/decision` | Human approval or rejection; approval logs contacts |

A full round trip in PowerShell:

```bash
$r = Invoke-RestMethod -Method Post -Uri http://localhost:8000/agent/run `
     -ContentType "application/json" -Body '{"question":"Plan this month''s retention campaign.","budget":500}'
$r.narrative
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/campaigns/$($r.campaign_id)/decision" `
     -ContentType "application/json" -Body '{"reviewer":"me","approve":true}'
```

After approval those customers are in a 30-day cooldown. Run the same request again and they will not appear.

## Start the dashboard

In a second terminal (with the environment activated):

```bash
streamlit run app/ui/streamlit_app.py
```

Open <http://localhost:8501>. The four tabs are Experiment, Benchmark, Targeting (budget slider), and Agent.

## Quality checks

```bash
ruff check .                  # lint
pytest -q                     # 70 tests, about 1 minute (tests/ is kept locally)
python -m pipelines.agent_eval  # 9 golden cases, must be 9/9
```

The agent evaluation runs on a copy of the database, so it does not change your offers, contacts, or memory. It writes `artifacts/reports/agent_eval.json`.

## MLflow: runs, versions, rollback

Every pipeline run is logged. To browse:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000
```

Open <http://localhost:5000>. Under Models, `uplift_scorer` shows the versions and the aliases `champion` and `previous_champion`. A new version gets `champion` only if all quality gates passed.

Roll back to the previous champion:

```bash
python -m scripts.rollback_model --status   # who holds which alias
python -m scripts.rollback_model            # swap champion and previous_champion
```

Then call `POST /model/reload` or restart the API. Check `GET /health`: `model_version` shows the new champion.

## Docker

One image runs every part.

```bash
docker compose build
docker compose run --rm pipeline       # first time: builds data, model, reports
docker compose up api ui mlflow        # serve
```

API on port 8000, dashboard on 8501, MLflow on 5000. The `artifacts` folder is shared with your computer, so results survive container restarts.

## Airflow (optional)

`infra/airflow/dags/uplift_pipeline.py` defines the weekly DAG `uplift_retention_weekly` (Mondays 06:00 UTC). It needs Airflow 3 with this project installed in the same environment:

```bash
pip install "apache-airflow>=3.0"
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/infra/airflow/dags
airflow standalone
```

Open <http://localhost:8080>, turn on the DAG, and trigger a run. Each task is one pipeline step; a failed SRM or balance check stops the DAG at `analyze`.

## GitHub Actions CI

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests: install, lint, tests (skipped when `tests/` is not in the repository), a reduced pipeline, a hard stop if any gate fails, the agent evaluation, and a rollback drill. The reports are uploaded as a build artifact.

## Optional: LLM-written briefs

By default the agent uses fixed templates. To let Claude rephrase the brief:

```bash
pip install anthropic
$env:ANTHROPIC_API_KEY="sk-ant-..."      # macOS/Linux: export ANTHROPIC_API_KEY=...
$env:AGENT_PROVIDER="anthropic"
python -m scripts.ask_agent "Plan this month's retention campaign."
```

The model only rewrites text. If it adds a number that is not in the template, or the call fails, the template is used. `LLM_MODEL` selects another model (default `claude-opus-5-5`).

## Rebuild the manuals

```bash
python docs/tools/build_manuals.py          # extract code snippets and compile all four PDFs
python docs/tools/build_manuals.py --check  # only check that every snippet still exists
```

Needs XeLaTeX and the `ctex` package (TeX Live or MiKTeX). The build runs in a temporary folder, so cloud-sync clients do not interfere.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named app` | Run commands from the project folder, and run `pip install --no-deps -e .` |
| LightGBM fails to load on Linux | Install the OpenMP runtime: `sudo apt-get install libgomp1` |
| Dashboard says ``No pipeline results yet'' | Run the pipeline first (Section [Run the pipeline](#run-the-pipeline)) |
| `/health` shows `model_loaded: false` | No model yet, or MLflow not reachable and no local file; run the pipeline |
| Port 8000 or 8501 already in use | `uvicorn ... --port 8001` or `streamlit run ... --server.port 8502` |
| Pipeline is slow | Use the quick settings in Section [Run the pipeline](#run-the-pipeline) |
| Agent proposes zero offers | No customer has positive predicted net value; this is a correct result, not an error |
| `run_real_benchmarks` says no datasets | Put the files in `data/external` (Section [Optional: public datasets](#optional-public-datasets)) |

## Command summary

```bash
pip install -r requirements.txt && pip install --no-deps -e .
python -m pipelines.run_pipeline
python -m scripts.ask_agent "Plan this month's retention campaign." --trace
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
ruff check . && pytest -q && python -m pipelines.agent_eval
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000
python -m scripts.rollback_model
python -m scripts.run_real_benchmarks
docker compose run --rm pipeline && docker compose up api ui mlflow
python docs/tools/build_manuals.py
```
