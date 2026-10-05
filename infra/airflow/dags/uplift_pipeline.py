"""Weekly Airflow DAG. Each task calls one step of pipelines/run_pipeline.py,
so the DAG holds no business logic of its own.

    generate -> validate -> analyze -> benchmark -> train -> gates -> register -> score -> agent

`analyze` raises if the A/B test fails SRM or balance, which stops every
downstream task. `register` promotes the model only if `gates` passed.
Code guide: section "Airflow" (sec:airflow).
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


def _run(step: str) -> dict:
    from app.core.config import get_settings
    from pipelines import run_pipeline

    out = getattr(run_pipeline, f"step_{step}")(get_settings())
    if step == "benchmark":
        run_pipeline.write_summary(get_settings())
        return {"headline": out["headline"]}  # keep XCom small
    return {k: v for k, v in out.items() if not str(k).startswith("_")}


@dag(
    dag_id="uplift_retention_weekly",
    schedule="0 6 * * 1",  # Mondays 06:00
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["uplift", "retention", "causal"],
)
def uplift_retention_weekly():
    steps = ["generate", "validate", "analyze", "benchmark", "train", "gates", "register", "score", "agent"]
    tasks = [task(task_id=s)(_run)(s) for s in steps]
    for upstream, downstream in zip(tasks, tasks[1:], strict=False):
        upstream >> downstream


uplift_retention_weekly()
