# One image for the API, the UI, and pipeline jobs. Code guide: section "Docker" (sec:docker).
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1

# Install pinned dependencies first so code changes do not invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY pipelines ./pipelines
COPY scripts ./scripts
COPY configs ./configs
COPY docs/knowledge ./docs/knowledge
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
