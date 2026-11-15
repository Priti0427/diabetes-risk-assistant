# Diabetes Risk Assistant — Streamlit app with the trained model baked in.
#
#   docker build -t diabetes-risk-assistant .
#   docker run -p 8501:8501 --env-file .env diabetes-risk-assistant
#
# The image downloads the dataset, trains every configuration in configs/config.yaml,
# selects the best run with MLflow, and serves the Streamlit UI on port 8501.
# Omit --env-file to run in offline rule-based mode (no LLM calls).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by xgboost; curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY scripts/ scripts/
COPY src/ src/

# Fetch data, train all experiments, export the best pipeline to models/best_model.joblib
RUN python scripts/get_data.py \
    && python -m src.train --all \
    && python scripts/compare_runs.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
