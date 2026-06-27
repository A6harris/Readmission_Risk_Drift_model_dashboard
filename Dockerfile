# One-command, reproducible image: builds all artifacts, then serves the
# dashboard. Build with `docker build -t readmission-monitoring .` and run with
# `docker run --rm -p 8501:8501 readmission-monitoring`, then open :8501.

FROM python:3.11-slim

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source (see .dockerignore for what's excluded).
COPY . .

# Build artifacts at image-build time: downloads the UCI dataset, trains the
# model, and generates the evaluation / fairness / SHAP / drift reports. The
# image is then self-contained and the dashboard starts instantly.
RUN python src/run_pipeline.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

CMD ["streamlit", "run", "src/monitor_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
