# Convenience targets. Uses the active `python` (a venv or the Docker image).
# On Windows, `python src/run_pipeline.py` works without make.

PYTHON ?= python

.PHONY: help setup pipeline data train evaluate fairness explain drift app test \
        docker-build docker-run clean

help:
	@echo "setup        - install dependencies"
	@echo "pipeline     - run the full pipeline (data -> drift)"
	@echo "app          - launch the Streamlit dashboard"
	@echo "test         - run the test suite"
	@echo "docker-build - build the Docker image"
	@echo "docker-run   - run the dashboard in Docker on :8501"

setup:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) src/run_pipeline.py

data:
	$(PYTHON) src/data_prep.py
train:
	$(PYTHON) src/train.py
evaluate:
	$(PYTHON) src/evaluate.py
fairness:
	$(PYTHON) src/fairness.py
explain:
	$(PYTHON) src/explain.py
drift:
	$(PYTHON) src/drift.py

app:
	$(PYTHON) -m streamlit run src/monitor_app.py

test:
	$(PYTHON) -m pytest tests/ -q

docker-build:
	docker build -t readmission-monitoring .

docker-run:
	docker run --rm -p 8501:8501 readmission-monitoring

clean:
	rm -rf data/processed/* models/*.joblib reports/*.html reports/figures/*.png
