# Deploying the dashboard

The app ([`src/monitor_app.py`](src/monitor_app.py)) is a *reader* of the
artifacts produced by the pipeline. Generated artifacts (the serialized model,
figures, and Evidently HTML) are gitignored, so a hosting platform has to build
them. There are two ways to handle that:

- **Self-provisioning (recommended for free hosts):** set the environment
  variable `AUTO_BOOTSTRAP=1`. On first load the app runs the full pipeline once
  (downloads the data, trains, generates reports), showing a spinner. Subsequent
  loads are instant. The committed JSON summaries mean the app still renders the
  headline numbers even before the bootstrap finishes.
- **Pre-built image:** use the [`Dockerfile`](Dockerfile), which bakes all
  artifacts into the image at build time (no bootstrap needed).

> ⚠️ The first bootstrap trains a model and runs SHAP + Evidently, which is
> CPU/RAM heavy. On constrained free tiers prefer the Docker route, or commit the
> artifacts you need.

---

## Option 1 — Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, "New app", and point it at your repo.
3. Set **Main file path** to `src/monitor_app.py`.
4. Under **Advanced settings**:
   - Python version: **3.11**.
   - Add a secret / environment variable: `AUTO_BOOTSTRAP = "1"`.
5. Deploy. `packages.txt` (it contains `libgomp1`, required by XGBoost) and
   `requirements.txt` are picked up automatically.

When it's live, add the URL to the badge at the top of the README.

## Option 2 — Hugging Face Spaces (Streamlit SDK)

1. Create a new Space → SDK: **Streamlit**.
2. Push this repo to the Space, and add this front-matter to the **Space's**
   `README.md` (keep the project README separate, or prepend this block):

   ```yaml
   ---
   title: Readmission Risk Monitoring
   emoji: 🏥
   colorFrom: pink
   colorTo: indigo
   sdk: streamlit
   app_file: src/monitor_app.py
   python_version: "3.11"
   ---
   ```
3. In the Space **Settings → Variables**, set `AUTO_BOOTSTRAP=1`.
4. `packages.txt` installs `libgomp1` automatically on Spaces too.

## Option 3 — Docker (anywhere)

```bash
docker build -t readmission-monitoring .
docker run --rm -p 8501:8501 readmission-monitoring
# open http://localhost:8501
```

The image builds every artifact during `docker build`, so the container starts
serving immediately with no bootstrap step.

---

## Closing the loop: automated retraining

[`src/retrain_trigger.py`](src/retrain_trigger.py) reads the drift summary and
retrains when the monitoring policy is breached, logging each decision to
`models/retrain_log.jsonl`:

```bash
python src/retrain_trigger.py --dry-run     # report only
python src/retrain_trigger.py               # retrain if any window breaches
```

Schedule it (cron, GitHub Actions, a platform scheduler) after each monitoring
run to make the deployment self-healing. In production it would retrain on fresh
data rather than the static demo dataset.
