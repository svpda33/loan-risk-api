# Loan Eligibility and Repayment Risk Decision Support API

One FastAPI service exposes two **separate research models**. Stage 1 predicts a historical accepted-versus-rejected proxy. Stage 2 predicts a default-risk flag for funded loans at the point just after loan terms are set. Neither response is an automatic lending decision.

## Included files

- `main.py`: inference-only API. It loads both models once at startup.
- `models/stage1_pipeline.joblib`: fitted preprocessing and Random Forest; threshold `0.80`.
- `models/stage2_full_eligible_lightgbm_pipeline.joblib`: LightGBM plus feature and category metadata; threshold `0.30`.
- `.python-version` and `requirements.txt`: the Python and library versions used for local verification.

The original CSV files and notebooks are **not required** at inference and are excluded by `.gitignore`. Joblib artifacts use pickle; load only the trusted files supplied with this project.

## Run locally

Use Python 3.14.3, then run from this directory:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/health`. The application uses `PORT` if provided, otherwise port `8000`, and runs **one Uvicorn worker**. Import `postman_collection.json` into Postman for sample requests.

## Input and response

`POST /predict/stage1` requires exactly `loan_amount`, `dti_percent`, `employment_length`, and `state`. DTI and employment length may be `null` when unavailable. Amount and DTI are numerical; DTI is a percentage such as `18`, not `0.18`. The Stage 1 artifact predicts the historical proxy label, not an original underwriting decision.

`POST /predict/stage2` requires the **24 fields in the Postman sample**. `credit_age_years` is the number of years from the earliest reported credit-line month to the loan issue month; the model expects this derived value, not a raw date. `revol_util`, `mort_acc`, `total_rev_hi_lim`, and `emp_length` may be `null`. Interest rate, DTI and revolving utilization are percentages. Categorical values must match categories seen during training; unsupported categories return HTTP 422. Stage 2 is for funded loans after terms are set and must not automatically reject an application.

Both endpoints return class probabilities, the saved threshold, a predicted class and an explanation. Input keys are required even when their value may be `null`. Extra fields and invalid types return HTTP 422.

## Railway deployment, when approved

1. Create a **private** GitHub repository containing `main.py`, `.python-version`, `requirements.txt`, `README.md`, `postman_collection.json`, `.gitignore`, and only the **two selected** model artifacts above. Before pushing, preview the files with `git status` and `git add -n .`. Never push `Datasets/`, notebooks, `.env`, or other model artifacts.
2. In Railway, create one service from that GitHub repository. Keep the project root as the service root. Railpack should detect `requirements.txt` and `.python-version`. Set the Start Command to `python main.py` if Railway does not detect it automatically. This starts Uvicorn on `0.0.0.0` and Railway's `PORT` with one worker.
3. Set the service healthcheck path to `/health`. No database, volume, credentials, or dataset path is needed. Keep one service instance and watch the build logs, startup logs, memory graph and credits.
4. Generate a public domain in Railway's Networking settings. Replace the Postman `baseUrl` variable with that URL and verify `/health`, `/docs`, and both prediction routes. Do this only after deciding to spend trial credits.

Railway currently lists a **1 GB trial** service limit and a **0.5 GB Free** service limit. Local Windows measurements are encouraging for both, but Linux startup, native-wheel installation and actual Railway memory must be checked in the first deployment. The Free plan's monthly credit is small and services may sleep when idle. If the service approaches its memory limit, stop it and review the measurements before continuing.

## Runtime notes

The app loads both artifacts once. No training or CSV loading occurs during startup or requests. For Stage 2, `main.py` converts the provided categorical strings to pandas categories using the levels saved in the artifact. There is no external preprocessing file. Python 3.14.3 and the pinned native packages should be kept together; changing versions requires a new compatibility check.
