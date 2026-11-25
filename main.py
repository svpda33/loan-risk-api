"""Inference-only API for the two independent loan research models."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints


MODEL_DIR = Path(__file__).resolve().parent / "models"
STAGE1_PATH = MODEL_DIR / "stage1_pipeline.joblib"
STAGE2_PATH = MODEL_DIR / "stage2_full_eligible_lightgbm_pipeline.joblib"
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StateCode = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Z]{2}$")]
predict_lock = Lock()


class RequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Stage1Request(RequestSchema):
    loan_amount: float = Field(gt=0, description="Requested amount in dollars")
    dti_percent: float | None = Field(..., ge=0, description="Debt-to-income percentage; null if unavailable")
    employment_length: NonBlank | None = Field(..., description="For example, '10+ years'; null if unavailable")
    state: StateCode = Field(description="Two-letter US state code")


class Stage2Request(RequestSchema):
    loan_amnt: float = Field(gt=0)
    int_rate: float = Field(gt=0, description="Annual loan interest rate in percent")
    annual_inc: float = Field(gt=0)
    dti: float = Field(ge=0, description="Debt-to-income percentage")
    delinq_2yrs: int = Field(ge=0)
    fico_range_low: float = Field(ge=300, le=850)
    inq_last_6mths: int = Field(ge=0)
    open_acc: int = Field(ge=0)
    pub_rec: int = Field(ge=0)
    revol_bal: float = Field(ge=0)
    revol_util: float | None = Field(..., ge=0)
    total_acc: int = Field(ge=0)
    installment: float = Field(gt=0, description="Scheduled monthly installment")
    credit_age_years: float = Field(ge=0, description="Years from earliest credit line to loan issue month")
    mort_acc: int | None = Field(..., ge=0)
    total_rev_hi_lim: float | None = Field(..., ge=0)
    term: Literal["36 months", "60 months"]
    emp_length: NonBlank | None = Field(...)
    home_ownership: NonBlank
    verification_status: NonBlank
    purpose: NonBlank
    addr_state: StateCode
    application_type: NonBlank
    sub_grade: NonBlank


class PredictionResponse(BaseModel):
    stage: str
    task: str
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    threshold: float
    note: str


def _check_schema(artifact: dict, expected: list[str], key: str) -> None:
    if artifact.get(key) != expected:
        raise RuntimeError(f"Artifact {key} does not match the API request schema")


def _load_models() -> tuple[dict, dict]:
    if not STAGE1_PATH.is_file() or not STAGE2_PATH.is_file():
        raise RuntimeError("Both model artifacts must be present in the models directory")
    # Joblib uses pickle. Load only the two trusted artifacts bundled with this app.
    stage1 = joblib.load(STAGE1_PATH)
    stage2 = joblib.load(STAGE2_PATH)
    _check_schema(stage1, list(Stage1Request.model_fields), "feature_columns")
    _check_schema(stage2, list(Stage2Request.model_fields), "features")
    if not callable(getattr(stage1.get("pipeline"), "predict_proba", None)):
        raise RuntimeError("Stage 1 artifact does not support probability prediction")
    if not callable(getattr(stage2.get("model"), "predict_proba", None)):
        raise RuntimeError("Stage 2 artifact does not support probability prediction")
    if set(stage2["categorical_features"]) != set(stage2["category_levels"]):
        raise RuntimeError("Stage 2 categorical levels are incomplete")
    if stage1.get("threshold") != 0.80 or stage2.get("threshold") != 0.30:
        raise RuntimeError("Saved model thresholds differ from the reviewed values")
    return stage1, stage2


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = perf_counter()
    app.state.stage1, app.state.stage2 = _load_models()
    app.state.load_seconds = perf_counter() - started
    yield
    app.state.stage1 = None
    app.state.stage2 = None


app = FastAPI(
    title="Loan Eligibility and Repayment Risk Decision Support",
    version="1.0.0",
    description="Research prototype with separate historical-approval-proxy and repayment-risk models.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_response(request: Request, exc: RequestValidationError):
    # Pydantic rejects non-finite input, but JSONResponse cannot serialize that
    # same value when FastAPI includes it in the validation error's "input" field.
    safe_errors = jsonable_encoder(
        exc.errors(),
        custom_encoder={float: lambda value: value if math.isfinite(value) else str(value)},
    )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.get("/")
def root():
    return {
        "name": app.title,
        "description": "Two separate research predictions; neither is an automatic lending decision.",
        "endpoints": ["/health", "/predict/stage1", "/predict/stage2", "/docs"],
    }


@app.get("/health")
def health(request: Request):
    ready = getattr(request.app.state, "stage1", None) is not None and getattr(request.app.state, "stage2", None) is not None
    if not ready:
        raise HTTPException(status_code=503, detail="Models are not ready")
    return {"status": "ok", "models": {"stage1": "loaded", "stage2": "loaded"}}


def _validate_category(field: str, value: str | None, allowed: list[str]) -> str:
    normalized = "Missing" if value is None else value.strip()
    if normalized not in allowed:
        raise HTTPException(
            status_code=422,
            detail={"field": field, "message": "Category was not seen during training", "allowed": allowed},
        )
    return normalized


@app.post("/predict/stage1", response_model=PredictionResponse)
def predict_stage1(payload: Stage1Request, request: Request):
    artifact = request.app.state.stage1
    names = artifact["feature_columns"]
    data = payload.model_dump()
    ohe = artifact["pipeline"].named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"]
    for field, allowed in zip(["employment_length", "state"], ohe.categories_):
        _validate_category(field, data[field], list(allowed))
    row = {name: (np.nan if data[name] is None else data[name]) for name in names}
    frame = pd.DataFrame([row], columns=names)
    with predict_lock:
        accepted_probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    predicted = int(accepted_probability >= artifact["threshold"])
    return PredictionResponse(
        stage="stage1",
        task="Historical accepted-versus-rejected proxy prediction",
        predicted_class=predicted,
        predicted_label="Accepted proxy (1)" if predicted else "Rejected proxy (0)",
        probabilities={"rejected_0": 1.0 - accepted_probability, "accepted_1": accepted_probability},
        threshold=float(artifact["threshold"]),
        note="This predicts a historical proxy, not a verified original underwriting decision or loan eligibility.",
    )


@app.post("/predict/stage2", response_model=PredictionResponse)
def predict_stage2(payload: Stage2Request, request: Request):
    artifact = request.app.state.stage2
    data = payload.model_dump()
    for field in artifact["categorical_features"]:
        data[field] = _validate_category(field, data[field], artifact["category_levels"][field])
    frame = pd.DataFrame([data], columns=artifact["features"])
    for field in artifact["numeric_features"]:
        frame[field] = frame[field].astype("float32")
    for field in artifact["categorical_features"]:
        frame[field] = pd.Categorical(frame[field], categories=artifact["category_levels"][field])
    with predict_lock:
        default_probability = float(artifact["model"].predict_proba(frame)[0, 1])
    predicted = int(default_probability >= artifact["threshold"])
    return PredictionResponse(
        stage="stage2",
        task="Repayment risk among funded loans",
        predicted_class=predicted,
        predicted_label="Default risk flag (1)" if predicted else "No default risk flag (0)",
        probabilities={"fully_paid_0": 1.0 - default_probability, "charged_off_or_default_1": default_probability},
        threshold=float(artifact["threshold"]),
        note="Decision support only. A prediction must not automatically reject a loan or be treated as a repayment guarantee.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)
