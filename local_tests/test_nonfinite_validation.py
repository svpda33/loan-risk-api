"""Local regression checks for validation response serialization."""

import json
import math

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    # These requests fail validation before model access; no model startup is needed.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("stage,field,value", [
    (1, "loan_amount", math.nan),
    (1, "loan_amount", math.inf),
    (1, "loan_amount", -math.inf),
    (2, "annual_inc", math.nan),
    (2, "annual_inc", math.inf),
    (2, "annual_inc", -math.inf),
])
def test_nonfinite_returns_validation_error(client, stage, field, value):
    if stage == 1:
        payload = {"loan_amount": 15000, "dti_percent": 18, "employment_length": "10+ years", "state": "CA"}
    else:
        payload = {"annual_inc": 72000}
    payload[field] = value
    response = client.post(
        f"/predict/stage{stage}",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(
        error.get("loc") == ["body", field] and error.get("input") == str(value)
        for error in detail
    )
    assert "Internal Server Error" not in response.text


def test_ordinary_type_error_still_returns_422(client):
    response = client.post(
        "/predict/stage1",
        json={"loan_amount": "bad", "dti_percent": 18, "employment_length": "10+ years", "state": "CA"},
    )
    assert response.status_code == 422
    assert any(error.get("loc") == ["body", "loan_amount"] for error in response.json()["detail"])


def test_malformed_json_still_returns_422(client):
    response = client.post("/predict/stage1", content='{"broken":', headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert any(error.get("type") == "json_invalid" for error in response.json()["detail"])
