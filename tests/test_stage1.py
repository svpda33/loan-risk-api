import pytest

from conftest import assert_prediction


PATH = "/predict/stage1"
KEYS = ("rejected_0", "accepted_1")
LABELS = {0: "Rejected proxy (0)", 1: "Accepted proxy (1)"}


@pytest.mark.case("API-003", "Valid Stage 1 prediction and response schema", "Repository sample payload", "200; complete prediction schema")
def test_stage1_valid(api_call, stage1_payload):
    response = api_call("POST", PATH, stage1_payload)
    assert response.status_code == 200
    assert_prediction(response.json(), "stage1", KEYS, "accepted_1", 0.80, LABELS)


@pytest.mark.case("API-004", "Stage 1 repeated prediction consistency", "Same payload twice", "200 twice; same prediction and probabilities")
def test_stage1_repeat(api_call, stage1_payload):
    first = api_call("POST", PATH, stage1_payload)
    second = api_call("POST", PATH, stage1_payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


@pytest.mark.case("API-005", "Stage 1 nullable fields", "DTI and employment length null", "200; valid prediction")
def test_stage1_nulls(api_call, stage1_payload):
    stage1_payload.update(dti_percent=None, employment_length=None)
    response = api_call("POST", PATH, stage1_payload)
    assert response.status_code == 200
    assert_prediction(response.json(), "stage1", KEYS, "accepted_1", 0.80, LABELS)
