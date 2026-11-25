import pytest

from conftest import assert_prediction


PATH = "/predict/stage2"
KEYS = ("fully_paid_0", "charged_off_or_default_1")
LABELS = {0: "No default risk flag (0)", 1: "Default risk flag (1)"}


@pytest.mark.case("API-006", "Valid Stage 2 prediction and response schema", "Repository sample payload", "200; complete prediction schema")
def test_stage2_valid(api_call, stage2_payload):
    response = api_call("POST", PATH, stage2_payload)
    assert response.status_code == 200
    assert_prediction(response.json(), "stage2", KEYS, "charged_off_or_default_1", 0.30, LABELS)


@pytest.mark.case("API-007", "Stage 2 repeated prediction consistency", "Same payload twice", "200 twice; same prediction and probabilities")
def test_stage2_repeat(api_call, stage2_payload):
    first = api_call("POST", PATH, stage2_payload)
    second = api_call("POST", PATH, stage2_payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


@pytest.mark.case("API-008", "Stage 2 nullable fields", "Nullable numeric and employment fields null", "200; valid prediction")
def test_stage2_nulls(api_call, stage2_payload):
    stage2_payload.update(revol_util=None, mort_acc=None, total_rev_hi_lim=None, emp_length=None)
    response = api_call("POST", PATH, stage2_payload)
    assert response.status_code == 200
    assert_prediction(response.json(), "stage2", KEYS, "charged_off_or_default_1", 0.30, LABELS)
