import json

import pytest

from phase2_helpers import case, validation_error


@pytest.mark.parametrize("stage", [
    case("API-056", "Stage 1 malformed JSON", "Truncated JSON request body", "422; JSON decode validation error", 1),
    case("API-057", "Stage 2 malformed JSON", "Truncated JSON request body", "422; JSON decode validation error", 2),
])
def test_malformed_json(api_call, stage):
    response = api_call("POST", f"/predict/stage{stage}", raw_body='{"broken":', content_type="application/json")
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), list)
    assert any(error.get("type") == "json_invalid" for error in body["detail"])


@pytest.mark.parametrize("stage", [
    case("API-058", "Stage 1 incorrect Content-Type", "Valid JSON bytes as text/plain", "422; body rejected", 1),
    case("API-059", "Stage 2 incorrect Content-Type", "Valid JSON bytes as text/plain", "422; body rejected", 2),
])
def test_incorrect_content_type(api_call, stage1_payload, stage2_payload, stage):
    payload = stage1_payload if stage == 1 else stage2_payload
    response = api_call("POST", f"/predict/stage{stage}", raw_body=json.dumps(payload), content_type="text/plain")
    assert response.status_code == 422
    assert isinstance(response.json().get("detail"), list)


@pytest.mark.parametrize("stage,field", [
    case("API-060", "Stage 1 missing nullable key", "Omit dti_percent entirely", "422; required field missing", 1, "dti_percent"),
    case("API-061", "Stage 2 missing nullable key", "Omit revol_util entirely", "422; required field missing", 2, "revol_util"),
])
def test_nullable_key_still_required(api_call, stage1_payload, stage2_payload, stage, field):
    payload = stage1_payload if stage == 1 else stage2_payload
    del payload[field]
    validation_error(api_call("POST", f"/predict/stage{stage}", payload), field, "missing")
