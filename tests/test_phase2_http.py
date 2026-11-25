import pytest

from phase2_helpers import case, stage1_prediction, stage2_prediction


@pytest.mark.parametrize("method,path", [
    case("API-068", "POST unsupported on health", "POST /health", "405; method not allowed", "POST", "/health"),
    case("API-069", "GET unsupported on Stage 1", "GET /predict/stage1", "405; method not allowed", "GET", "/predict/stage1"),
    case("API-070", "GET unsupported on Stage 2", "GET /predict/stage2", "405; method not allowed", "GET", "/predict/stage2"),
])
def test_unsupported_methods(api_call, method, path):
    response = api_call(method, path)
    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}
    assert "application/json" in response.headers["Content-Type"].lower()


@pytest.mark.parametrize("stage,valid", [
    case("API-071", "Stage 1 successful response media type", "Valid Stage 1 JSON", "200; application/json", 1, True),
    case("API-072", "Stage 2 successful response media type", "Valid Stage 2 JSON", "200; application/json", 2, True),
    case("API-073", "Stage 1 validation response media type", "Stage 1 loan_amount='bad'", "422; application/json", 1, False),
    case("API-074", "Stage 2 validation response media type", "Stage 2 annual_inc='bad'", "422; application/json", 2, False),
])
def test_json_content_types(api_call, stage1_payload, stage2_payload, stage, valid):
    payload = stage1_payload if stage == 1 else stage2_payload
    if not valid:
        payload["loan_amount" if stage == 1 else "annual_inc"] = "bad"
    response = api_call("POST", f"/predict/stage{stage}", payload)
    assert "application/json" in response.headers["Content-Type"].lower()
    if valid:
        (stage1_prediction if stage == 1 else stage2_prediction)(response)
    else:
        assert response.status_code == 422
        assert isinstance(response.json().get("detail"), list)


@pytest.mark.parametrize("stage", [
    case("API-075", "Stage 1 error disclosure check", "Invalid state category ZZ", "422; no trace or model path", 1),
    case("API-076", "Stage 2 error disclosure check", "Invalid purpose category", "422; no trace or model path", 2),
])
def test_error_no_internal_diagnostics(api_call, stage1_payload, stage2_payload, stage):
    payload = stage1_payload if stage == 1 else stage2_payload
    payload["state" if stage == 1 else "purpose"] = "ZZ" if stage == 1 else "unknown_purpose"
    response = api_call("POST", f"/predict/stage{stage}", payload)
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body.get("detail"), dict)
    lowered = response.text.lower()
    assert all(token not in lowered for token in ("traceback", "stage1_pipeline.joblib", "stage2_full_eligible_lightgbm_pipeline.joblib", "c:\\users\\", "/workspace/"))
