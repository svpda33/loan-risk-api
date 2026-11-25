import pytest

from phase2_helpers import case, stage1_prediction, stage2_prediction, validation_error


@pytest.mark.parametrize("stage,field,value", [
    case("API-047", "Stage 1 empty employment length", "employment_length=''", "422; nonblank validation", 1, "employment_length", ""),
    case("API-048", "Stage 1 whitespace employment length", "employment_length='   '", "422; nonblank validation", 1, "employment_length", "   "),
    case("API-049", "Stage 2 empty purpose", "purpose=''", "422; nonblank validation", 2, "purpose", ""),
    case("API-050", "Stage 2 whitespace purpose", "purpose='   '", "422; nonblank validation", 2, "purpose", "   "),
])
def test_nonblank_strings(api_call, stage1_payload, stage2_payload, stage, field, value):
    payload = stage1_payload if stage == 1 else stage2_payload
    payload[field] = value
    validation_error(api_call("POST", f"/predict/stage{stage}", payload), field, "string_too_short")


@pytest.mark.case("API-051", "Invalid state code format", "Stage 1 state='ca'", "422; string pattern mismatch")
def test_state_format(api_call, stage1_payload):
    stage1_payload["state"] = "ca"
    validation_error(api_call("POST", "/predict/stage1", stage1_payload), "state", "string_pattern_mismatch")


@pytest.mark.case("API-052", "Unseen employment category", "Stage 1 employment_length='11 years'", "422; unseen category detail")
def test_unseen_employment_category(api_call, stage1_payload):
    stage1_payload["employment_length"] = "11 years"
    response = api_call("POST", "/predict/stage1", stage1_payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["field"] == "employment_length"
    assert detail["message"] == "Category was not seen during training"
    assert "10+ years" in detail["allowed"]


@pytest.mark.case("API-053", "Alternative valid Stage 1 categories", "employment_length='1 year'; state='NY'", "200; valid prediction")
def test_stage1_alternative_categories(api_call, stage1_payload):
    stage1_payload.update(employment_length="1 year", state="NY")
    stage1_prediction(api_call("POST", "/predict/stage1", stage1_payload))


@pytest.mark.case("API-054", "Alternative valid Stage 2 categories", "RENT; credit_card; A1; Not Verified", "200; valid prediction")
def test_stage2_alternative_categories(api_call, stage2_payload):
    stage2_payload.update(home_ownership="RENT", purpose="credit_card", sub_grade="A1", verification_status="Not Verified")
    stage2_prediction(api_call("POST", "/predict/stage2", stage2_payload))


@pytest.mark.case("API-055", "Second valid loan term", "term='60 months'", "200; valid prediction")
def test_sixty_month_term(api_call, stage2_payload):
    stage2_payload["term"] = "60 months"
    stage2_prediction(api_call("POST", "/predict/stage2", stage2_payload))
