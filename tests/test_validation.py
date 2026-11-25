import pytest


def assert_422(response, field, custom=False):
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body, dict) and "detail" in body
    detail = body["detail"]
    if custom:
        assert isinstance(detail, dict)
        assert detail["field"] == field
        assert detail["message"] == "Category was not seen during training"
        assert isinstance(detail["allowed"], list) and detail["allowed"]
    else:
        assert isinstance(detail, list) and detail
        assert any(field in str(error.get("loc", ())) for error in detail)


@pytest.mark.case("API-009", "Stage 1 unseen category", "state=ZZ (valid code format, unseen category)", "422; category error for state")
def test_stage1_invalid_category(api_call, stage1_payload):
    stage1_payload["state"] = "ZZ"
    assert_422(api_call("POST", "/predict/stage1", stage1_payload), "state", custom=True)


@pytest.mark.case("API-010", "Stage 2 unseen category", "purpose=unknown_purpose", "422; category error for purpose")
def test_stage2_invalid_category(api_call, stage2_payload):
    stage2_payload["purpose"] = "unknown_purpose"
    assert_422(api_call("POST", "/predict/stage2", stage2_payload), "purpose", custom=True)


@pytest.mark.case("API-011", "Stage 1 missing required field", "Omit loan_amount", "422; missing field detail")
def test_stage1_missing(api_call, stage1_payload):
    del stage1_payload["loan_amount"]
    assert_422(api_call("POST", "/predict/stage1", stage1_payload), "loan_amount")


@pytest.mark.case("API-012", "Stage 2 missing required field", "Omit loan_amnt", "422; missing field detail")
def test_stage2_missing(api_call, stage2_payload):
    del stage2_payload["loan_amnt"]
    assert_422(api_call("POST", "/predict/stage2", stage2_payload), "loan_amnt")


@pytest.mark.case("API-013", "Stage 1 incorrect data type", "loan_amount as string", "422; type detail")
def test_stage1_wrong_type(api_call, stage1_payload):
    stage1_payload["loan_amount"] = "15000"
    assert_422(api_call("POST", "/predict/stage1", stage1_payload), "loan_amount")


@pytest.mark.case("API-014", "Stage 2 incorrect data type", "delinq_2yrs as string", "422; type detail")
def test_stage2_wrong_type(api_call, stage2_payload):
    stage2_payload["delinq_2yrs"] = "0"
    assert_422(api_call("POST", "/predict/stage2", stage2_payload), "delinq_2yrs")


@pytest.mark.case("API-015", "Stage 1 empty JSON", "POST {}", "422; required fields reported")
def test_stage1_empty(api_call):
    response = api_call("POST", "/predict/stage1", {})
    assert_422(response, "loan_amount")


@pytest.mark.case("API-016", "Stage 2 empty JSON", "POST {}", "422; required fields reported")
def test_stage2_empty(api_call):
    response = api_call("POST", "/predict/stage2", {})
    assert_422(response, "loan_amnt")


@pytest.mark.case("API-017", "Stage 1 nonpositive amount", "loan_amount=0; Field(gt=0)", "422; numeric bound detail")
def test_stage1_amount_bound(api_call, stage1_payload):
    stage1_payload["loan_amount"] = 0
    assert_422(api_call("POST", "/predict/stage1", stage1_payload), "loan_amount")


@pytest.mark.case("API-018", "Stage 1 negative DTI", "dti_percent=-1; Field(ge=0)", "422; numeric bound detail")
def test_stage1_dti_bound(api_call, stage1_payload):
    stage1_payload["dti_percent"] = -1
    assert_422(api_call("POST", "/predict/stage1", stage1_payload), "dti_percent")


@pytest.mark.case("API-019", "Stage 2 FICO below minimum", "fico_range_low=299; Field(ge=300)", "422; numeric bound detail")
def test_stage2_fico_bound(api_call, stage2_payload):
    stage2_payload["fico_range_low"] = 299
    assert_422(api_call("POST", "/predict/stage2", stage2_payload), "fico_range_low")


@pytest.mark.case("API-020", "Stage 2 invalid term", "term=48 months; Literal permits 36 or 60", "422; literal detail")
def test_stage2_term(api_call, stage2_payload):
    stage2_payload["term"] = "48 months"
    assert_422(api_call("POST", "/predict/stage2", stage2_payload), "term")


@pytest.mark.case("API-021", "Stage 1 extra field", "Unexpected field added", "422; extra field forbidden")
def test_stage1_extra(api_call, stage1_payload):
    stage1_payload["unexpected"] = 1
    assert_422(api_call("POST", "/predict/stage1", stage1_payload), "unexpected")


@pytest.mark.case("API-022", "Stage 2 extra field", "Unexpected field added", "422; extra field forbidden")
def test_stage2_extra(api_call, stage2_payload):
    stage2_payload["unexpected"] = 1
    assert_422(api_call("POST", "/predict/stage2", stage2_payload), "unexpected")
