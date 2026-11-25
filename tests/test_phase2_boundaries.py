import pytest

from phase2_helpers import case, stage1_prediction, stage2_prediction, validation_error


@pytest.mark.parametrize("fico,valid", [
    case("API-023", "FICO lower valid boundary", "fico_range_low=300", "200; valid prediction", 300, True),
    case("API-024", "FICO upper valid boundary", "fico_range_low=850", "200; valid prediction", 850, True),
    case("API-025", "FICO above upper boundary", "fico_range_low=851", "422; less_than_equal", 851, False),
])
def test_fico_boundaries(api_call, stage2_payload, fico, valid):
    stage2_payload["fico_range_low"] = fico
    response = api_call("POST", "/predict/stage2", stage2_payload)
    if valid:
        stage2_prediction(response)
    else:
        validation_error(response, "fico_range_low", "less_than_equal")


@pytest.mark.parametrize("field", [
    case("API-026", "Zero loan amount", "Stage 2 loan_amnt=0", "422; greater_than", "loan_amnt"),
    case("API-027", "Zero interest rate", "Stage 2 int_rate=0", "422; greater_than", "int_rate"),
    case("API-028", "Zero annual income", "Stage 2 annual_inc=0", "422; greater_than", "annual_inc"),
    case("API-029", "Zero installment", "Stage 2 installment=0", "422; greater_than", "installment"),
])
def test_stage2_positive_fields_reject_zero(api_call, stage2_payload, field):
    stage2_payload[field] = 0
    validation_error(api_call("POST", "/predict/stage2", stage2_payload), field, "greater_than")


@pytest.mark.parametrize("field", [
    case("API-030", "Small positive loan amount", "Stage 2 loan_amnt=0.01", "200; valid prediction", "loan_amnt"),
    case("API-031", "Small positive interest rate", "Stage 2 int_rate=0.01", "200; valid prediction", "int_rate"),
    case("API-032", "Small positive annual income", "Stage 2 annual_inc=0.01", "200; valid prediction", "annual_inc"),
    case("API-033", "Small positive installment", "Stage 2 installment=0.01", "200; valid prediction", "installment"),
])
def test_stage2_positive_fields_accept_small_value(api_call, stage2_payload, field):
    stage2_payload[field] = 0.01
    stage2_prediction(api_call("POST", "/predict/stage2", stage2_payload))


@pytest.mark.case("API-034", "Stage 1 zero DTI", "dti_percent=0; ge=0", "200; valid prediction")
def test_stage1_zero_dti(api_call, stage1_payload):
    stage1_payload["dti_percent"] = 0
    stage1_prediction(api_call("POST", "/predict/stage1", stage1_payload))


@pytest.mark.case("API-035", "Stage 2 zero DTI", "dti=0; ge=0", "200; valid prediction")
def test_stage2_zero_dti(api_call, stage2_payload):
    stage2_payload["dti"] = 0
    stage2_prediction(api_call("POST", "/predict/stage2", stage2_payload))
