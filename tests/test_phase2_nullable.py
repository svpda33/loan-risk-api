import pytest

from phase2_helpers import case, stage1_prediction, stage2_prediction


@pytest.mark.parametrize("field", [
    case("API-062", "Individual Stage 1 nullable DTI", "dti_percent=null only", "200; valid prediction", "dti_percent"),
    case("API-063", "Individual Stage 1 nullable employment", "employment_length=null only", "200; valid prediction", "employment_length"),
])
def test_stage1_individual_nulls(api_call, stage1_payload, field):
    stage1_payload[field] = None
    stage1_prediction(api_call("POST", "/predict/stage1", stage1_payload))


@pytest.mark.parametrize("field", [
    case("API-064", "Individual Stage 2 nullable utilization", "revol_util=null only", "200; valid prediction", "revol_util"),
    case("API-065", "Individual Stage 2 nullable mortgage count", "mort_acc=null only", "200; valid prediction", "mort_acc"),
    case("API-066", "Individual Stage 2 nullable revolving limit", "total_rev_hi_lim=null only", "200; valid prediction", "total_rev_hi_lim"),
    case("API-067", "Individual Stage 2 nullable employment", "emp_length=null only", "200; valid prediction", "emp_length"),
])
def test_stage2_individual_nulls(api_call, stage2_payload, field):
    stage2_payload[field] = None
    stage2_prediction(api_call("POST", "/predict/stage2", stage2_payload))
