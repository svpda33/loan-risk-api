import json
import math

import pytest

from phase2_helpers import case, validation_error


@pytest.mark.parametrize("stage,field,value", [
    case("API-036", "Stage 1 NaN", "loan_amount=NaN token", "422; non-finite rejected", 1, "loan_amount", math.nan),
    case("API-037", "Stage 1 positive infinity", "loan_amount=Infinity token", "422; non-finite rejected", 1, "loan_amount", math.inf),
    case("API-038", "Stage 1 negative infinity", "loan_amount=-Infinity token", "422; non-finite rejected", 1, "loan_amount", -math.inf),
    case("API-039", "Stage 2 NaN", "annual_inc=NaN token", "422; non-finite rejected", 2, "annual_inc", math.nan),
    case("API-040", "Stage 2 positive infinity", "annual_inc=Infinity token", "422; non-finite rejected", 2, "annual_inc", math.inf),
    case("API-041", "Stage 2 negative infinity", "annual_inc=-Infinity token", "422; non-finite rejected", 2, "annual_inc", -math.inf),
])
def test_nonfinite_numbers(api_call, stage1_payload, stage2_payload, stage, field, value):
    payload = stage1_payload if stage == 1 else stage2_payload
    payload[field] = value
    # Python's encoder can emit these tokens; Pydantic's allow_inf_nan=False must reject them.
    raw = json.dumps(payload, allow_nan=True)
    validation_error(api_call("POST", f"/predict/stage{stage}", raw_body=raw, content_type="application/json"), field)


@pytest.mark.parametrize("stage,field,error_type", [
    case("API-042", "Stage 1 negative amount", "loan_amount=-1", "422; greater_than", 1, "loan_amount", "greater_than"),
    case("API-043", "Stage 2 negative DTI", "dti=-1", "422; greater_than_equal", 2, "dti", "greater_than_equal"),
    case("API-044", "Stage 2 negative delinquency count", "delinq_2yrs=-1", "422; greater_than_equal", 2, "delinq_2yrs", "greater_than_equal"),
])
def test_negative_numeric_bounds(api_call, stage1_payload, stage2_payload, stage, field, error_type):
    payload = stage1_payload if stage == 1 else stage2_payload
    payload[field] = -1
    validation_error(api_call("POST", f"/predict/stage{stage}", payload), field, error_type)


@pytest.mark.parametrize("field,value", [
    case("API-045", "Boolean rejected as integer", "delinq_2yrs=true", "422; strict integer type", "delinq_2yrs", True),
    case("API-046", "Decimal rejected as integer", "delinq_2yrs=1.5", "422; strict integer type", "delinq_2yrs", 1.5),
])
def test_strict_integer_types(api_call, stage2_payload, field, value):
    stage2_payload[field] = value
    validation_error(api_call("POST", "/predict/stage2", stage2_payload), field)
