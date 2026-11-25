"""Assertion and case helpers for additional live API checks."""

import pytest

from conftest import assert_prediction


def case(case_id, description, scenario, expected, *values):
    return pytest.param(*values, id=case_id, marks=pytest.mark.case(case_id, description, scenario, expected))


def validation_error(response, field, error_type=None):
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body, dict) and isinstance(body.get("detail"), list)
    errors = [e for e in body["detail"] if e.get("loc") == ["body", field]]
    assert errors, body
    if error_type is not None:
        assert any(e.get("type") == error_type for e in errors), errors


def stage1_prediction(response):
    assert response.status_code == 200
    assert_prediction(
        response.json(), "stage1", ("rejected_0", "accepted_1"), "accepted_1", 0.80,
        {0: "Rejected proxy (0)", 1: "Accepted proxy (1)"},
    )


def stage2_prediction(response):
    assert response.status_code == 200
    assert_prediction(
        response.json(), "stage2", ("fully_paid_0", "charged_off_or_default_1"),
        "charged_off_or_default_1", 0.30,
        {0: "No default risk flag (0)", 1: "Default risk flag (1)"},
    )
