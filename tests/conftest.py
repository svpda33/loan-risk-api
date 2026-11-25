"""Live HTTP fixtures and an execution report for the deployed API."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import pytest
import requests


BASE_URL = os.environ.get("LOAN_API_BASE_URL", "https://loan-risk-api-g7ij.onrender.com").rstrip("/")
REPORT_PATH = Path(os.environ.get(
    "LOAN_API_REPORT_PATH",
    str(Path(__file__).resolve().parents[1] / "reports" / "api_testing_report.md"),
)).resolve()
BASELINE_XML = Path(__file__).resolve().parents[1] / "reports" / "api_testing_baseline_22.xml"
CASES = {}
RESULTS = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pytest_configure(config):
    config.addinivalue_line("markers", "case(case_id, description, scenario, expected): live API test metadata")


def pytest_collection_modifyitems(items):
    for item in items:
        marker = item.get_closest_marker("case")
        if marker:
            CASES[item.nodeid] = marker.args
            item.user_properties.append(("case_id", marker.args[0]))


def pytest_runtest_logreport(report):
    if report.when not in ("call", "setup"):
        return
    if report.when == "setup" and report.passed:
        return
    entry = RESULTS.setdefault(report.nodeid, {})
    entry["outcome"] = report.outcome.upper()
    entry["timestamp"] = utc_now()
    if report.failed:
        entry["failure"] = str(report.longrepr).splitlines()[-1][:300]
    for key, value in report.user_properties:
        if key in ("response_time", "http_status", "attempts", "attempt_log", "actual") and key in entry:
            entry[key] = f"{entry[key]}, {value}"
        else:
            entry[key] = value


def pytest_sessionfinish(session, exitstatus):
    REPORT_PATH.parent.mkdir(exist_ok=True)
    counts = {"PASSED": 0, "FAILED": 0, "SKIPPED": 0, "NOT EXECUTED": 0}
    phase_counts = {"Baseline IDs 001–022": counts.copy(), "Phase 2 IDs 023+": counts.copy()}
    baseline = ElementTree.parse(BASELINE_XML).getroot().find("testsuite") if BASELINE_XML.exists() else None
    lines = [
        "# Loan Risk API testing report", "",
        "## Project overview", "",
        "The FastAPI research prototype exposes a Stage 1 historical acceptance proxy and a Stage 2 funded-loan repayment-risk model. Its predictions are decision support, not automatic lending decisions.", "",
        "## Testing environment", "",
        f"- Target: `{BASE_URL}`",
        f"- Generated (UTC): {utc_now()}",
        f"- Runner: Python {sys.version.split()[0]}, pytest {pytest.__version__}, requests {requests.__version__}.",
        "- Transport: real HTTPS requests; no mocks. Response time is wall time for each call, including retries and wait time when applicable.",
        "", "## Initial 22-test baseline", "",
        (f"The preserved historical JUnit run recorded {baseline.get('tests')} tests, {int(baseline.get('tests')) - int(baseline.get('failures')) - int(baseline.get('errors')) - int(baseline.get('skipped'))} passed, {baseline.get('failures')} failed, and {baseline.get('skipped')} skipped at {baseline.get('timestamp')}. See [baseline Markdown](api_testing_baseline_22.md) and [baseline JUnit](api_testing_baseline_22.xml)." if baseline is not None else "Historical baseline file was unavailable; no historical results are inferred."),
        "", "## Complete current execution", "",
    ]
    header = "| Test case ID | Test description | Input scenario | Expected result | Actual result | HTTP status | Result | Execution timestamp (UTC) | Response time (s) | Attempts | Attempt log |"
    divider = "|---|---|---|---|---|---|---|---|---|---|---|"
    last_phase = None
    for nodeid, (case_id, description, scenario, expected) in sorted(
        CASES.items(), key=lambda pair: int(pair[1][0].split("-")[1])
    ):
        result = RESULTS.get(nodeid, {})
        outcome = result.get("outcome", "NOT EXECUTED")
        counts[outcome] += 1
        phase = "Baseline IDs 001–022" if int(case_id.split("-")[1]) <= 22 else "Phase 2 IDs 023+"
        phase_counts[phase][outcome] += 1
        if phase != last_phase:
            lines += [f"### {phase}", "", header, divider]
            last_phase = phase
        actual = result.get("actual", result.get("failure", "No request completed"))
        if outcome == "FAILED" and result.get("failure"):
            actual += "; " + result["failure"]
        fields = (
            case_id, description, scenario, expected, actual,
            result.get("http_status", "—"), outcome,
            result.get("timestamp", "—"), result.get("response_time", "—"),
            result.get("attempts", "—"), result.get("attempt_log", "—"),
        )
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in fields) + " |")
    nonfinite_500 = [
        case[0] for nodeid, case in CASES.items()
        if case[0] in {f"API-{number:03d}" for number in range(36, 42)}
        and RESULTS.get(nodeid, {}).get("http_status") == 500
    ]
    lines += [
        "", "## Overall test summary", "",
        f"- Original cases in current run: {sum(phase_counts['Baseline IDs 001–022'].values())}; passed {phase_counts['Baseline IDs 001–022']['PASSED']}, failed {phase_counts['Baseline IDs 001–022']['FAILED']}, skipped {phase_counts['Baseline IDs 001–022']['SKIPPED']}.",
        f"- Phase 2 cases in current run: {sum(phase_counts['Phase 2 IDs 023+'].values())}; passed {phase_counts['Phase 2 IDs 023+']['PASSED']}, failed {phase_counts['Phase 2 IDs 023+']['FAILED']}, skipped {phase_counts['Phase 2 IDs 023+']['SKIPPED']}.",
        f"- Total collected: {len(CASES)}; passed {counts['PASSED']}, failed {counts['FAILED']}, skipped {counts['SKIPPED']}, not yet executed {counts['NOT EXECUTED']}.",
        "", "## Discovered defects and triage", "",
        ("**D-001 — Non-finite numeric tokens:** " + ", ".join(sorted(nonfinite_500)) + " returned HTTP 500 with `Internal Server Error` instead of a client validation response. The tested `NaN` and infinity tokens are nonstandard JSON; the response indicates an unhandled input path. The server-side root cause has not been confirmed." if nonfinite_500 else "No HTTP 500 result was observed for the non-finite numeric cases in this run."),
        ("No failed assertions in this run; this does not prove absence of defects." if counts["FAILED"] == 0 else "Any failures beyond D-001 require separate triage before being classified as application defects."),
        "Skipped cases indicate unavailable infrastructure or transport after bounded retries; they are not application passes.",
        "", "## Limitations", "",
        f"This suite uses small synthetic payloads and a single sequential run against {BASE_URL}. It does not establish sustained availability, security completeness, model quality, fairness, or performance service levels. Cold-start recovery is implemented but only verified if retries occur during this run.",
        "", "## Conclusion", "",
        ("All collected cases executed and passed in this run." if counts["PASSED"] == len(CASES) and CASES else "Complete verification was not achieved; consult the case statuses above."),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pytest_unconfigure(config):
    """Keep local machine names out of the generated JUnit artifact."""
    configured_path = getattr(config.option, "xmlpath", None)
    if not configured_path:
        return
    junit_path = Path(configured_path)
    if not junit_path.exists():
        return
    tree = ElementTree.parse(junit_path)
    for suite in tree.getroot().iter("testsuite"):
        suite.attrib.pop("hostname", None)
    tree.write(junit_path, encoding="utf-8", xml_declaration=True)


@pytest.fixture(scope="session")
def http_session():
    with requests.Session() as session:
        session.headers.update({"Accept": "application/json"})
        yield session


@pytest.fixture
def api_call(http_session, request):
    def call(method, path, payload=None, *, raw_body=None, content_type=None):
        assert not (payload is not None and raw_body is not None)
        started = time.perf_counter()
        response = None
        last_error = None
        attempt_log = []
        for attempt in range(3):
            attempt_started = time.perf_counter()
            try:
                kwargs = {"timeout": (10, 90)}
                if payload is not None:
                    kwargs["json"] = payload
                if raw_body is not None:
                    kwargs["data"] = raw_body
                if content_type is not None:
                    kwargs["headers"] = {"Content-Type": content_type}
                response = http_session.request(method, BASE_URL + path, **kwargs)
                attempt_log.append(f"{attempt + 1}:HTTP {response.status_code} ({time.perf_counter() - attempt_started:.3f}s)")
                if response.status_code not in (502, 503, 504):
                    break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                response = None
                attempt_log.append(f"{attempt + 1}:{type(exc).__name__} ({time.perf_counter() - attempt_started:.3f}s)")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
        seconds = round(time.perf_counter() - started, 3)
        request.node.user_properties.append(("response_time", seconds))
        request.node.user_properties.append(("attempts", len(attempt_log)))
        request.node.user_properties.append(("attempt_log", "; ".join(attempt_log)))
        if response is None:
            reason = f"Transport unavailable after {len(attempt_log)} attempts: {type(last_error).__name__}"
            request.node.user_properties.append(("actual", reason))
            pytest.skip(reason)
        request.node.user_properties.append(("http_status", response.status_code))
        if response.status_code in (502, 503, 504):
            try:
                application_not_ready = response.status_code == 503 and response.json() == {"detail": "Models are not ready"}
            except ValueError:
                application_not_ready = False
            if not application_not_ready:
                reason = f"Infrastructure HTTP {response.status_code} after {len(attempt_log)} attempts"
                request.node.user_properties.append(("actual", reason))
                pytest.skip(reason)
        try:
            body = response.json()
            if isinstance(body, dict) and "probabilities" in body:
                observed = {key: body.get(key) for key in ("stage", "predicted_class", "predicted_label", "probabilities", "threshold")}
            elif isinstance(body, dict) and isinstance(body.get("detail"), list):
                observed = {"validation_errors": [
                    {"field": error.get("loc", [None])[-1], "type": error.get("type")}
                    for error in body["detail"]
                ]}
            elif isinstance(body, dict) and isinstance(body.get("detail"), dict):
                observed = {"detail": {key: body["detail"].get(key) for key in ("field", "message")}}
            else:
                observed = body
            summary = json.dumps(observed, ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            summary = response.text[:280]
        request.node.user_properties.append(("actual", summary))
        return response
    return call


@pytest.fixture
def stage1_payload():
    return {"loan_amount": 15000, "dti_percent": 18, "employment_length": "10+ years", "state": "CA"}


@pytest.fixture
def stage2_payload():
    return {
        "loan_amnt": 15000, "int_rate": 13.5, "annual_inc": 72000, "dti": 18,
        "delinq_2yrs": 0, "fico_range_low": 690, "inq_last_6mths": 1,
        "open_acc": 12, "pub_rec": 0, "revol_bal": 8500, "revol_util": 42,
        "total_acc": 25, "installment": 510, "credit_age_years": 12,
        "mort_acc": 1, "total_rev_hi_lim": 20000, "term": "36 months",
        "emp_length": "10+ years", "home_ownership": "MORTGAGE",
        "verification_status": "Verified", "purpose": "debt_consolidation",
        "addr_state": "CA", "application_type": "Individual", "sub_grade": "C2",
    }


def assert_prediction(body, stage, keys, positive_key, threshold, labels):
    import math
    assert isinstance(body, dict)
    assert set(body) == {"stage", "task", "predicted_class", "predicted_label", "probabilities", "threshold", "note"}
    assert body["stage"] == stage
    assert isinstance(body["task"], str) and body["task"]
    assert type(body["predicted_class"]) is int and body["predicted_class"] in (0, 1)
    assert isinstance(body["predicted_label"], str)
    assert isinstance(body["note"], str) and body["note"]
    assert type(body["threshold"]) in (int, float) and body["threshold"] == pytest.approx(threshold)
    probabilities = body["probabilities"]
    assert isinstance(probabilities, dict) and set(probabilities) == set(keys)
    assert all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1, abs=1e-9)
    expected_class = int(probabilities[positive_key] >= body["threshold"])
    assert body["predicted_class"] == expected_class
    assert body["predicted_label"] == labels[expected_class]
