import pytest


@pytest.mark.case("API-001", "Root endpoint", "GET /", "200; service name and route list")
def test_root(api_call):
    response = api_call("GET", "/")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert isinstance(body.get("name"), str) and body["name"]
    assert isinstance(body.get("description"), str) and body["description"]
    assert set(("/health", "/predict/stage1", "/predict/stage2", "/docs")) <= set(body["endpoints"])


@pytest.mark.case("API-002", "Health and model loading", "GET /health", "200; both stage models loaded")
def test_health(api_call):
    response = api_call("GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"stage1": "loaded", "stage2": "loaded"}}
