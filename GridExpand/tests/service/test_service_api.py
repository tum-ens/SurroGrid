"""HTTP API of the service without a database: guards, plugin files, scenarios, validation."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_index(client):
    assert client.get("/api/health").json() == {"ok": True, "service": "gridexpand", "api": 1}
    assert client.get("/").json()["plugin_manifest"] == "ui/manifest.json"


def test_host_allowlist(settings, fake_solvers):
    from gridexpand.service.app import create_app

    with TestClient(create_app(settings)) as c:
        assert c.get("/api/health", headers={"Host": "127.0.0.1:18766"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "localhost:18766"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "127.0.0.1:9999"}).status_code == 421
        assert c.get("/api/health", headers={"Host": "evil.example:18766"}).status_code == 421


def test_state_changing_calls_need_the_ui_header(settings, fake_solvers):
    from gridexpand.service.app import create_app

    with TestClient(create_app(settings)) as c:
        response = c.post("/api/jobs/pipeline", json={})
        assert response.status_code == 403 and "X-GridExpand-UI" in response.json()["detail"]
        assert c.post("/api/jobs/nope/cancel").status_code == 403


def test_cors_only_when_configured(settings, fake_solvers):
    from dataclasses import replace

    from gridexpand.service.app import create_app

    preflight = {"Origin": "http://127.0.0.1:18765", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "x-gridexpand-ui,content-type"}
    with TestClient(create_app(settings)) as c:
        assert "access-control-allow-origin" not in c.options("/api/jobs/pipeline", headers=preflight).headers
    dev = replace(settings, cors_origins=("http://127.0.0.1:18765",))
    with TestClient(create_app(dev)) as c:
        response = c.options("/api/jobs/pipeline", headers=preflight)
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:18765"
        assert c.get("/ui/plugin.js", headers={"Origin": "http://127.0.0.1:18765"}).headers[
            "access-control-allow-origin"] == "http://127.0.0.1:18765"


def test_plugin_manifest_and_modules(client):
    manifest = client.get("/ui/manifest.json").json()
    assert manifest["schema"] == 1 and manifest["name"] == "gridexpand"
    assert manifest["entry"] == "ui/plugin.js" and manifest["api"] == "api/"
    assert manifest["csrf_header"] == "X-GridExpand-UI"
    for path in ("/ui/plugin.js", "/ui/lib.js", "/ui/maplayer.js", "/ui/panels/runs.js", "/ui/panels/results.js",
                 "/ui/panels/scenarios.js"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-cache"
        assert "javascript" in response.headers["content-type"]
    assert "export function register(host)" in client.get("/ui/plugin.js").text


def test_scenarios(client):
    items = {item["name"]: item for item in client.get("/api/scenarios").json()}
    assert items["schweinfurt_2045.yaml"]["valid"] and items["schweinfurt_2045.yaml"]["id"] == "schweinfurt_2045"
    assert items["schweinfurt_2045.yaml"]["heat_source"] == "teaser"
    assert items["schweinfurt_2045.yaml"]["adoption"]["heat"]["building_share"] == 0.75
    assert items["schweinfurt_2045.yaml"]["scenario_key"].startswith("scenario_schweinfurt_2045_")
    assert items["broken.yaml"]["valid"] is False and items["broken.yaml"]["error"]
    assert "milestone_year" in client.get("/api/scenarios/schweinfurt_2045.yaml").text
    assert client.get("/api/scenarios/other.yaml").status_code == 404
    assert client.get("/api/scenarios/..%2F..%2F.env").status_code == 404


def test_pipeline_validation(client):
    base = {"plz": 85653, "pylovo_version_id": "1", "scenario": "schweinfurt_2045.yaml"}
    assert client.post("/api/jobs/pipeline", json=base | {"model_cases": ["post-inflex-heuristic"]}).status_code == 422
    assert client.post("/api/jobs/pipeline", json=base | {"timeframe_mode": "one_day"}).status_code == 422
    assert client.post("/api/jobs/pipeline", json=base | {"unknown": 1}).status_code == 422
    assert client.post("/api/jobs/pipeline", json=base | {"scenario": "nope.yaml"}).status_code == 404
    assert client.post("/api/jobs/pipeline", json=base | {"scenario": "broken.yaml"}).status_code == 400
    response = client.post("/api/jobs/pipeline", json=base | {"model_cases": ["pre", "post-hems-optimized"]})
    assert response.status_code == 409 and "solver" in response.json()["detail"]  # fake: Gurobi not usable


def test_status_without_database(client):
    status = client.get("/api/status").json()
    assert status["database"]["connected"] is False and status["database"]["error"]
    assert status["database"]["port"] == "9"  # the unreachable test database
    assert status["solvers"]["post_cases_supported"] is False
    assert {asset["name"] for asset in status["data_assets"]} >= {"elec_lps.h5", "mobility_demand_pool.csv"}
    assert status["jobs"] == {"running": 0, "queued": 0}
    assert client.get("/api/results/analyses").status_code == 503


def test_unknown_job(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/api/jobs").json() == []


def test_root_path_with_and_without_the_prefix(settings, fake_solvers):
    """Behind a proxy that strips /gridexpand, and directly with or without the prefix."""
    from dataclasses import replace

    from gridexpand.service.app import create_app

    with TestClient(create_app(replace(settings, root_path="/gridexpand"))) as c:
        for prefix in ("", "/gridexpand"):
            assert c.get(f"{prefix}/api/health").status_code == 200
            assert c.get(f"{prefix}/api/health").headers["cache-control"] == "no-store"
            assert c.get(f"{prefix}/ui/manifest.json").json()["entry"] == "ui/plugin.js"
            assert "export function register" in c.get(f"{prefix}/ui/plugin.js").text
            assert c.get(f"{prefix}/ui/panels/runs.js").status_code == 200
            assert c.post(f"{prefix}/api/jobs/pipeline", json={}).status_code == 403  # no CSRF bypass
        assert "/gridexpand/openapi.json" in c.get("/docs").text
        assert c.get("/openapi.json").json()["servers"] == [{"url": "/gridexpand"}]
