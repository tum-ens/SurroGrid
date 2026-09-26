"""HTTP API without a database: guards, health and contract, scenarios, validation."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_index(client):
    health = client.get("/api/health").json()
    assert {k: health[k] for k in ("ok", "service", "api")} == {"ok": True, "service": "gridexpand-api", "api": 1}
    assert health["version"] and "revision" in health
    assert client.get("/").json()["openapi"] == "openapi.json"


def test_host_allowlist(settings, fake_solvers):
    from gridexpand.api.app import create_app

    with TestClient(create_app(settings)) as c:
        assert c.get("/api/health", headers={"Host": "127.0.0.1:18766"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "localhost:18766"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "127.0.0.1:9999"}).status_code == 421
        assert c.get("/api/health", headers={"Host": "evil.example:18766"}).status_code == 421


def test_state_changing_calls_need_the_ui_header(settings, fake_solvers):
    from gridexpand.api.app import create_app

    with TestClient(create_app(settings)) as c:
        response = c.post("/api/jobs/pipeline", json={})
        assert response.status_code == 403 and "X-GridExpand-UI" in response.json()["detail"]
        assert c.post("/api/jobs/nope/cancel").status_code == 403


def test_cors_only_when_configured(settings, fake_solvers):
    from dataclasses import replace

    from gridexpand.api.app import create_app

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


def test_no_ui_is_served(client):
    """The UI lives in GridPlanner; the API serves no browser files."""
    for path in ("/ui/manifest.json", "/ui/plugin.js", "/static/index.html"):
        assert client.get(path).status_code == 404, path


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


def test_status_without_database_configuration(client, monkeypatch, tmp_path):
    from gridexpand.db import engine as db_engine
    from gridexpand.api import db as service_db

    monkeypatch.setattr(db_engine, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(db_engine, "_env_loaded", True)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.setattr(service_db, "_engine", None)
    database = client.get("/api/status").json()["database"]
    assert database["connected"] is False and "DB_NAME" in database["error"]
    assert database["database"] is None and database["host"] is None
    response = client.get("/api/results/analyses")
    assert response.status_code == 503 and "No database configured" in response.json()["detail"]


def test_unknown_job(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/api/jobs").json() == []


def test_root_path_with_and_without_the_prefix(settings, fake_solvers):
    """Behind a proxy that strips /gridexpand, and directly with or without the prefix."""
    from dataclasses import replace

    from gridexpand.api.app import create_app

    with TestClient(create_app(replace(settings, root_path="/gridexpand"))) as c:
        for prefix in ("", "/gridexpand"):
            assert c.get(f"{prefix}/api/health").status_code == 200
            assert c.get(f"{prefix}/api/health").headers["cache-control"] == "no-store"
            assert c.post(f"{prefix}/api/jobs/pipeline", json={}).status_code == 403  # no CSRF bypass
        assert "/gridexpand/openapi.json" in c.get("/docs").text
        assert c.get("/openapi.json").json()["servers"] == [{"url": "/gridexpand"}]


def test_terminal_run_for_one_grid(client, monkeypatch):
    from gridexpand.api import queries

    grid = {"grid_result_id": 7, "plz": 85653, "kcid": 1, "bcid": 3, "candidate_index": 2, "n_buildings": 3, "results": []}
    monkeypatch.setattr(queries, "ags_for_plz", lambda plz: [9184137])
    monkeypatch.setattr(queries, "grid_candidates", lambda ags, version, min_buildings, plz=None: [grid])
    body = {"plz": 85653, "pylovo_version_id": "1", "scenario": "schweinfurt_2045.yaml", "model_cases": ["pre"],
            "grid_result_id": 7}
    response = client.post("/api/jobs/terminal", json=body)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["title"].startswith("Grid 1/3 · PLZ 85653") and data["grids"] == 1
    assert "kcid: 1" in data["run_yaml"] and "bcid: 3" in data["run_yaml"] and "min_buildings: 1" in data["run_yaml"]
    assert "scenario: ../scenarios/schweinfurt_2045.yaml" in data["portable_run_yaml"]
    assert data["commands"][0]["command"].startswith("tmux new-session -d -s ui_85653-1-3_schweinfurt_2045_")
    runs = client.get("/api/jobs/terminal").json()
    assert [r["run_id"] for r in runs] == [data["run_id"]] and runs[0]["status"] == "not started"
    assert client.post("/api/jobs/terminal", json=body | {"grid_result_id": 8}).status_code == 400
