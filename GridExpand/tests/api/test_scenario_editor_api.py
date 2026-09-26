"""HTTP API of the scenario editor: form, preview, save into the user directory, delete, guards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

BASE = "schweinfurt_2045.yaml"
CHANGE = {"asset_sizing.pv.demand_multiplier": 2.5}


def save(client, **body):
    return client.post("/api/scenarios", json={"base": BASE, "file_name": "mine.yaml", "scenario_id": "mine"} | body)


def test_user_directory_is_the_last_scenario_directory(settings, tmp_path, monkeypatch):
    from gridexpand.paths import SCENARIO_CONFIG_DIR
    from gridexpand.api.settings import ENV_USER_SCENARIO_DIR, ServiceSettings

    assert settings.scenario_dirs[-1] == settings.user_scenario_dir == (tmp_path / "user_scenarios").resolve()
    assert settings.shipped_scenario_dirs == (tmp_path / "scenarios",)
    monkeypatch.setenv(ENV_USER_SCENARIO_DIR, str(tmp_path / "gp"))
    from_env = ServiceSettings.from_env()
    assert from_env.user_scenario_dir == (tmp_path / "gp").resolve() and from_env.scenario_dirs[0] == SCENARIO_CONFIG_DIR
    moved = ServiceSettings(scenario_dirs=(tmp_path / "gp", SCENARIO_CONFIG_DIR), user_scenario_dir=tmp_path / "gp")
    assert moved.scenario_dirs == (SCENARIO_CONFIG_DIR, (tmp_path / "gp").resolve())
    with pytest.raises(ValueError, match="must not be the repository"):
        ServiceSettings(user_scenario_dir=SCENARIO_CONFIG_DIR)


def test_serve_option_for_the_user_directory(capsys):
    from pathlib import Path

    from gridexpand.paths import SCENARIO_CONFIG_DIR
    from gridexpand.api.cli import build_parser, main

    assert build_parser().parse_args(["--user-scenario-dir", "gp/scenarios"]).user_scenario_dir == Path("gp/scenarios")
    assert build_parser().parse_args([]).user_scenario_dir is None
    assert main(["--user-scenario-dir", str(SCENARIO_CONFIG_DIR)]) == 2  # refused before anything starts
    assert "must not be the repository" in capsys.readouterr().err


def test_form(client, settings):
    form = client.get(f"/api/scenarios/{BASE}/form").json()
    assert form["name"] == BASE and form["user"] is False and form["writable"] is False
    assert form["id"] == "schweinfurt_2045" and form["scenario_key"].startswith("scenario_schweinfurt_2045_")
    assert form["text"].startswith("# Scientific and methodological assumptions") and len(form["text_sha256"]) == 64
    assert form["proposal"] == {"file_name": "schweinfurt_2045_custom.yaml", "scenario_id": "schweinfurt_2045_custom"}
    assert form["user_dir"] == {"path": str(settings.user_scenario_dir), "exists": False, "writable": True, "reason": None}
    sections = {s["id"]: {f["key"]: f for f in s["fields"]} for s in form["editable_fields"]}
    assert list(sections) == ["electrification", "economics", "pv", "battery", "heat", "mobility", "time_aggregation"]
    pv = sections["pv"]["asset_sizing.pv.demand_multiplier"]
    assert pv["value"] == 2.0 and pv["unit"] == "kWp/(MWh/a)" and pv["help"].startswith("Central compromise")
    assert sections["electrification"]["electrification.heat.building_share"]["type"] == "percent"
    broken = client.get("/api/scenarios/broken.yaml/form").json()
    assert broken["valid"] is False and broken["editable_fields"] == [] and broken["text"]
    assert client.get("/api/scenarios/nope.yaml/form").status_code == 404


def test_preview(client):
    listed = {item["name"]: item for item in client.get("/api/scenarios").json()}[BASE]
    same = client.post("/api/scenarios/preview", json={"base": BASE, "changes": {"asset_sizing.pv.demand_multiplier": 2.0}}).json()
    assert same["ok"] and same["unchanged"] and same["diff"] == "" and same["changes"] == []
    assert same["configuration_hash"] == listed["configuration_hash"] and same["scenario_key"] == listed["scenario_key"]
    changed = client.post("/api/scenarios/preview", json={"base": BASE, "changes": CHANGE, "scenario_id": "mine",
                                                          "file_name": "mine.yaml"}).json()
    assert changed["ok"] and not changed["unchanged"] and changed["id"] == "mine"
    assert changed["scenario_key"].startswith("scenario_mine_") and changed["configuration_hash"] != listed["configuration_hash"]
    assert "-    demand_multiplier: 2.0\n+    demand_multiplier: 2.5\n" in changed["diff"]
    assert changed["target"] == {"file_name": "mine.yaml", "allowed": True, "exists": False, "message": None}
    bad = client.post("/api/scenarios/preview", json={"base": BASE, "changes": {"electrification.heat.building_share": 2}}).json()
    assert not bad["ok"] and bad["issues"][0]["key"] == "electrification.heat.building_share" and bad["issues"][0]["line"]
    assert client.post("/api/scenarios/preview", json={"base": "nope.yaml"}).status_code == 404
    assert client.post("/api/scenarios/preview", json={"base": BASE, "extra": 1}).status_code == 422
    stale = client.post("/api/scenarios/preview", json={"base": BASE, "base_sha256": "0" * 64})
    assert stale.status_code == 409 and "changed on disk" in stale.json()["detail"]


def test_save_writes_a_user_file_and_lists_it(client, settings):
    response = save(client, changes=CHANGE)
    assert response.status_code == 201, response.text
    entry = response.json()
    path = settings.user_scenario_dir / "mine.yaml"
    assert entry["name"] == "mine.yaml" and entry["user"] is True and entry["path"] == str(path)
    assert entry["id"] == "mine" and entry["scenario_key"].startswith("scenario_mine_") and entry["backup"] is None
    text = path.read_text(encoding="utf-8")
    shipped = (settings.shipped_scenario_dirs[0] / BASE).read_text(encoding="utf-8")
    assert text == shipped.replace("id: schweinfurt_2045\n", "id: mine\n").replace(
        "demand_multiplier: 2.0", "demand_multiplier: 2.5")
    items = {item["name"]: item for item in client.get("/api/scenarios").json()}
    assert items["mine.yaml"]["user"] is True and items[BASE]["user"] is False
    assert items["mine.yaml"]["scenario_key"] == entry["scenario_key"]
    form = client.get("/api/scenarios/mine.yaml/form").json()
    assert form["user"] and form["writable"] and form["proposal"] == {"file_name": "mine.yaml", "scenario_id": "mine"}
    # the base file is untouched
    assert (settings.shipped_scenario_dirs[0] / BASE).read_text(encoding="utf-8") == shipped


def test_save_needs_overwrite_for_an_own_file_and_keeps_a_backup(client, settings):
    assert save(client, changes=CHANGE).status_code == 201
    again = save(client, base="mine.yaml", changes={"asset_sizing.pv.demand_multiplier": 3.0})
    assert again.status_code == 409 and "already exists" in again.json()["detail"]["message"]
    replaced = save(client, base="mine.yaml", changes={"asset_sizing.pv.demand_multiplier": 3.0}, overwrite=True)
    assert replaced.status_code == 201
    backup = replaced.json()["backup"]
    assert backup.startswith("mine.yaml.bak-")
    assert "demand_multiplier: 2.5" in (settings.user_scenario_dir / backup).read_text(encoding="utf-8")
    assert "demand_multiplier: 3.0" in (settings.user_scenario_dir / "mine.yaml").read_text(encoding="utf-8")
    assert "mine.yaml.bak" not in "".join(item["name"] for item in client.get("/api/scenarios").json())
    unchanged = save(client, base="mine.yaml", overwrite=True)
    assert unchanged.status_code == 201 and unchanged.json()["backup"] is None


@pytest.mark.parametrize(("file_name", "status"), [
    ("schweinfurt_2045.yaml", 409),  # shipped
    ("../escape.yaml", 400), ("sub/dir.yaml", 400), (".hidden.yaml", 400), ("-x.yaml", 400),
    ("notes.txt", 400), ("ümlaut.yaml", 400), ("x" * 120 + ".yaml", 422),
])
def test_save_refuses_names_outside_the_rules(client, settings, file_name, status):
    response = save(client, changes=CHANGE, file_name=file_name)
    assert response.status_code == status, response.text
    assert not settings.user_scenario_dir.exists() or not any(settings.user_scenario_dir.iterdir())
    assert not (settings.user_scenario_dir.parent / "escape.yaml").exists()


def test_save_refuses_invalid_scenarios_and_ids(client, settings):
    invalid = save(client, changes={"asset_sizing.pv.demand_multiplier": -1})
    assert invalid.status_code == 400
    detail = invalid.json()["detail"]
    assert "must be positive" in detail["message"] and detail["issues"][0]["key"] == "asset_sizing.pv.demand_multiplier"
    assert save(client, changes=CHANGE, scenario_id="bad id").status_code == 400
    assert save(client, text="scenario: [").status_code == 400
    assert not (settings.user_scenario_dir / "mine.yaml").exists()


def test_save_from_edited_text(client, settings):
    text = client.get(f"/api/scenarios/{BASE}").text.replace("commuting_probability: 0.62", "commuting_probability: 0.5")
    response = save(client, text=text, changes={"time_aggregation.enabled": False})
    assert response.status_code == 201, response.text
    saved = (settings.user_scenario_dir / "mine.yaml").read_text(encoding="utf-8")
    assert "commuting_probability: 0.5\n" in saved and "  enabled: false\n" in saved and "id: mine\n" in saved
    assert "# TSAM assumptions are scenario methodology" in saved


def test_files_used_by_an_active_job_are_not_replaced_or_deleted(client, monkeypatch):
    assert save(client, changes=CHANGE).status_code == 201
    job = SimpleNamespace(id="abc", status="running", active=True, params={"scenario": "mine.yaml"})
    monkeypatch.setattr(client.app.state.jobs, "list", lambda: [job])
    busy = save(client, base="mine.yaml", changes={"asset_sizing.pv.demand_multiplier": 3.0}, overwrite=True)
    assert busy.status_code == 409 and "Job abc" in busy.json()["detail"]
    assert client.delete("/api/scenarios/mine.yaml").status_code == 409
    assert save(client, changes=CHANGE, file_name="other.yaml", scenario_id="other").status_code == 201


def test_delete_only_user_files(client, settings):
    assert save(client, changes=CHANGE).status_code == 201
    assert client.delete(f"/api/scenarios/{BASE}").status_code == 403
    assert (settings.shipped_scenario_dirs[0] / BASE).exists()
    deleted = client.delete("/api/scenarios/mine.yaml")
    assert deleted.status_code == 200 and deleted.json()["backup"].startswith("mine.yaml.bak-")
    assert not (settings.user_scenario_dir / "mine.yaml").exists()
    assert (settings.user_scenario_dir / deleted.json()["backup"]).exists()
    assert "mine.yaml" not in {item["name"] for item in client.get("/api/scenarios").json()}
    assert client.delete("/api/scenarios/mine.yaml").status_code == 404
    assert client.delete("/api/scenarios/..%2Fscenarios%2Fschweinfurt_2045.yaml").status_code in (403, 404)


def test_state_changing_scenario_calls_need_the_ui_header(settings, fake_solvers):
    from fastapi.testclient import TestClient

    from gridexpand.api.app import create_app

    with TestClient(create_app(settings)) as c:
        body = {"base": BASE, "file_name": "mine.yaml", "scenario_id": "mine", "changes": CHANGE}
        assert c.post("/api/scenarios", json=body).status_code == 403
        assert c.post("/api/scenarios/preview", json=body).status_code == 403
        assert c.delete("/api/scenarios/mine.yaml").status_code == 403
        assert c.get(f"/api/scenarios/{BASE}/form").status_code == 200
    assert not settings.user_scenario_dir.exists()
