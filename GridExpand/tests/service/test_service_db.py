"""Read-only API tests on the sandbox database (opt-in, see conftest.py).

The sandbox holds the finished harness run: pylovo v1 of PLZ 85653 (AGS 9184137) with four
grids and the three model cases of the sandbox scenario.
"""

from __future__ import annotations

import pytest

PLZ, VERSION = 85653, "1"


@pytest.fixture()
def db_client(sandbox_db, client):
    return client


def test_status_reads_the_database(db_client):
    database = db_client.get("/api/status").json()["database"]
    assert database["connected"] and database["schema"]["pylovo_ready"] and database["schema"]["surrogrid_ready"]
    versions = {v["version_id"]: v for v in database["pylovo_versions"]}
    assert versions[VERSION]["grids"] >= 1


def test_grids_use_the_runner_numbering(db_client):
    from gridexpand.scenario.synthetic_ags_runner import get_candidates

    data = db_client.get("/api/grids", params={"plz": PLZ, "pylovo_version_id": VERSION, "min_buildings": 5}).json()
    assert data["ags"] == 9184137 and data["ags_options"] == [9184137]
    expected = [c for c in get_candidates("9184137", 5, "all", VERSION) if c["plz"] == PLZ]
    assert [(c["candidate_index"], c["grid_result_id"]) for c in data["candidates"]] == \
        [(c["candidate_index"], c["grid_result_id"]) for c in expected]
    assert all("results" in c for c in data["candidates"])


def test_analyses_grids_geojson_and_powerflow(db_client):
    analyses = db_client.get("/api/results/analyses", params={"plz": PLZ, "pylovo_version_id": VERSION}).json()
    assert analyses, "the sandbox should contain expansion analyses"
    for a in analyses:
        assert a["total_cost_eur"] == pytest.approx(a["line_cost_eur"] + a["transformer_cost_eur"])
        assert a["plz_list"] == [PLZ] and a["timeframe_mode"]
    post = next((a for a in analyses if a["stage"] == "post"), analyses[0])
    key = post["analysis_key"]
    grids = db_client.get(f"/api/results/analyses/{key}/grids", params={"plz": PLZ}).json()
    assert len(grids) == post["grids"]
    assert sum(g["total_cost_eur"] for g in grids) == pytest.approx(post["total_cost_eur"])
    geo = db_client.get(f"/api/results/analyses/{key}/geojson", params={"plz": PLZ}).json()
    assert len(geo["lines"]["features"]) == post["lines_total"]
    assert len(geo["transformers"]["features"]) == post["transformers_total"]
    assert sum(f["properties"]["action"] != "none" for f in geo["lines"]["features"]) == post["lines_to_reinforce"]
    lon, lat = geo["lines"]["features"][0]["geometry"]["coordinates"][0]
    assert 5 < lon < 16 and 47 < lat < 56  # EPSG:4326 in Germany
    assert geo["bounds"][0] < geo["bounds"][2]
    assert db_client.get("/api/results/analyses/nope/grids").status_code == 404
    rows = db_client.get("/api/results/powerflow", params={"plz": PLZ, "pylovo_version_id": VERSION}).json()
    assert rows and {r["stage"] for r in rows} <= {"pre", "post"}
    assert {r["model_case"] for r in rows} >= {"pre"}
