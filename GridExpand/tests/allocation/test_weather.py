"""gridexpand.common.weather with a mocked PVGIS API (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

import gridexpand.common.weather as weather


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _payload(offset=0.5):
    hours = [
        {"time(UTC)": f"2015{month:02d}01:{hour:02d}00", "T2m": 1.0 + hour, "RH": 80.0,
         "G(h)": 10.0 * hour, "Gb(n)": 5.0, "Gd(h)": 2.0, "SP": 98000.0, "WS10m": 3.0,
         "WD10m": 180.0}
        for month, hour in ((1, 0), (1, 1), (12, 23))
    ]
    return {
        "inputs": {"location": {"irradiance_time_offset": offset, "elevation": 512.0}},
        "outputs": {"tmy_hourly": hours, "months_selected": []},
    }


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(weather.time, "sleep", sleeps.append)
    return sleeps


def test_tmy_is_moved_to_reference_year_and_utc_plus_one(monkeypatch, no_sleep):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, dict(params), timeout))
        return _Response(200, _payload())

    monkeypatch.setattr(weather.requests, "get", fake_get)
    df, altitude = weather.get_pvgis_tmy_sarah3_dataframe(48.0, 11.7, reference_year=2009)
    assert altitude == 512.0
    assert calls == [(weather.PVGIS_URL, {
        "lat": 48.0, "lon": 11.7, "raddatabase": "PVGIS-SARAH3", "outputformat": "json",
        "usehorizon": 1, "database": "SARAH3"}, weather.REQUEST_TIMEOUT_S)]
    assert {"ghi", "dni", "dhi", "temp_air", "relative_humidity", "pressure",
            "wind_speed", "wind_direction"} <= set(df.columns)
    # The last instantaneous measurement (31 Dec, 23:00 UTC + 1 h + 0.5 h) moves to the start.
    assert list(df["temp_air"]) == [24.0, 1.0, 2.0]
    assert str(df["time(UTC+1)"].iloc[0]) == "2008-12-02 00:00:00+01:00"
    assert str(df["time(inst)"].iloc[1]) == "2009-01-01 01:30:00+01:00"
    assert no_sleep == []


def test_negative_offset_keeps_order(monkeypatch, no_sleep):
    monkeypatch.setattr(weather.requests, "get", lambda url, params, timeout: _Response(200, _payload(-0.5)))
    df, _ = weather.get_pvgis_tmy_sarah3_dataframe(48.0, 11.7)
    assert list(df["temp_air"]) == [1.0, 2.0, 24.0]


def test_client_errors_raise_without_retry(monkeypatch, no_sleep):
    """B6: an HTTP error used to return None (TypeError on unpacking)."""
    calls = []

    def fake_get(url, params, timeout):
        calls.append(1)
        return _Response(400, text="Location over the sea")

    monkeypatch.setattr(weather.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="Location over the sea"):
        weather.get_pvgis_tmy_sarah3_dataframe(54.0, 5.0)
    assert calls == [1] and no_sleep == []


def test_transient_errors_are_retried(monkeypatch, no_sleep):
    responses = iter([requests.Timeout("slow"), _Response(503, text="busy"), _Response(200, _payload())])

    def fake_get(url, params, timeout):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(weather.requests, "get", fake_get)
    df, _ = weather.get_pvgis_tmy_sarah3_dataframe(48.0, 11.7)
    assert len(df) == 3
    assert no_sleep == [2, 4]


def test_persistent_failure_raises(monkeypatch, no_sleep):
    def fake_get(url, params, timeout):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(weather.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        weather.get_pvgis_tmy_sarah3_dataframe(48.0, 11.7, max_attempts=3)
    assert no_sleep == [2, 4]


def test_unexpected_payload_raises(monkeypatch, no_sleep):
    monkeypatch.setattr(weather.requests, "get", lambda url, params, timeout: _Response(200, {"message": "x"}))
    with pytest.raises(RuntimeError, match="structure"):
        weather.get_pvgis_tmy_sarah3_dataframe(48.0, 11.7)


def test_dew_point_magnus_tetens():
    temp = pd.Series([20.0, 0.0])
    dew = weather.get_dew_point(temp, pd.Series([100.0, 50.0]))
    assert np.isclose(dew.iloc[0], 20.0)
    assert np.isclose(dew.iloc[1], 237.7 * np.log(0.5) / (17.27 - np.log(0.5)))


def test_read_weather_hdf_adds_the_dew_point_and_returns_the_altitude(tmp_path):
    import numpy as np
    import pandas as pd

    from gridexpand.common import weather

    path = tmp_path / "w.h5"
    frame = pd.DataFrame({"temp_air": [5.0, -3.0], "relative_humidity": [80.0, 90.0], "ghi": [0.0, 100.0]})
    with pd.HDFStore(path, mode="w") as store:
        store.put("raw_data/weather", frame)
        store.put("raw_data/region", pd.DataFrame([{"lat": 48.6, "lon": 12.3, "altitude": 376.0, "plz": 84051}]))
    loaded, altitude = weather.read_weather_hdf(path)
    assert altitude == 376.0
    assert np.allclose(loaded["dew_point"], weather.get_dew_point(frame["temp_air"], frame["relative_humidity"]))
    assert list(loaded.columns[:3]) == ["temp_air", "relative_humidity", "ghi"]
