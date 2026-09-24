"""Write one provider's shared weather HDF for an aligned paired dataset.

The weather is a PVGIS SARAH3 TMY at the centroid of the dataset's buildings.
The file carries ``raw_data/weather`` and ``raw_data/region`` for the PV
profile library and ``urbs_in/weather`` for Step 3. Its name must contain the
five-digit postcode used for the heat-pump design outdoor temperature.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ..paths import GRIDEXPAND_DIR

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402
from common.timeframe import build_full_year_metadata, write_hdf_metadata  # noqa: E402
import src.functions.weather as weather_functions  # noqa: E402


def write_aligned_weather(paired_dir: Path, plz: int, output: Path, reference_year: int) -> Path:
    if re.search(rf"_{int(plz):05d}_", output.name) is None:
        raise ValueError(f"The weather filename must contain _{plz:05d}_.")
    plan = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv")
    database = SurroGridDatabase()
    with database.engine.connect() as conn:
        centroid = conn.execute(
            text(
                """
                SELECT AVG(ST_Y(ST_Transform(centroid, 4326))) AS lat,
                       AVG(ST_X(ST_Transform(centroid, 4326))) AS lon
                FROM pylovo.buildings_result
                WHERE version_id::text = :version AND objectid = ANY(:ids)
                """
            ),
            {
                "version": _version(paired_dir),
                "ids": plan["building_objectid"].astype(str).tolist(),
            },
        ).mappings().one()
    result = weather_functions.get_pvgis_tmy_sarah3_dataframe(
        float(centroid["lat"]), float(centroid["lon"]), reference_year=int(reference_year)
    )
    if result is None:
        raise RuntimeError("PVGIS returned no TMY weather.")
    weather, altitude, _ = result
    region = pd.DataFrame(
        [{"lat": float(centroid["lat"]), "lon": float(centroid["lon"]),
          "altitude": float(altitude), "plz": int(plz)}]
    )
    urbs = weather[["temp_air", "ghi"]].rename(columns={"temp_air": "Tamb", "ghi": "Irradiation"})
    urbs = urbs.reset_index(drop=True)
    urbs.index.name = "t"
    urbs.columns = pd.MultiIndex.from_product([["ambient"], urbs.columns])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    write_hdf_metadata(output, build_full_year_metadata())
    with pd.HDFStore(output, mode="a") as store:
        store.put("raw_data/weather", weather)
        store.put("raw_data/region", region)
        store.put("urbs_in/weather", urbs)
    return output


def _version(paired_dir: Path) -> str:
    metadata = json.loads((paired_dir / "paired_scenario_metadata.json").read_text(encoding="utf-8"))
    return str(metadata["pylovo_version_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--plz", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-year", type=int, default=2009)
    args = parser.parse_args()
    print(write_aligned_weather(args.paired_dir.resolve(), args.plz, args.output.resolve(), args.reference_year))


if __name__ == "__main__":
    main()
