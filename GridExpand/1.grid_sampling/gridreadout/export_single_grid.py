#!/usr/bin/env python3
"""Export exactly one pylovo grid to Step-1 HDF5 format.

This helper is intended for low-cost pilot runs (for example one Munich PLZ)
without modifying the sampling notebooks.
"""

import argparse
from typing import Optional

import pandas as pd

import src.db_read as dbrd
import src.grid_topol as grdtpl
import src.save_grid as svgrd
import src.weather as wth


def _select_grid_for_plz(
    db: dbrd.DataBase,
    plz: str,
    kcid: Optional[int],
    bcid: Optional[int],
    candidate_index: int,
    min_buildings: int,
) -> tuple[int, int, int, pd.DataFrame, str, int]:
    df_all_candidates = db.read_grid_identifiers_from_positions(min_buildings=min_buildings)
    n_total_candidates = len(df_all_candidates)
    source_label = f"transformer_positions (>= {min_buildings} buildings)"

    df_candidates = (
        df_all_candidates[df_all_candidates["plz"].astype(str) == str(plz)][["plz", "kcid", "bcid"]]
        .drop_duplicates()
        .sort_values(["kcid", "bcid"])
        .reset_index(drop=True)
    )

    if df_candidates.empty:
        raise ValueError(f"No pylovo grids found for PLZ {plz} in source {source_label}.")

    if (kcid is None) != (bcid is None):
        raise ValueError("Please provide both --kcid and --bcid, or neither.")

    if kcid is not None and bcid is not None:
        mask = (df_candidates["kcid"] == int(kcid)) & (df_candidates["bcid"] == int(bcid))
        df_match = df_candidates[mask]
        if df_match.empty:
            raise ValueError(
                f"No grid found for PLZ={plz}, KCID={kcid}, BCID={bcid}. "
                "Use --list-candidates to inspect valid combinations."
            )
        row = df_match.iloc[0]
    else:
        if candidate_index < 0 or candidate_index >= len(df_candidates):
            raise ValueError(
                f"Candidate index {candidate_index} out of range 0..{len(df_candidates)-1}."
            )
        row = df_candidates.iloc[candidate_index]

    return (
        int(row["plz"]),
        int(row["kcid"]),
        int(row["bcid"]),
        df_candidates,
        source_label,
        n_total_candidates,
    )


def _build_region_row(db: dbrd.DataBase, plz: int, kcid: int, bcid: int) -> pd.DataFrame:
    grid_specs = {"plz": plz, "kcid": kcid, "bcid": bcid}

    df_region = db.read_regional_stats(plz)
    if df_region.empty:
        df_region = pd.DataFrame([{"plz": plz}])
    else:
        df_region = df_region.iloc[[0]].copy().reset_index(drop=True)

    location = db.read_trafo_pos(grid_specs)
    df_region["lat"] = location["lat"]
    df_region["lon"] = location["lon"]
    df_region["kcid"] = kcid
    df_region["bcid"] = bcid

    return df_region


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one pylovo grid for pilot runs.")
    parser.add_argument("--plz", required=True, help="Target German PLZ (e.g. 80803).")
    parser.add_argument("--kcid", type=int, help="Optional KCID to pin one exact grid.")
    parser.add_argument("--bcid", type=int, help="Optional BCID to pin one exact grid.")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="If KCID/BCID are not provided, pick this 0-based candidate.",
    )
    parser.add_argument(
        "--cell-id",
        default=None,
        help="Prefix for output filename before PLZ/KCID/BCID. Defaults to pilot<PLZ>.",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip PVGIS/Open-Meteo weather retrieval and do not write /raw_data/weather.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Only list available (PLZ, KCID, BCID) combinations and exit.",
    )
    parser.add_argument(
        "--min-buildings",
        type=int,
        default=5,
        help="Minimum number of buildings required for candidate grids (default: 5).",
    )
    args = parser.parse_args()

    db = dbrd.DataBase()

    plz, kcid, bcid, df_candidates, source_label, n_total_candidates = _select_grid_for_plz(
        db=db,
        plz=args.plz,
        kcid=args.kcid,
        bcid=args.bcid,
        candidate_index=args.candidate_index,
        min_buildings=args.min_buildings,
    )

    if args.list_candidates:
        print(f"Candidate source: {source_label}")
        print(
            f"Candidates for PLZ {plz}: {len(df_candidates)} "
            f"(global pool: {n_total_candidates})"
        )
        print(df_candidates.to_string(index=True))
        return

    cell_id = args.cell_id if args.cell_id else f"pilot{plz}"
    grid_specs = {"cell_id": cell_id, "plz": plz, "kcid": kcid, "bcid": bcid}
    print(f"Selected grid: PLZ={plz}, KCID={kcid}, BCID={bcid}, cell_id={cell_id}")

    net = db.read_single_ppgrid(grid_specs)
    net = grdtpl.assign_min_linelen(net)
    net = grdtpl.remove_duplicate_loads(net)

    df_buildings = db.read_buildings(grid_specs, net.bus)
    df_region = _build_region_row(db, plz, kcid, bcid)

    df_weather = None
    if not args.skip_weather:
        lat = float(df_region.iloc[0]["lat"])
        lon = float(df_region.iloc[0]["lon"])

        weather_tuple = wth.get_pvgis_tmy_sarah3_dataframe(lat, lon)
        if weather_tuple is None:
            raise RuntimeError("PVGIS weather retrieval failed.")
        df_weather, altitude, selected_months = weather_tuple

        df_weather["dew_point"] = wth.get_dew_point(
            df_weather["temp_air"], df_weather["relative_humidity"]
        )

        df_soil = wth.get_open_meteo_soil_temperature(lat, lon, selected_months)
        soil_series = df_soil.iloc[:, 0] if isinstance(df_soil, pd.DataFrame) else df_soil
        df_weather["soil_temp"] = pd.Series(soil_series).reset_index(drop=True)
        df_region["altitude"] = float(altitude)

    sf = svgrd.SaveFile(grid_specs)
    sf.save_topology(net, "/raw_data/")
    sf.save_df(df_region, "/raw_data/region")
    sf.save_df(df_buildings, "/raw_data/buildings")
    if df_weather is not None:
        sf.save_df(df_weather, "/raw_data/weather")

    print(f"Export complete: {sf.path}")
    if args.skip_weather:
        print("Weather skipped. For Step 2 set weather_data_exists=False or add weather later.")


if __name__ == "__main__":
    main()
