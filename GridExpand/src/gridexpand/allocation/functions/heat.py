from gridexpand.allocation.config import config
import functools

import pandas as pd
import numpy as np
import warnings

from gridexpand.common.reproducibility import physical_building_id, stable_seed

# TABULA/TEASER construction-year classes (first year, last year, representative year). Class A (before 1860)
# is merged into B because the DistrictGenerator clamps earlier years to 1860.
TABULA_CLASSES = {
    "B": (1860, 1918, 1889), "C": (1919, 1948, 1933), "D": (1949, 1957, 1953), "E": (1958, 1968, 1963),
    "F": (1969, 1978, 1973), "G": (1979, 1983, 1981), "H": (1984, 1994, 1989), "I": (1995, 2001, 1998),
    "J": (2002, 2009, 2005), "K": (2010, 2015, 2012), "L": (2016, 2100, 2018),
}
# Residential buildings in thousands per class B-J (A added to B), German census 2011, IWU (Loga et al. 2015,
# Deutsche Wohngebaeudetypologie, Tab. 4; EFH=SFH, RH=TH, MFH=MFH, GMH=AB). K and L have no count; they get
# the yearly rate of class J.
IWU_STOCK_BUILDINGS = {
    "SFH": {"B": 1296, "C": 1131, "D": 859, "E": 1509, "F": 1507, "G": 704, "H": 1160, "I": 1035, "J": 775},
    "TH": {"B": 640, "C": 710, "D": 447, "E": 633, "F": 611, "G": 335, "H": 652, "I": 619, "J": 384},
    "MFH": {"B": 496, "C": 388, "D": 356, "E": 586, "F": 412, "G": 146, "H": 309, "I": 244, "J": 85},
    "AB": {"B": 29.3, "C": 7.4, "D": 17.3, "E": 34.0, "F": 50.1, "G": 15.0, "H": 28.7, "I": 20.9, "J": 7.6},
}
# IWU consumption calibration (typical measured / standard calculated heat demand) per type and class, from
# Loga et al. (2015), App. C.3 (GMH for AB up to F, MFH values for younger AB). Classes K and L: 1.
IWU_CONSUMPTION_FACTOR = {
    "SFH": {"B": 0.59, "C": 0.65, "D": 0.58, "E": 0.59, "F": 0.69, "G": 0.80, "H": 0.76, "I": 0.90, "J": 0.97},
    "TH": {"B": 0.68, "C": 0.75, "D": 0.69, "E": 0.85, "F": 0.79, "G": 0.79, "H": 0.87, "I": 0.97, "J": 0.99},
    "MFH": {"B": 0.73, "C": 0.67, "D": 0.70, "E": 0.79, "F": 0.78, "G": 0.82, "H": 0.81, "I": 0.95, "J": 1.0},
    "AB": {"B": 0.80, "C": 0.74, "D": 0.75, "E": 0.79, "F": 0.83, "G": 0.82, "H": 0.81, "I": 0.95, "J": 1.0},
}


def tabula_class(year):
    """Return the TABULA class letter of a construction year (before 1860 counts as B)."""
    for letter, (first, last, _) in TABULA_CLASSES.items():
        if year <= last and (year >= first or letter == "B"):
            return letter
    return "L"


def band_years(band):
    """Return (first, last) year of a Zensus age band such as ``-1919``, ``1949-1978`` or ``2020-``."""
    text = str(band).strip()
    first, _, last = text.partition("-")
    first = int(first) if first.strip() else 1860
    last = int(last) if last.strip() else max(first, 2025)
    return max(first, 1860), last


def stock_class_shares(building_type, band):
    """Shares of the TABULA classes within a Zensus band, weighted by the IWU 2011 building counts.

    Within a class the buildings are assumed uniform over its years, so a class that only partly overlaps the
    band enters with the overlapping share of its count. K and L use the yearly rate of class J.
    """
    counts = IWU_STOCK_BUILDINGS.get(building_type, IWU_STOCK_BUILDINGS["AB"])
    first, last = band_years(band)
    weights = {}
    for letter, (c0, c1, _) in TABULA_CLASSES.items():
        overlap = min(last, c1) - max(first, c0) + 1
        if overlap <= 0:
            continue
        if letter in counts:
            weights[letter] = counts[letter] * overlap / (c1 - c0 + 1)
        else:
            weights[letter] = counts["J"] / 8.0 * overlap
    total = sum(weights.values())
    return {letter: w / total for letter, w in weights.items()} if total > 0 else {tabula_class(last): 1.0}


def teaser_year(row, band, teaser, base_seed):
    """TEASER construction year of one building: last band year (legacy) or a stock-share draw (S5)."""
    if teaser.age_mapping == "band_end":
        return int(pd.Series([band]).str.extract(r'(\d+)(?!.*\d)').astype(int).iloc[0, 0])
    shares = stock_class_shares(row["building_type"], band)
    rng = np.random.default_rng(stable_seed(base_seed, physical_building_id(row), "heat", "teaser_age_class"))
    letters = list(shares)
    letter = letters[int(rng.choice(len(letters), p=[shares[x] for x in letters]))]
    return TABULA_CLASSES[letter][2]


def teaser_retrofit(row, year, level, teaser, base_seed):
    """TABULA variant: the scenario level, raised to 1 for a sampled share of old buildings (S7, off by default)."""
    share = (teaser.retrofit_share_before_1979 if year < 1979
             else teaser.retrofit_share_1979_1994 if year <= 1994 else 0.0)
    if share <= 0.0 or level >= 1:
        return level
    rng = np.random.default_rng(stable_seed(base_seed, physical_building_id(row), "heat", "teaser_retrofit_mix"))
    return 1 if rng.random() < share else level


def energy_only_scaling(profile, factor):
    """Scale the annual sum of an hourly profile by ``factor`` and keep its peak (S8).

    q' = peak * (q / peak) ** gamma with gamma >= 1 found by bisection; the transform keeps zero hours and
    the peak hour and lowers the part-load hours most.
    """
    q = np.asarray(profile, float)
    peak, total = q.max(initial=0.0), q.sum()
    if factor >= 1.0 or peak <= 0.0 or total <= 0.0:
        return q * factor if factor < 1.0 else q
    x = q / peak
    target = factor * total
    if peak * np.count_nonzero(x >= 1.0 - 1e-12) >= target:
        return q * factor
    lo, hi = 1.0, 2.0
    while (peak * x ** hi).sum() > target:
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (peak * x ** mid).sum() > target:
            lo = mid
        else:
            hi = mid
    return peak * x ** hi


def teaser_weather_arrays(df_weather, teaser=None):
    """Weather input of the TEASER path: [beam, diffuse, temperature] arrays.

    The DistrictGenerator solar code (Perez model, Duffie & Beckman eq. 1.8.1) expects the beam irradiance on
    the horizontal. ``solar_beam: horizontal`` passes GHI - DHI; ``normal`` (legacy) passes DNI. The internal
    1R1C model has its own call and is not affected.
    """
    teaser = teaser if teaser is not None else _default_teaser()
    if teaser.solar_beam == "horizontal":
        beam = (np.asarray(df_weather["ghi"], float) - np.asarray(df_weather["dhi"], float)).clip(min=0.0)
    else:
        beam = np.array(df_weather["dni"])
    return [beam, np.array(df_weather["dhi"]), np.array(df_weather["temp_air"])]


def _default_teaser():
    from gridexpand.scenario.scenario_config import TeaserHeatConfig

    return TeaserHeatConfig()


# Per-building LoD2 surface sums stored with the grid inputs (h5 key raw_data/building_surfaces).
BUILDING_SURFACE_COLUMNS = (
    "wall_surface_area_m2", "roof_surface_area_m2", "shared_wall_area_m2",
    "wall_above_eaves_m2", "shared_wall_above_eaves_m2",
)


def lod2_inputs(df_buildings, teaser):
    """Per-building LoD2 envelope inputs of the TEASER path, scaled to the residential share.

    Returns (walls incl. windows, roof or top-floor ceiling, ground floor, temperature reduction factor of the
    roof element). Uses ``wall_surface_area_m2``, ``roof_surface_area_m2``, ``shared_wall_area_m2``,
    ``wall_above_eaves_m2`` and ``shared_wall_above_eaves_m2`` (optional) and the footprint ``floor_area``.
    With ``attic: unheated`` a pitched roof (roof surface > ``pitched_roof_area_ratio`` x footprint) closes
    the heated zone at the eaves: the ceiling (= footprint) with ``attic_ceiling_b`` replaces the roof, and the
    walls above the eaves leave the envelope. Missing values give NaN, which keeps the TABULA envelope.
    """
    n = len(df_buildings)
    def column(name):
        if name not in df_buildings.columns:
            return np.full(n, np.nan)
        return pd.to_numeric(df_buildings[name], errors="coerce").to_numpy(dtype=float)
    footprint = column("floor_area")
    gross = footprint * column("floor_number")
    share = np.clip(column("residential_effective_floor_area_m2") / gross, 0.0, 1.0)
    share = np.where(np.isfinite(share), share, 1.0)
    wall, roof = column("wall_surface_area_m2"), column("roof_surface_area_m2")
    shared = np.nan_to_num(column("shared_wall_area_m2"), nan=0.0)
    above = np.nan_to_num(column("wall_above_eaves_m2"), nan=0.0)
    shared_above = np.nan_to_num(column("shared_wall_above_eaves_m2"), nan=0.0)
    attic = (np.full(n, teaser.attic == "unheated")
             & (roof > teaser.pitched_roof_area_ratio * footprint))
    walls = np.where(attic, wall - above, wall)
    if teaser.subtract_shared_walls:
        walls = walls - np.where(attic, shared - shared_above, shared)
    roof_element = np.where(attic, footprint, roof)
    roof_b = np.where(attic, teaser.attic_ceiling_b, 1.0)
    return walls * share, roof_element * share, footprint * share, roof_b


##############################################################
################## Obtaining GHD + HP COP ####################
##############################################################
def _get_cop(heating_type, df_heat_space, df_heat_water, air_temp):
    """Return the demand-weighted air-source heat pump COP of one bus."""
    hp_cop_func = config.ASHP_COP

    ### Floor heat sink temperature function:
    if heating_type == "radiator":
        def heating_func(T_amb):
            return np.array(40 - T_amb)
    elif heating_type == "floor":
        def heating_func(T_amb):
            return np.array(30 - 0.5 * T_amb)
    else:
        raise ValueError("Unknown heating system type!")

    ### Calculate T_sink-T_amb:
    dT_space = heating_func(air_temp) - air_temp
    dT_water = 50 - air_temp

    ### Clip to minimum temperature difference of 15K:
    dT_space = pd.DataFrame(dT_space).clip(lower=15)
    dT_water = pd.DataFrame(dT_water).clip(lower=15)

    ### Compute COPs:
    cop_space = hp_cop_func(dT_space)
    cop_space.columns=["0"]
    cop_water = hp_cop_func(dT_water)
    cop_water.columns=["0"]

    ### Compute final cop as weighed average of space/water cop with space/water heat demand
    df_heat_space.columns=["0"]
    df_heat_water.columns=["0"]
    df_heat_total = df_heat_space + df_heat_water

    numerator = cop_space.values * df_heat_space.values + cop_water.values * df_heat_water.values
    cop_total = np.divide(numerator, 
                          df_heat_total.values, 
                          out=np.full_like(numerator, cop_space), 
                          where=df_heat_total.values != 0)

    return pd.DataFrame(cop_total)

##############################################################
############## Generation, Publicly Callable #################
##############################################################
@functools.cache
def site_data() -> pd.DataFrame:
    """Return the DistrictGenerator postcode climate table, read once per process."""
    return pd.read_csv(
        f"{config.DISTGEN_DATA_PATH}/site_data.txt",
        delimiter="\t",
        dtype={"Zip": str},
    )


def get_norm_outside_temperature(zip_code):
    """Return the exact postcode-specific norm outside temperature."""
    postcode = str(zip_code).zfill(5)
    sites = site_data()
    match = sites[sites["Zip"].eq(postcode)]
    if len(match) != 1:
        raise ValueError(
            f"Expected one exact postcode climate entry for {postcode}, found {len(match)}."
        )
    return float(match.iloc[0]["T_ne"])

def sample_statistics(df_buildings, base_seed=0):
    # Sample ages for non-residential buildings:
    missing_age = df_buildings["construction_year"].isna()
    for index, row in df_buildings.loc[missing_age].iterrows():
        rng = np.random.default_rng(stable_seed(
            base_seed, physical_building_id(row), "heat", "construction_year"
        ))
        df_buildings.at[index, "construction_year"] = rng.choice(
            config.AGE_GHD_DISTRIBUTION["age"],
            p=config.AGE_GHD_DISTRIBUTION["prob"],
        )
    
    # # Sample heat pump type:
    # df_buildings["hp_type"] = np.random.choice(
    #     config.HP_TYPE_DIST["type"], 
    #     size=len(df_buildings), 
    #     p=config.HP_TYPE_DIST["prob"])

    # Sample floor heating type:
    df_buildings["heating_type"] = df_buildings.apply(
        lambda row: np.random.default_rng(stable_seed(
            base_seed, physical_building_id(row), "heat", "heating_type"
        )).choice(
            ["radiator", "floor"], p=[config.PROB_RADIATOR, config.PROB_FLOOR]
        ),
        axis=1,
    )

    return df_buildings

def generate_heat_demands(df_buildings, df_elec_demand, weather_data, zip, base_seed=0, teaser=None):
    """Simulate TEASER space heat and OpenDHW hot water (kWh per hour and bus).

    ``teaser`` is the scenario's ``TeaserHeatConfig`` (None: the corrected defaults; ``TeaserHeatConfig.legacy()``
    reproduces SurroGrid 2.0.0). ``weather_data`` comes from ``teaser_weather_arrays`` with the same config.
    The INFDB ``ro_heat`` source does not use this function (see
    ``Grid.generate_heat``).
    """
    teaser = teaser if teaser is not None else _default_teaser()
    if "residential_effective_floor_area_m2" not in df_buildings.columns:
        raise ValueError(
            "Residential heat requires residential_effective_floor_area_m2 from the component manifest."
        )

    # Import the legacy generator lazily. The INFDB ro_heat path must not load
    # TEASER or execute any DistrictGenerator code.
    from gridexpand.allocation.external.districtgenerator.classes import Datahandler

    # The first heat scenario models residential components only. In
    # particular, do not infer non-residential DHW from a mixed building's
    # source building_use label.

    # Setting up input for heat load generator
    scenario = df_buildings.copy()
    scenario = scenario[["bus", "building_type", "construction_year", "floor_area", "floor_number", "households", "occ_list"]]
    scenario.reset_index(inplace=True)
    scenario.rename(inplace=True, columns={"index": "id", "building_type": "building", "households": "nb_flat", "occ_list": "nb_occ", "construction_year": "year", "floor_area": "area", "floor_number": "floors"})
    scenario["NWG"] = scenario["building"].apply(lambda x: 1 if x not in ["SFH","MFH","TH","AB"] else 0)
    if teaser.age_mapping == "band_end":
        scenario["year"] = scenario["year"].str.extract(r'(\d+)(?!.*\d)').astype(int)
    else:
        bands = list(scenario["year"])
        scenario["year"] = [
            teaser_year(row, bands[i], teaser, base_seed)
            for i, (_, row) in enumerate(df_buildings.iterrows())
        ]
    # Component area is already effective total floor area. Passing the
    # physical footprint and multiplying by floor count would double count a
    # mixed building's non-residential share. S1: TEASER's net_leased_area is
    # the heated reference area, heated_area_ratio x this gross area.
    scenario["area"] = pd.to_numeric(
        df_buildings["residential_effective_floor_area_m2"], errors="coerce"
    ).to_numpy() * teaser.heated_area_ratio
    if teaser.envelope_source == "lod2":
        wall, roof, ground, roof_b = lod2_inputs(df_buildings, teaser)
        scenario["lod2_wall_m2"], scenario["lod2_roof_m2"], scenario["lod2_ground_m2"] = wall, roof, ground
        scenario["lod2_roof_b"] = roof_b
    scenario["floors"] = 1
    scenario["nb_occ"] = scenario["nb_occ"].apply(lambda x: [int(round(y,0)) for y in x])
    # TABULA variant applied relative to each building's own year class
    # (0 standard/as built, 1 retrofit, 2 advanced retrofit), from the scenario.
    level = int(getattr(config, "TEASER_RETROFIT_LEVEL", 0))
    years = list(scenario["year"])
    scenario["retrofit"] = [
        teaser_retrofit(row, years[i], level, teaser, base_seed)
        for i, (_, row) in enumerate(df_buildings.iterrows())
    ]
    # Occupancy and DHW draw on the global RNGs; seed them per physical
    # building so the realization is shared by all model cases and does not
    # depend on the CPU partition.
    scenario["seed"] = [
        stable_seed(base_seed, physical_building_id(row), "heat", "teaser")
        for _, row in df_buildings.iterrows()
    ]

    # Extract location data
    zip_code = str(zip)
    sites = site_data()
    if zip_code not in set(sites["Zip"]):
        raise ValueError(
            f"No exact postcode climate entry for {zip_code}. Numeric postcode "
            "proximity is not a geographic climate fallback."
        )
    # print(site_data)
    # Simulate heating
    heat_data = Datahandler(scenario, scenario_name = "example", zip_code = zip_code, options = teaser)
    heat_data.generateEnvironment(weather_data, sites)
    heat_data.initializeBuildings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        heat_data.generateBuildings()
        df_space_heat, df_dhw, df_gains = heat_data.generateDemands(df_elec_demand)

    if teaser.consumption_factor == "iwu":
        # S8: IWU consumption calibration on the annual energy, peak kept.
        for row in scenario.itertuples():
            if row.retrofit != 0 or row.bus not in df_space_heat.columns:
                continue
            btype = row.building if row.building in IWU_CONSUMPTION_FACTOR else "AB"
            factor = IWU_CONSUMPTION_FACTOR[btype].get(tabula_class(row.year), 1.0)
            df_space_heat[row.bus] = energy_only_scaling(df_space_heat[row.bus].to_numpy(), factor)

    # Postprocess demands
    df_dhw.columns = pd.MultiIndex.from_product([df_dhw.columns, ["water_heat"]])
    df_space_heat.columns = pd.MultiIndex.from_product([df_space_heat.columns, ["space_heat"]])

    return df_space_heat, df_dhw

def require_heat_profiles(buses, df_heat_space, df_heat_water):
    """Raise if a selected heat bus has no space-heat or hot-water profile.

    Heat sizing treats a missing profile column as zero demand, so a building
    dropped by a generator would otherwise be sized at 0 kW without an error.
    """
    missing = [
        bus
        for bus in dict.fromkeys(buses)
        if (bus, "space_heat") not in df_heat_space.columns
        or (bus, "water_heat") not in df_heat_water.columns
    ]
    if missing:
        raise ValueError(
            f"Heat generation produced no profile for selected heat bus(es) {missing[:10]}."
        )

def generate_hp_cop(df_buildings, df_heat_space, df_heat_water, df_weather):
    """Return one air-source heat pump COP series per heat bus.

    Demand columns are already shared-bus aggregates; the heating system
    (radiator or floor) of the first building on a bus sets the sink
    temperature.
    """
    air_temp = df_weather["temp_air"]
    cop_dict = {}
    for bus, group in df_buildings.groupby("bus", sort=True):
        if (bus, "space_heat") not in df_heat_space.columns:
            continue
        heating_type = group.iloc[0]["heating_type"]
        space_heat = pd.DataFrame(df_heat_space[bus, "space_heat"])
        water_heat = pd.DataFrame(df_heat_water[bus, "water_heat"])
        cop_dict[bus] = _get_cop(heating_type, space_heat, water_heat, air_temp)
    if not cop_dict:
        return pd.DataFrame(index=df_heat_space.index)
    cops_air = pd.concat([cop for _,cop in cop_dict.items()], axis=1)
    cops_air.columns = pd.MultiIndex.from_tuples([(col, "heatpump_air") for col in cop_dict.keys()])
    return cops_air
