from gridexpand.allocation.config import config
import functools

import pandas as pd
import numpy as np
import warnings

from gridexpand.common.reproducibility import physical_building_id, stable_seed

from gridexpand.allocation.functions.infdb_ro_heat import generate_opendhw, load_space_heat

##############################################################
################## Obtaining GHD + HP COP ####################
##############################################################
def _get_cop(heating_type, df_heat_space, df_heat_water, air_temp):
    """Return the demand-weighted air-source heat pump COP of one bus."""
    hp_cop_func = config.ASHP_COP

    ### Floor heat sink temperature function:
    if heating_type=="radiator": heating_func = lambda T_amb: np.array(40-T_amb)
    elif heating_type=="floor": heating_func = lambda T_amb: np.array(30-0.5*T_amb)
    else: raise ValueError("Unknown heating system type!")

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

def generate_heat_demands(df_buildings, df_elec_demand, weather_data, zip, base_seed=0):
    if "residential_effective_floor_area_m2" not in df_buildings.columns:
        raise ValueError(
            "Residential heat requires residential_effective_floor_area_m2 from the component manifest."
        )
    if getattr(config, "SPACE_HEAT_SOURCE", "teaser") == "infdb_ro_heat":
        space_heat, audit = load_space_heat(df_buildings)
        if audit["space_heat_source_fallback"]:
            print(
                "WARNING: preliminary ro_heat fallback used: "
                f"{audit['space_heat_source_fallback']} for "
                f"{audit['space_heat_source_fallback_buildings']} building(s)."
            )
        return space_heat, generate_opendhw(df_buildings, base_seed=base_seed)

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
    scenario["year"] = scenario["year"].str.extract(r'(\d+)(?!.*\d)').astype(int)
    # Component area is already effective total floor area. Passing the
    # physical footprint and multiplying by floor count would double count a
    # mixed building's non-residential share.
    scenario["area"] = pd.to_numeric(
        df_buildings["residential_effective_floor_area_m2"], errors="coerce"
    ).to_numpy()
    scenario["floors"] = 1
    scenario["nb_occ"] = scenario["nb_occ"].apply(lambda x: [int(round(y,0)) for y in x])
    # TABULA variant applied relative to each building's own year class
    # (0 standard/as built, 1 retrofit, 2 advanced retrofit), from the scenario.
    scenario["retrofit"] = int(getattr(config, "TEASER_RETROFIT_LEVEL", 0))
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
    heat_data = Datahandler(scenario, scenario_name = "example", zip_code = zip_code)
    heat_data.generateEnvironment(weather_data, sites)
    heat_data.initializeBuildings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        heat_data.generateBuildings()
        df_space_heat, df_dhw, df_gains = heat_data.generateDemands(df_elec_demand)

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
