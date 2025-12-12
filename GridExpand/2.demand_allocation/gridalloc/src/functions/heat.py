from config import config
import pandas as pd
import numpy as np
from src.external.districtgenerator.classes import *
import warnings

##############################################################
################## Obtaining GHD + HP COP ####################
##############################################################
def _get_single_dhw_timeseries_ghd(type, area, floors, df_normalized_lps_ghd):
    return df_normalized_lps_ghd[type]*area*floors

def _get_dhw_demand_ghd(df_buildings):
    df_normalized_lps_ghd = pd.read_csv(config.DHW_GHD_PATH, skiprows=1, header=[0])*1000

    # Apply function and create a new DataFrame
    data_dict_ghd = {row["bus"]: _get_single_dhw_timeseries_ghd(row['type'], row["area"], row["floors"], df_normalized_lps_ghd) for idx, row in df_buildings.iterrows() if row["use"]!="Residential"}

    # Convert to DataFrame
    df_dhw_demand = pd.DataFrame(data_dict_ghd).reset_index(drop=True)
    df_dhw_demand.columns = pd.MultiIndex.from_product([df_dhw_demand.columns, ["water_heat"]])
    return df_dhw_demand

def _get_cop(hp_type, heating_type, df_heat_space, df_heat_water, air_temp, soil_temp):
    ### Select heat pump function:
    if hp_type=="ASHP": hp_cop_func = config.ASHP_COP
    elif hp_type=="GSHP": hp_cop_func = config.GSHP_COP
    # elif hp_type=="WSHP": hp_cop_func = config.WSHP_COP
    else: raise ValueError("Unknown heat pump type!")

    ### Floor heat sink temperature function:
    if heating_type=="radiator": heating_func = lambda T_amb: np.array(40-T_amb)
    elif heating_type=="floor": heating_func = lambda T_amb: np.array(30-0.5*T_amb)
    else: raise ValueError("Unknown heating system type!")

    ### Calculate T_sink-T_amb:
    if hp_type=="ASHP":
        dT_space = heating_func(air_temp) - air_temp
        dT_water = 50 - air_temp
    # if hp_type=="GSHP":
    #     dT_space = heating_func(air_temp) - soil_temp - 5  # -5 to account for heat transfer of soil to brine
    #     dT_water = 50 - soil_temp
    # if hp_type=="WSHP":
    #     dT_space = heating_func(air_temp) - 10 - 5         # assume 10°C water temperature - 5 loss for intermediate heat exchangers 
    #     dT_water = np.ones((8760))*(50 - 10 - 5)

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
def sample_statistics(df_buildings):
    # Sample ages for non-residential buildings:
    df_buildings.loc[df_buildings["constructi"].isna(), ["constructi"]] = np.random.choice(
        config.AGE_GHD_DISTRIBUTION["age"], 
        size=len(df_buildings[df_buildings["constructi"].isna()]["constructi"]), 
        p=config.AGE_GHD_DISTRIBUTION["prob"])
    
    # # Sample heat pump type:
    # df_buildings["hp_type"] = np.random.choice(
    #     config.HP_TYPE_DIST["type"], 
    #     size=len(df_buildings), 
    #     p=config.HP_TYPE_DIST["prob"])

    # Sample floor heating type:
    df_buildings["heating_type"] = np.random.choice(
        ["radiator","floor"], 
        size=len(df_buildings), 
        p=[config.PROB_RADIATOR, config.PROB_FLOOR])

    return df_buildings

def generate_heat_demands(df_buildings, df_elec_demand, weather_data, zip):
    # Domestic hot water only for non-residential buildings
    df_dhw_ghd = _get_dhw_demand_ghd(df_buildings)

    # Setting up input for heat load generator
    scenario = df_buildings.copy()
    scenario = scenario[["bus", "type", "constructi", "area", "floors", "houses_per_building","occ_list"]]
    scenario.reset_index(inplace=True)
    scenario.rename(inplace=True, columns={"index":"id","type":"building", "houses_per_building":"nb_flat", "occ_list":"nb_occ", "constructi":"year"})
    scenario["NWG"] = scenario["building"].apply(lambda x: 1 if x not in ["SFH","MFH","TH","AB"] else 0)
    scenario["year"] = scenario["year"].str.extract(r'(\d+)(?!.*\d)').astype(int)
    scenario["area"] = scenario["area"] * scenario["floors"]
    scenario["nb_occ"] = scenario["nb_occ"].apply(lambda x: [int(round(y,0)) for y in x])
    scenario["retrofit"] = 0

    # Extract location data
    site_data = pd.read_csv(f"{config.DISTGEN_DATA_PATH}/site_data.txt", delimiter='\t', dtype={'Zip': str})
    # print(site_data)
    # Simulate heating
    heat_data = Datahandler(scenario, scenario_name = "example", zip_code = str(zip))
    heat_data.generateEnvironment(weather_data, site_data)
    heat_data.initializeBuildings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        heat_data.generateBuildings()
        df_space_heat, df_dhw, df_gains = heat_data.generateDemands(df_elec_demand)

    # Postprocess demands
    df_dhw.columns = pd.MultiIndex.from_product([df_dhw.columns, ["water_heat"]])
    df_space_heat.columns = pd.MultiIndex.from_product([df_space_heat.columns, ["space_heat"]])
    df_dhw[df_dhw_ghd.columns] = df_dhw_ghd  # Combine with nonres building results for DHW

    return df_space_heat, df_dhw

def generate_hp_cop(df_buildings, df_heat_space, df_heat_water, df_weather):
    air_temp = df_weather["temp_air"]
    soil_temp = df_weather["soil_temp"]

    ### Generation
    cop_dict = {}

    # Air-source
    for _, row in df_buildings.iterrows():
        hp_type = "ASHP"
        bus = row["bus"]
        # hp_type = row["hp_type"]
        heating_type = row["heating_type"]
        
        space_heat = pd.DataFrame(df_heat_space[bus, "space_heat"])
        water_heat = pd.DataFrame(df_heat_water[bus, "water_heat"])

        cop_dict[bus] = _get_cop(hp_type, heating_type, space_heat, water_heat, air_temp, soil_temp)
    cops_air = pd.concat([cop for _,cop in cop_dict.items()], axis=1)
    cops_air.columns = pd.MultiIndex.from_tuples([(col, "heatpump_air") for col in cop_dict.keys()])

    # # Ground-source
    # for _, row in df_buildings.iterrows():
    #     hp_type = "GSHP"
    #     bus = row["bus"]
    #     # hp_type = row["hp_type"]
    #     heating_type = row["heating_type"]
        
    #     space_heat = pd.DataFrame(df_heat_space[bus, "space_heat"])
    #     water_heat = pd.DataFrame(df_heat_water[bus, "water_heat"])

    #     cop_dict[bus] = _get_cop(hp_type, heating_type, space_heat, water_heat, air_temp, soil_temp)
    # cops_grd = pd.concat([cop for _,cop in cop_dict.items()], axis=1)
    # cops_grd.columns = pd.MultiIndex.from_tuples([(col, "heatpump_grd") for col in cop_dict.keys()])

    # cops = pd.concat([cops_air, cops_grd], axis=1)
    cops = cops_air
    return cops

def create_pro_heat(consumer_list):
    # Heatpump AIR
    df_pro_base = pd.DataFrame(consumer_list, columns=['Site'])
    df_pro_base[["Process","inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
        "heatpump_air", config.HP_AIR_INST_CAP, config.HP_AIR_CAP_UP, config.HP_AIR_INV_COST_FIX, config.HP_AIR_INV_COST, 
            config.HP_AIR_FIX_COST, config.HP_AIR_VAR_COST, config.HP_AIR_WACC, config.HP_AIR_DEPRECIATION, config.HP_AIR_PF_MIN
    )
    # # Heatpump GROUND
    # df_pro_grd = df_pro_base.copy()
    # df_pro_grd[["Process","inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
    #     "heatpump_grd", config.HP_GRD_INST_CAP, config.HP_GRD_CAP_UP, config.HP_GRD_INV_COST_FIX, config.HP_GRD_INV_COST, 
    #         config.HP_GRD_FIX_COST, config.HP_GRD_VAR_COST, config.HP_GRD_WACC, config.HP_GRD_DEPRECIATION, config.HP_GRD_PF_MIN
    # )
    # Heatpump booster
    df_pro_bst = df_pro_base.copy()
    df_pro_bst[["Process","inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
        "heatpump_booster", config.HP_BST_INST_CAP, config.HP_BST_CAP_UP, config.HP_BST_INV_COST_FIX, config.HP_BST_INV_COST, 
            config.HP_BST_FIX_COST, config.HP_BST_VAR_COST, config.HP_BST_WACC, config.HP_BST_DEPRECIATION, config.HP_BST_PF_MIN)
    # Heat dummy space
    df_pro_dmys = df_pro_base.copy()
    df_pro_dmys[["Process","inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
        "Heat_dummy_space", config.HDM_INST_CAP, config.HDM_CAP_UP, config.HDM_INV_COST_FIX, config.HDM_INV_COST, 
            config.HDM_FIX_COST, config.HDM_VAR_COST, config.HDM_WACC, config.HDM_DEPRECIATION, config.HDM_PF_MIN)
    # Heat dummy water
    df_pro_dmyw = df_pro_dmys.copy()
    df_pro_dmyw["Process"] = "Heat_dummy_water"

    # df_pro = pd.concat([df_pro_base, df_pro_grd, df_pro_bst, df_pro_dmys, df_pro_dmyw], axis=0)
    df_pro = pd.concat([df_pro_base, df_pro_bst, df_pro_dmys, df_pro_dmyw], axis=0)
    return df_pro.reset_index(drop=True)

def create_com_heat(consumer_list):
    df_com_base = pd.DataFrame(consumer_list, columns=['Site'])
    df_com_base[["Commodity","Type","price"]] = ("common_heat", "Stock", np.nan)

    df_com_sh = df_com_base.copy()
    df_com_sh[["Commodity","Type"]] = ("space_heat", "Demand")

    df_com_wh = df_com_base.copy()
    df_com_wh[["Commodity","Type"]] = ("water_heat", "Demand")

    df_com = pd.concat([df_com_base, df_com_sh, df_com_wh], axis=0)
    return df_com.reset_index(drop=True)

def create_pro_com_heat():
    # df_pro_com = pd.DataFrame({
    #     'Process':   ["Heat_dummy_space", "Heat_dummy_space", "Heat_dummy_water", "Heat_dummy_water", "heatpump_air", "heatpump_air", "heatpump_air","heatpump_grd", "heatpump_grd", "heatpump_grd", "heatpump_booster", "heatpump_booster"],
    #     'Commodity': ["common_heat", "space_heat", "common_heat", "water_heat", "electricity-reactive", "electricity", "common_heat", "electricity-reactive", "electricity", "common_heat", "electricity", "common_heat"],
    #     'Direction': ["In", "Out", "In", "Out", "In", "In", "Out",  "In", "In", "Out",  "In", "Out"],
    #     'ratio':     [1, 1, 1, 1, config.HP_AIR_Q_IN_RATIO, 1, 1, config.HP_GRD_Q_IN_RATIO, 1, 1, 1, 1]
    # })
    df_pro_com = pd.DataFrame({
        'Process':   ["Heat_dummy_space", "Heat_dummy_space", "Heat_dummy_water", "Heat_dummy_water", "heatpump_air", "heatpump_air", "heatpump_booster", "heatpump_booster"],
        'Commodity': ["common_heat",      "space_heat",       "common_heat",      "water_heat",       "electricity",  "common_heat",  "electricity",      "common_heat"],
        'Direction': ["In",               "Out",              "In",               "Out",              "In",           "Out",          "In",               "Out"],
        'ratio':     [1,                   1,                  1,                  1,                  1,             1,               1,                  1]
    })
    return df_pro_com.reset_index(drop=True)

def create_sto_heat(consumer_list):
    df_sto = pd.DataFrame(consumer_list, columns=['Site'])
    df_sto[["Storage","Commodity","inst-cap-c","cap-up-c","inst-cap-p","cap-up-p","eff-in","eff-out","discharge","ep-ratio",
            "inv-cost-p","inv-cost-c","fix-cost-p","fix-cost-c","var-cost-p","wacc","depreciation"]] = (
            "heat_storage", "common_heat", config.TS_INST_CAP_C, config.TS_CAP_UP_C, config.TS_INST_CAP_P, 
            config.TS_CAP_UP_P, config.TS_EFF_IN, config.TS_EFF_OUT, config.TS_DISCHARGE, config.TS_EP_RATIO,
            config.TS_INV_COST_P, config.TS_INV_COST_C, config.TS_FIX_COST_P, config.TS_FIX_COST_C,
            config.TS_VAR_COST_P, config.TS_WACC, config.TS_DEPRECIATION)
    return df_sto.reset_index(drop=True)