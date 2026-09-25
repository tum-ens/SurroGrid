from gridexpand.paths import POWERFLOW_INPUT_DIR, POWERFLOW_OUTPUT_DIR


class Config():
    ### Data directories (absolute; see gridexpand.paths)
    DATA_DIR = f"{POWERFLOW_INPUT_DIR}/"
    STORAGE_DIR = f"{POWERFLOW_OUTPUT_DIR}/"

    ### Power factors
    # Assumed PV compensation bound: |Q| <= P_PV * tan(arccos(PF_PV_MIN)).
    # This is a model assumption, not a voltage-dependent inverter controller.
    PF_PV_MIN = 0.95
    PF_HP = 0.95        # Heat pump,     source: example data sheet - https://www.solarwatt.de/canto/download/bnu1pcavot0oh2bem4qpi4k63i
    PF_ELC = 0.959      # Electricity,   source: https://www.researchgate.net/publication/285577915_Representative_electrical_load_profiles_of_residential_buildings_in_Germany_with_a_temporal_resolution_of_one_second

config = Config()
