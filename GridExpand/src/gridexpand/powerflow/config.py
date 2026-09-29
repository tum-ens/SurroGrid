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

    ### Station voltage (station_voltage.py): the reference follows pylovo's validation power flow
    # (LV_REFERENCE_VOLTAGE_PU in config_generation.yaml; POWER_FLOW_VOLTAGE_LIMITS in
    # config_analysis.yaml), the DIN EN 50160 band split between MV and LV of Niederle et al. (2026).
    # The tap is GridExpand's first, free voltage measure; pylovo keeps it neutral.
    LV_REFERENCE_VOLTAGE_PU = 0.96  # LV busbar of the station
    MAX_TAP_STEPS = 2               # off-load tap steps towards a higher LV voltage
    TAP_STEP_PERCENT = 2.5          # per step, on the HV side as in the pandapower standard types
    MIN_VM_PU = 0.90                # a tap step is used while a bus is below it
    MAX_VM_PU = 1.10                # a tap step that pushes a bus above it is not used

config = Config()
