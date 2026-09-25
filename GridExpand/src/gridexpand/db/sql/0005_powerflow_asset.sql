-- 0005 building assets simulated by a power-flow run (map layer of the GridExpand UI).
--
-- Step 4 writes one row per bus and technology from the Step 3 result it reads
-- (urbs cap_pro / cap_sto_c / cap_sto_p): the capacities the power flow used,
-- chosen by the heuristic rules or by the optimisation. Only post cases have
-- rows; capacities below 1 W / 1 Wh are left out. PV rows carry the building
-- (one PV process per building); the other technologies are sized per bus.
CREATE TABLE surrogrid.powerflow_asset (
    powerflow_run_id bigint NOT NULL REFERENCES surrogrid.powerflow_run (powerflow_run_id) ON DELETE CASCADE,
    bus integer NOT NULL,
    technology text NOT NULL,
    building_objectid text NOT NULL DEFAULT '',
    units integer NOT NULL DEFAULT 1,
    power_kw double precision,
    energy_kwh double precision,
    CONSTRAINT pk_powerflow_asset PRIMARY KEY (powerflow_run_id, bus, technology, building_objectid),
    CONSTRAINT ck_powerflow_asset_technology CHECK (technology IN ('pv', 'battery', 'heat_pump', 'heating_rod', 'heat_storage', 'ev'))
);
