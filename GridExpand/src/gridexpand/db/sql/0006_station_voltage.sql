-- 0006 station voltage of the Step 4 power flow (powerflow/station_voltage.py).
--
-- Step 4 sets the station's LV busbar to 0.96 p.u. and lifts it with up to two
-- off-load tap steps of 2.5 % while a bus is below 0.90 p.u., never beyond
-- 1.10 p.u. (the convention of pylovo's validation power flow). Each summary
-- row records the busbar voltage and the tap of its stage. Rows written before
-- this migration stay NULL: those runs solved with the external grid at the
-- voltage stored in the grid (1.0 p.u. for the real grids and for pylovo
-- versions generated before the convention).
ALTER TABLE surrogrid.powerflow_summary
    ADD COLUMN IF NOT EXISTS lv_busbar_vm_pu double precision,
    ADD COLUMN IF NOT EXISTS tap_steps integer;
ALTER TABLE surrogrid.real_powerflow_summary
    ADD COLUMN IF NOT EXISTS lv_busbar_vm_pu double precision,
    ADD COLUMN IF NOT EXISTS tap_steps integer;
