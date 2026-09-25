> **Historical record** (not maintained). Run: one full-year Step 2 realization of Forchheim grid
> `9474126-07_91301_1_12`, `post-hems-heuristic`, residential scope (run id not recorded). pylovo version: not
> recorded. Written: 2026-08-14. Moved from
> `docs/scenario_pipeline/SCENARIO_METHOD.md` (section "Space-heating buffer") on 2026-09-25. The heat
> profiles of that run predate the per-building seeding of TEASER occupancy and hot water (2026-09-25), so a
> re-run gives different aggregate values; the sizing rules are maintained in
> [method.md](../method.md#residential-heat-assets). See [the documentation index](../README.md) for the
> current code.

# Forchheim heat-asset pilot (one grid, full year)

One implemented full-year realization on Forchheim grid
`9474126-07_91301_1_12` (`post-hems-heuristic`, residential scope) produced 20
central systems and 477.874 MWhth/a of useful heat. Its checks found 2,619.23
full-load hours, 39.221 kWel of fixed HP capacity, 258.182 kWel of fixed
auxiliary capacity, 1,906.4 litres (11.086 kWhth) of explicit space-heating
buffer, an exact 20 l/kWth ratio, retained within-day OpenDHW variation, and
zero uncovered heat in INFLEX dispatch. The coincident auxiliary peak was
216.587 kW and the auxiliary heater supplied 10.54% of annual useful heat. The
high auxiliary peak is consistent with the documented absence of DHW-tank
buffering. Building design-load intensities
ranged from 13.7 to 109.6 W/m2 (median 53.1 W/m2), which is retained in the
audit for outlier review rather than silently clipped. Deliberately changing the
run-level profile seed can shift these aggregate pilot values; the sizing and coverage
invariants are the acceptance criteria. The optimized-mode check wrote
zero installed capacity, positive costs, and finite building-specific bounds
for HP, auxiliary heater, and buffer.
