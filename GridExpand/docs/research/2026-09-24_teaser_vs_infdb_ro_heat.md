> **Historical record** (not maintained). Run: TEASER (retrofit level 0) versus INFDB `ro_heat` heat demand on
> 1,203 common SWF buildings (run id not recorded). pylovo version: not recorded. Written: 2026-09-24. Moved
> from `docs/scenario_pipeline/SCENARIO_METHOD.md` (section "TEASER refurbishment level") on 2026-09-25; the
> refurbishment-level choice itself is maintained in [method.md](../method.md#teaser-refurbishment-level). See
> [the documentation index](../README.md) for the current code.

# TEASER versus INFDB `ro_heat` space heat

**Comparison with INFDB `ro_heat`** (checked in `infdb/tools/ro-heat`, September 2026):

- *Refurbishment:* simulated per component (wall, roof, window) from lifespans
  (40/50/30 a, σ 10 a) up to 2023. It is capped at 33 %, 63 % and 90 % of
  buildings, respectively.
- *Envelope:* uses real LoD2 wall and roof areas and applies a heated-area
  ratio of 0.8.
- *Model:* the heating-load series come from an EnTiSe RC model with
  ventilation losses and solar gains set to zero.
- *Weather:* the actual 2023 Open-Meteo year, not a TMY.
- *Results:* on 1,203 common SWF buildings, TEASER at level 0 is about 1.9×
  INFDB in annual space heat and about 3× in peak. The ratio is still about
  1.65 for buildings from 2010 onwards. Missing ventilation losses are a
  plausible reason for this residual gap; that is not decomposed.
- *Consequence:* INFDB is therefore not used as the reference for peaks.
