"""Station voltage of the Step 4 power flow: LV busbar reference and off-load tap.

Step 4 solves every grid with the external grid on the station's LV busbar; the MV/LV transformer
is represented by its rating only. The busbar voltage follows the convention of pylovo's validation
power flow (``pylovo/src/pylovo/station_voltage.py``), which splits the DIN EN 50160 band between MV
and LV as in Niederle et al. (2026):

1. the LV busbar sits at ``LV_REFERENCE_VOLTAGE_PU`` (0.96 p.u.);
2. while a bus is below ``MIN_VM_PU`` in any timestep, the off-load tap lifts the LV side by one more
   step, at most ``MAX_TAP_STEPS``. A step that pushes a bus above ``MAX_VM_PU`` in any timestep, or
   that adds non-converged timesteps, is not used.

An off-load tap cannot follow the load, so the tap is chosen per grid and stage over the whole
horizon; a post stage may use another tap than the status quo (re-tapping is the first, free voltage
measure). The pandapower standard types have the tap on the HV side, so ``k`` steps lift the LV
busbar to ``reference / (1 - k * step)``. The busbar voltage replaces whatever the stored grid holds
(pylovo stores the MV-side voltage of its own check, the real grids 1.0 p.u.).

The evaluated buses decide the tap: the voltage scope of the summary (``summary_grid_scope``).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandapower as pp

from gridexpand.powerflow.config import config
from gridexpand.powerflow.engine import PowerflowMatrices


@dataclass(frozen=True)
class StationVoltage:
    """Busbar voltage and tap of one solved stage."""

    lv_busbar_vm_pu: float
    tap_steps: int

    def as_summary(self) -> dict[str, float | int]:
        return {"lv_busbar_vm_pu": self.lv_busbar_vm_pu, "tap_steps": self.tap_steps}

    def describe(self) -> str:
        return f"LV busbar {self.lv_busbar_vm_pu:.4f} p.u., tap {self.tap_steps:+d} step(s)"


def assumptions() -> dict[str, float | int]:
    """The convention, for the run assumptions."""
    return {
        "lv_reference_voltage_pu": config.LV_REFERENCE_VOLTAGE_PU,
        "max_tap_steps": config.MAX_TAP_STEPS,
        "tap_step_percent": config.TAP_STEP_PERCENT,
        "min_vm_pu": config.MIN_VM_PU,
        "max_vm_pu": config.MAX_VM_PU,
    }


def busbar_voltage_pu(tap_steps: int) -> float:
    """LV busbar voltage after ``tap_steps`` steps of the HV-side off-load tap."""
    return config.LV_REFERENCE_VOLTAGE_PU / (1.0 - tap_steps * config.TAP_STEP_PERCENT / 100.0)


def voltage_extremes(matrices: PowerflowMatrices, voltage_buses) -> tuple[float, float]:
    """Minimum and maximum voltage of the evaluated buses over the converged timesteps (NaN if none)."""
    positions = matrices.bus_index.get_indexer(list(voltage_buses))
    values = matrices.vm_pu[:, positions[positions >= 0]]
    if values.size == 0 or np.isnan(values).all():
        return float("nan"), float("nan")
    return float(np.nanmin(values)), float(np.nanmax(values))


def solve(grid, voltage_buses, run: Callable[[object], PowerflowMatrices]) -> tuple[PowerflowMatrices, StationVoltage]:
    """Solve at the LV reference voltage and lift the tap as long as a bus is below the band.

    Args:
        grid: prepared net with the external grid on the LV busbar (not modified).
        voltage_buses: evaluated buses; their extremes over all timesteps decide the tap.
        run: solves the time series of a net (``engine.run_timeseries`` with the caller's options).

    Returns:
        The matrices of the chosen tap and its :class:`StationVoltage`.

    Raises:
        ValueError: the external grid does not sit on an LV bus.
    """
    grid = deepcopy(grid)
    if grid.ext_grid.empty or (grid.bus.loc[grid.ext_grid["bus"], "vn_kv"] > 1.0).any():
        raise ValueError("The station voltage needs the external grid on the LV busbar.")
    grid.ext_grid["vm_pu"] = busbar_voltage_pu(0)
    matrices = run(grid)
    steps = 0
    while steps < config.MAX_TAP_STEPS:
        low, _ = voltage_extremes(matrices, voltage_buses)
        if not low < config.MIN_VM_PU:  # within the band, or no converged timestep
            break
        grid.ext_grid["vm_pu"] = busbar_voltage_pu(steps + 1)
        try:
            candidate = run(grid)
        except pp.LoadflowNotConverged:
            break
        _, high = voltage_extremes(candidate, voltage_buses)
        if len(candidate.failed) > len(matrices.failed) or high > config.MAX_VM_PU:
            break
        matrices, steps = candidate, steps + 1
    return matrices, StationVoltage(busbar_voltage_pu(steps), steps)
