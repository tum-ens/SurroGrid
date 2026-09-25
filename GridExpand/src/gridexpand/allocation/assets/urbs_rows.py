"""urbs process and storage row builders shared by the asset materializations."""

from __future__ import annotations


def process_row(site, name, installed, upper, *, fixed, parameters) -> dict:
    """Return one urbs ``process`` row.

    Args:
        site: urbs site (bus).
        name: Process name.
        installed: Installed capacity in kW.
        upper: Capacity upper bound in kW.
        fixed: Fixed (heuristic) assets carry no investment cost.
        parameters: ``technologies.processes[<name>]`` of the scenario YAML.
    """
    return {
        "Site": int(site),
        "Process": name,
        "inst-cap": float(installed),
        "cap-up": float(upper),
        "inv-cost-fix": 0.0 if fixed else parameters["fixed_investment_cost_eur"],
        "inv-cost": 0.0 if fixed else parameters["investment_cost_eur_per_kw"],
        "fix-cost": parameters["fixed_cost_eur_per_hour"],
        "var-cost": parameters["variable_cost_eur_per_kwh"],
        "wacc": parameters["wacc"],
        "depreciation": parameters["depreciation_years"],
        "pf-min": parameters["minimum_power_factor"],
    }


def storage_parameter_fields(parameters, *, fixed, ep_ratio) -> dict:
    """Return the efficiency and cost columns of one urbs ``storage`` row.

    The keys follow the capacity columns (``inst-cap-c`` ... ``cap-up-p``) in
    urbs column order.

    Args:
        parameters: ``technologies.storages[<name>]`` of the scenario YAML.
        fixed: Fixed (heuristic) assets carry no investment cost.
        ep_ratio: Energy-to-power ratio written to ``ep-ratio``.
    """
    return {
        "eff-in": parameters["charge_efficiency"],
        "eff-out": parameters["discharge_efficiency"],
        "discharge": parameters["self_discharge_per_timestep"],
        "ep-ratio": ep_ratio,
        "inv-cost-p": 0.0 if fixed else parameters["investment_cost_eur_per_kw"],
        "inv-cost-c": 0.0 if fixed else parameters["investment_cost_eur_per_kwh"],
        "fix-cost-p": 0.0 if fixed else parameters["fixed_investment_cost_power_eur"],
        "fix-cost-c": 0.0 if fixed else parameters["fixed_investment_cost_energy_eur"],
        "var-cost-p": parameters["variable_cost_eur_per_kwh"],
        "wacc": parameters["wacc"],
        "depreciation": parameters["depreciation_years"],
    }
