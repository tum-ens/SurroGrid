> **Historical record** (not maintained). Run: none (code inspection of commit `7228308`). pylovo version: not applicable. Written: 2026-09-10. Moved from `docs/powerflow_summary.md` on 2026-09-25. Paths, commands and module names below are those of the code at that time; see [the documentation index](../README.md) for the current code.

# Original reactive-power sign error and implications for voltage results

## Scope and conclusion

This note concerns only the original implementation inspected on local `main`, commit `722830874dcff693dee80b9475e1a555d1f76f81`. It explains the interface between URBS dispatch, reactive-demand reconstruction and pandapower. No scenario or power-flow run was performed for this note.

**The original code assigns negative reactive power to household and heat-pump demands intended to be inductive, then passes those values to pandapower load elements without changing their signs. Pandapower interprets them as reactive injection, not inductive absorption. This is a physical modeling error, not merely a different plotting convention.**

The error is confirmed in the inspected source. Establishing its numerical effect on the published paper requires identifying the exact archived code, dependency versions, input networks and outputs used for the publication. The inspected branch alone does not establish the magnitude of that effect.

All repository paths and line numbers below refer to the pinned commit, **not to the subsequently edited working tree**. Historical source can be inspected without switching branches:

```bash
git show 722830874dcff693dee80b9475e1a555d1f76f81:GridExpand/4.powerflow/src/demands.py
```

## 1. URBS balance signs versus electrical signs

URBS models process throughput and commodity conservation. In `GridExpand/3.urbs/urbs/model.py:350`, `tau_pro` and `e_pro_in` are nonnegative; `e_pro_out` is declared real and related to throughput by the process equations at lines 759 and 765:

```text
e_pro_in  = tau_pro * r_in
e_pro_out = tau_pro * r_out
```

The commodity balance in `urbs/features/modelhelper.py:118` counts process consumption positively and production negatively. The vertex constraint in `urbs/model.py:612` then uses:

```text
commodity_balance = process inputs - process outputs + storage balance
0 = -commodity_balance + purchases - sales - demand
```

Purchases and sales enter through `urbs/features/BuySellPrice.py:172`. These signs identify sources and sinks in an accounting equation. A demand appearing with a minus sign in that equation does **not** imply that an inductive appliance must have negative Q in a power-flow load element.

For the inspected scenario-construction path, `GridExpand/2.demand_allocation/gridalloc/src/functions/electricity.py:265` leaves the reactive commodity commented out. The unit-ratio import/feed-in definitions are at line 282. Legacy URBS reactive-output constraints are also commented out at `urbs/model.py:474`.

Consequently, this path does not pass an optimized URBS reactive-power dispatch to pandapower. It reconstructs Q afterward from active component powers and assumed power factors. The error is in that reconstruction/interface, not a requirement imposed by URBS.

## 2. The original reconstruction and adapter

Exact locations in `GridExpand/4.powerflow/`:

| Source location | Original operation |
| --- | --- |
| `src/demands.py:35`, `_process_pre_demands` | `Q_HH = -P_HH * tan(acos(PF_ELC))` |
| `src/demands.py:61`, `_extract_relevant_demands` | `P_net = tau_import - tau_feed_in` |
| `src/demands.py:86`, `_obtain_post_reactive_power` | `Q_HP = -P_HP * tan(acos(PF_HP))` |
| `src/demands.py:91` | PV limit and local reactive compensation |
| `src/powerflow.py:89`, `run_full_pf` | Rename reactive demand to `q_mvar`; divide P and Q by 1,000 |
| `src/powerflow.py:63`, `run_single_pf` | Assign these values to `net.load` and call `pp.runpp` |
| `src/powerflow.py:97` | Store calculated bus voltage magnitude `res_bus.vm_pu` |

The original `config.py:7` sets `PF_ELC=0.959`, `PF_HP=0.95` and `PF_PV_MIN=0.95`.

The adapter therefore supplies:

```text
load.p_mw   = P_net_kW / 1000
load.q_mvar = Q_net_kvar / 1000
```

There is **no Q-sign reversal at the interface**. Positive active consumption and negative reconstructed Q describe active consumption combined with reactive injection.

## 3. What pandapower means by those values

For current convention documentation, see pandapower's versioned [load documentation](https://pandapower.readthedocs.io/en/v3.1.2/elements/load.html) and [static-generator documentation](https://pandapower.readthedocs.io/en/v3.1.2/elements/sgen.html). These references establish the convention; they do not establish which dependency version produced the paper.

| Element convention | Positive P | Positive Q |
| --- | --- | --- |
| `load`: consumer convention | Active consumption | Reactive absorption |
| `sgen` / `ext_grid`: generator convention | Active injection | Reactive injection |

For current directed into a consumer:

```text
S = V * conjugate(I) = P + jQ
Q_inductive = +P * tan(acos(power_factor))
```

Lagging current gives positive Q; capacitive behavior gives negative Q. Reversing Q while leaving P unchanged changes the physical load, rather than consistently switching reference conventions.

The documented load equation is `Q_load = q_mvar * scaling * (p_const + z_const*V² + i_const*V)`. The multiplier does not turn negative input Q into positive inductive absorption. For constant-power loads with scaling 1, Q equals the supplied setpoint.

The code comments describing negative Q as inductive are therefore incompatible with the element actually receiving the data. They suggest a sign-convention misunderstanding, but do not prove the historical reasoning that produced it.

## 4. PV compensation also changes direction

Let:

```text
q = P_HH*tan(acos(PF_ELC)) + P_HP*tan(acos(PF_HP)) >= 0
c = P_PV*tan(acos(PF_PV_MIN)) >= 0
```

For finite, correctly aligned component data, the original code implements:

```text
Q_appliances_original = -q
Q_PV_original        = clip(+q, -c, +c)
Q_net_original       = -max(q-c, 0)
```

Keeping the same local compensation policy but correcting the consumer convention gives:

```text
Q_appliances_correct = +q
Q_PV_correct         = clip(-q, -c, +c)
Q_net_correct        = +max(q-c, 0)
```

Thus the original PV contribution represents absorption that offsets fictitious capacitive appliances. The intended PV contribution is injection that offsets inductive appliances. In this net-load representation, that intended PV contribution is negative; a separate generator supplying it would use positive generator Q.

Example: 10 kW household demand, 5 kW heat-pump demand and 4 kW PV give appliance magnitude 4.599 kvar and PV compensation limit 1.315 kvar. Original net Q is **-3.284 kvar**; corrected net Q is **+3.284 kvar**.

Full local compensation makes net Q zero in both formulations. With no PV generation, compensation is zero and the full appliance-Q sign error remains. The policy is a generation-dependent local compensation assumption, not a voltage controller or proof of inverter compliance.

## 5. Why voltage results can change even when |Q| does not

For a simple radial branch, in consistent per-unit quantities and neglecting losses to first order:

```text
V_downstream ≈ V_upstream - (R*P + X*Q)/V_upstream
```

P and Q here are downstream demand flows, positive toward consumption. With positive branch reactance X and residual inductive demand q_res:

```text
V_original - V_correct ≈ 2*X*q_res/V_upstream
```

This is a first-order explanation, not a numerical correction formula for the published networks. On a feeder, the corresponding contributions accumulate along the supply path.

Under fixed upstream voltage and otherwise unchanged conditions, the original negative Q tends to:

- Counteract voltage drop during active-power import, potentially hiding undervoltage.
- Reinforce voltage rise during active-power export, potentially exaggerating overvoltage.

This does not imply every bus or every metric changes monotonically in a complete AC network. Branch losses, shunts, controls, topology and operating points matter. Where PV compensation removes all residual reconstructed Q, this particular sign error has no local net-Q effect.

Although `sqrt(P²+Q²)` is unchanged by a pure sign reversal at a fixed load setpoint, bus voltages and network flows need not be. Accordingly, unchanged apparent-power magnitudes are not evidence that voltage, line-current or loss results are unaffected.

Nor can errors be assumed to cancel between published scenarios: their residual Q, PV compensation, spatial demand distributions and critical hours may differ.

## 6. Assessing the publication consequences

A controlled comparison should preserve the original study and change only this sign convention:

1. Identify the exact publication code revision, dependencies, networks, dispatch and timestep selection.
2. Reproduce the original outputs first.
3. Correct household/HP Q and recompute PV compensation under the same policy; keep active dispatch and all other assumptions unchanged.
4. Recalculate AC power flow. Compare bus-by-hour voltage differences, voltage extrema, violation counts/durations and which buses or scenarios cross the paper's thresholds.
5. Reassess the specific figures and conclusions supported by those metrics.

Preserve the original grid preparation too. In the inspected source, `src/powerflow.py:42` removes the transformer and replaces it with a switch. Introducing transformer impedance while correcting Q would confound the comparison.

For this reconstructed-Q path, saved active dispatch can normally be reused; correcting Q alone does not require re-optimizing URBS. Verify that the publication workflow did not feed power-flow outcomes back into optimization or scenario selection.

No magnitude of voltage bias, number of affected violations, or change to the paper's conclusions has been established by this source inspection. The stored voltage magnitudes cannot be repaired by relabeling Q or negating a plotted quantity.

## Suggested explanation

“Inspection of the original power-flow interface identified a reactive-power sign error. Household and heat-pump demands intended to represent inductive absorption were assigned negative reactive power and passed directly to pandapower load elements, where negative reactive power denotes injection. URBS commodity-balance signs do not require this inversion. The local photovoltaic compensation consequently operated with the reversed reactive direction. This changes the reactive contribution to voltage drop or rise even where the magnitude of apparent demand is unchanged. Its quantitative impact on the reported voltage results must be determined by a controlled recalculation with the original study inputs.”
