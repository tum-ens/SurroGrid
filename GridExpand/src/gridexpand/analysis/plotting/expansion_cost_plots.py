"""Expansion-cost plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap

from matplotlib.colors import to_rgb, to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Analyses written before migration 0007 have Cables and Transformers only; later ones split the
# station-level cost.
COMPONENT_ORDER = (
    "Cables", "Transformers", "Transformer exchange", "Load transfer", "New substations", "Voltage measures",
)
# The small station-level measures, plotted together as "Other" (the caption names them).
OTHER_COMPONENTS = ("Load transfer", "Voltage measures")


# Short network names under the total bars (a full "Synthetic" is wider than its bar slot).
SOURCE_TICK_LABELS = {"Synthetic": "Syn.", "Real": "Real"}
# Fill opacity and outline of each network, as the violins of the power-flow overview: Real lighter and dashed.
_BAR_SOURCE_STYLES = (
    {"alpha": 0.85, "linestyle": "-"},
    {"alpha": 0.30, "linestyle": (0, (3.0, 1.6))},
    {"alpha": 0.55, "linestyle": "-."},
    {"alpha": 0.15, "linestyle": ":"},
)
_MILLION_EUR = "Expansion cost [M\u20ac]"


def _cost_table(
    expansion_cost_comparison: pd.DataFrame, stage_order: Sequence[str], source_order: Sequence[str]
) -> tuple[pd.Series, list[str], list[str], list[str]]:
    """M€ per (network, case, component) with ``OTHER_COMPONENTS`` as "Other", and the networks, cases and
    components in plot order."""
    cost_data = expansion_cost_comparison.copy()
    if "data_source" not in cost_data.columns:
        cost_data["data_source"] = "Synthetic"
    cost_data["data_source"] = cost_data["data_source"].astype(str)
    cost_data["component"] = cost_data["component"].where(~cost_data["component"].isin(OTHER_COMPONENTS), "Other")
    present_sources = list(dict.fromkeys(cost_data["data_source"]))
    sources = [source for source in source_order if source in present_sources]
    sources += [source for source in present_sources if source not in sources]
    stages = [stage for stage in stage_order if stage in set(cost_data["stage"])]
    order = [component for component in COMPONENT_ORDER if component not in OTHER_COMPONENTS] + ["Other"]
    components = [component for component in order if component in set(cost_data["component"])]
    million_eur = cost_data.groupby(["data_source", "stage", "component"])["cost_eur"].sum().div(1_000_000.0)
    return million_eur, sources, stages, components


def _print_rc(font_size: float) -> dict[str, object]:
    return {
        "font.size": font_size, "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1, "legend.fontsize": font_size,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    }


def _total_label(value: float, base: float | None) -> str:
    """Cost of a total bar, with its change against ``base`` when given."""
    text = f"{value:.2f}"
    if base:
        text += f"\n{(value - base) / base * 100:+.0f} %".replace("-", "\u2212")
    return text


def _label_case_axis(ax, stage_names: list[str], source_ticks: dict[float, str], font_size: float) -> None:
    """Case names under the clusters, the networks under their bars."""
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(["\n".join(textwrap.wrap(name, 12)) for name in stage_names])
    if len(set(source_ticks.values())) > 1:
        ax.set_xticks(list(source_ticks), minor=True)
        ax.set_xticklabels(list(source_ticks.values()), minor=True, fontsize=font_size - 1.5, color="#555555")
        ax.tick_params(axis="x", which="major", pad=font_size + 4)
    ax.set_xlim(-0.6, len(stage_names) - 0.4)


def _style_axes(ax) -> None:
    ax.set_ylabel(_MILLION_EUR)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, save_path: str | Path | None, save_formats: tuple[str, ...]) -> None:
    if save_path is None:
        return
    base_path = Path(save_path)
    base_path = base_path.with_suffix("") if base_path.suffix else base_path
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for image_format in save_formats:
        fig.savefig(base_path.with_suffix(f".{image_format.lstrip('.')}"), bbox_inches="tight")


def plot_expansion_cost_overview_static(
    expansion_cost_comparison: pd.DataFrame,
    *,
    stage_order: Sequence[str],
    color_map: Mapping[str, str],
    stage_labels: Mapping[str, str] | None = None,
    asset_stage_order: Sequence[str] | None = None,
    source_order: Sequence[str] = ("Synthetic", "Real"),
    source_tick_labels: Mapping[str, str] | None = None,
    reduction_stages: tuple[str, str] | None = None,
    value_labels: bool = True,
    title: str | None = None,
    width_mm: float = 180.0,
    height_mm: float | None = None,
    font_size: float = 8.0,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("pdf", "svg"),
):
    """Publication figure of the expansion cost: total per case (left) and per asset (right).

    Colors follow the cases (``color_map``), the networks (``data_source``) sit side by side
    in each case: the first of ``source_order`` solid, the next lighter with a dashed
    outline, as in ``plot_powerflow_asset_cutoff_overview_static``. The total panel labels
    every bar with its cost and the networks under the bars (``source_tick_labels``);
    ``reduction_stages=(from, to)`` adds the change of the total from ``from`` to ``to``
    to the ``to`` bars. ``asset_stage_order`` limits the asset panel to these cases (default:
    all; e.g. without a status quo that costs nothing). ``value_labels`` labels the asset
    bars as well. ``OTHER_COMPONENTS`` form one "Other" asset.

    Input: the ``costs`` frame of ``expansion_cost_comparison`` (``stage``, ``data_source``,
    ``component``, ``cost_eur``). Sized for print like the power-flow overview.
    """
    if expansion_cost_comparison.empty:
        print("No expansion cost summaries available yet.")
        return None
    million_eur, sources, stages, components = _cost_table(expansion_cost_comparison, stage_order, source_order)
    asset_stages = [stage for stage in (asset_stage_order or stages) if stage in stages]
    totals = million_eur.groupby(level=["data_source", "stage"]).sum()
    stage_labels = stage_labels or {}
    tick_labels = {**SOURCE_TICK_LABELS, **(source_tick_labels or {})}
    styles = {source: _BAR_SOURCE_STYLES[index % len(_BAR_SOURCE_STYLES)] for index, source in enumerate(sources)}

    def _stage_label(stage: str) -> str:
        return str(stage_labels.get(stage, stage)).replace("status-quo", "status quo")

    def _bar(ax, x, value, stage, source, width):
        style = styles[source]
        color = color_map.get(stage, "#777777")
        ax.bar(x, value, width=width, color=to_rgba(color, style["alpha"]), edgecolor=color,
               linewidth=0.8, linestyle=style["linestyle"])

    def _value(source, stage, component=None) -> float:
        key = (source, stage) if component is None else (source, stage, component)
        series = totals if component is None else million_eur
        return float(series.get(key, 0.0))

    width_in = width_mm / 25.4
    size = (width_in, (height_mm / 25.4) if height_mm else width_in * 0.42)
    label_kw = {"fontsize": font_size - 1.5, "color": "#333333", "ha": "center", "va": "bottom"}
    with plt.rc_context(_print_rc(font_size)):
        fig, (ax_total, ax_assets) = plt.subplots(
            1, 2, figsize=size, gridspec_kw={"width_ratios": [1.0, 2.5]},
        )
        # Total: one cluster per case, the networks side by side and labeled under the bars.
        slot = 0.8 / max(1, len(sources))
        source_ticks: dict[float, str] = {}
        total_max = max([_value(source, stage) for source in sources for stage in stages] + [0.0])
        for stage_index, stage in enumerate(stages):
            for source_index, source in enumerate(sources):
                x = stage_index + (source_index - (len(sources) - 1) / 2) * slot
                value = _value(source, stage)
                _bar(ax_total, x, value, stage, source, slot * 0.9)
                source_ticks[x] = tick_labels.get(source, source)
                base = _value(source, reduction_stages[0]) if reduction_stages and stage == reduction_stages[1] else None
                ax_total.text(x, value + total_max * 0.015, _total_label(value, base), linespacing=1.1, **label_kw)
        _label_case_axis(ax_total, [_stage_label(stage) for stage in stages], source_ticks, font_size)
        ax_total.set_ylim(0, total_max * 1.22 if total_max else 1.0)
        ax_total.set_title("Total", fontweight="bold")

        # Per asset: one cluster per component, the cases in order, the networks side by side in each case.
        n_bars = len(asset_stages) * len(sources)
        bar_slot = 0.84 / max(1, n_bars + 0.5 * (len(asset_stages) - 1))
        asset_max = max([_value(source, stage, component) for source in sources for stage in asset_stages
                         for component in components] + [0.0])
        for component_index, component in enumerate(components):
            left = component_index - 0.42
            for stage_index, stage in enumerate(asset_stages):
                for source_index, source in enumerate(sources):
                    offset = (stage_index * (len(sources) + 0.5) + source_index + 0.5) * bar_slot
                    value = _value(source, stage, component)
                    _bar(ax_assets, left + offset, value, stage, source, bar_slot * 0.88)
                    if value_labels and value >= 0.005:
                        ax_assets.text(left + offset, value + asset_max * 0.015, f"{value:.2f}", rotation=90,
                                       **label_kw)
        ax_assets.set_xticks(range(len(components)))
        ax_assets.set_xticklabels(["\n".join(textwrap.wrap(component, 12)) for component in components])
        ax_assets.set_xlim(-0.6, len(components) - 0.4)
        ax_assets.set_ylim(0, asset_max * 1.25 if asset_max else 1.0)
        ax_assets.set_title("By asset", fontweight="bold")
        for ax in (ax_total, ax_assets):
            _style_axes(ax)

        handles = [Patch(facecolor=to_rgba(color_map.get(stage, "#777777"), 0.85), edgecolor="none",
                         label=_stage_label(stage)) for stage in stages]
        if len(sources) > 1:
            handles += [Patch(facecolor=to_rgba("#555555", styles[source]["alpha"]), edgecolor="#555555",
                              linewidth=0.8, linestyle=styles[source]["linestyle"], label=source)
                        for source in sources]
        fig.tight_layout(w_pad=1.5)
        top = max(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
                  for ax in (ax_total, ax_assets))
        legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                            bbox_to_anchor=(0.5, top), handlelength=1.6, columnspacing=1.4, borderaxespad=0.2)
        if title:
            top = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
            fig.suptitle(title, y=top, va="bottom", fontweight="bold")
        _save(fig, save_path, save_formats)
    return fig


def plot_expansion_cost_stacked_static(
    expansion_cost_comparison: pd.DataFrame,
    *,
    stage_order: Sequence[str],
    color_map: Mapping[str, str],
    stage_labels: Mapping[str, str] | None = None,
    source_order: Sequence[str] = ("Synthetic", "Real"),
    reduction_stages: tuple[str, str] | None = None,
    width_mm: float = 120.0,
    height_mm: float | None = None,
    font_size: float = 8.0,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("pdf", "svg"),
):
    """Variant of ``plot_expansion_cost_overview_static``: only the total bars, stacked by asset.

    One cluster per case, the networks side by side in ``source_order`` and labeled under the
    bars. Each bar stacks the assets from cables at the bottom to "Other" at the top, in shades
    of the case color from full to light. The legend on top splits each asset's swatch into the
    shades of the cases with visible bars (at least 1 % of the largest total). Segments tall
    enough for it carry their cost, the bars their total. With ``reduction_stages=(from, to)``,
    a ``to`` bar below the ``from`` total of its network gets a dotted box up to that total and
    an arrow with the relative reduction; the boxes share the label "Reduction through <to>".
    """
    if expansion_cost_comparison.empty:
        print("No expansion cost summaries available yet.")
        return None
    million_eur, sources, stages, components = _cost_table(expansion_cost_comparison, stage_order, source_order)
    totals = million_eur.groupby(level=["data_source", "stage"]).sum()
    stage_labels = stage_labels or {}
    shades = np.linspace(1.0, 0.22, len(components))

    def _tint(color: str, share: float) -> tuple[float, float, float]:
        return tuple(share * channel + (1.0 - share) for channel in to_rgb(color))

    width_in = width_mm / 25.4
    size = (width_in, (height_mm / 25.4) if height_mm else width_in * 0.62)
    with plt.rc_context(_print_rc(font_size)):
        fig, ax = plt.subplots(figsize=size)
        slot = 0.8 / max(1, len(sources))
        total_max = max([float(value) for value in totals] + [0.0])
        source_ticks: dict[float, str] = {}
        reductions: list[tuple[float, float]] = []
        for stage_index, stage in enumerate(stages):
            color = color_map.get(stage, "#777777")
            for source_index, source in enumerate(sources):
                x = stage_index + (source_index - (len(sources) - 1) / 2) * slot
                bottom = 0.0
                for component, shade in zip(components, shades):
                    value = float(million_eur.get((source, stage, component), 0.0))
                    if value <= 0:
                        continue
                    ax.bar(x, value, bottom=bottom, width=slot * 0.9, color=_tint(color, shade), edgecolor="white",
                           linewidth=0.6)
                    if value >= total_max * 0.06:
                        ax.text(x, bottom + value / 2, f"{value:.2f}", ha="center", va="center",
                                fontsize=font_size - 2, color="white" if shade > 0.9 else "#333333")
                    bottom += value
                ax.text(x, bottom + total_max * 0.015, f"{bottom:.2f}", ha="center", va="bottom",
                        fontsize=font_size - 1.5, color="#333333")
                source_ticks[x] = SOURCE_TICK_LABELS.get(source, source)
                base = float(totals.get((source, reduction_stages[0]), 0.0)) if (
                    reduction_stages and stage == reduction_stages[1]) else 0.0
                if base > bottom:
                    # The saved cost as a dotted box on the bar, the arrow from the `from` total ends above the total.
                    ax.bar(x, base - bottom, bottom=bottom, width=slot * 0.9, fill=False, edgecolor="#666666",
                           linewidth=0.8, linestyle=":")
                    arrow_end = bottom + total_max * 0.08
                    ax.annotate("", xy=(x, arrow_end), xytext=(x, base),
                                arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.8,
                                            "shrinkA": 0, "shrinkB": 0})
                    ax.text(x, (base + arrow_end) / 2, f"{(bottom - base) / base * 100:+.0f} %".replace("-", "\u2212"),
                            ha="center", va="center", fontsize=font_size - 1, fontweight="bold", color="#333333",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
                    reductions.append((x, base))
        if reductions:
            to_label = str(stage_labels.get(reduction_stages[1], reduction_stages[1]))
            ax.text(float(np.mean([x for x, _ in reductions])), max(base for _, base in reductions) + total_max * 0.03,
                    f"Reduction through {to_label}", ha="center", va="bottom", fontsize=font_size - 1,
                    color="#333333")
        stage_names = [str(stage_labels.get(stage, stage)).replace("status-quo", "status quo") for stage in stages]
        _label_case_axis(ax, stage_names, source_ticks, font_size)
        ax.set_ylim(0, total_max * 1.18 if total_max else 1.0)
        _style_axes(ax)

        # One swatch per asset, split into the shades of the cases with visible bars, left to right as in the axis.
        shown = [stage for stage in stages
                 if max(float(totals.get((source, stage), 0.0)) for source in sources) >= total_max * 0.01] or stages
        handles = [tuple(Patch(facecolor=_tint(color_map.get(stage, "#777777"), shade), edgecolor="none")
                         for stage in shown) for shade in shades]
        fig.tight_layout()
        top = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
        fig.legend(handles=handles, labels=components, loc="lower center", ncol=len(components), frameon=False,
                   bbox_to_anchor=(0.5, top), handlelength=2.4, columnspacing=1.4, borderaxespad=0.2,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
        _save(fig, save_path, save_formats)
    return fig
