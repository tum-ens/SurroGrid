"""Small Plotly export utilities used by postprocessing notebooks."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def _apply_plotly_slider_step(fig: go.Figure, active_slider_step: int | str) -> None:
    sliders = fig.layout.sliders
    if not sliders:
        raise ValueError("active_slider_step was provided, but the Plotly figure has no sliders.")
    slider = sliders[0]
    steps = list(slider.steps)
    if isinstance(active_slider_step, str):
        labels = [str(step.label) for step in steps]
        try:
            step_index = labels.index(active_slider_step)
        except ValueError as exc:
            available = ", ".join(labels)
            raise ValueError(
                f"Unknown slider step label {active_slider_step!r}. Available labels: {available}."
            ) from exc
    else:
        step_index = int(active_slider_step)
    if step_index < 0 or step_index >= len(steps):
        raise IndexError(f"Slider step index {step_index} is outside [0, {len(steps) - 1}].")

    step = steps[step_index]
    args = list(step.args) if step.args is not None else []
    if args:
        trace_update = args[0] or {}
        visible = trace_update.get("visible") if isinstance(trace_update, dict) else None
        if visible is not None:
            for trace, is_visible in zip(fig.data, visible):
                trace.visible = bool(is_visible)
    if len(args) > 1 and isinstance(args[1], dict):
        fig.update_layout(args[1])
    fig.layout.sliders[0].active = step_index

def save_plotly_figure(
    fig: go.Figure,
    output_path: str | Path,
    formats: tuple[str, ...] = ("png", "svg"),
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
    active_slider_step: int | str | None = None,
) -> list[Path]:
    """Save a Plotly figure to one or more static image formats.

    ``output_path`` can be either a path without suffix, such as
    ``output/asset-percentiles``, or a concrete file path. Static Plotly export
    requires the ``kaleido`` package, which is included in the Step 5
    environment. For figures with sliders, ``active_slider_step`` selects the
    static state to export by zero-based step index or by exact step label.
    """
    export_fig = go.Figure(fig)
    if active_slider_step is not None:
        _apply_plotly_slider_step(export_fig, active_slider_step)

    output_path = Path(output_path)
    if output_path.suffix:
        base_path = output_path.with_suffix("")
        if not formats:
            formats = (output_path.suffix.lstrip("."),)
    else:
        base_path = output_path
    base_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image_format in formats:
        fmt = image_format.lower().lstrip(".")
        target = base_path.with_suffix(f".{fmt}")
        try:
            export_fig.write_image(str(target), format=fmt, width=width, height=height, scale=scale)
        except ValueError as exc:
            raise RuntimeError(
                "Static Plotly export failed. Make sure the Step 5 environment "
                "contains kaleido by running `uv sync` in GridExpand/5.postprocessing."
            ) from exc
        saved_paths.append(target)
    return saved_paths
