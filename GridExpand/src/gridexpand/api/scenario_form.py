"""Scenario editor: curated form fields, comment-preserving edits and validation of scenario YAMLs.

The editor never changes a file in place. :func:`preview` applies form changes (dotted YAML paths of
:data:`SECTIONS`) and/or a whole edited text to a copy of a base file and validates the result with
:func:`~gridexpand.scenario.config_loader.load_scenario_config`, the loader of every run;
:mod:`gridexpand.api.scenarios` writes accepted texts as new files into the user scenario directory.
Form changes go through ruamel.yaml's round trip, so comments, key order and formatting of the base
file stay as they are.
"""

from __future__ import annotations

import copy
import difflib
import io
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import CommentMark
from ruamel.yaml.error import YAMLError as RuamelError
from ruamel.yaml.tokens import CommentToken

from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key

SCENARIO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
FIELD_TYPES = ("float", "int", "percent", "enum", "bool")
_MISSING = object()


def _field(key: str, label: str, type_: str, unit: str | None = None, hint: str = "", **extra: Any) -> dict[str, Any]:
    assert type_ in FIELD_TYPES, type_
    return {"key": key, "label": label, "type": type_, "unit": unit, "hint": hint, **extra}


def _adoption_fields(tech: str, label: str) -> list[dict[str, Any]]:
    mode = f"electrification.{tech}.adoption_mode"
    return [
        _field(mode, f"{label}: selection", "enum", options=[
            {"value": "deterministic_share", "label": "Share of eligible buildings"},
            {"value": "source_inventory", "label": "Source inventory (paired data)"},
        ], hint="deterministic_share or source_inventory (paired DSO data only)."),
        _field(f"electrification.{tech}.building_share", f"{label}: building share", "percent", "%",
               min=0, max=1, step=0.05, requires={"key": mode, "value": "deterministic_share"}, after="adoption_mode",
               hint="Seeded share of the eligible buildings (only with deterministic_share)."),
    ]


# Curated form fields (dotted paths of the scenario YAML, checked against the loader and every shipped
# file by the tests). ``min``/``max`` mirror the loader's rules; the loader stays the only validator.
# ``requires``: the key exists only while another field has the given value (it is removed otherwise).
# ``optional`` + ``default`` + ``after``: the loader's default applies when the key is absent; a change
# to another value inserts the key after ``after``. ``mirror``: keys that are always written with it.
SECTIONS: list[dict[str, Any]] = [
    {"id": "electrification", "title": "Electrification",
     "note": "Which buildings get heat pumps, EVs and PV + battery. deterministic_share selects a seeded share of the "
             "eligible buildings; source_inventory takes the buildings of a source inventory (paired DSO data; "
             "synthetic runs fail with it).",
     "fields": [
        *_adoption_fields("heat", "Heat pumps"),
        *_adoption_fields("mobility", "Electric vehicles"),
        *_adoption_fields("pv_battery", "PV + battery"),
    ]},
    {"id": "economics", "title": "Economics", "fields": [
        _field("economics.electricity.import_price_eur_per_kwh", "Electricity import price", "float", "€/kWh",
               min=0, step=0.001, hint="Retail electricity price of the building energy model (Step 3)."),
        _field("economics.electricity.pv_feed_in_tariff_eur_per_kwh", "PV feed-in tariff", "float", "€/kWh",
               min=0, step=0.001, hint="Remuneration of PV exports (0: unremunerated export)."),
    ]},
    {"id": "pv", "title": "PV sizing", "fields": [
        _field("asset_sizing.pv.demand_multiplier", "Demand multiplier", "float", "kWp/(MWh/a)", min=0, step=0.1,
               hint="Heuristic PV rule: kWp per MWh/a of base electricity demand, capped by the roof potential."),
        _field("asset_sizing.pv.module_capacity_kw_per_m2", "Module capacity", "float", "kWp/m²", min=0, step=0.001,
               hint="Module peak power per m² of usable LoD2 roof area."),
        _field("asset_sizing.pv.flat_roof_utilization", "Flat roof utilisation", "percent", "%", min=0, max=1,
               step=0.01, hint="Usable fraction of flat LoD2 roof area."),
        _field("asset_sizing.pv.slanted_roof_utilization", "Slanted roof utilisation", "percent", "%", min=0,
               max=1, step=0.01, hint="Usable fraction of slanted LoD2 roof area."),
        _field("asset_sizing.pv.fallback_capacity_kwp", "Fallback capacity", "float", "kWp", min=0, step=0.5,
               hint="PV capacity of a selected building without a usable LoD2 roof."),
        _field("asset_sizing.pv.maximum_fallback_share", "Max. fallback share", "percent", "%", min=0, max=1,
               step=0.05, hint="Largest share of PV buildings that may use the fallback; a run above it fails "
                               "(regions without LoD2 roofs need 100 %)."),
    ]},
    {"id": "battery", "title": "Battery sizing", "fields": [
        _field("asset_sizing.battery.heuristic_usable_kwh_per_pv_kwp", "Heuristic: per PV kWp", "float",
               "kWh/kWp", min=0, max=1.5, step=0.05,
               hint="Heuristic battery (HTW 2025 rule): usable kWh per kWp of PV (at most 1.5)."),
        _field("asset_sizing.battery.heuristic_usable_kwh_per_annual_mwh", "Heuristic: per annual MWh", "float",
               "kWh/(MWh/a)", min=0, max=1.5, step=0.05,
               hint="Heuristic battery: usable kWh per MWh/a of base demand (at most 1.5)."),
        _field("asset_sizing.battery.optimized_upper_kwh_per_pv_kwp", "Optimised bound: per PV kWp", "float",
               "kWh/kWp", min=0, max=1.5, step=0.05,
               hint="Upper bound of the optimised battery: kWh per kWp of PV (at most 1.5)."),
        _field("asset_sizing.battery.optimized_upper_kwh_per_annual_mwh", "Optimised bound: per annual MWh",
               "float", "kWh/(MWh/a)", min=0, max=1.5, step=0.05,
               hint="Upper bound of the optimised battery: kWh per MWh/a of base demand (at most 1.5)."),
        _field("asset_sizing.battery.energy_to_power_hours", "Energy-to-power ratio", "float", "h", min=0,
               step=0.5, hint="E/P ratio: the battery power is its energy divided by this value."),
        _field("asset_sizing.battery.minimum_pv_kwp_per_annual_mwh", "Min. PV for a battery", "float",
               "kWp/(MWh/a)", min=0, step=0.05,
               hint="A building gets a battery only if its PV kWp exceeds this factor times its annual demand."),
    ]},
    {"id": "heat", "title": "Heat", "fields": [
        _field("asset_sizing.heat.space_heat_source", "Space-heat source", "enum", options=[
            {"value": "teaser", "label": "TEASER building models"},
            {"value": "infdb_ro_heat", "label": "InfDB ro_heat"},
        ], hint="Space-heat demand from TEASER, or from the InfDB ro_heat schema (must exist in the database)."),
        _field("asset_sizing.heat.teaser_retrofit_level", "TEASER retrofit level", "enum", options=[
            {"value": 0, "label": "0 · as built"},
            {"value": 1, "label": "1 · usual refurbishment"},
            {"value": 2, "label": "2 · advanced refurbishment"},
        ], optional=True, default=0, after="space_heat_source",
            hint="TABULA variant of the TEASER source (optional key, default 0)."),
        _field("asset_sizing.heat.heat_pump_design_share", "Heat-pump design share", "percent", "%", min=0, max=1,
               step=0.05, hint="Heat-pump thermal output at the norm outdoor temperature as share of the design "
                               "heat load."),
        _field("asset_sizing.heat.buffer_volume_l_per_kw_th", "Buffer volume", "float", "l/kW_th", min=0, step=1,
               hint="Space-heating buffer litres per kW of heat-pump thermal power."),
        _field("asset_sizing.heat.indoor_design_temperature_c", "Indoor design temperature", "float", "°C",
               min=0, step=0.5, hint="Indoor reference temperature of the degree-day method."),
        _field("asset_sizing.heat.heating_limit_temperature_c", "Heating limit temperature", "float", "°C",
               min=0, step=0.5, hint="Heating limit of the degree-day method (below the indoor temperature)."),
    ]},
    {"id": "mobility", "title": "Mobility", "fields": [
        _field("mobility.commuting_probability", "Commuting probability", "percent", "%", min=0, max=1, step=0.01,
               hint="Share of vehicles with a commuter driving schedule."),
        _field("technologies.processes.home_charger.installed_capacity_kw", "Home charger power", "float", "kW",
               min=0, step=0.1, mirror=["technologies.processes.home_charger.capacity_upper_kw"],
               hint="Charger power per EV: urbs inst-cap and cap-up of every charging station (both are written; "
                    "the model does not size chargers)."),
    ]},
    {"id": "time_aggregation", "title": "Time aggregation (Step 3)", "fields": [
        _field("time_aggregation.enabled", "TSAM typical periods", "bool",
               hint="Step 3 reduces the time series to typical periods (TSAM); meant for full-year runs."),
        _field("time_aggregation.number_of_typical_periods", "Typical periods", "int", min=1, step=1,
               hint="Number of typical periods TSAM keeps."),
        _field("time_aggregation.hours_per_period", "Hours per period", "int", "h", min=1, step=1,
               hint="Length of one typical period."),
    ]},
]
FIELDS: dict[str, dict[str, Any]] = {f["key"]: f for section in SECTIONS for f in section["fields"]}
LABELS: dict[str, str] = {"scenario.id": "Scenario id", **{key: f["label"] for key, f in FIELDS.items()}, **{
    other: f"{f['label']} ({other.rsplit('.', 1)[1]})" for f in FIELDS.values() for other in f.get("mirror", ())}}


class ScenarioEditError(ValueError):
    """Changes that cannot be applied; ``issues`` holds readable reasons."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        super().__init__(issues[0]["message"] if issues else "invalid changes")
        self.issues = issues


# --------------------------------------------------------------------------- round trip
def _represent_null(representer, _data):
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")


def _represent_float(representer, data: float):
    """New floats in the shortest form that PyYAML (the loader) also reads as floats (``1.0e-05``)."""
    if not math.isfinite(data):
        return representer.represent_float(data)
    text = repr(float(data))
    if "e" in text:
        mantissa, exponent = text.split("e")
        mantissa = mantissa if "." in mantissa else f"{mantissa}.0"
        text = f"{mantissa}e{'-' if exponent.startswith('-') else '+'}{exponent.lstrip('+-').zfill(2)}"
    return representer.represent_scalar("tag:yaml.org,2002:float", text)


def _guess_indent(text: str) -> tuple[int, int, int]:
    """``(mapping, sequence, offset)`` indentation of a YAML text (default: that of the shipped files)."""
    mapping = sequence = offset = None
    parent_key: int | None = None
    for line in text.splitlines():
        body = line.strip()
        if not body or body.startswith("#"):
            continue
        spaces = len(line) - len(line.lstrip(" "))
        if parent_key is not None and spaces >= parent_key:
            if body.startswith("- ") and sequence is None:
                offset = spaces - parent_key
                sequence = offset + 2 + len(body[2:]) - len(body[2:].lstrip(" "))
            elif spaces > parent_key and not body.startswith("-") and mapping is None:
                mapping = spaces - parent_key
        if mapping is not None and sequence is not None:
            break
        parent_key = spaces if body.split("#", 1)[0].rstrip().endswith(":") else None
    return mapping or 2, sequence or 4, offset if offset is not None else 2


def _round_trip(text: str) -> tuple[YAML, Any]:
    """A ruamel round-trip loader configured for ``text`` (indentation guessed), and its document."""
    mapping, sequence, offset = _guess_indent(text)
    rt = YAML(typ="rt")
    rt.preserve_quotes = True
    rt.width = 4096
    rt.indent(mapping=mapping, sequence=sequence, offset=offset)
    rt.representer.add_representer(type(None), _represent_null)
    rt.representer.add_representer(float, _represent_float)
    return rt, rt.load(text)


def _dump(rt: YAML, doc: Any) -> str:
    out = io.StringIO()
    rt.dump(doc, out)
    return out.getvalue()


def _parent(doc: Any, dotted: str) -> tuple[CommentedMap | None, str]:
    *path, leaf = dotted.split(".")
    node = doc
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None, leaf
        node = node[part]
    return (node if isinstance(node, CommentedMap) else None), leaf


def get_value(data: Any, dotted: str, default: Any = _MISSING) -> Any:
    """Value at a dotted path of parsed YAML (``default`` when a part is missing)."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _post_token(mapping: CommentedMap, key: str):
    entry = mapping.ca.items.get(key)
    return entry[2] if entry and len(entry) > 2 else None


def _split_token(token) -> tuple[str, str]:
    """``(own line part, following lines)`` of a comment token after a value."""
    if token is None:
        return "", ""
    value = token.value
    cut = value.find("\n")
    return (value, "") if cut < 0 else (value[: cut + 1], value[cut + 1:])


def _set_post(mapping: CommentedMap, key: str, value: str, like) -> None:
    entry = mapping.ca.items.setdefault(key, [None, None, None, None])
    if not value or value == "\n":
        entry[2] = None
    elif like is not None and entry[2] is like:
        like.value = value
    else:
        mark = getattr(like, "start_mark", None)
        column = getattr(mark, "column", 0) if not value.startswith("\n") else 0
        entry[2] = CommentToken(value, CommentMark(column), None)
    if not any(entry):
        del mapping.ca.items[key]


def _insert_after(mapping: CommentedMap, after: str | None, key: str, value: Any) -> None:
    """Insert ``key`` after ``after`` (or last); comments below ``after`` move below the new key."""
    keys = list(mapping)
    anchor = after if after in mapping else (keys[-1] if keys else None)
    position = keys.index(anchor) + 1 if anchor is not None else 0
    token = _post_token(mapping, anchor) if anchor is not None else None
    mapping.insert(position, key, value)
    if token is not None:
        own, following = _split_token(token)
        _set_post(mapping, anchor, own, token)
        if following:
            _set_post(mapping, key, "\n" + following, None)


def _delete(mapping: CommentedMap, key: str) -> None:
    """Remove ``key``; comments below it move to the previous key, its own help comment goes with it."""
    keys = list(mapping)
    index = keys.index(key)
    _, following = _split_token(_post_token(mapping, key))
    if index > 0:
        previous = keys[index - 1]
        token = _post_token(mapping, previous)
        own, _ = _split_token(token)
        _set_post(mapping, previous, (own or "\n") + following if following else own, token)
    mapping.ca.items.pop(key, None)
    del mapping[key]


# --------------------------------------------------------------------------- values
def _same(old: Any, new: Any, *, strict: bool = True) -> bool:
    """Whether two YAML values are equal.

    ``strict``: in the sense of the configuration hash, where ``2`` and ``2.0`` differ; otherwise
    numbers compare by value (an edit to an equal number keeps the file's spelling).
    """
    if old is _MISSING or new is _MISSING:
        return old is new
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool) and old == new
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return float(old) == float(new) and (not strict or isinstance(old, float) == isinstance(new, float))
    return old == new


def coerce(field: dict[str, Any], value: Any) -> Any:
    """The YAML value of a form input, typed as the field requires.

    Raises:
        ValueError: With a readable message if the value has the wrong type.
    """
    kind, label = field["type"], field["label"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be true or false.")
        return value
    if kind == "enum":
        for option in field["options"]:
            wanted = option["value"]
            if value == wanted and isinstance(value, bool) == isinstance(wanted, bool) and (
                    isinstance(wanted, str) == isinstance(value, str)):
                return wanted
        raise ValueError(f"{label} must be one of {[o['value'] for o in field['options']]}.")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a number.")
    if kind == "int":
        if float(value) != int(value):
            raise ValueError(f"{label} must be a whole number.")
        return int(value)
    return float(value)


def _active(field: dict[str, Any], data: Any) -> bool:
    rule = field.get("requires")
    return not rule or get_value(data, rule["key"], None) == rule["value"]


def field_values(data: Any) -> dict[str, Any]:
    """Current value of every catalogue field (loader default for absent optional keys, else ``None``)."""
    values = {}
    for key, field in FIELDS.items():
        value = get_value(data, key)
        values[key] = field.get("default") if value is _MISSING else value
    return values


def apply_changes(text: str, changes: dict[str, Any], scenario_id: str | None = None) -> str:
    """Return ``text`` with catalogue fields (and ``scenario.id``) set, keeping comments and layout.

    Values equal to the current ones are skipped, so a no-op change returns ``text`` unchanged (and the
    configuration hash stays the same). Dependent keys follow their rule (``requires``, ``mirror``).

    Raises:
        ScenarioEditError: Unknown fields, wrong value types, or a text that is no YAML mapping.
    """
    issues = []
    typed: dict[str, Any] = {}
    for key, value in changes.items():
        field = FIELDS.get(key)
        if field is None:
            issues.append({"level": "error", "key": key, "message": f"{key} is not an editable field."})
            continue
        try:
            typed[key] = coerce(field, value)
        except ValueError as exc:
            issues.append({"level": "error", "key": key, "message": str(exc)})
    if scenario_id is not None and not SCENARIO_ID_RE.match(scenario_id):
        issues.append({"level": "error", "key": "scenario.id", "message": "The scenario id must start with a letter "
                       "or digit and contain only letters, digits, '_' and '-' (at most 64 characters)."})
    if issues:
        raise ScenarioEditError(issues)
    if not typed and scenario_id is None:
        return text
    try:
        rt, doc = _round_trip(text)
    except RuamelError as exc:
        raise ScenarioEditError([{"level": "error", "message": f"YAML syntax error: {exc}"}]) from exc
    if not isinstance(doc, CommentedMap):
        raise ScenarioEditError([{"level": "error", "message": "The scenario YAML must be a mapping."}])
    expected = copy.deepcopy(yaml.safe_load(text))
    changed = False

    def put(dotted: str, value: Any, field: dict[str, Any] | None = None) -> None:
        nonlocal changed
        mapping, leaf = _parent(doc, dotted)
        if mapping is None:
            raise ScenarioEditError([{"level": "error", "key": dotted,
                                      "message": f"{dotted.rsplit('.', 1)[0]} is missing in this file."}])
        current = mapping.get(leaf, _MISSING)
        if _same(current, value, strict=False):
            return
        if current is _MISSING and field and field.get("optional") and _same(field.get("default"), value,
                                                                              strict=False):
            return
        if current is _MISSING:
            _insert_after(mapping, (field or {}).get("after"), leaf, value)
        else:
            mapping[leaf] = value
        get_value(expected, dotted.rsplit(".", 1)[0])[leaf] = value
        changed = True

    for key, value in typed.items():
        if FIELDS[key].get("requires"):
            continue  # after their controlling fields
        put(key, value, FIELDS[key])
        for other in FIELDS[key].get("mirror", ()):
            if get_value(doc, other) is not _MISSING:
                put(other, value)
    for key, field in FIELDS.items():
        if not field.get("requires"):
            continue
        mapping, leaf = _parent(doc, key)
        if mapping is None:
            continue
        if _active(field, doc):
            if key in typed:
                put(key, typed[key], field)
        elif leaf in mapping and field["requires"]["key"] in typed:
            _delete(mapping, leaf)
            del get_value(expected, key.rsplit(".", 1)[0])[leaf]
            changed = True
    if scenario_id is not None:
        put("scenario.id", scenario_id)
    if not changed:
        return text
    new_text = _dump(rt, doc)
    if yaml.safe_load(new_text) != expected:  # never write what the loader would read differently
        raise ScenarioEditError([{"level": "error", "message": "The changes could not be written safely into "
                                                               "this file; edit it in the YAML view."}])
    return new_text


# --------------------------------------------------------------------------- help texts
def _inline_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if ch in "\"'" and quote is None:
            quote = ch
        elif ch == quote:
            quote = None
        elif ch == "#" and quote is None and i > 0 and line[i - 1] in " \t":
            return line[i + 1:].strip()
    return ""


def help_texts(text: str, doc: Any | None = None) -> dict[str, str]:
    """Help text per catalogue field: the comment lines directly above its key plus an inline comment."""
    if doc is None:
        try:
            _, doc = _round_trip(text)
        except RuamelError:
            return {}
    lines = text.splitlines()
    result = {}
    for key in FIELDS:
        mapping, leaf = _parent(doc, key)
        if mapping is None or leaf not in mapping:
            continue
        row = mapping.lc.key(leaf)[0]
        above: list[str] = []
        i = row - 1
        while i >= 0 and lines[i].lstrip().startswith("#"):
            body = lines[i].strip().lstrip("#").strip()
            if body and not set(body) <= set("-=#*"):
                above.insert(0, body)
            i -= 1
        inline = _inline_comment(lines[row]) if row < len(lines) else ""
        parts = [*above, inline] if inline else above
        if parts:
            result[key] = " ".join(parts)
    return result


def form_sections(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The catalogue sections with value, presence and YAML help per field, and the parsed data.

    Fields a file does not have are left out, except optional ones (loader default) and fields that
    depend on another field (for example ``building_share`` of a ``source_inventory`` technology).
    """
    _, doc = _round_trip(text)
    data = yaml.safe_load(text)
    helps = help_texts(text, doc)
    sections = []
    for section in SECTIONS:
        fields = []
        for field in section["fields"]:
            value = get_value(data, field["key"])
            parent_exists = get_value(data, field["key"].rsplit(".", 1)[0]) is not _MISSING
            if value is _MISSING and not (parent_exists and (field.get("optional") or field.get("requires"))):
                continue
            fields.append(field | {
                "value": field.get("default") if value is _MISSING else value,
                "present": value is not _MISSING,
                "help": helps.get(field["key"], ""),
            })
        if fields:
            sections.append({"id": section["id"], "title": section["title"], "note": section.get("note"),
                             "fields": fields})
    return sections, data


# --------------------------------------------------------------------------- validation
_DOTTED_RE = re.compile(r"\b([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+)\b")
_UNKNOWN_RE = re.compile(r"^Unknown (.+?) option\(s\): \[(.*)\]")


def _line_of(doc: Any, dotted: str) -> tuple[str | None, int | None]:
    """The longest existing prefix of ``dotted`` and its 1-based line in the document."""
    parts = dotted.split(".")
    for n in range(len(parts), 0, -1):
        mapping, leaf = _parent(doc, ".".join(parts[:n]))
        if mapping is not None and leaf in mapping:
            return ".".join(parts[:n]), mapping.lc.key(leaf)[0] + 1
    return None, None


def _loader_issue(exc: Exception, doc: Any) -> dict[str, Any]:
    if isinstance(exc, KeyError):
        missing = exc.args[0] if exc.args else exc
        issue: dict[str, Any] = {"level": "error", "message": f"Missing required key '{missing}'."}
        candidates = [key for key in FIELDS if key.rsplit(".", 1)[-1] == missing]
        if len(candidates) == 1:
            issue["key"] = candidates[0]
            _, line = _line_of(doc, candidates[0])
            if line:
                issue["line"] = line
        return issue
    message = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
    issue = {"level": "error", "message": message}
    match = _DOTTED_RE.search(message)
    dotted = match.group(1) if match else None
    unknown = _UNKNOWN_RE.match(message)
    names = re.findall(r"'([^']+)'", unknown.group(2)) if unknown else []
    if names:  # point at the unknown key itself
        dotted = names[0] if unknown.group(1) == "top-level scenario" else f"{unknown.group(1)}.{names[0]}"
    if dotted:
        if dotted in FIELDS:
            issue["key"] = dotted
        _, line = _line_of(doc, dotted)
        if line:
            issue["line"] = line
    return issue


def validate_text(text: str) -> dict[str, Any]:
    """Validate a scenario text with the loader of the runs (through a temporary file).

    Returns:
        ``{ok, issues, id, configuration_hash, scenario_key, data}``; ``data`` is the parsed YAML
        (``None`` when the syntax is invalid).
    """
    result: dict[str, Any] = {"ok": False, "issues": [], "id": None, "configuration_hash": None,
                              "scenario_key": None, "data": None}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        issue = {"level": "error", "message": f"YAML syntax error: {getattr(exc, 'problem', None) or exc}"}
        if mark is not None:
            issue["line"] = mark.line + 1
        result["issues"].append(issue)
        return result
    result["data"] = data
    if not isinstance(data, dict):
        result["issues"].append({"level": "error", "message": "The scenario YAML must be a mapping."})
        return result
    try:
        _, doc = _round_trip(text)
    except RuamelError:
        doc = None
    with tempfile.TemporaryDirectory(prefix="gridexpand-scenario-") as tmp:
        path = Path(tmp) / "scenario.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            scenario, config_hash = load_scenario_config(path)
        except Exception as exc:  # noqa: BLE001 - every loader error becomes a readable issue
            result["issues"].append(_loader_issue(exc, doc))
            return result
    result.update(ok=True, id=scenario.scenario_id, configuration_hash=config_hash,
                  scenario_key=scenario_identity_key(scenario.scenario_id, config_hash))
    for tech in ("heat", "mobility", "pv_battery"):
        if scenario.electrification.for_technology(tech).adoption_mode == "source_inventory":
            key = f"electrification.{tech}.adoption_mode"
            result["issues"].append({"level": "warning", "key": key, "line": _line_of(doc, key)[1],
                                     "message": f"{key} is source_inventory: it needs source evidence of the paired "
                                                "DSO data, so synthetic runs of this service fail with it."})
    if "CHANGE_ME" in scenario.scenario_id:
        result["issues"].append({"level": "warning", "key": "scenario.id",
                                 "message": "The scenario id still contains CHANGE_ME (listed as a template)."})
    return result


def _leaves(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict) and data:
        out = {}
        for key, value in data.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: data} if prefix else {}


def changed_values(old: Any, new: Any) -> list[dict[str, Any]]:
    """Leaf values that differ between two parsed scenarios (``kind``: changed, added, removed)."""
    before, after = _leaves(old if isinstance(old, dict) else {}), _leaves(new if isinstance(new, dict) else {})
    items = []
    for key in [*before, *(k for k in after if k not in before)]:
        a, b = before.get(key, _MISSING), after.get(key, _MISSING)
        if _same(a, b):
            continue
        items.append({"key": key, "label": LABELS.get(key),
                      "kind": "added" if a is _MISSING else "removed" if b is _MISSING else "changed",
                      "old": None if a is _MISSING else a, "new": None if b is _MISSING else b})
    return items


def unified_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    """Unified diff of two scenario texts (two lines of context)."""
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        old_name, new_name, n=2))


def preview(base_text: str, *, base_name: str, changes: dict[str, Any] | None = None, text: str | None = None,
            scenario_id: str | None = None, target_name: str | None = None) -> dict[str, Any]:
    """Apply form changes (on top of ``text`` or the base file) and validate the result.

    Returns:
        ``{ok, issues, text, diff, changes, values, id, configuration_hash, scenario_key}``;
        ``changes`` lists every leaf value that differs from the base file.
    """
    source = base_text if text is None else text
    try:
        new_text = apply_changes(source, changes or {}, scenario_id)
    except ScenarioEditError as exc:
        return {"ok": False, "issues": exc.issues, "text": source, "diff": unified_diff(
                    base_text, source, f"{base_name} (base)", target_name or "new scenario"),
                "changes": [], "values": {}, "id": None, "configuration_hash": None, "scenario_key": None}
    result = validate_text(new_text)
    data = result.pop("data")
    try:
        base_data = yaml.safe_load(base_text)
    except yaml.YAMLError:
        base_data = None
    return result | {
        "text": new_text,
        "diff": unified_diff(base_text, new_text, f"{base_name} (base)", target_name or "new scenario"),
        "changes": changed_values(base_data, data),
        "values": field_values(data) if isinstance(data, dict) else {},
    }
