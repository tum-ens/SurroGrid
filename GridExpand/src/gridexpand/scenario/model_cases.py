"""Stable public names and behavior of scenario model cases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCase:
    name: str
    asset_sizing: str
    dispatch: str


MODEL_CASES = {
    "pre": ModelCase("pre", "reference", "reference"),
    "post-inflex-heuristic": ModelCase("post-inflex-heuristic", "heuristic", "rule_based"),
    "post-hems-optimized": ModelCase("post-hems-optimized", "optimization", "optimized"),
    "post-hems-heuristic": ModelCase("post-hems-heuristic", "heuristic", "optimized"),
}

POST_MODEL_CASES = tuple(name for name in MODEL_CASES if name != "pre")


def get_model_case(name: str) -> ModelCase:
    try:
        return MODEL_CASES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model case {name!r}.") from exc
