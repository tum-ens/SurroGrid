"""Locations of the test fixtures (scenario and run YAMLs that are not shipped in config/)."""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_SCENARIOS = FIXTURES / "scenarios"
FIXTURE_RUNS = FIXTURES / "runs"
