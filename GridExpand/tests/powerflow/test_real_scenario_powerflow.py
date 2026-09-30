"""Real-grid Step 4 CLI: option checks (parsed before any database or grid access)."""

from __future__ import annotations

import pytest

from gridexpand.powerflow import run_real_swf_scenario_powerflow as module


@pytest.mark.parametrize(
    "argv",
    [
        ["--post-only"],  # post-only needs a Step 3 result
        ["--post-only", "--urbs-result-hdf", "r.h5", "--post-demand-mode", "pre-only"],
    ],
)
def test_post_only_needs_a_post_stage(argv):
    with pytest.raises(SystemExit):
        module.main(argv)
