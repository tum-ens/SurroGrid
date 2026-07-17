# Legacy scenario-calibration workflows

These modules preserve the exploratory, one-sided real-SWF workflow that preceded the paired real/synthetic scenario. They are retained for provenance only and are not used by the current paired runner.

- `real_swf_urbs_input.py`: materialized real-grid-only URBS inputs before stable paired scenario units and exact heat-profile readiness were enforced.
- `real_swf_sector_readiness.py`: audited the earlier one-sided sector-profile handover.

For current runs, use `allocation/paired_allocation.py`, `profiles/paired_profile_readiness.py`, and `pipeline/paired_urbs_input.py`.
