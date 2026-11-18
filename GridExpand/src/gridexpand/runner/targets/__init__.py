"""Network targets of the per-grid runner: given real grids and pylovo grids."""

from . import real, synthetic

TARGET_ADAPTERS = {real.TARGET_NETWORK: real, synthetic.TARGET_NETWORK: synthetic}


def adapter_for(target_network: str):
    """The target of a job: every real network label (``real``, ``real_dso_a``, ...) runs as ``real``."""
    return real if str(target_network).startswith("real") else TARGET_ADAPTERS[str(target_network)]


def adapters_for_scope(scope: str):
    """``both`` means the real grids of the dataset and their synthetic counterparts."""
    if scope == "both":
        return (real, synthetic)
    try:
        return (TARGET_ADAPTERS[scope],)
    except KeyError as exc:
        raise ValueError(f"Unknown runner target {scope!r}.") from exc
