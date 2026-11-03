"""Network-source adapters available to paired validation."""

from . import dso_a, synthetic, dso_b

TARGET_ADAPTERS = {
    dso_a.TARGET_NETWORK: dso_a,
    dso_b.TARGET_NETWORK: dso_b,
    synthetic.TARGET_NETWORK: synthetic,
}
REAL_ADAPTERS = {"dso_a": dso_a, "dso_b": dso_b}


def adapters_for_scope(scope: str, provider: str = "dso_a"):
    """``both`` means the provider's real grids and their synthetic counterparts."""
    if scope == "both":
        return (REAL_ADAPTERS[provider], synthetic)
    try:
        return (TARGET_ADAPTERS[scope],)
    except KeyError as exc:
        raise ValueError(f"Unknown paired target scope {scope!r}.") from exc
