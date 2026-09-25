"""Network-source adapters available to paired validation."""

from . import swf, synthetic, uzw

TARGET_ADAPTERS = {
    swf.TARGET_NETWORK: swf,
    uzw.TARGET_NETWORK: uzw,
    synthetic.TARGET_NETWORK: synthetic,
}
REAL_ADAPTERS = {"swf": swf, "uzw": uzw}


def adapters_for_scope(scope: str, provider: str = "swf"):
    """``both`` means the provider's real grids and their synthetic counterparts."""
    if scope == "both":
        return (REAL_ADAPTERS[provider], synthetic)
    try:
        return (TARGET_ADAPTERS[scope],)
    except KeyError as exc:
        raise ValueError(f"Unknown paired target scope {scope!r}.") from exc
