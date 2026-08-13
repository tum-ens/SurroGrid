"""Network-source adapters available to paired validation."""

from . import swf, synthetic

TARGET_ADAPTERS = {
    swf.TARGET_NETWORK: swf,
    synthetic.TARGET_NETWORK: synthetic,
}


def adapters_for_scope(scope: str):
    if scope == "both":
        return (swf, synthetic)
    try:
        return (TARGET_ADAPTERS[scope],)
    except KeyError as exc:
        raise ValueError(f"Unknown paired target scope {scope!r}.") from exc
