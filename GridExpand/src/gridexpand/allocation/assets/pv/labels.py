"""Stable names shared by PV profiles and urbs tables."""


def profile_label(tilt: float, azimuth: float) -> str:
    return f"solar_{float(tilt):g}_{float(azimuth):g}"
