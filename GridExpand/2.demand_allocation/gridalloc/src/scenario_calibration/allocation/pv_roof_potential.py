"""Compatibility exports for the shared LoD2 rooftop-PV catalog."""

from ...assets.pv.roof_catalog import (
    DEFAULT_AZIMUTH_BIN_DEG as PROFILE_AZIMUTH_BIN_DEG,
    DEFAULT_FALLBACK_CAPACITY_KW as FALLBACK_PV_CAPACITY_KW,
    DEFAULT_FLAT_ROOF_UTILIZATION as FLAT_ROOF_UTILIZATION,
    DEFAULT_MODULE_CAPACITY_KW_PER_M2 as PV_AREA_FACTOR_KW_PER_M2,
    DEFAULT_SLANTED_ROOF_UTILIZATION as SLANTED_ROOF_UTILIZATION,
    DEFAULT_TILT_BIN_DEG as PROFILE_TILT_BIN_DEG,
    FALLBACK_AZIMUTH_DEG,
    FALLBACK_TILT_DEG,
    ROOF_SURFACE_QUERY,
    add_missing_building_fallbacks,
    assert_fallback_share,
    building_roof_capacity,
    empty_roof_catalog,
    load_lod2_roof_catalog,
    normalize_lod2_roof_sections,
    read_lod2_roof_catalog_hdf,
)

__all__ = [
    "FALLBACK_PV_CAPACITY_KW", "FLAT_ROOF_UTILIZATION",
    "PV_AREA_FACTOR_KW_PER_M2", "SLANTED_ROOF_UTILIZATION",
    "PROFILE_TILT_BIN_DEG", "PROFILE_AZIMUTH_BIN_DEG",
    "FALLBACK_TILT_DEG", "FALLBACK_AZIMUTH_DEG", "ROOF_SURFACE_QUERY",
    "add_missing_building_fallbacks", "assert_fallback_share",
    "building_roof_capacity", "empty_roof_catalog", "load_lod2_roof_catalog",
    "normalize_lod2_roof_sections", "read_lod2_roof_catalog_hdf",
]
