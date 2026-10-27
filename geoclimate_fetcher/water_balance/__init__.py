"""
Water Balance module — global, multi-product, on one common analysis grid.

Phase 1: evapotranspiration (actual ET ensemble + reference ET).
Later phases add precipitation, runoff and storage change (P − ET − Q = ΔS).
"""

from .grid import DEFAULT_RES_DEG, GRID_OPTIONS, area_weighted_mean, crs_transform, to_grid
from .products import (
    ALL_PRODUCTS,
    ET_PRODUCTS,
    REF_ET_PRODUCTS,
    ProductSpec,
    covers,
    fetch_availability,
    monthly_collection,
    monthly_image,
    native_projection,
    period_image,
)
from .ensemble import (
    annual_totals,
    basin_monthly_series,
    basin_period_means,
    ensemble_image,
    ensemble_series,
    gridded_period_image,
)
from .basins import BASIN_LEVELS, basin_features, find_basin

VERSION = "0.1.0"
