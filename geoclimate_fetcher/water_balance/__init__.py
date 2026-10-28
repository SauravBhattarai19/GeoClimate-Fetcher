"""
Water Balance module — global, multi-product, on one common analysis grid.

Phase 1: evapotranspiration (actual ET ensemble + reference ET).
Phase 2: precipitation ensemble and P − ET.
Phase 3: runoff and terrestrial water storage change (GRACE).
Phase 4 closes the budget P − ET − Q = ΔS.
"""

from .grid import DEFAULT_RES_DEG, GRID_OPTIONS, area_weighted_mean, crs_transform, to_grid
from .products import (
    ALL_PRODUCTS,
    DS_PRODUCTS,
    ET_PRODUCTS,
    P_PRODUCTS,
    PRODUCTS_BY_KIND,
    Q_PRODUCTS,
    REF_ET_PRODUCTS,
    ProductSpec,
    covers,
    covers_latitudes,
    fetch_availability,
    monthly_collection,
    monthly_image,
    native_projection,
    period_image,
    storage_collection,
    storage_image,
)
from .ensemble import (
    annual_totals,
    basin_monthly_series,
    basin_period_means,
    centred_change,
    ensemble_image,
    ensemble_series,
    gridded_period_image,
)
from .balance import balance_image, balance_series, discharge_to_monthly_mm, pair_differences
from .basins import BASIN_LEVELS, basin_features, find_basin

VERSION = "0.3.0"
