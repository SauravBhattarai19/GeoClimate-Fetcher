"""
Water Balance Interface Module — Phase 3: P, ET, Runoff and Storage Change

Global, multi-product precipitation, evapotranspiration, runoff and terrestrial
water storage change on one common analysis grid — the components of a basin
water balance (P − ET − Q = ΔS).

Uses shared platform components:
- GeometrySelectionWidget for custom AOIs, plus a HydroBASINS river-basin picker
- get_ee_tile_url for cached GEE map tiles
- DownloadHelper for GeoTIFF export (local or Google Drive)
GEE is already initialised by the main app.

Tabs 1–3 show the variable (P, ET, Q or ΔS) chosen above them:
  1. Map                 one product, native resolution or analysis grid
  2. Basin Time Series   every product monthly + ensemble median / IQR
  3. Product Agreement   ensemble median, spread, CV and product count maps
  4. P − ET              water availability from every P × ET product pairing
  5. Reference ET        FAO-56 ET₀ and ET / ET₀
"""

import datetime
import math
import re

import ee
import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from app_components.download_component import DownloadHelper
from app_components.ee_tile_cache import get_ee_tile_url
from geoclimate_fetcher.core import GeometrySelectionWidget
from geoclimate_fetcher.serves_calculator import (
    PREDEFINED_REGION_COORDS,
    PREDEFINED_REGION_LABELS,
    get_predefined_region,
)
from geoclimate_fetcher.water_balance import (
    ALL_PRODUCTS,
    BASIN_LEVELS,
    DEFAULT_RES_DEG,
    DS_PRODUCTS,
    ET_PRODUCTS,
    GRID_OPTIONS,
    P_PRODUCTS,
    Q_PRODUCTS,
    REF_ET_PRODUCTS,
    VERSION,
    annual_totals,
    area_weighted_mean,
    balance_image,
    balance_series,
    basin_features,
    basin_monthly_series,
    basin_period_means,
    covers,
    covers_latitudes,
    discharge_to_monthly_mm,
    ensemble_image,
    ensemble_series,
    fetch_availability,
    find_basin,
    gridded_period_image,
    native_projection,
    pair_differences,
    period_image,
    to_grid,
)
from geoclimate_fetcher.water_balance.basins import basin_collection
from geoclimate_fetcher.water_balance.grid import res_to_meters
from geoclimate_fetcher.water_balance.temporal import MONTH_NAMES, month_range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WB_PREFIX = "wb_"

_PERIOD_TYPES = {
    "month":       "📅 Single Month",
    "year":        "📆 Calendar Year",
    "mean_annual": "📊 Mean Annual (multi-year)",
    "climatology": "🔁 Monthly Climatology (multi-year)",
}

_GRID_LABELS = {
    0.05: "0.05° (~5.5 km)",
    0.1:  "0.1° (~11 km) — default",
    0.25: "0.25° (~28 km)",
    0.5:  "0.5° (~55 km)",
}

_DEFAULT_YEAR = 2023                 # every P, ET, Q and ΔS product covers 2018–2023
_DEFAULT_RANGE = (2018, 2023)
_LARGE_AOI_KM2 = 5_000_000           # above this, fine products may hit GEE time limits
_GRACE_MIN_KM2 = 200_000             # GRACE's ~300 km effective resolution

_ET_PALETTE = ["#fff7bc", "#d9f0a3", "#78c679", "#31a354", "#2c7fb8", "#253494"]
_P_PALETTE = ["#ffffd9", "#c7e9b4", "#41b6c4", "#1d91c0", "#225ea8", "#081d58"]
_BALANCE_PALETTE = ["#8c510a", "#d8b365", "#f6e8c3", "#f5f5f5", "#c7eae5", "#5ab4ac", "#01665e"]
_Q_PALETTE = ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"]
_DS_PALETTE = ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"]
_ET0_PALETTE = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
_STD_PALETTE = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
_CV_PALETTE = ["#1a9850", "#a6d96a", "#ffffbf", "#fdae61", "#d73027"]
_COUNT_PALETTE = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]

# Colour-scale maxima per unit (mm/month or mm/yr); spread maps use 30% of these.
# Diverging variables run from −max to +max.
_VIS_MAX = {
    "et":         {"mm/month": 200, "mm/yr": 1600},
    "p":          {"mm/month": 300, "mm/yr": 3000},
    "q":          {"mm/month": 150, "mm/yr": 1500},
    "ds":         {"mm/month": 100, "mm/yr": 200},
    "ref_et":     {"mm/month": 250, "mm/yr": 2500},
    "p_minus_et": {"mm/month": 150, "mm/yr": 1500},   # symmetric around zero
}

# Variables that share the Map / Time Series / Agreement tabs
_VARIABLES = {
    "et": {"label": "Evapotranspiration", "symbol": "ET", "icon": "🌿",
           "products": ET_PRODUCTS, "palette": _ET_PALETTE,
           "coverage_hint": "MOD16, for example, has no values over barren land, urban areas and water."},
    "p":  {"label": "Precipitation", "symbol": "P", "icon": "🌧️",
           "products": P_PRODUCTS, "palette": _P_PALETTE,
           "coverage_hint": "CHIRPS stops at 50° and PERSIANN-CDR at 60° latitude."},
    "q":  {"label": "Runoff", "symbol": "Q", "icon": "🏞️",
           "products": Q_PRODUCTS, "palette": _Q_PALETTE,
           "coverage_hint": "Land-surface models have no runoff over open water and ice sheets."},
    "ds": {"label": "Storage Change", "symbol": "ΔS", "icon": "🛰️",
           "products": DS_PRODUCTS, "palette": _DS_PALETTE, "diverging": True,
           "coverage_hint": "GRACE has no data from Jul 2017 to May 2018."},
}

_ENSEMBLE_LAYERS = {
    "median": "Ensemble median",
    "stdDev": "Spread (standard deviation)",
    "cv":     "Coefficient of variation (%)",
    "count":  "Products with data",
}

_CONTINENT_KEYS = ["globe", "europe", "asia", "north_america", "south_america", "africa", "australia"]
_SUB_KEYS = [k for k in PREDEFINED_REGION_LABELS if k not in _CONTINENT_KEYS]

# Session keys holding results — cleared when the grid, period or AOI change.
# Series and agreement results also record their products and are ignored once
# the product selection no longer matches.
_RESULT_KEYS = ("wb_ts_et", "wb_ts_p", "wb_ts_q", "wb_ts_ds", "wb_agreement_et", "wb_agreement_p",
                "wb_agreement_q", "wb_agreement_ds", "wb_balance_map",
                "wb_ref_ts", "wb_aoi_means", "wb_tile_cache", "wb_sample_cache")
_AOI_KEYS = ("wb_geometry", "wb_geometry_complete", "wb_area_km2", "wb_centroid",
             "wb_aoi_name", "wb_aoi_fc", "wb_lat_range", "wb_method_selection")
_SELECTION_PARAMS = ("months", "products", "members", "excluded")


# ===========================================================================
# Availability (live catalog, cached daily)
# ===========================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def _availability(key: str) -> dict:
    """Live date range for a product, or {'error': ...} if no asset answers."""
    try:
        return fetch_availability(ALL_PRODUCTS[key])
    except Exception as exc:
        return {"error": str(exc)}


def _available(products: dict) -> dict:
    """{key: availability} for products that answered."""
    out = {}
    for key in products:
        av = _availability(key)
        if "error" not in av:
            out[key] = av
    return out


def _covering(products: dict, params: dict) -> list:
    """Keys of products with every month of the analysis span."""
    avail = _available(products)
    return [k for k, av in avail.items() if covers(av, params["first"], params["last"])]


def _lat_label(bounds: tuple) -> str:
    return "–".join(f"{abs(v):g}°{'S' if v < 0 else 'N'}" for v in bounds)


def _exclusions(products: dict, params: dict) -> dict:
    """
    {key: reason} for products that cannot join the ensemble: not every month of
    the span, or a latitude band that does not contain the whole AOI.
    """
    lat_range = st.session_state.get("wb_lat_range")
    out = {}
    for key, spec in products.items():
        av = _availability(key)
        if "error" in av:
            out[key] = "unavailable"
        elif not covers(av, params["first"], params["last"]):
            out[key] = f"data {av['first_month']:%Y-%m} → {av['last_month']:%Y-%m} only"
        elif lat_range and not covers_latitudes(spec, *lat_range):
            out[key] = f"{_lat_label(spec.lat_bounds)} only"
    return out


def _stored(name: str, var: str, params: dict):
    """A stored result for `var`, if it was computed for the current product selection."""
    result = st.session_state.get(f"{name}_{var}")
    if result and result.get("products") == params["products"][var]:
        return result
    return None


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def render_water_balance():
    """Main entry point — called by interface/router.py."""
    st.markdown("## 🌊 Water Balance")
    st.markdown(
        "**Phase 3 — Precipitation, Evapotranspiration, Runoff & Storage Change** · global "
        "multi-product P, ET, Q and ΔS harmonised to one analysis grid and calendar-month "
        f"totals (mm) · v{VERSION}"
    )
    st.divider()

    _auto_expand_sidebar()
    params = _render_sidebar()

    params_key = str(sorted((k, str(v)) for k, v in params.items() if k not in _SELECTION_PARAMS))
    if st.session_state.get("wb_params_key") != params_key:
        for k in _RESULT_KEYS:
            st.session_state.pop(k, None)
        st.session_state["wb_params_key"] = params_key

    if _render_geometry_selection():
        st.divider()
        var = st.radio(
            "Variable", list(_VARIABLES), horizontal=True, key="wb_variable",
            format_func=lambda v: f"{_VARIABLES[v]['icon']} {_VARIABLES[v]['label']}",
            help="Shown in the Map, Basin Time Series and Product Agreement tabs.",
        )
        _render_variable_notes(var)
        tab_map, tab_ts, tab_agree, tab_bal, tab_ref = st.tabs([
            "🗺️ Map", "📈 Basin Time Series", "🤝 Product Agreement", "💧 P − ET", "🌡️ Reference ET",
        ])
        with tab_map:
            _render_map_tab(params, var)
        with tab_ts:
            _render_time_series_tab(params, var)
        with tab_agree:
            _render_agreement_tab(params, var)
        with tab_bal:
            _render_balance_tab(params)
        with tab_ref:
            _render_reference_tab(params)

    _render_products_expander()


# ===========================================================================
# SIDEBAR
# ===========================================================================

def _auto_expand_sidebar():
    """Open the sidebar on the first visit to this module."""
    if not st.session_state.get("wb_sidebar_opened"):
        components.html(
            """
            <script>
                (function () {
                    function tryExpand(n) {
                        var btn = window.parent.document.querySelector(
                            '[data-testid="collapsedControl"]'
                        );
                        if (btn) { btn.click(); }
                        else if (n > 0) { setTimeout(function(){ tryExpand(n-1); }, 250); }
                    }
                    setTimeout(function(){ tryExpand(6); }, 400);
                })();
            </script>
            """,
            height=0,
        )
        st.session_state.wb_sidebar_opened = True


def _render_sidebar() -> dict:
    """All Water Balance parameters. Returns a dict of settings."""
    st.sidebar.header("🌊 Water Balance")
    st.sidebar.caption("Phase 3 · P, ET, runoff & storage change")

    # ── Analysis grid ──────────────────────────────────────────────────────
    st.sidebar.subheader("🧭 Analysis Grid")
    res = st.sidebar.selectbox(
        "Grid resolution (WGS84)",
        list(GRID_OPTIONS),
        index=list(GRID_OPTIONS).index(DEFAULT_RES_DEG),
        format_func=lambda r: _GRID_LABELS[r],
        key="wb_res",
        help="Every product is resampled onto this grid (cell edges anchored at −180°, 90°): "
             "finer products are area-averaged, coarser ones sampled without interpolation.",
    )

    # ── Period ─────────────────────────────────────────────────────────────
    st.sidebar.subheader("📅 Period")
    period_type = st.sidebar.radio(
        "Period type",
        list(_PERIOD_TYPES),
        format_func=lambda k: _PERIOD_TYPES[k],
        key="wb_period_type",
    )

    with st.spinner("Checking product availability…"):
        avail = _available({**ET_PRODUCTS, **P_PRODUCTS})
    max_year = max((av["last_month"].year for av in avail.values()),
                   default=datetime.date.today().year)
    years = list(range(max_year, 1957, -1))
    default_idx = years.index(_DEFAULT_YEAR) if _DEFAULT_YEAR in years else 0

    if period_type == "month":
        year = st.sidebar.selectbox("Year", years, index=default_idx, key="wb_year")
        month = st.sidebar.selectbox("Month", range(1, 13), index=6,
                                     format_func=lambda m: MONTH_NAMES[m - 1], key="wb_month")
        first = last = datetime.date(year, month, 1)
        months, per, unit = [first], 1, "mm/month"
        label = f"{MONTH_NAMES[month - 1]} {year}"
    elif period_type == "year":
        year = st.sidebar.selectbox("Year", years, index=default_idx, key="wb_year")
        first, last = datetime.date(year, 1, 1), datetime.date(year, 12, 1)
        months, per, unit = month_range(first, last), 12, "mm/yr"
        label = str(year)
    else:
        c1, c2 = st.sidebar.columns(2)
        y0 = c1.selectbox("Start year", sorted(years), key="wb_y0",
                          index=sorted(years).index(_DEFAULT_RANGE[0]))
        y1 = c2.selectbox("End year", years, key="wb_y1",
                          index=years.index(_DEFAULT_RANGE[1]) if _DEFAULT_RANGE[1] in years else 0)
        if y1 < y0:
            st.sidebar.error("End year must not be before start year.")
            y1 = y0
        first, last = datetime.date(y0, 1, 1), datetime.date(y1, 12, 1)
        if period_type == "mean_annual":
            months, per, unit = month_range(first, last), 12, "mm/yr"
            label = f"Mean annual {y0}–{y1}"
        else:
            month = st.sidebar.selectbox("Month", range(1, 13), index=6,
                                         format_func=lambda m: MONTH_NAMES[m - 1], key="wb_clim_month")
            months = [datetime.date(y, month, 1) for y in range(y0, y1 + 1)]
            per, unit = 1, "mm/month"
            label = f"{MONTH_NAMES[month - 1]} mean {y0}–{y1}"

    params = {
        "res": res, "period_type": period_type, "first": first, "last": last,
        "months": months, "months_per_period": per, "unit": unit, "label": label,
    }

    # ── Products ───────────────────────────────────────────────────────────
    st.sidebar.caption(
        f"Ensemble members must cover every month of {first:%b %Y} – {last:%b %Y} and the "
        "whole area, so products entering or leaving mid-period cannot create artificial jumps."
    )
    lat_range = st.session_state.get("wb_lat_range")
    lat_tag = f"{lat_range[0]:.0f}_{lat_range[1]:.0f}" if lat_range else "any"
    params.update(products={}, members={}, excluded={})
    for var, cfg in _VARIABLES.items():
        products = cfg["products"]
        st.sidebar.subheader(f"{cfg['icon']} {cfg['label']} Products")
        excluded = _exclusions(products, params)
        eligible = [k for k in products if k not in excluded]
        st.sidebar.caption(f"**{len(eligible)} of {len(products)}** eligible for this period and area.")
        # Keyed on span and latitude so the default resets to the eligible set when they change
        selected = st.sidebar.multiselect(
            "Products",
            list(products),
            default=eligible,
            format_func=lambda k, p=products: p[k].label,
            key=f"wb_products_{var}_{first:%Y%m}_{last:%Y%m}_{lat_tag}",
        )
        params["products"][var] = tuple(selected)
        params["members"][var] = tuple(k for k in selected if k in eligible)
        params["excluded"][var] = excluded
    return params


# ===========================================================================
# AREA OF INTEREST
# ===========================================================================

def _lat_range(geometry):
    """(south, north) of a geometry's bounding box, or None if it cannot be computed."""
    try:
        ring = ee.Geometry(geometry).bounds(maxError=1000).coordinates().get(0).getInfo()
        lats = [c[1] for c in ring]
        return (min(lats), max(lats))
    except Exception:
        return None


def _set_aoi(geometry, name: str, area_km2: float, centroid: list, outline_fc=None):
    st.session_state["wb_geometry"] = geometry
    st.session_state["wb_lat_range"] = _lat_range(geometry)
    st.session_state["wb_aoi_name"] = name
    st.session_state["wb_area_km2"] = round(area_km2, 2)
    st.session_state["wb_centroid"] = centroid
    st.session_state["wb_aoi_fc"] = outline_fc or ee.FeatureCollection([ee.Feature(geometry)])
    for k in _RESULT_KEYS:
        st.session_state.pop(k, None)


def _render_geometry_selection() -> bool:
    """Basin / custom / predefined AOI selection. Returns True when an AOI is ready."""
    if st.session_state.get("wb_geometry") is not None:
        name = st.session_state.get("wb_aoi_name", "Custom area")
        area_km2 = st.session_state.get("wb_area_km2")
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.success(f"✅ Area of Interest: **{name}**")
        if area_km2:
            c2.metric("Area", f"{area_km2:,.0f} km²")
        if c3.button("🗑️ Reset Area", key="wb_reset_geometry"):
            for k in _AOI_KEYS + _RESULT_KEYS:
                st.session_state.pop(k, None)
            st.rerun()
        _render_aoi_warnings(area_km2)
        return True

    tab_basin, tab_custom, tab_region = st.tabs(
        ["🏞️ River Basin (HydroSHEDS)", "📍 Custom AOI", "🌍 Predefined Region"]
    )
    with tab_basin:
        _render_basin_picker()
    with tab_region:
        _render_predefined_region_selector()
    with tab_custom:
        def on_geometry_selected(geometry):
            try:
                area_km2 = geometry.area(maxError=100).divide(1e6).getInfo()
                centroid = geometry.centroid(maxError=100).coordinates().getInfo()
            except Exception:
                area_km2, centroid = 0, None
            _set_aoi(geometry, "Custom area", area_km2, centroid)

        widget = GeometrySelectionWidget(session_prefix=_WB_PREFIX, title="🗺️ Define Study Area")
        if widget.render_complete_interface(on_geometry_selected=on_geometry_selected):
            st.rerun()
            return True
    return False


def _render_aoi_warnings(area_km2):
    if not area_km2:
        return
    res = st.session_state.get("wb_res", DEFAULT_RES_DEG)
    cell_km2 = (res * 111.32) ** 2
    if area_km2 < 4 * cell_km2:
        st.info(
            f"ℹ️ This area spans only ~{max(area_km2 / cell_km2, 0.1):.1f} grid cells at {res}°. "
            "Basin means use the overlapping cells, weighted by overlap — consider a finer grid "
            "or a larger basin."
        )
    if area_km2 > _LARGE_AOI_KM2:
        st.warning(
            "⚠️ Very large area: time series for the 300 m – 5 km products (PML, MOD16, WaPOR, "
            "SSEBop, CHIRPS, TerraClimate) may exceed Earth Engine's interactive time limit. "
            "A coarser grid (0.25° / 0.5°) helps."
        )


def _render_basin_picker():
    st.markdown(
        "Click a point on a river (or enter coordinates) to select the **HydroBASINS** unit "
        "containing it — optionally with everything that drains into it."
    )
    c1, c2 = st.columns([2, 1])
    level = c1.selectbox("Basin level", list(BASIN_LEVELS), index=list(BASIN_LEVELS).index(6),
                         format_func=lambda lv: BASIN_LEVELS[lv], key="wb_basin_level")
    upstream = c2.checkbox("Include all upstream area", value=True, key="wb_basin_upstream",
                           help="Trace every HydroBASINS unit draining into the selected one — "
                                "the full catchment above that point.")

    # The map writes clicks into the coordinate inputs' state before they are drawn
    st.session_state.setdefault("wb_basin_lat", 27.7)
    st.session_state.setdefault("wb_basin_lon", 85.3)
    pick = [st.session_state["wb_basin_lat"], st.session_state["wb_basin_lon"]]

    m = folium.Map(location=pick, zoom_start=5, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite (Esri)",
    ).add_to(m)
    try:
        outlines = basin_collection(level).style(color="00FFFF", fillColor="00000000", width=1)
        url = get_ee_tile_url(outlines, {}, "wb_basin_tile_cache", f"hybas_{level}",
                              failed_msg="Basin outlines unavailable")
        folium.TileLayer(tiles=url, attr="WWF HydroSHEDS", name=f"HydroBASINS level {level}",
                         overlay=True).add_to(m)
    except Exception:
        pass
    folium.Marker(pick, tooltip="Selected point").add_to(m)
    data = st_folium(m, height=420, width=None, returned_objects=["last_clicked"], key="wb_basin_map")
    if data and data.get("last_clicked"):
        clicked = [round(data["last_clicked"]["lat"], 4), round(data["last_clicked"]["lng"], 4)]
        if clicked != pick:
            st.session_state["wb_basin_lat"], st.session_state["wb_basin_lon"] = clicked
            st.rerun()

    c3, c4, c5 = st.columns([1, 1, 2])
    lat = c3.number_input("Latitude", -90.0, 90.0, format="%.4f", key="wb_basin_lat")
    lon = c4.number_input("Longitude", -180.0, 180.0, format="%.4f", key="wb_basin_lon")
    c5.write("")
    if c5.button("🏞️ Use basin at this point", type="primary", width="stretch", key="wb_use_basin"):
        with st.spinner("Finding basin…"):
            try:
                basin = find_basin(lon, lat, level, upstream=upstream)
            except ValueError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Basin lookup failed: {exc}")
                return
        fc = basin_features(level, basin["ids"])
        kind = f"upstream catchment, {basin['n_units']} units" if upstream else "single unit"
        _set_aoi(fc.geometry(), f"HydroBASINS L{level} #{basin['hybas_id']} ({kind})",
                 basin["area_km2"], [lon, lat], outline_fc=fc)
        st.rerun()


def _render_predefined_region_selector():
    st.markdown("Pick a continental or regional bounding box as the study area.")
    group = st.radio("Category:", ["🌍 Continents", "🗺️ Sub-continental Regions"],
                     horizontal=True, key="wb_reg_group")
    keys = _CONTINENT_KEYS if group == "🌍 Continents" else _SUB_KEYS
    labels = [PREDEFINED_REGION_LABELS[k] for k in keys]
    sel_label = st.selectbox("Region:", labels, key="wb_reg_selectbox")
    sel_key = keys[labels.index(sel_label)]

    west, south, east, north = PREDEFINED_REGION_COORDS[sel_key]
    lat_mid, lon_mid = (south + north) / 2, (west + east) / 2
    approx_area = abs((east - west) * (north - south)) * (111.32 ** 2) * math.cos(math.radians(lat_mid))
    st.caption(f"Bounds: {west}°–{east}° lon, {south}°–{north}° lat  |  ~{approx_area:,.0f} km²")

    if st.button(f"🌍 Use {sel_label}", type="primary", key="wb_use_predefined_region"):
        _set_aoi(get_predefined_region(sel_key), sel_label, approx_area, [lon_mid, lat_mid])
        st.rerun()


# ===========================================================================
# MAP RENDERING
# ===========================================================================

def _render_map(image, vis: dict, layer_key: str, caption: str,
                sample_scale: float, value_unit: str) -> bool:
    """Folium map with a cached GEE tile layer, AOI outline, colorbar and pixel inspector."""
    centroid = st.session_state.get("wb_centroid") or [0, 20]
    area_km2 = st.session_state.get("wb_area_km2", 10_000) or 10_000
    init_zoom = 9 if area_km2 < 1_000 else (7 if area_km2 < 50_000 else (5 if area_km2 < 1e6 else 3))

    state_key = f"wb_map_state_{layer_key.split('::')[0]}"  # one viewport per map, kept across layers
    saved = st.session_state.get(state_key, {})
    m = folium.Map(location=[saved.get("lat", centroid[1]), saved.get("lon", centroid[0])],
                   zoom_start=saved.get("zoom", init_zoom), tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite (Esri)", overlay=False, control=True, show=False,
    ).add_to(m)

    tile_key = f"{st.session_state.get('wb_params_key', '')}::{layer_key}"
    layer_ok = False
    try:
        url = get_ee_tile_url(
            image, vis, "wb_tile_cache", tile_key,
            failed_msg="Map preview unavailable (GEE memory limit). Use the GeoTIFF export — "
                       "batch export has higher memory limits.",
        )
        folium.TileLayer(tiles=url, attr="Google Earth Engine", name=caption,
                         overlay=True, control=True).add_to(m)
        layer_ok = True
    except Exception as exc:
        st.warning(f"Could not render map layer: {exc}")

    aoi_fc = st.session_state.get("wb_aoi_fc")
    if aoi_fc is not None:
        try:
            outline = aoi_fc.style(color="000000", fillColor="00000000", width=2)
            url = get_ee_tile_url(outline, {}, "wb_tile_cache", "aoi_outline::" +
                                  st.session_state.get("wb_aoi_name", ""), failed_msg="")
            folium.TileLayer(tiles=url, attr="AOI", name="Area of interest",
                             overlay=True, control=True).add_to(m)
        except Exception:
            pass

    LinearColormap(colors=vis["palette"], vmin=vis["min"], vmax=vis["max"],
                   caption=caption).add_to(m)
    folium.LayerControl().add_to(m)

    folium_key = "wb_folium_" + "".join(c if c.isalnum() else "_" for c in layer_key.split("::")[0])
    data = st_folium(m, width=None, height=480, returned_objects=["last_clicked"], key=folium_key)
    if data and data.get("center") and data.get("zoom"):
        st.session_state[state_key] = {"lat": data["center"]["lat"], "lon": data["center"]["lng"],
                                       "zoom": int(data["zoom"])}

    if layer_ok and data and data.get("last_clicked"):
        lat, lng = data["last_clicked"]["lat"], data["last_clicked"]["lng"]
        cache = st.session_state.setdefault("wb_sample_cache", {})
        sk = f"{tile_key}::{lat:.4f}::{lng:.4f}"
        if sk not in cache:
            try:
                vals = image.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([lng, lat]),
                                          sample_scale).values().getInfo()
                cache[sk] = next((v for v in vals if isinstance(v, (int, float))), None)
            except Exception:
                cache[sk] = None
        val = cache[sk]
        st.caption(f"📍 **{lat:.4f}°, {lng:.4f}°** → " +
                   (f"**{val:,.1f} {value_unit}**" if val is not None else "no data (masked or outside AOI)"))
    elif layer_ok:
        st.caption("🖱️ Click the map to inspect a pixel value.")
    return layer_ok


def _vis(kind: str, unit: str, palette: list, fraction: float = 1.0, diverging: bool = False) -> dict:
    top = _VIS_MAX[kind][unit] * fraction
    return {"min": -top if diverging else 0, "max": top, "palette": palette}


def _render_variable_notes(var: str):
    """What the chosen variable means, and when GRACE cannot resolve the area."""
    if var == "q":
        st.caption(
            "ℹ️ Runoff is the water each grid cell sheds (surface + subsurface) in land-surface "
            "models — not routed river discharge. Over months to years it should match the "
            "discharge at the basin outlet; compare with a gauge record in the Basin Time Series tab."
        )
    elif var == "ds":
        st.caption(
            "ℹ️ ΔS is the monthly change in total water storage (snow, soil, groundwater and surface "
            "water): [S(next month) − S(previous month)] / 2, so it lines up with monthly fluxes."
        )
        area_km2 = st.session_state.get("wb_area_km2") or 0
        if area_km2 < _GRACE_MIN_KM2:
            st.warning(
                f"⚠️ GRACE resolves areas of roughly {_GRACE_MIN_KM2:,} km² and up; this area is "
                f"{area_km2:,.0f} km². Its GRACE values mostly reflect the surrounding region — "
                "treat them as indicative, or select a larger basin."
            )


def _render_export(image, params: dict, file_tag: str, key: str, button_label: str):
    """GeoTIFF export of a gridded image at the analysis-grid resolution."""
    with st.expander("💾 Export GeoTIFF", expanded=False):
        st.caption(f"Exported on the analysis grid: {params['res']}° in EPSG:4326, values in {params['unit']}.")
        helper = DownloadHelper()
        pref = helper.render_smart_download_options(export_format="GeoTIFF", key=f"{key}_pref")
        filename = re.sub(r"[^\w\-]", "_", f"wb_{file_tag}_{params['label']}_{params['res']}deg")[:80]
        if st.button(f"💾 Export {button_label}", type="primary", width="stretch", key=key):
            with st.spinner("Exporting…"):
                result = helper.execute_smart_download(
                    image=image.toFloat(), filename=filename,
                    region=st.session_state["wb_geometry"],
                    scale=res_to_meters(params["res"]),
                    export_preference=pref, crs="EPSG:4326",
                )
                if not result.get("success"):
                    st.error(f"Export failed: {result.get('message', 'Unknown error')}")


# ===========================================================================
# TAB 1 — MAP
# ===========================================================================

def _render_map_tab(params: dict, var: str):
    cfg = _VARIABLES[var]
    products, sym = cfg["products"], cfg["symbol"]
    covering = _covering(products, params)
    if not covering:
        st.warning(f"No {cfg['label'].lower()} product covers the whole selected period.")
        return

    c1, c2 = st.columns([2, 1])
    options = [k for k in params["products"][var] if k in covering] or covering
    key = c1.selectbox("Product", options, format_func=lambda k: products[k].label,
                       key=f"wb_map_product_{var}")
    display = c2.radio("Display", ["Native resolution", f"Analysis grid ({params['res']}°)"],
                       key="wb_map_display", help="Native shows the product as published; the "
                       "grid view is exactly what the time series and ensemble use.")
    spec = products[key]
    asset = _availability(key)["asset"]
    geom = st.session_state["wb_geometry"]

    native = period_image(spec, asset, params["months"], params["months_per_period"])
    gridded = to_grid(native, native_projection(spec, asset), spec.native_res_deg, params["res"])
    on_grid = display != "Native resolution"
    image = (gridded if on_grid else native).clip(geom)

    st.caption(f"**{spec.label}** · {spec.algorithm} · native {spec.native_res_label} · "
               f"{params['label']} ({params['unit']})")
    if spec.notes:
        st.caption(f"ℹ️ {spec.notes}")
    if key in params["excluded"][var]:
        st.warning(f"⚠️ {spec.label} covers {params['excluded'][var][key]} — part of this area has no data.")

    sample_scale = res_to_meters(params["res"]) if on_grid else spec.native_res_deg * 111_320
    _render_map(image, _vis(var, params["unit"], cfg["palette"], diverging=cfg.get("diverging", False)),
                f"map_{var}::{key}::{on_grid}", f"{sym} — {spec.label} ({params['unit']})",
                sample_scale, params["unit"])

    mean_key = f"{key}::{params['res']}"
    means = st.session_state.setdefault("wb_aoi_means", {})
    if st.button("📊 Area-weighted AOI mean", key=f"wb_map_mean_btn_{var}"):
        with st.spinner("Averaging over the AOI on the analysis grid…"):
            try:
                means[mean_key] = area_weighted_mean(gridded, geom, params["res"]).getInfo()
            except Exception as exc:
                st.error(f"Could not compute the AOI mean: {exc}")
    if mean_key in means:
        res = means[mean_key]
        m1, m2 = st.columns(2)
        m1.metric(f"Mean {sym} ({params['unit']})",
                  f"{res['mean']:,.1f}" if res.get("mean") is not None else "no data")
        m2.metric("Area with data", f"{100 * (res.get('coverage') or 0):.0f}%")

    _render_export(gridded.clip(geom), params, f"{var}_{key}", f"wb_map_export_{var}",
                   f"{spec.label} {sym}")


# ===========================================================================
# TAB 2 — BASIN TIME SERIES
# ===========================================================================

def _membership_caption(var: str, params: dict) -> str:
    """Markdown line naming the ensemble members, plus the selected outsiders and why."""
    products = _VARIABLES[var]["products"]
    members = params["members"][var]
    text = "Ensemble members: " + (", ".join(products[k].label for k in members) if members else "none")
    outsiders = [k for k in params["products"][var] if k not in members]
    if outsiders:
        text += "  \nShown but not in the ensemble: " + ", ".join(
            f"{products[k].label} ({params['excluded'][var].get(k, 'partial coverage')})" for k in outsiders)
    return text


def _render_time_series_tab(params: dict, var: str):
    cfg = _VARIABLES[var]
    products, sym = cfg["products"], cfg["symbol"]
    span = month_range(params["first"], params["last"])
    if len(span) < 12:
        st.info("Choose **Calendar Year** or a multi-year period in the sidebar for a monthly time series.")
        return
    if not params["products"][var]:
        st.warning(f"Select at least one {cfg['label'].lower()} product in the sidebar.")
        return

    members = list(params["members"][var])
    st.markdown(
        f"Monthly basin-mean {sym}, **{span[0]:%b %Y} – {span[-1]:%b %Y}**, area-weighted on the "
        f"{params['res']}° grid."
    )
    st.caption(_membership_caption(var, params))

    observed = _render_observed_upload(span) if var == "q" else None

    n = len(params["products"][var])
    if st.button(f"▶ Compute monthly {sym} time series ({n} products × {len(span)} months)",
                 type="primary", width="stretch", key=f"wb_ts_run_{var}"):
        st.session_state[f"wb_ts_{var}"] = _compute_series(products, params["products"][var], span, params)

    result = _stored("wb_ts", var, params)
    if not result:
        return
    monthly, coverage, errors = result["monthly"], result["coverage"], result["errors"]
    for key, err in errors.items():
        st.error(f"{products[key].label}: {err}")
    if monthly.empty:
        return

    ens = ensemble_series(monthly, members) if members else None
    _plot_monthly(monthly, ens, products, f"{sym} (mm/month)", members, observed)
    if result.get("storage") is not None and not result["storage"].empty:
        _plot_storage(result["storage"], products)

    low = [k for k in monthly.columns if coverage[k].mean() < 0.9]
    if low:
        st.warning(
            "⚠️ Low spatial coverage — basin mean uses only cells with data: "
            + ", ".join(f"{products[k].label} ({100 * coverage[k].mean():.0f}%)" for k in low)
            + ". " + cfg["coverage_hint"]
        )

    annual = annual_totals(monthly)
    if not annual.dropna(how="all").empty:
        _render_annual_summary(annual, members, products, sym)
    if observed is not None:
        _render_observed_summary(observed, annual, members)

    table = monthly.rename(columns=lambda k: f"{k}_mm").copy()
    if ens is not None:
        table = table.join(ens.add_prefix("ensemble_"))
    if observed is not None:
        table = table.join(observed.rename("observed_mm"))
    if result.get("storage") is not None:
        table = table.join(result["storage"].rename(columns=lambda k: f"{k}_storage_mm"))
    table = table.join(coverage.rename(columns=lambda k: f"{k}_coverage"))
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "month"
    st.download_button(
        "📥 Download monthly table (CSV)", table.to_csv().encode(),
        file_name=re.sub(r"[^\w\-]", "_", f"wb_{var}_monthly_{params['label']}_{params['res']}deg") + ".csv",
        mime="text/csv", key=f"wb_ts_csv_{var}",
    )


def _compute_series(products: dict, keys, span: list, params: dict) -> dict:
    """Basin monthly series for each product, over the months it has data for."""
    geom = st.session_state["wb_geometry"]
    series, coverage, storage, errors = {}, {}, {}, {}
    bar = st.progress(0.0, text="Starting…")
    for i, key in enumerate(keys):
        spec, av = products[key], _availability(key)
        months = [m for m in span if "error" not in av and av["first_month"] <= m <= av["last_month"]]
        if not months:
            errors[key] = "no data in this period"
            continue

        def on_chunk(done, total, i=i, label=spec.label):
            bar.progress((i + done / total) / len(keys), text=f"{label}: {done}/{total} months")

        try:
            df = basin_monthly_series(spec, av["asset"], months, geom, params["res"], on_chunk=on_chunk)
            series[key], coverage[key] = df["mean"], df["coverage"]
            if "storage" in df:
                storage[key] = df["storage"]
        except Exception as exc:
            errors[key] = str(exc)[:300]
    bar.empty()
    index = pd.to_datetime(span)
    return {
        "products": tuple(keys),
        "monthly": pd.DataFrame(series).reindex(index).astype(float),
        "coverage": pd.DataFrame(coverage).reindex(index).astype(float),
        "storage": pd.DataFrame(storage).reindex(index).astype(float) if storage else None,
        "errors": errors,
    }


def _plot_monthly(monthly: pd.DataFrame, ens, products: dict, y_title: str, members: list,
                  observed=None):
    fig = go.Figure()
    if ens is not None and ens["n"].max() >= 2:
        fig.add_trace(go.Scatter(x=ens.index, y=ens["q75"], line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ens.index, y=ens["q25"], line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(120,120,120,0.25)", name="Ensemble IQR",
                                 hoverinfo="skip"))
    for key in monthly.columns:
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly[key], mode="lines", name=products[key].label,
            line=dict(color=products[key].color, width=1.4,
                      dash="solid" if key in members else "dot"),
            hovertemplate=f"{products[key].label}: %{{y:.1f}} mm<extra></extra>",
        ))
    if ens is not None:
        fig.add_trace(go.Scatter(x=ens.index, y=ens["median"], mode="lines", name="Ensemble median",
                                 line=dict(color="#111111", width=3),
                                 hovertemplate="Ensemble median: %{y:.1f} mm<extra></extra>"))
    if observed is not None:
        obs = observed.reindex(monthly.index)
        fig.add_trace(go.Scatter(x=obs.index, y=obs, mode="lines+markers", name="Observed (gauge)",
                                 line=dict(color="#e41a1c", width=2.5, dash="dash"), marker=dict(size=4),
                                 hovertemplate="Observed: %{y:.1f} mm<extra></extra>"))
    fig.update_layout(height=430, hovermode="x unified", yaxis_title=y_title,
                      margin=dict(l=60, r=20, t=30, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")


def _plot_storage(storage: pd.DataFrame, products: dict):
    """Basin storage of each product as anomalies from its own mean over the period."""
    anomalies = storage - storage.mean()
    fig = go.Figure()
    for key in anomalies.columns:
        fig.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies[key], mode="lines+markers", name=products[key].label,
            line=dict(color=products[key].color, width=2), marker=dict(size=4), connectgaps=False,
            hovertemplate=f"{products[key].label}: %{{y:.0f}} mm<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#999999", line_width=1)
    fig.update_layout(title="Total water storage anomaly (relative to each product's period mean)",
                      height=360, hovermode="x unified", yaxis_title="mm",
                      margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5))
    st.plotly_chart(fig, width="stretch")
    st.caption("Gaps are months without data (for GRACE, Jul 2017 – May 2018 between the missions). "
               "ΔS above is the centred monthly difference of these storage series.")


def _render_observed_upload(span: list):
    """Optional gauge record → monthly runoff depth (mm) over the AOI, or None."""
    with st.expander("📏 Compare with observed river discharge (optional)", expanded=False):
        st.caption(
            "Upload a gauge record — CSV with a date column and a discharge column in m³/s, at any "
            "time step. It is converted to runoff depth over the selected area, so pick the basin "
            "at the gauge with **Include all upstream area** on. Months with < 80% of days are dropped."
        )
        upload = st.file_uploader("Discharge CSV", type=["csv"], key="wb_q_obs_file")
        if upload is None:
            return None
        try:
            raw = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            return None
        cols = list(raw.columns)
        if len(cols) < 2:
            st.error("The CSV needs a date column and a discharge column.")
            return None
        date_guess = next((c for c in cols if re.search(r"date|time", str(c), re.I)), cols[0])
        value_cols = [c for c in cols if c != date_guess]
        value_guess = next((c for c in value_cols if re.search(r"q|discharge|flow", str(c), re.I)),
                           value_cols[0])
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Date column", cols, index=cols.index(date_guess), key="wb_q_obs_date")
        value_col = c2.selectbox("Discharge column (m³/s)", cols, index=cols.index(value_guess),
                                 key="wb_q_obs_value")
        flow = pd.Series(pd.to_numeric(raw[value_col], errors="coerce").values,
                         index=pd.to_datetime(raw[date_col], errors="coerce"))
        flow = flow[flow.index.notna()].dropna()
        area_km2 = st.session_state.get("wb_area_km2") or 0
        if flow.empty or area_km2 <= 0:
            st.warning("No valid date / discharge rows, or the area size is unknown.")
            return None
        observed = discharge_to_monthly_mm(flow, area_km2)
        in_span = observed.reindex(pd.to_datetime(span)).notna().sum()
        st.caption(
            f"{len(flow):,} records, {flow.index.min():%Y-%m-%d} → {flow.index.max():%Y-%m-%d} · "
            f"{in_span} complete months in the analysis period · converted over {area_km2:,.0f} km²."
        )
        if in_span == 0:
            st.warning("The record has no complete months inside the analysis period.")
        return observed


def _render_observed_summary(observed: pd.Series, annual: pd.DataFrame, members: list):
    """Observed annual runoff against the ensemble, over years complete in both."""
    obs_annual = annual_totals(observed.to_frame("observed"))["observed"].reindex(annual.index)
    member_cols = [k for k in members if k in annual.columns]
    if not member_cols:
        return
    model = annual[member_cols].median(axis=1)
    both = pd.DataFrame({"obs": obs_annual, "model": model}).dropna()
    if both.empty:
        st.info("No calendar year is complete in both the gauge record and the products.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed runoff", f"{both['obs'].mean():,.0f} mm/yr", help=f"{len(both)} complete years")
    c2.metric("Ensemble median runoff", f"{both['model'].mean():,.0f} mm/yr")
    c3.metric("Ensemble / observed", f"{both['model'].mean() / both['obs'].mean():.2f}"
              if both["obs"].mean() > 0 else "—")
    st.caption("Generated runoff and gauged discharge differ through routing delays, reservoirs, "
               "withdrawals and transfers, so expect closer agreement over longer periods.")


def _render_annual_summary(annual: pd.DataFrame, members: list, products: dict, sym: str):
    st.markdown("#### Annual totals")
    mean_annual = annual.mean()
    member_means = mean_annual[[k for k in members if k in mean_annual.index]].dropna()
    if len(member_means):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Ensemble median {sym}", f"{member_means.median():,.0f} mm/yr")
        c2.metric("Range across products", f"{member_means.min():,.0f} – {member_means.max():,.0f} mm/yr")
        c3.metric("Ensemble members", len(member_means))

    order = mean_annual.dropna().sort_values().index
    fig = go.Figure(go.Bar(
        x=[products[k].label for k in order], y=mean_annual[order],
        marker_color=[products[k].color for k in order],
        hovertemplate="%{x}: %{y:.0f} mm/yr<extra></extra>",
    ))
    if len(member_means):
        fig.add_hline(y=member_means.median(), line_dash="dash", line_color="#111111",
                      annotation_text="ensemble median", annotation_position="top left")
    fig.update_layout(height=340, yaxis_title=f"Mean annual {sym} (mm/yr)",
                      margin=dict(l=60, r=20, t=20, b=80))
    st.plotly_chart(fig, width="stretch")

    shown = annual.dropna(how="all").rename(columns=lambda k: products[k].label)
    shown.index.name = "Year"
    st.dataframe(shown.style.format("{:.0f}", na_rep="—"), width="stretch")


# ===========================================================================
# TAB 3 — PRODUCT AGREEMENT
# ===========================================================================

def _gridded_members(var: str, params: dict) -> dict:
    """{key: period image on the analysis grid} for the ensemble members of `var`."""
    products = _VARIABLES[var]["products"]
    return {
        k: gridded_period_image(products[k], _availability(k)["asset"], params["months"],
                                params["months_per_period"], params["res"])
        for k in params["members"][var]
    }


def _render_agreement_tab(params: dict, var: str):
    cfg = _VARIABLES[var]
    products, sym = cfg["products"], cfg["symbol"]
    members = list(params["members"][var])
    if len(members) < 2:
        st.info("Product agreement needs at least two selected products that cover the whole "
                "period and area.")
        return
    st.markdown(
        f"Per-cell statistics of {sym} across **{len(members)} products** for **{params['label']}** "
        f"({params['unit']}), each resampled to the {params['res']}° grid."
    )
    if st.button("▶ Compute product agreement", type="primary", width="stretch",
                 key=f"wb_agree_run_{var}"):
        geom = st.session_state["wb_geometry"]
        gridded = _gridded_members(var, params)
        bar = st.progress(0.0, text="Averaging each product over the AOI…")
        means = basin_period_means(
            gridded, geom, params["res"],
            on_product=lambda done, total, key: bar.progress(
                done / total, text=f"{products[key].label} done ({done}/{total})"),
        )
        bar.empty()
        st.session_state[f"wb_agreement_{var}"] = {
            "ensemble": ensemble_image(list(gridded.values())).clip(geom),
            "means": means, "members": members, "products": params["products"][var],
        }

    result = _stored("wb_agreement", var, params)
    if not result:
        return

    for k, v in result["means"].items():
        if v.get("error"):
            st.error(f"{products[k].label}: AOI mean failed — {v['error']}")
    means = {k: v for k, v in result["means"].items() if v.get("mean") is not None}
    if means:
        values = pd.Series({k: v["mean"] for k, v in means.items()}).sort_values()
        median = values.median()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Median of products ({params['unit']})", f"{median:,.1f}")
        c2.metric("Range", f"{values.min():,.1f} – {values.max():,.1f}")
        c3.metric("Spread (max / min)", f"{values.max() / values.min():.1f}×" if values.min() > 0 else "—")
        fig = go.Figure(go.Bar(
            x=[products[k].label for k in values.index], y=values.values,
            marker_color=[products[k].color for k in values.index],
            customdata=[100 * means[k]["coverage"] for k in values.index],
            hovertemplate="%{x}: %{y:.1f} mm<br>area with data %{customdata:.0f}%<extra></extra>",
        ))
        fig.add_hline(y=median, line_dash="dash", line_color="#111111",
                      annotation_text="median", annotation_position="top left")
        fig.update_layout(height=320, yaxis_title=f"AOI mean {sym} ({params['unit']})",
                          margin=dict(l=60, r=20, t=20, b=80))
        st.plotly_chart(fig, width="stretch")

    # CV is meaningless for ΔS, whose mean sits near zero
    layers = [k for k in _ENSEMBLE_LAYERS if not (k == "cv" and cfg.get("diverging"))]
    layer = st.radio("Map layer", layers, format_func=lambda k: _ENSEMBLE_LAYERS[k],
                     horizontal=True, key=f"wb_agree_layer_{var}")
    unit = params["unit"]
    vis = {
        "median": _vis(var, unit, cfg["palette"], diverging=cfg.get("diverging", False)),
        "stdDev": _vis(var, unit, _STD_PALETTE, fraction=0.3),
        "cv":     {"min": 0, "max": 100, "palette": _CV_PALETTE},
        "count":  {"min": 0, "max": len(result["members"]), "palette": _COUNT_PALETTE},
    }[layer]
    value_unit = {"cv": "%", "count": "products"}.get(layer, unit)
    _render_map(result["ensemble"].select(layer), vis,
                f"agree_{var}::{layer}::{','.join(result['members'])}",
                f"{sym} — {_ENSEMBLE_LAYERS[layer]}", res_to_meters(params["res"]), value_unit)
    st.caption(f"High spread or CV marks where products disagree most — treat {sym} there as "
               "uncertain. Product count drops where a product has no data.")

    _render_export(result["ensemble"].select(["median", "stdDev", "cv", "count"]), params,
                   f"{var}_ensemble", f"wb_agree_export_{var}", "ensemble (median, stdDev, CV, count)")


# ===========================================================================
# TAB 4 — P − ET
# ===========================================================================

def _render_balance_tab(params: dict):
    span = month_range(params["first"], params["last"])
    p_members, et_members = list(params["members"]["p"]), list(params["members"]["et"])
    st.markdown(
        "**P − ET** is the water left after evaporation. Over several years it approximates "
        "runoff plus storage change (P − ET = Q + ΔS); negative values mean the basin "
        "evaporates more than it receives — from irrigation, groundwater, inflow from upstream "
        "or product bias. Its uncertainty comes from every pairing of a P member with an ET member."
    )
    if not p_members or not et_members:
        st.warning("P − ET needs at least one precipitation and one ET ensemble member — check the sidebar.")
        return

    _render_balance_map(params, p_members, et_members)

    st.markdown("#### Basin time series")
    if len(span) < 12:
        st.info("Choose **Calendar Year** or a multi-year period in the sidebar for the P − ET time series.")
        return
    missing = [v for v in ("p", "et") if not _stored("wb_ts", v, params)]
    if missing:
        names = " and ".join(_VARIABLES[v]["symbol"] for v in missing)
        st.caption(f"Uses the monthly series from the Basin Time Series tab; the {names} series "
                   "is not computed yet for the current selection.")
        if not st.button(f"▶ Compute {names} time series", type="primary", width="stretch",
                         key="wb_bal_run"):
            return
        for v in missing:
            keys = params["products"][v]
            st.session_state[f"wb_ts_{v}"] = _compute_series(_VARIABLES[v]["products"], keys, span, params)

    p_res, et_res = _stored("wb_ts", "p", params), _stored("wb_ts", "et", params)
    for v, res in (("p", p_res), ("et", et_res)):
        for key, err in res["errors"].items():
            st.error(f"{_VARIABLES[v]['products'][key].label}: {err}")
    bal = balance_series(p_res["monthly"], et_res["monthly"], p_members, et_members)
    if bal["n"].max() == 0:
        st.warning("No month has both P and ET data.")
        return

    # Same statistic as the Basin Time Series tab: median across products of mean annual totals
    pairs_annual = annual_totals(
        pair_differences(p_res["monthly"], et_res["monthly"], p_members, et_members)).mean().dropna()
    annual = pd.Series({
        "p": annual_totals(p_res["monthly"])[[k for k in p_members if k in p_res["monthly"]]].mean().median(),
        "et": annual_totals(et_res["monthly"])[[k for k in et_members if k in et_res["monthly"]]].mean().median(),
    })
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P (ensemble median)", f"{annual['p']:,.0f} mm/yr" if pd.notna(annual["p"]) else "—")
    c2.metric("ET (ensemble median)", f"{annual['et']:,.0f} mm/yr" if pd.notna(annual["et"]) else "—")
    if len(pairs_annual):
        c3.metric("P − ET (median of pairs)", f"{pairs_annual.median():,.0f} mm/yr",
                  help=f"Range across {len(pairs_annual)} P × ET pairs: "
                       f"{pairs_annual.min():,.0f} to {pairs_annual.max():,.0f} mm/yr")
    if pd.notna(annual["p"]) and annual["p"] > 0 and pd.notna(annual["et"]):
        c4.metric("ET / P (evaporative index)", f"{annual['et'] / annual['p']:.2f}",
                  help="Share of precipitation returned to the atmosphere.")
    if len(pairs_annual):
        st.caption(f"P − ET ranges from **{pairs_annual.min():,.0f}** to **{pairs_annual.max():,.0f} mm/yr** "
                   f"across the {len(pairs_annual)} P × ET product pairs (complete calendar years only).")

    _plot_balance(bal)
    _plot_seasonal_balance(bal)

    table = bal.rename(columns={"p": "P_median", "et": "ET_median",
                                **{c: f"P_minus_ET_{c}" for c in ("median", "q25", "q75", "min", "max", "n")}})
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "month"
    st.download_button(
        "📥 Download P − ET table (CSV)", table.to_csv().encode(),
        file_name=re.sub(r"[^\w\-]", "_", f"wb_p_minus_et_{params['label']}_{params['res']}deg") + ".csv",
        mime="text/csv", key="wb_bal_csv",
    )


def _render_balance_map(params: dict, p_members: list, et_members: list):
    st.markdown(f"#### P − ET map · {params['label']} ({params['unit']})")
    signature = (params["products"]["p"], params["products"]["et"])
    if st.button("▶ Map P − ET (ensemble medians)", width="stretch", key="wb_bal_map_run"):
        gridded_p, gridded_et = _gridded_members("p", params), _gridded_members("et", params)
        st.session_state["wb_balance_map"] = {
            "image": balance_image(list(gridded_p.values()), list(gridded_et.values()))
            .clip(st.session_state["wb_geometry"]),
            "signature": signature,
        }
    stored = st.session_state.get("wb_balance_map")
    if not stored or stored["signature"] != signature:
        st.caption("Median P minus median ET on the analysis grid, for the period in the sidebar.")
        return
    limit = _VIS_MAX["p_minus_et"][params["unit"]]
    _render_map(stored["image"], {"min": -limit, "max": limit, "palette": _BALANCE_PALETTE},
                f"balance::{','.join(p_members)}|{','.join(et_members)}",
                f"P − ET ({params['unit']})", res_to_meters(params["res"]), params["unit"])
    st.caption("Brown: ET exceeds P (water deficit). Teal: P exceeds ET (water available for runoff "
               "and recharge).")
    _render_export(stored["image"], params, "p_minus_et", "wb_bal_export", "P − ET")


def _plot_balance(bal: pd.DataFrame):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=bal.index, y=bal["p"], mode="lines", name="P (median)",
                             line=dict(color="#2171b5", width=2.2),
                             hovertemplate="P: %{y:.1f} mm<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bal.index, y=bal["et"], mode="lines", name="ET (median)",
                             line=dict(color="#31a354", width=2.2),
                             hovertemplate="ET: %{y:.1f} mm<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(
        x=bal.index, y=bal["median"], name="P − ET (median of pairs)",
        marker_color=["#01665e" if v >= 0 else "#8c510a" for v in bal["median"].fillna(0)],
        error_y=dict(type="data", array=bal["q75"] - bal["median"],
                     arrayminus=bal["median"] - bal["q25"], color="#555555", thickness=1),
        hovertemplate="P − ET: %{y:.1f} mm<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="mm/month", row=1, col=1)
    fig.update_yaxes(title_text="P − ET (mm/month)", row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(l=60, r=20, t=30, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Error bars: interquartile range of P − ET across all P × ET product pairs.")


def _plot_seasonal_balance(bal: pd.DataFrame):
    seasonal = bal.groupby(bal.index.month)[["p", "et", "median", "q25", "q75"]].mean()
    labels = [MONTH_NAMES[m - 1][:3] for m in seasonal.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=seasonal["median"], name="P − ET",
        marker_color=["#01665e" if v >= 0 else "#8c510a" for v in seasonal["median"].fillna(0)],
        error_y=dict(type="data", array=seasonal["q75"] - seasonal["median"],
                     arrayminus=seasonal["median"] - seasonal["q25"], color="#555555", thickness=1),
        hovertemplate="%{x} P − ET: %{y:.1f} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=labels, y=seasonal["p"], mode="lines+markers", name="P",
                             line=dict(color="#2171b5", width=2)))
    fig.add_trace(go.Scatter(x=labels, y=seasonal["et"], mode="lines+markers", name="ET",
                             line=dict(color="#31a354", width=2)))
    fig.update_layout(title="Mean seasonal cycle", height=360, yaxis_title="mm/month",
                      margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5))
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# TAB 5 — REFERENCE ET
# ===========================================================================

def _render_reference_tab(params: dict):
    covering = _covering(REF_ET_PRODUCTS, params)
    if not covering:
        st.warning("No reference-ET product covers the whole selected period.")
        return
    st.markdown(
        "Reference ET (ET₀) is the evaporative demand of a well-watered short grass surface. "
        "The ratio **ET / ET₀** shows how far water supply limits evaporation "
        "(≈1 for well-watered vegetation, ≪1 in water-limited regions)."
    )
    key = st.radio("Product", covering, format_func=lambda k: REF_ET_PRODUCTS[k].label,
                   horizontal=True, key="wb_ref_product")
    spec = REF_ET_PRODUCTS[key]
    asset = _availability(key)["asset"]
    image = period_image(spec, asset, params["months"], params["months_per_period"]).clip(
        st.session_state["wb_geometry"])
    st.caption(f"**{spec.label}** · {spec.algorithm} · native {spec.native_res_label} · "
               f"{params['label']} ({params['unit']})")
    _render_map(image, _vis("ref_et", params["unit"], _ET0_PALETTE), f"ref::{key}",
                f"ET₀ — {spec.label} ({params['unit']})", spec.native_res_deg * 111_320, params["unit"])

    span = month_range(params["first"], params["last"])
    if len(span) < 12:
        st.info("Choose **Calendar Year** or a multi-year period for the ET / ET₀ time series.")
        return
    if st.button("▶ Compute basin ET₀ time series", type="primary", width="stretch", key="wb_ref_run"):
        st.session_state["wb_ref_ts"] = _compute_series(REF_ET_PRODUCTS, covering, span, params)

    ref = st.session_state.get("wb_ref_ts")
    if not ref or ref["monthly"].empty:
        return
    for k, err in ref["errors"].items():
        st.error(f"{REF_ET_PRODUCTS[k].label}: {err}")

    fig = go.Figure()
    for k in ref["monthly"].columns:
        fig.add_trace(go.Scatter(x=ref["monthly"].index, y=ref["monthly"][k], mode="lines",
                                 name=REF_ET_PRODUCTS[k].label,
                                 line=dict(color=REF_ET_PRODUCTS[k].color, width=2)))
    et = _stored("wb_ts", "et", params)
    ens = None
    if et and not et["monthly"].empty and params["members"]["et"]:
        ens = ensemble_series(et["monthly"], list(params["members"]["et"]))
        fig.add_trace(go.Scatter(x=ens.index, y=ens["median"], mode="lines", name="ET ensemble median",
                                 line=dict(color="#111111", width=3)))
    fig.update_layout(height=400, hovermode="x unified", yaxis_title="mm/month",
                      margin=dict(l=60, r=20, t=30, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

    annual_ref = annual_totals(ref["monthly"]).mean()
    cols = st.columns(len(annual_ref) + (1 if ens is not None else 0))
    for col, (k, v) in zip(cols, annual_ref.items()):
        col.metric(f"Mean annual {REF_ET_PRODUCTS[k].label}", f"{v:,.0f} mm/yr" if pd.notna(v) else "—")
    if ens is not None:
        et_annual = annual_totals(ens[["median"]])["median"].mean()
        ref_annual = annual_ref.get("agera5_et0", annual_ref.iloc[0])
        if pd.notna(et_annual) and pd.notna(ref_annual) and ref_annual > 0:
            cols[-1].metric("ET / ET₀ (ensemble median)", f"{et_annual / ref_annual:.2f}")
    else:
        st.caption("Compute the ET series in the **Basin Time Series** tab to add the ET ensemble "
                   "and the ET / ET₀ ratio.")


# ===========================================================================
# PRODUCTS & METHODS
# ===========================================================================

_VARIABLE_NAMES = {"et": "Actual ET", "ref_et": "Reference ET", "p": "Precipitation",
                   "q": "Runoff", "ds": "Storage change"}


def _render_products_expander():
    groups = (P_PRODUCTS, ET_PRODUCTS, Q_PRODUCTS, DS_PRODUCTS, REF_ET_PRODUCTS)
    with st.expander("📚 Products, methods & live availability", expanded=False):
        rows = []
        for products in groups:
            for key, spec in products.items():
                av = _availability(key)
                rows.append({
                    "Product": spec.label,
                    "Variable": _VARIABLE_NAMES[spec.kind],
                    "Type": spec.method,
                    "Algorithm": spec.algorithm,
                    "Native": spec.native_res_label,
                    "Cadence": {"8day": "8-day", "3hourly": "3-hourly"}.get(spec.cadence, spec.cadence),
                    "Coverage": _lat_label(spec.lat_bounds) if spec.lat_bounds else "Global",
                    "Available": ("⚠️ " + av["error"][:60]) if "error" in av
                                 else f"{av['first_month']:%Y-%m} → {av['last_month']:%Y-%m}",
                    "Licence": spec.license,
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("""
**How products are harmonised**
- **One grid:** WGS84, cell edges anchored at (−180°, 90°), resolution chosen in the sidebar.
  Finer products are area-averaged into each cell (cells with < 50% valid fine pixels are masked);
  coarser products are sampled nearest-neighbour, so no detail is invented.
- **One time step:** calendar-month totals in mm. 8-day and dekadal composites are split across
  month boundaries by the days they overlap each month (verified to conserve annual totals exactly).
- **ET units:** MOD16 8-day sums ×0.1; PML mm/day ×0.01 (open water from its Ew band); WaPOR dekadal
  mean mm/day ×0.1; ERA5-Land metres with the sign flipped; GLDAS / FLDAS / SMAP kg m⁻² s⁻¹ × 86 400.
- **P units:** CHIRPS and PERSIANN-CDR mm/day; IMERG monthly mean mm/h × 24; CPC 0.1 mm/day;
  ERA5-Land metres × 1000; GLDAS forcing kg m⁻² s⁻¹ × 86 400; TerraClimate mm/month.
- **Q units:** surface + subsurface runoff. GLDAS-2.1 3-hourly accumulations × 8; GLDAS-2.2 / FLDAS /
  SMAP kg m⁻² s⁻¹ × 86 400; ERA5-Land metres × 1000; TerraClimate mm/month.
- **ΔS:** each product's storage is averaged per calendar month (GRACE solutions are assigned to the
  month holding their midpoint), then ΔS(m) = [S(m+1) − S(m−1)] / 2, falling back to a one-sided
  difference where one neighbour is missing. GRACE mascons are in cm (× 10); GLDAS-2.2 TWS in mm.
- **Basin means** are weighted by true cell area (a 0.1° cell shrinks with latitude).
- **Ensemble:** only products covering every month of the period, and (for CHIRPS and PERSIANN-CDR)
  the whole latitude span of the area, are members. Products differ widely — the spread is the
  uncertainty, not noise to hide.
- **Not included:** FLDAS and AgERA5 precipitation (re-use CHIRPS and ERA5), GSMaP (hourly only,
  too heavy for multi-year basin series), the GRACE spherical-harmonic land grids (end 2017 and need
  RL05 gain factors), and storage from models without groundwater (ERA5-Land, SMAP).

**Coming next:** the closed basin budget P − ET − Q − ΔS, runoff ratio and Budyko analysis (Phase 4).
""")
        st.markdown("**Citations**")
        for products in groups:
            for spec in products.values():
                st.markdown(f"- **{spec.label}** ({_VARIABLE_NAMES[spec.kind]}) — {spec.citation} "
                            f"[catalog]({spec.url})")
        st.caption("River basins: Lehner, B., Grill, G. (2013). Global river hydrography and network "
                   "routing: baseline data and new approaches to study the world's large river systems. "
                   "Hydrological Processes, 27(15), 2171–2186.")
