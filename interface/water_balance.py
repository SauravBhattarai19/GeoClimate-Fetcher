"""
Water Balance Interface Module — Phase 1: Evapotranspiration

Global, multi-product evapotranspiration on one common analysis grid, the first
component of a basin water balance (P − ET − Q = ΔS).

Uses shared platform components:
- GeometrySelectionWidget for custom AOIs, plus a HydroBASINS river-basin picker
- get_ee_tile_url for cached GEE map tiles
- DownloadHelper for GeoTIFF export (local or Google Drive)
GEE is already initialised by the main app.

Tabs:
  1. ET Map              one product, native resolution or analysis grid
  2. Basin Time Series   every product monthly + ensemble median / IQR
  3. Product Agreement   ensemble median, spread, CV and product count maps
  4. Reference ET        FAO-56 ET₀ and ET / ET₀
"""

import datetime
import math
import re

import ee
import folium
import pandas as pd
import plotly.graph_objects as go
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
    BASIN_LEVELS,
    DEFAULT_RES_DEG,
    ET_PRODUCTS,
    GRID_OPTIONS,
    REF_ET_PRODUCTS,
    VERSION,
    annual_totals,
    area_weighted_mean,
    basin_features,
    basin_monthly_series,
    basin_period_means,
    covers,
    ensemble_image,
    ensemble_series,
    fetch_availability,
    find_basin,
    gridded_period_image,
    native_projection,
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

_DEFAULT_YEAR = 2023                 # every ET product covers 2018–2023
_DEFAULT_RANGE = (2018, 2023)
_LARGE_AOI_KM2 = 5_000_000           # above this, fine products may hit GEE time limits

_ET_PALETTE = ["#fff7bc", "#d9f0a3", "#78c679", "#31a354", "#2c7fb8", "#253494"]
_ET0_PALETTE = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
_STD_PALETTE = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
_CV_PALETTE = ["#1a9850", "#a6d96a", "#ffffbf", "#fdae61", "#d73027"]
_COUNT_PALETTE = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]

# Colour-scale maxima per unit (mm/month or mm/yr)
_VIS_MAX = {
    "et":     {"mm/month": 200, "mm/yr": 1600},
    "ref_et": {"mm/month": 250, "mm/yr": 2500},
    "stdDev": {"mm/month": 60,  "mm/yr": 400},
}

_ENSEMBLE_LAYERS = {
    "median": "Ensemble median",
    "stdDev": "Spread (standard deviation)",
    "cv":     "Coefficient of variation (%)",
    "count":  "Products with data",
}

_CONTINENT_KEYS = ["globe", "europe", "asia", "north_america", "south_america", "africa", "australia"]
_SUB_KEYS = [k for k in PREDEFINED_REGION_LABELS if k not in _CONTINENT_KEYS]

# Session keys holding results — cleared when parameters or the AOI change
_RESULT_KEYS = ("wb_ts", "wb_agreement", "wb_ref_ts", "wb_aoi_means",
                "wb_tile_cache", "wb_sample_cache")
_AOI_KEYS = ("wb_geometry", "wb_geometry_complete", "wb_area_km2", "wb_centroid",
             "wb_aoi_name", "wb_aoi_fc", "wb_method_selection")


# ===========================================================================
# Availability (live catalog, cached daily)
# ===========================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def _availability(key: str) -> dict:
    """Live date range for a product, or {'error': ...} if no asset answers."""
    spec = ET_PRODUCTS.get(key) or REF_ET_PRODUCTS[key]
    try:
        return fetch_availability(spec)
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


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def render_water_balance():
    """Main entry point — called by interface/router.py."""
    st.markdown("## 🌊 Water Balance")
    st.markdown(
        "**Phase 1 — Evapotranspiration** · global multi-product ET harmonised to one "
        f"analysis grid and calendar-month totals (mm) · v{VERSION}"
    )
    st.divider()

    _auto_expand_sidebar()
    params = _render_sidebar()

    params_key = str(sorted((k, str(v)) for k, v in params.items() if k != "months"))
    if st.session_state.get("wb_params_key") != params_key:
        for k in _RESULT_KEYS:
            st.session_state.pop(k, None)
        st.session_state["wb_params_key"] = params_key

    if _render_geometry_selection():
        st.divider()
        tab_map, tab_ts, tab_agree, tab_ref = st.tabs([
            "🗺️ ET Map", "📈 Basin Time Series", "🤝 Product Agreement", "🌡️ Reference ET",
        ])
        with tab_map:
            _render_et_map_tab(params)
        with tab_ts:
            _render_time_series_tab(params)
        with tab_agree:
            _render_agreement_tab(params)
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
    st.sidebar.caption("Phase 1 · Evapotranspiration")

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
        avail = _available(ET_PRODUCTS)
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
    st.sidebar.subheader("🛰️ ET Products")
    covering = _covering(ET_PRODUCTS, params)
    st.sidebar.caption(
        f"**{len(covering)} of {len(ET_PRODUCTS)}** products cover every month of "
        f"{first:%b %Y} – {last:%b %Y}. Only these join the ensemble, so products "
        "entering or leaving mid-period cannot create artificial jumps."
    )
    # Keyed on the span so the default resets to the covering set when the period changes
    selected = st.sidebar.multiselect(
        "Products",
        list(ET_PRODUCTS),
        default=covering,
        format_func=lambda k: ET_PRODUCTS[k].label,
        key=f"wb_products_{first:%Y%m}_{last:%Y%m}",
    )
    params["products"] = tuple(selected)
    params["members"] = tuple(k for k in selected if k in covering)
    return params


# ===========================================================================
# AREA OF INTEREST
# ===========================================================================

def _set_aoi(geometry, name: str, area_km2: float, centroid: list, outline_fc=None):
    st.session_state["wb_geometry"] = geometry
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
            "⚠️ Very large area: time series for the 500 m – 1 km products (PML, MOD16, WaPOR, "
            "SSEBop) may exceed Earth Engine's interactive time limit. A coarser grid "
            "(0.25° / 0.5°) helps."
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

    state_key = f"wb_map_state_{layer_key.split('::')[0]}"
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


def _vis(kind: str, unit: str, palette: list) -> dict:
    return {"min": 0, "max": _VIS_MAX[kind][unit], "palette": palette}


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
# TAB 1 — ET MAP
# ===========================================================================

def _render_et_map_tab(params: dict):
    covering = _covering(ET_PRODUCTS, params)
    if not covering:
        st.warning("No ET product covers the whole selected period.")
        return

    c1, c2 = st.columns([2, 1])
    options = [k for k in params["products"] if k in covering] or covering
    key = c1.selectbox("Product", options, format_func=lambda k: ET_PRODUCTS[k].label,
                       key="wb_map_product")
    display = c2.radio("Display", ["Native resolution", f"Analysis grid ({params['res']}°)"],
                       key="wb_map_display", help="Native shows the product as published; the "
                       "grid view is exactly what the time series and ensemble use.")
    spec = ET_PRODUCTS[key]
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

    sample_scale = res_to_meters(params["res"]) if on_grid else spec.native_res_deg * 111_320
    _render_map(image, _vis("et", params["unit"], _ET_PALETTE),
                f"etmap::{key}::{on_grid}", f"ET — {spec.label} ({params['unit']})",
                sample_scale, params["unit"])

    mean_key = f"{key}::{params['res']}"
    means = st.session_state.setdefault("wb_aoi_means", {})
    if st.button("📊 Area-weighted AOI mean", key="wb_map_mean_btn"):
        with st.spinner("Averaging over the AOI on the analysis grid…"):
            try:
                means[mean_key] = area_weighted_mean(gridded, geom, params["res"]).getInfo()
            except Exception as exc:
                st.error(f"Could not compute the AOI mean: {exc}")
    if mean_key in means:
        res = means[mean_key]
        m1, m2 = st.columns(2)
        m1.metric(f"Mean ET ({params['unit']})",
                  f"{res['mean']:,.1f}" if res.get("mean") is not None else "no data")
        m2.metric("Area with data", f"{100 * (res.get('coverage') or 0):.0f}%")

    _render_export(gridded.clip(geom), params, f"et_{key}", "wb_map_export", f"{spec.label} ET")


# ===========================================================================
# TAB 2 — BASIN TIME SERIES
# ===========================================================================

def _render_time_series_tab(params: dict):
    span = month_range(params["first"], params["last"])
    if len(span) < 12:
        st.info("Choose **Calendar Year** or a multi-year period in the sidebar for a monthly time series.")
        return
    if not params["products"]:
        st.warning("Select at least one ET product in the sidebar.")
        return

    members = list(params["members"])
    outsiders = [k for k in params["products"] if k not in members]
    st.markdown(
        f"Monthly basin-mean ET, **{span[0]:%b %Y} – {span[-1]:%b %Y}**, area-weighted on the "
        f"{params['res']}° grid. Ensemble members: "
        + (", ".join(ET_PRODUCTS[k].label for k in members) if members else "none")
    )
    if outsiders:
        st.caption("Shown but not in the ensemble (partial coverage of the period): "
                   + ", ".join(ET_PRODUCTS[k].label for k in outsiders))

    if st.button(f"▶ Compute monthly time series ({len(params['products'])} products × {len(span)} months)",
                 type="primary", width="stretch", key="wb_ts_run"):
        st.session_state["wb_ts"] = _compute_series(ET_PRODUCTS, params["products"], span, params)

    result = st.session_state.get("wb_ts")
    if not result:
        return
    monthly, coverage, errors = result["monthly"], result["coverage"], result["errors"]
    for key, err in errors.items():
        st.error(f"{ET_PRODUCTS[key].label}: {err}")
    if monthly.empty:
        return

    ens = ensemble_series(monthly, members) if members else None
    _plot_monthly(monthly, ens, ET_PRODUCTS, "ET (mm/month)", members)

    low = [k for k in monthly.columns if coverage[k].mean() < 0.9]
    if low:
        st.warning(
            "⚠️ Low spatial coverage — basin mean uses only cells with data: "
            + ", ".join(f"{ET_PRODUCTS[k].label} ({100 * coverage[k].mean():.0f}%)" for k in low)
            + ". MOD16, for example, has no values over barren land, urban areas and water."
        )

    annual = annual_totals(monthly)
    if not annual.dropna(how="all").empty:
        _render_annual_summary(annual, members)

    table = monthly.rename(columns=lambda k: f"{k}_mm").copy()
    if ens is not None:
        table = table.join(ens.add_prefix("ensemble_"))
    table = table.join(coverage.rename(columns=lambda k: f"{k}_coverage"))
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "month"
    st.download_button(
        "📥 Download monthly table (CSV)", table.to_csv().encode(),
        file_name=re.sub(r"[^\w\-]", "_", f"wb_et_monthly_{params['label']}_{params['res']}deg") + ".csv",
        mime="text/csv", key="wb_ts_csv",
    )


def _compute_series(products: dict, keys, span: list, params: dict) -> dict:
    """Basin monthly series for each product, over the months it has data for."""
    geom = st.session_state["wb_geometry"]
    series, coverage, errors = {}, {}, {}
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
        except Exception as exc:
            errors[key] = str(exc)[:300]
    bar.empty()
    index = pd.to_datetime(span)
    return {
        "monthly": pd.DataFrame(series).reindex(index).astype(float),
        "coverage": pd.DataFrame(coverage).reindex(index).astype(float),
        "errors": errors,
    }


def _plot_monthly(monthly: pd.DataFrame, ens, products: dict, y_title: str, members: list):
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
    fig.update_layout(height=430, hovermode="x unified", yaxis_title=y_title,
                      margin=dict(l=60, r=20, t=30, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")


def _render_annual_summary(annual: pd.DataFrame, members: list):
    st.markdown("#### Annual totals")
    mean_annual = annual.mean()
    member_means = mean_annual[[k for k in members if k in mean_annual.index]].dropna()
    if len(member_means):
        c1, c2, c3 = st.columns(3)
        c1.metric("Ensemble median ET", f"{member_means.median():,.0f} mm/yr")
        c2.metric("Range across products", f"{member_means.min():,.0f} – {member_means.max():,.0f} mm/yr")
        c3.metric("Ensemble members", len(member_means))

    order = mean_annual.dropna().sort_values().index
    fig = go.Figure(go.Bar(
        x=[ET_PRODUCTS[k].label for k in order], y=mean_annual[order],
        marker_color=[ET_PRODUCTS[k].color for k in order],
        hovertemplate="%{x}: %{y:.0f} mm/yr<extra></extra>",
    ))
    if len(member_means):
        fig.add_hline(y=member_means.median(), line_dash="dash", line_color="#111111",
                      annotation_text="ensemble median", annotation_position="top left")
    fig.update_layout(height=340, yaxis_title="Mean annual ET (mm/yr)",
                      margin=dict(l=60, r=20, t=20, b=80))
    st.plotly_chart(fig, width="stretch")

    shown = annual.dropna(how="all").rename(columns=lambda k: ET_PRODUCTS[k].label)
    shown.index.name = "Year"
    st.dataframe(shown.style.format("{:.0f}", na_rep="—"), width="stretch")


# ===========================================================================
# TAB 3 — PRODUCT AGREEMENT
# ===========================================================================

def _render_agreement_tab(params: dict):
    members = list(params["members"])
    if len(members) < 2:
        st.info("Product agreement needs at least two selected products that cover the whole period.")
        return
    st.markdown(
        f"Per-cell statistics across **{len(members)} products** for **{params['label']}** "
        f"({params['unit']}), each resampled to the {params['res']}° grid."
    )
    if st.button("▶ Compute product agreement", type="primary", width="stretch", key="wb_agree_run"):
        geom = st.session_state["wb_geometry"]
        gridded = {
            k: gridded_period_image(ET_PRODUCTS[k], _availability(k)["asset"], params["months"],
                                    params["months_per_period"], params["res"])
            for k in members
        }
        bar = st.progress(0.0, text="Averaging each product over the AOI…")
        means = basin_period_means(
            gridded, geom, params["res"],
            on_product=lambda done, total, key: bar.progress(
                done / total, text=f"{ET_PRODUCTS[key].label} done ({done}/{total})"),
        )
        bar.empty()
        st.session_state["wb_agreement"] = {
            "ensemble": ensemble_image(list(gridded.values())).clip(geom),
            "means": means, "members": members,
        }

    result = st.session_state.get("wb_agreement")
    if not result:
        return

    for k, v in result["means"].items():
        if v.get("error"):
            st.error(f"{ET_PRODUCTS[k].label}: AOI mean failed — {v['error']}")
    means = {k: v for k, v in result["means"].items() if v.get("mean") is not None}
    if means:
        values = pd.Series({k: v["mean"] for k, v in means.items()}).sort_values()
        median = values.median()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Median of products ({params['unit']})", f"{median:,.1f}")
        c2.metric("Range", f"{values.min():,.1f} – {values.max():,.1f}")
        c3.metric("Spread (max / min)", f"{values.max() / values.min():.1f}×" if values.min() > 0 else "—")
        fig = go.Figure(go.Bar(
            x=[ET_PRODUCTS[k].label for k in values.index], y=values.values,
            marker_color=[ET_PRODUCTS[k].color for k in values.index],
            customdata=[100 * means[k]["coverage"] for k in values.index],
            hovertemplate="%{x}: %{y:.1f} mm<br>area with data %{customdata:.0f}%<extra></extra>",
        ))
        fig.add_hline(y=median, line_dash="dash", line_color="#111111",
                      annotation_text="median", annotation_position="top left")
        fig.update_layout(height=320, yaxis_title=f"AOI mean ET ({params['unit']})",
                          margin=dict(l=60, r=20, t=20, b=80))
        st.plotly_chart(fig, width="stretch")

    layer = st.radio("Map layer", list(_ENSEMBLE_LAYERS), format_func=lambda k: _ENSEMBLE_LAYERS[k],
                     horizontal=True, key="wb_agree_layer")
    unit = params["unit"]
    vis = {
        "median": _vis("et", unit, _ET_PALETTE),
        "stdDev": _vis("stdDev", unit, _STD_PALETTE),
        "cv":     {"min": 0, "max": 100, "palette": _CV_PALETTE},
        "count":  {"min": 0, "max": len(result["members"]), "palette": _COUNT_PALETTE},
    }[layer]
    value_unit = {"cv": "%", "count": "products"}.get(layer, unit)
    _render_map(result["ensemble"].select(layer), vis, f"agree::{layer}",
                _ENSEMBLE_LAYERS[layer], res_to_meters(params["res"]), value_unit)
    st.caption("High CV marks where products disagree most — treat ET there as uncertain. "
               "Product count drops where a product has no data (e.g. MOD16 over deserts).")

    _render_export(result["ensemble"].select(["median", "stdDev", "cv", "count"]), params,
                   "et_ensemble", "wb_agree_export", "ensemble (median, stdDev, CV, count)")


# ===========================================================================
# TAB 4 — REFERENCE ET
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
    et = st.session_state.get("wb_ts")
    ens = None
    if et and not et["monthly"].empty and params["members"]:
        ens = ensemble_series(et["monthly"], list(params["members"]))
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
        st.caption("Run the **Basin Time Series** tab to add the ET ensemble and the ET / ET₀ ratio.")


# ===========================================================================
# PRODUCTS & METHODS
# ===========================================================================

def _render_products_expander():
    with st.expander("📚 Products, methods & live availability", expanded=False):
        rows = []
        for products in (ET_PRODUCTS, REF_ET_PRODUCTS):
            for key, spec in products.items():
                av = _availability(key)
                rows.append({
                    "Product": spec.label,
                    "Variable": "Actual ET" if spec.kind == "et" else "Reference ET",
                    "Type": spec.method,
                    "Algorithm": spec.algorithm,
                    "Native": spec.native_res_label,
                    "Cadence": {"8day": "8-day", "3hourly": "3-hourly"}.get(spec.cadence, spec.cadence),
                    "Available": ("⚠️ " + av["error"][:60]) if "error" in av
                                 else f"{av['first_month']:%Y-%m} → {av['last_month']:%Y-%m}",
                    "Licence": spec.license,
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown(f"""
**How products are harmonised**
- **One grid:** WGS84, cell edges anchored at (−180°, 90°), resolution chosen in the sidebar.
  Finer products are area-averaged into each cell (cells with < 50% valid fine pixels are masked);
  coarser products are sampled nearest-neighbour, so no detail is invented.
- **One time step:** calendar-month totals in mm. 8-day and dekadal composites are split across
  month boundaries by the days they overlap each month (verified to conserve annual totals exactly).
- **Units:** MOD16 8-day sums ×0.1; PML mm/day ×0.01 (open water from its Ew band); WaPOR dekadal
  mean mm/day ×0.1; ERA5-Land metres with the sign flipped; GLDAS / FLDAS / SMAP kg m⁻² s⁻¹ × 86 400.
- **Basin means** are weighted by true cell area (a 0.1° cell shrinks with latitude).
- **Ensemble:** only products covering every month of the period are members (fixed membership).
  Global ET products differ widely — the spread is the uncertainty, not noise to hide.

**Coming next:** precipitation (Phase 2), runoff and storage change from GRACE (Phase 3),
and the closed basin budget P − ET − Q − ΔS (Phase 4).
""")
        st.markdown("**Citations**")
        for products in (ET_PRODUCTS, REF_ET_PRODUCTS):
            for spec in products.values():
                st.markdown(f"- **{spec.label}** — {spec.citation} [catalog]({spec.url})")
        st.caption("River basins: Lehner, B., Grill, G. (2013). Global river hydrography and network "
                   "routing: baseline data and new approaches to study the world's large river systems. "
                   "Hydrological Processes, 27(15), 2171–2186.")
