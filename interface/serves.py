"""
SERVES v2.0 Interface Module — Fully Integrated
Soil Moisture Estimation using the SERVES algorithm (Python/GEE)

Uses shared platform components:
- GeometrySelectionWidget for AOI selection (same as GLOBAL-SAM, Hydrology, etc.)
- DownloadHelper for GeoTIFF / CSV export
- GEE already initialised by main app — no auth gate here

Analysis modes:
  1. Single Date
  2. Monthly Average  (single year or multi-year composite)
  3. Seasonal Composite (single or multi-year)
  4. Annual Average    (single or multi-year)
  5. Custom Date Range
  6. Time Series       (monthly / bi-weekly / weekly)
  7. Regional Climatology (global & continental, MODIS)
"""

import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import plotly.graph_objects as go
import ee
import datetime
import io
import csv

from geoclimate_fetcher.core import GeometrySelectionWidget
from app_components.download_component import DownloadHelper
from geoclimate_fetcher.serves_calculator import (
    VIS_PARAMS,
    PREDEFINED_REGION_COORDS,
    PREDEFINED_REGION_LABELS,
    MONTH_NAMES,
    SEASON_LABELS,
    SOIL_DEPTH_MAPPING,
    DEFAULT_FIELD_CAPACITY,
    DEFAULT_WILTING_POINT,
    get_month_name,
    get_predefined_region,
    run_serves,
    run_serves_for_period,
    run_serves_time_series,
    run_serves_regional_climatology,
    run_serves_multi_year_monthly,
    run_serves_multi_year_seasonal,
    run_serves_multi_year_annual,
    run_serves_longterm_stats,
    run_serves_temporal_variability,
    get_month_date_range,
    get_season_date_range,
    get_image_statistics,
    extract_time_series_means,
    VERSION,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SERVES_PREFIX = "serves_"
_YEAR_RANGE = list(range(2024, 2012, -1))   # 2024 → 2013
_YEAR_OPTIONS = [str(y) for y in _YEAR_RANGE]

_ANALYSIS_MODES = {
    "single":         "📅 Single Date",
    "monthly":        "📆 Monthly Average",
    "seasonal":       "🌿 Seasonal Composite",
    "annual":         "📊 Annual Average",
    "custom_range":   "📋 Custom Date Range",
    "time_series":    "📈 Time Series",
    "regional":       "🌍 Regional Climatology",
    "longterm_stats": "📉 Long-term Statistics",
}

_SATELLITE_OPTIONS = {
    "landsat":   "Landsat 8/9 (30 m)",
    "sentinel2": "Sentinel-2 (10 m)",
    "modis":     "MODIS MOD13A2 (1 km)",
}

_COMPOSITE_OPTIONS = {
    "median":  "Median (recommended)",
    "closest": "Closest to date",
    "mean":    "Mean",
    "max":     "Maximum (greenest)",
}

_SOIL_DEPTH_LABELS = {
    "b0":   "0–5 cm",
    "b10":  "5–15 cm",
    "b30":  "15–30 cm (recommended)",
    "b60":  "30–60 cm",
    "b100": "60–100 cm",
    "b200": "100–200 cm",
}

_INTERVAL_OPTIONS = {
    "monthly": "Monthly",
    "16day":   "Bi-weekly (16-day)",
    "weekly":  "Weekly",
}


def _get_min_year(satellite: str) -> int:
    """Earliest year with reliable data per satellite used by SERVES."""
    return {"modis": 2000, "sentinel2": 2016}.get(satellite, 2013)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def render_serves():
    """Main entry point — called by interface/router.py."""
    st.markdown("## 🛰️ SERVES v2.0 — Soil Moisture Estimation")
    st.markdown(
        "**Soil-moisture Estimation of Root zone through Vegetation index-based "
        "Evapotranspiration fraction and Soil properties** · "
        f"v{VERSION} · Jackson State University & ERDC/CHL"
    )
    st.divider()

    _auto_expand_sidebar()
    params = _render_sidebar()

    # Auto-clear results whenever any sidebar param changes
    params_key = str(sorted((k, str(v)) for k, v in params.items()))
    if st.session_state.get("serves_params_key") != params_key:
        st.session_state.pop("serves_results", None)
        st.session_state.pop("serves_ts_data", None)
        st.session_state.pop("serves_tile_cache", None)
        st.session_state.pop("serves_sample_cache", None)
        st.session_state["serves_params_key"] = params_key

    mode = params["mode"]

    if mode == "regional":
        _render_regional_mode(params)
    else:
        geometry_ready = _render_geometry_selection()
        if geometry_ready:
            st.divider()
            _render_analysis_section(params)

    _render_about_expander()


# ===========================================================================
# SIDEBAR AUTO-EXPAND
# ===========================================================================

def _auto_expand_sidebar():
    """Open sidebar on the first visit to this module."""
    if not st.session_state.get("serves_sidebar_opened"):
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
        st.session_state.serves_sidebar_opened = True


# ===========================================================================
# SIDEBAR
# ===========================================================================

def _render_sidebar() -> dict:
    """Render all SERVES parameters in the sidebar. Returns a dict of all settings."""
    st.sidebar.header("🛰️ SERVES Parameters")

    # ── Analysis Mode ──────────────────────────────────────────────────────
    st.sidebar.subheader("🔬 Analysis Mode")
    mode_labels = list(_ANALYSIS_MODES.values())
    mode_keys   = list(_ANALYSIS_MODES.keys())
    current_mode = st.session_state.get("serves_mode", "single")
    default_idx  = mode_keys.index(current_mode) if current_mode in mode_keys else 0

    selected_label = st.sidebar.selectbox(
        "Mode:", mode_labels, index=default_idx, key="serves_mode_select"
    )
    mode = mode_keys[mode_labels.index(selected_label)]
    st.session_state["serves_mode"] = mode

    # ── Mode-specific date parameters ──────────────────────────────────────
    params = {"mode": mode}
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Date / Period")
    params.update(_render_date_params(mode))

    # ── Satellite & Composite ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛰️ Image Settings")

    sat_labels = list(_SATELLITE_OPTIONS.values())
    sat_keys   = list(_SATELLITE_OPTIONS.keys())

    if mode == "regional":
        # Separate key so regional default (MODIS) never bleeds into other modes
        reg_sat_default = st.session_state.get("serves_satellite_regional", "modis")
        reg_sat_idx     = sat_keys.index(reg_sat_default) if reg_sat_default in sat_keys else sat_keys.index("modis")
        sel_sat = st.sidebar.selectbox(
            "Satellite:", sat_labels, index=reg_sat_idx, key="serves_satellite_regional_sel"
        )
        params["satellite"] = sat_keys[sat_labels.index(sel_sat)]
        st.session_state["serves_satellite_regional"] = params["satellite"]
    else:
        sat_idx = sat_keys.index(st.session_state.get("serves_satellite", "landsat"))
        sel_sat = st.sidebar.selectbox(
            "Satellite:", sat_labels, index=sat_idx, key="serves_satellite_sel"
        )
        params["satellite"] = sat_keys[sat_labels.index(sel_sat)]
        st.session_state["serves_satellite"] = params["satellite"]

    # Warn when a high-res satellite is chosen for a continental predefined region
    if (mode == "regional"
            and params.get("regional_aoi_source", "predefined") == "predefined"
            and params["satellite"] != "modis"):
        _CONTINENT_SCALE = {
            "globe", "asia", "north_america", "south_america",
            "africa", "europe", "russia",
        }
        if params.get("region") in _CONTINENT_SCALE:
            st.sidebar.warning(
                "⚠️ Landsat/Sentinel-2 over a full continent will likely timeout. "
                "**MODIS recommended** for continent-scale analysis."
            )
        else:
            st.sidebar.caption(
                "⏱️ Landsat/Sentinel-2 over large sub-continental regions may be slow. "
                "MODIS is faster."
            )

    if mode == "single":
        comp_labels = list(_COMPOSITE_OPTIONS.values())
        comp_keys   = list(_COMPOSITE_OPTIONS.keys())
        comp_idx    = comp_keys.index(st.session_state.get("serves_composite", "median"))
        sel_comp    = st.sidebar.selectbox(
            "Composite Method:", comp_labels, index=comp_idx, key="serves_composite_sel"
        )
        params["composite_method"] = comp_keys[comp_labels.index(sel_comp)]
        st.session_state["serves_composite"] = params["composite_method"]

        params["search_window"] = st.sidebar.slider(
            "Search window (±days):", 5, 45, 16, key="serves_search_window"
        )
    else:
        params["composite_method"] = "median"
        params["search_window"]    = 16

    # ── Soil Parameters ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌱 Soil Parameters")

    soil_mode_options = ["SoilGrids (spatially variable)", "Uniform defaults"]
    soil_mode_idx = 0 if st.session_state.get("serves_soil_mode", "soilgrids") == "soilgrids" else 1
    sel_soil_mode = st.sidebar.radio(
        "Source:", soil_mode_options, index=soil_mode_idx, key="serves_soil_mode_radio"
    )
    params["soil_parameter_mode"] = "soilgrids" if sel_soil_mode == soil_mode_options[0] else "uniform"
    st.session_state["serves_soil_mode"] = params["soil_parameter_mode"]

    if params["soil_parameter_mode"] == "soilgrids":
        depth_labels = list(_SOIL_DEPTH_LABELS.values())
        depth_keys   = list(_SOIL_DEPTH_LABELS.keys())
        depth_idx    = depth_keys.index(st.session_state.get("serves_soil_depth", "b30"))
        sel_depth    = st.sidebar.selectbox(
            "Depth layer:", depth_labels, index=depth_idx, key="serves_depth_sel"
        )
        params["soil_depth"] = depth_keys[depth_labels.index(sel_depth)]
        st.session_state["serves_soil_depth"] = params["soil_depth"]
        params["field_capacity"] = None
        params["wilting_point"]  = None
    else:
        params["soil_depth"]     = "b30"
        params["field_capacity"] = st.sidebar.number_input(
            "Field Capacity (FC, cm³/cm³):",
            min_value=0.05, max_value=0.70,
            value=float(st.session_state.get("serves_fc", DEFAULT_FIELD_CAPACITY)),
            step=0.01, format="%.3f", key="serves_fc_input"
        )
        params["wilting_point"]  = st.sidebar.number_input(
            "Wilting Point (WP, cm³/cm³):",
            min_value=0.01, max_value=0.40,
            value=float(st.session_state.get("serves_wp", DEFAULT_WILTING_POINT)),
            step=0.01, format="%.3f", key="serves_wp_input"
        )
        st.session_state["serves_fc"] = params["field_capacity"]
        st.session_state["serves_wp"] = params["wilting_point"]

        if params["field_capacity"] <= params["wilting_point"]:
            st.sidebar.error("FC must be greater than WP.")

    # ── Quality Masks ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎭 Quality Masks")

    params["assign_water_to_fc"]        = st.sidebar.checkbox(
        "Assign water bodies → field capacity", value=True, key="serves_water_to_fc"
    )
    params["assign_negative_ndvi_to_fc"] = st.sidebar.checkbox(
        "Assign negative NDVI → field capacity", value=True, key="serves_neg_ndvi_fc"
    )
    params["mask_water"]    = st.sidebar.checkbox(
        "Mask water bodies (traditional)", value=False, key="serves_mask_water"
    )
    params["mask_non_veg"]  = st.sidebar.checkbox(
        "Mask non-vegetated areas", value=False, key="serves_mask_nonveg"
    )

    # ── Export Scale ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Export / Stats")
    params["scale"] = st.sidebar.number_input(
        "Resolution (m):", min_value=30, max_value=5000,
        value=int(st.session_state.get("serves_scale", 250)),
        step=10, key="serves_scale_input",
        help="Pixel resolution for export and statistics. Lower = higher quality but slower."
    )
    st.session_state["serves_scale"] = params["scale"]

    return params


def _render_date_params(mode: str) -> dict:
    """Render mode-specific date widgets. Returns partial params dict."""
    p = {}
    today = datetime.date.today()
    y_options = list(range(today.year, 2012, -1))

    if mode == "single":
        p["target_date"] = str(st.sidebar.date_input(
            "Target Date:", value=datetime.date(2023, 7, 15),
            max_value=today, key="serves_target_date"
        ))

    elif mode == "monthly":
        p["month"] = st.sidebar.selectbox(
            "Month:", range(1, 13),
            format_func=get_month_name, index=6, key="serves_month"
        )
        p["multi_year"] = st.sidebar.checkbox(
            "Multi-year composite", value=False, key="serves_monthly_multiyear"
        )
        if p["multi_year"]:
            p["start_year"] = st.sidebar.selectbox(
                "Start Year:", y_options, index=y_options.index(2018), key="serves_sy_monthly"
            )
            p["end_year"]   = st.sidebar.selectbox(
                "End Year:", y_options, index=y_options.index(2023), key="serves_ey_monthly"
            )
            p["year"] = None
        else:
            p["year"] = st.sidebar.selectbox(
                "Year:", y_options, index=y_options.index(2023), key="serves_year_monthly"
            )
            p["start_year"] = p["end_year"] = None

    elif mode == "seasonal":
        seas_labels = list(SEASON_LABELS.values())
        seas_keys   = list(SEASON_LABELS.keys())
        sel_seas    = st.sidebar.selectbox("Season:", seas_labels, index=2, key="serves_season")
        p["season"] = seas_keys[seas_labels.index(sel_seas)]
        p["hemisphere"] = "north" if st.sidebar.radio(
            "Hemisphere:", ["Northern", "Southern"], index=0, key="serves_hemi"
        ) == "Northern" else "south"
        p["multi_year"] = st.sidebar.checkbox(
            "Multi-year composite", value=False, key="serves_seasonal_multiyear"
        )
        if p["multi_year"]:
            p["start_year"] = st.sidebar.selectbox(
                "Start Year:", y_options, index=y_options.index(2020), key="serves_sy_seas"
            )
            p["end_year"]   = st.sidebar.selectbox(
                "End Year:", y_options, index=y_options.index(2023), key="serves_ey_seas"
            )
            p["year"] = None
        else:
            p["year"] = st.sidebar.selectbox(
                "Year:", y_options, index=y_options.index(2023), key="serves_year_seas"
            )
            p["start_year"] = p["end_year"] = None

    elif mode == "annual":
        p["multi_year"] = st.sidebar.checkbox(
            "Multi-year composite", value=False, key="serves_annual_multiyear"
        )
        if p["multi_year"]:
            p["start_year"] = st.sidebar.selectbox(
                "Start Year:", y_options, index=y_options.index(2020), key="serves_sy_ann"
            )
            p["end_year"]   = st.sidebar.selectbox(
                "End Year:", y_options, index=y_options.index(2023), key="serves_ey_ann"
            )
            p["year"] = None
        else:
            p["year"] = st.sidebar.selectbox(
                "Year:", y_options, index=y_options.index(2023), key="serves_year_ann"
            )
            p["start_year"] = p["end_year"] = None

    elif mode == "custom_range":
        p["start_date"] = str(st.sidebar.date_input(
            "Start Date:", value=datetime.date(2023, 1, 1),
            max_value=today, key="serves_start_custom"
        ))
        p["end_date"] = str(st.sidebar.date_input(
            "End Date:", value=datetime.date(2023, 12, 31),
            max_value=today, key="serves_end_custom"
        ))

    elif mode == "time_series":
        p["start_date"] = str(st.sidebar.date_input(
            "Start Date:", value=datetime.date(2022, 1, 1),
            max_value=today, key="serves_start_ts"
        ))
        p["end_date"] = str(st.sidebar.date_input(
            "End Date:", value=datetime.date(2023, 12, 31),
            max_value=today, key="serves_end_ts"
        ))
        int_labels = list(_INTERVAL_OPTIONS.values())
        int_keys   = list(_INTERVAL_OPTIONS.keys())
        sel_int    = st.sidebar.selectbox("Interval:", int_labels, index=0, key="serves_interval")
        p["interval"] = int_keys[int_labels.index(sel_int)]
        st.sidebar.caption("⚠️ Long time series (>36 steps) can be slow. Consider monthly interval.")

    elif mode == "regional":
        # AOI source — predefined continental region OR user-drawn/uploaded shapefile
        aoi_src_labels = ["🗺️ Predefined Region", "✏️ My Area (Draw / Upload)"]
        aoi_src_keys   = ["predefined", "custom"]
        sel_aoi_src    = st.sidebar.radio(
            "AOI Source:", aoi_src_labels, index=0, key="serves_reg_aoi_source"
        )
        p["regional_aoi_source"] = aoi_src_keys[aoi_src_labels.index(sel_aoi_src)]

        if p["regional_aoi_source"] == "predefined":
            reg_labels = list(PREDEFINED_REGION_LABELS.values())
            reg_keys   = list(PREDEFINED_REGION_LABELS.keys())
            sel_reg    = st.sidebar.selectbox(
                "Region:", reg_labels, index=1, key="serves_region"
            )
            p["region"] = reg_keys[reg_labels.index(sel_reg)]
        else:
            p["region"] = None  # geometry comes from the AOI widget in the main panel

        pt_labels = ["📅 Full Year (Annual)", "📆 Specific Month", "🌿 Specific Season"]
        pt_keys   = ["annual", "monthly", "seasonal"]
        sel_pt    = st.sidebar.selectbox(
            "Period:", pt_labels, index=0, key="serves_reg_period_type"
        )
        p["reg_period_type"] = pt_keys[pt_labels.index(sel_pt)]

        if p["reg_period_type"] == "monthly":
            p["month"] = st.sidebar.selectbox(
                "Month:", range(1, 13),
                format_func=get_month_name, index=6, key="serves_reg_month"
            )
        else:
            p["month"] = 1

        if p["reg_period_type"] == "seasonal":
            seas_labels = list(SEASON_LABELS.values())
            seas_keys   = list(SEASON_LABELS.keys())
            sel_seas    = st.sidebar.selectbox(
                "Season:", seas_labels, index=2, key="serves_reg_season"
            )
            p["reg_season"] = seas_keys[seas_labels.index(sel_seas)]
            p["reg_hemisphere"] = "north" if st.sidebar.radio(
                "Hemisphere:", ["Northern", "Southern"], index=0, key="serves_reg_hemi"
            ) == "Northern" else "south"
        else:
            p["reg_season"]     = "summer"
            p["reg_hemisphere"] = "north"

        p["start_year"] = st.sidebar.selectbox(
            "Start Year:", y_options, index=y_options.index(2019), key="serves_reg_sy"
        )
        p["end_year"]   = st.sidebar.selectbox(
            "End Year:", y_options, index=y_options.index(2023), key="serves_reg_ey"
        )

        stat_labels = ["Mean", "Minimum", "Maximum", "Median", "Std Deviation"]
        stat_keys   = ["mean", "min", "max", "median", "std_dev"]
        sel_stat    = st.sidebar.selectbox(
            "Summary Statistic:", stat_labels, index=3, key="serves_reg_stat"
        )
        p["regional_statistic"] = stat_keys[stat_labels.index(sel_stat)]
        if p["regional_aoi_source"] == "custom":
            st.sidebar.caption("Custom AOI — any satellite selectable above.")

    elif mode == "longterm_stats":
        # ── Statistics subtype ────────────────────────────────────────────────
        subtype_labels = [
            "📊 Inter-annual  (annual composites)",
            "〰️ Temporal Variability / CV  (monthly composites)",
        ]
        subtype_keys = ["interannual", "temporal_cv"]
        sel_sub = st.sidebar.radio(
            "Statistics type:", subtype_labels, index=0, key="serves_lt_subtype",
        )
        p["lt_stats_subtype"] = subtype_keys[subtype_labels.index(sel_sub)]

        st.sidebar.markdown("---")

        # ── Year range ────────────────────────────────────────────────────────
        current_sat = st.session_state.get("serves_satellite", "landsat")
        min_year    = _get_min_year(current_sat)
        lt_options  = list(range(today.year, min_year - 1, -1))

        default_sy = 2014 if 2014 in lt_options else lt_options[-1]
        default_ey = 2023 if 2023 in lt_options else lt_options[0]

        p["lt_start_year"] = st.sidebar.selectbox(
            "Start Year:", lt_options,
            index=lt_options.index(default_sy),
            key="serves_lt_start_year",
        )
        p["lt_end_year"] = st.sidebar.selectbox(
            "End Year:", lt_options,
            index=lt_options.index(default_ey),
            key="serves_lt_end_year",
        )

        # ── Period-within-year only for inter-annual mode ─────────────────────
        if p["lt_stats_subtype"] == "interannual":
            pt_labels = ["Full Year", "Specific Month", "Specific Season"]
            pt_keys   = ["annual", "monthly", "seasonal"]
            sel_pt    = st.sidebar.selectbox(
                "Period within each year:", pt_labels, index=0, key="serves_lt_period_type"
            )
            p["lt_period_type"] = pt_keys[pt_labels.index(sel_pt)]

            if p["lt_period_type"] == "monthly":
                p["lt_month"] = st.sidebar.selectbox(
                    "Month:", range(1, 13), format_func=get_month_name,
                    index=6, key="serves_lt_month",
                )
            else:
                p["lt_month"] = 1

            if p["lt_period_type"] == "seasonal":
                seas_labels = list(SEASON_LABELS.values())
                seas_keys   = list(SEASON_LABELS.keys())
                sel_seas    = st.sidebar.selectbox(
                    "Season:", seas_labels, index=2, key="serves_lt_season"
                )
                p["lt_season"] = seas_keys[seas_labels.index(sel_seas)]
                p["lt_hemisphere"] = "north" if st.sidebar.radio(
                    "Hemisphere:", ["Northern", "Southern"], index=0, key="serves_lt_hemi"
                ) == "Northern" else "south"
            else:
                p["lt_season"]     = "summer"
                p["lt_hemisphere"] = "north"

            st.sidebar.caption("⚠️ One composite per year. Tiles load on-demand via GEE.")
        else:
            # temporal_cv — always uses all 12 months, no period selector needed
            p["lt_period_type"] = "annual"
            p["lt_month"]       = 1
            p["lt_season"]      = "summer"
            p["lt_hemisphere"]  = "north"
            n_months = (p["lt_end_year"] - p["lt_start_year"] + 1) * 12
            st.sidebar.caption(
                f"⚠️ Builds {n_months} monthly composites "
                f"({p['lt_start_year']}–{p['lt_end_year']}). "
                "All GEE ops are lazy — tiles load on-demand."
            )

    return p


# ===========================================================================
# GEOMETRY SELECTION  (shared widget — same as GLOBAL-SAM, Hydrology, etc.)
# ===========================================================================

def _render_geometry_selection() -> bool:
    """Render AOI selection using the shared GeometrySelectionWidget. Returns True if ready."""
    if st.session_state.get("serves_geometry") is not None:
        st.success("✅ Area of Interest is ready for analysis")
        # Use cached area — avoids a .getInfo() call on every page render
        area_km2 = st.session_state.get("serves_area_km2")
        if area_km2 is not None:
            st.metric("Selected Area", f"{area_km2:.2f} km²")
        else:
            st.info("Geometry ready for analysis")

        if st.button("🗑️ Reset Area", key="serves_reset_geometry"):
            for k in ["serves_geometry", "serves_geometry_complete", "serves_area_km2",
                      "serves_centroid", "serves_results", "serves_ts_data",
                      "serves_tile_cache", "serves_sample_cache"]:
                st.session_state.pop(k, None)
            st.rerun()
        return True

    # Not yet selected — show the widget
    def on_geometry_selected(geometry):
        st.session_state["serves_geometry"] = geometry
        try:
            area_km2 = geometry.area().divide(1e6).getInfo()
            st.session_state["serves_area_km2"] = round(area_km2, 2)
        except Exception:
            pass
        try:
            centroid = geometry.centroid(maxError=1).coordinates().getInfo()
            st.session_state["serves_centroid"] = centroid   # [lon, lat]
        except Exception:
            pass

    widget = GeometrySelectionWidget(
        session_prefix=_SERVES_PREFIX,
        title="🗺️ Define Study Area",
    )
    if widget.render_complete_interface(on_geometry_selected=on_geometry_selected):
        st.rerun()
        return True

    return False


# ===========================================================================
# ANALYSIS SECTION  (AOI-based modes)
# ===========================================================================

def _render_analysis_section(params: dict):
    """Render the Run button and results for all AOI-based analysis modes."""
    mode       = params["mode"]
    study_area = st.session_state.get("serves_geometry")

    # Build a human-readable period label for the button
    period_label = _build_period_label(params)

    col_run, col_reset = st.columns([3, 1])
    with col_run:
        run_clicked = st.button(
            f"▶ Run SERVES Analysis — {period_label}",
            type="primary", width="stretch", key="serves_run_btn"
        )
    with col_reset:
        if st.button("🗑️ Clear Results", key="serves_clear_results"):
            st.session_state.pop("serves_results", None)
            st.session_state.pop("serves_ts_data", None)
            st.session_state.pop("serves_tile_cache", None)
            st.session_state.pop("serves_sample_cache", None)
            st.rerun()

    if run_clicked:
        _execute_analysis(params, study_area)

    # Always display cached results if available
    if st.session_state.get("serves_results"):
        _render_results(params, study_area)


def _execute_analysis(params: dict, study_area):
    """Run the SERVES computation and cache results in session state."""
    mode    = params["mode"]
    options = _build_options(params)

    with st.spinner("🌍 Running SERVES analysis on Google Earth Engine…"):
        try:
            if mode == "single":
                result = run_serves(study_area, params["target_date"], options)
                image  = result["soil_moisture"]
                count  = result["metadata"]["image_count"].getInfo()
                if not count:
                    st.error("❌ No satellite images found for this date/window. "
                             "Try a wider search window or a different date.")
                    return
                st.session_state["serves_results"] = {
                    "type":         "single",
                    "image":        image,
                    "full_image":   result["image"],
                    "ndvi":         result["ndvi"],
                    "et_fraction":  result["et_fraction"],
                    "metadata":     result["metadata"],
                    "params":       params,
                    "period_label": params["target_date"],
                }

            elif mode == "monthly":
                if params.get("multi_year"):
                    result = run_serves_multi_year_monthly(
                        study_area,
                        params["start_year"], params["end_year"], params["month"],
                        options,
                    )
                    period = f"{get_month_name(params['month'])} {params['start_year']}–{params['end_year']}"
                    count  = result["image_count"]
                else:
                    dr     = get_month_date_range(params["year"], params["month"])
                    result = run_serves_for_period(study_area, dr["start"], dr["end"], options)
                    period = f"{get_month_name(params['month'])} {params['year']}"

                st.session_state["serves_results"] = {
                    "type": "period", "image": result["soil_moisture"],
                    "ndvi": result.get("ndvi"), "period_label": period,
                    "params": params,
                }

            elif mode == "seasonal":
                season = params["season"]
                hemi   = params.get("hemisphere", "north")
                if params.get("multi_year"):
                    result = run_serves_multi_year_seasonal(
                        study_area,
                        params["start_year"], params["end_year"],
                        season, hemi, options,
                    )
                    period = (f"{SEASON_LABELS[season]} "
                              f"{params['start_year']}–{params['end_year']}")
                    count  = result["image_count"]
                else:
                    dr     = get_season_date_range(params["year"], season, hemi)
                    result = run_serves_for_period(study_area, dr["start"], dr["end"], options)
                    period = f"{SEASON_LABELS[season]} {params['year']}"

                st.session_state["serves_results"] = {
                    "type": "period", "image": result["soil_moisture"],
                    "ndvi": result.get("ndvi"), "period_label": period,
                    "params": params,
                }

            elif mode == "annual":
                if params.get("multi_year"):
                    result = run_serves_multi_year_annual(
                        study_area, params["start_year"], params["end_year"], options
                    )
                    period = f"Annual {params['start_year']}–{params['end_year']}"
                    count  = result["image_count"]
                else:
                    start = ee.Date.fromYMD(params["year"], 1, 1)
                    end   = ee.Date.fromYMD(params["year"] + 1, 1, 1)
                    result = run_serves_for_period(study_area, start, end, options)
                    period = f"Year {params['year']}"

                st.session_state["serves_results"] = {
                    "type": "period", "image": result["soil_moisture"],
                    "ndvi": result.get("ndvi"), "period_label": period,
                    "params": params,
                }

            elif mode == "custom_range":
                result = run_serves_for_period(
                    study_area, params["start_date"], params["end_date"], options
                )
                period = f"{params['start_date']} → {params['end_date']}"
                st.session_state["serves_results"] = {
                    "type": "period", "image": result["soil_moisture"],
                    "ndvi": result.get("ndvi"), "period_label": period,
                    "params": params,
                }

            elif mode == "time_series":
                result = run_serves_time_series(
                    study_area,
                    params["start_date"], params["end_date"],
                    params["interval"], options,
                )
                if result["num_steps"] == 0:
                    st.error("❌ Date range produces zero time steps.")
                    return

                # Extract time series means
                with st.spinner("📊 Extracting time series statistics…"):
                    scale = params.get("scale", 500)
                    dates, means = extract_time_series_means(
                        result["image_collection"], study_area, scale=scale
                    )
                st.session_state["serves_ts_data"] = {"dates": dates, "means": means}

                # Preview: first image
                first_img = ee.Image(result["image_collection"].first())
                period    = f"{params['start_date']} → {params['end_date']} ({params['interval']})"
                st.session_state["serves_results"] = {
                    "type":       "time_series",
                    "image":      first_img,
                    "collection": result["image_collection"],
                    "period_label": period,
                    "num_steps":  result["num_steps"],
                    "params":     params,
                }

            elif mode == "longterm_stats":
                sy = params["lt_start_year"]
                ey = params["lt_end_year"]
                if sy >= ey:
                    st.error("❌ Start year must be before end year.")
                    return

                subtype = params.get("lt_stats_subtype", "interannual")
                period  = _build_period_label(params)

                if subtype == "temporal_cv":
                    result = run_serves_temporal_variability(
                        study_area, sy, ey, options
                    )
                    st.session_state["serves_results"] = {
                        "type":           "longterm_stats",
                        "lt_stats_subtype": "temporal_cv",
                        "image":          result["mean"],
                        "mean":           result["mean"],
                        "std_dev":        result["std_dev"],
                        "cv":             result["cv"],
                        "total_months":   result["total_months"],
                        "period_label":   period,
                        "params":         params,
                    }
                else:
                    result = run_serves_longterm_stats(
                        study_area,
                        sy, ey,
                        options,
                        period_type = params["lt_period_type"],
                        month       = params.get("lt_month", 1),
                        season      = params.get("lt_season", "summer"),
                        hemisphere  = params.get("lt_hemisphere", "north"),
                    )
                    st.session_state["serves_results"] = {
                        "type":             "longterm_stats",
                        "lt_stats_subtype": "interannual",
                        "image":            result["mean"],
                        "mean":             result["mean"],
                        "min":              result["min"],
                        "max":              result["max"],
                        "median":           result["median"],
                        "std_dev":          result["std_dev"],
                        "collection":       result["collection"],
                        "year_count":       result["year_count"],
                        "period_label":     period,
                        "params":           params,
                    }

            st.rerun()

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            st.info(
                "Tips: reduce the area, narrow the date range, increase scale, "
                "or try MODIS for large regions."
            )


# ===========================================================================
# REGIONAL CLIMATOLOGY MODE
# ===========================================================================

def _render_regional_mode(params: dict):
    """Render the run button and results for Regional Climatology mode."""
    aoi_source  = params.get("regional_aoi_source", "predefined")
    pt          = params.get("reg_period_type", "annual")
    if pt == "monthly":
        period_label = f"{get_month_name(params['month'])} {params['start_year']}–{params['end_year']}"
    elif pt == "seasonal":
        period_label = f"{SEASON_LABELS.get(params.get('reg_season','summer'))} {params['start_year']}–{params['end_year']}"
    else:
        period_label = f"Annual {params['start_year']}–{params['end_year']}"

    if aoi_source == "custom":
        # Show the standard AOI widget; bail if geometry not yet set
        geometry_ready = _render_geometry_selection()
        if not geometry_ready:
            return
        study_area   = st.session_state.get("serves_geometry")
        region_label = "Custom Area"
        st.divider()
    else:
        study_area   = None  # backend resolves from region_name
        region_label = PREDEFINED_REGION_LABELS.get(params.get("region", "europe"), "Europe")
        st.markdown("### 🌍 Regional Climatology Mode")
        st.info(
            "Compute multi-year monthly soil moisture climatology over any predefined region. "
            "**MODIS (1 km)** is recommended for continent-scale regions — Landsat and Sentinel-2 "
            "are available for sub-continental regions but will be slower."
        )

    col_run, col_reset = st.columns([3, 1])
    with col_run:
        run_clicked = st.button(
            f"▶ Run Regional Climatology — {region_label} · {period_label}",
            type="primary", width="stretch", key="serves_reg_run_btn",
        )
    with col_reset:
        if st.button("🗑️ Clear", key="serves_reg_clear"):
            st.session_state.pop("serves_results", None)
            st.session_state.pop("serves_tile_cache", None)
            st.session_state.pop("serves_sample_cache", None)
            st.rerun()

    if run_clicked:
        options = _build_options(params)
        with st.spinner(f"🌍 Computing regional climatology for {region_label}…"):
            try:
                result = run_serves_regional_climatology(
                    params.get("region", "europe"),
                    params["start_year"], params["end_year"],
                    params["month"], options,
                    study_area=study_area,
                    period_type=params.get("reg_period_type", "annual"),
                    season=params.get("reg_season", "summer"),
                    hemisphere=params.get("reg_hemisphere", "north"),
                )
                stat_key = params.get("regional_statistic", "median")
                stat_map = {
                    "mean": result["mean"], "min": result["min"],
                    "max": result["max"],   "median": result["median"],
                    "std_dev": result["std_dev"],
                }
                st.session_state["serves_results"] = {
                    "type":         "regional",
                    "image":        stat_map[stat_key],
                    "region":       params.get("region"),
                    "region_label": region_label,
                    "study_area":   result["study_area"],
                    "period_label": f"{region_label} · {period_label}",
                    "stat_key":     stat_key,
                    "year_count":   result["year_count"],
                    "params":       params,
                }
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Regional analysis failed: {exc}")

    if st.session_state.get("serves_results", {}).get("type") == "regional":
        _render_regional_results()


def _render_regional_results():
    cached = st.session_state["serves_results"]
    params = cached["params"]
    image  = cached["image"]
    study_area = cached["study_area"]

    st.success(f"✅ Regional Climatology complete: **{cached['period_label']}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Region",    cached.get("region_label") or
                             PREDEFINED_REGION_LABELS.get(cached.get("region"), "Custom"))
    col2.metric("Years",     str(cached["year_count"]))
    col3.metric("Statistic", cached["stat_key"].capitalize())

    st.markdown("#### 🗺️ Soil Moisture Map")
    map_ok = _render_map_with_layer(
        image, VIS_PARAMS["soil_moisture"], study_area,
        f"SM {cached['stat_key'].capitalize()}",
        region_name=cached.get("region"),   # None for custom AOI → uses centroid
        caption=f"Soil Moisture {cached['stat_key'].capitalize()} (cm³/cm³)",
    )

    # Export
    st.markdown("---")
    with st.expander("💾 Export GeoTIFF", expanded=not map_ok):
        _render_export_section(image, study_area, params, label=cached["period_label"])


# ===========================================================================
# RESULTS DISPLAY  (AOI-based modes)
# ===========================================================================

def _render_results(params: dict, study_area):
    """Display map, statistics, time series chart, and export for the cached result."""
    cached = st.session_state["serves_results"]
    rtype  = cached["type"]
    image  = cached["image"]

    st.success(f"✅ Analysis complete: **{cached['period_label']}**")

    if rtype == "time_series":
        _render_time_series_results(cached, study_area, params)
    elif rtype == "longterm_stats":
        _render_longterm_stats_results(cached, study_area, params)
    else:
        _render_static_results(cached, study_area, params)


def _render_longterm_stats_results(cached: dict, study_area, params: dict):
    """Dispatch to sub-renderer based on statistics subtype."""
    subtype = cached.get("lt_stats_subtype", "interannual")
    if subtype == "temporal_cv":
        _render_temporal_cv_results(cached, study_area, params)
    else:
        _render_interannual_results(cached, study_area, params)


def _render_interannual_results(cached: dict, study_area, params: dict):
    """Mean / min / max / std dev across annual composites — inter-annual variability."""
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Valid Years",  str(cached["year_count"]))
    sc2.metric("Year Range",   f"{params['lt_start_year']} – {params['lt_end_year']}")
    sc3.metric("Period Type",  params["lt_period_type"].capitalize())


    st.markdown("#### 🗺️ Pixel-wise Statistical Maps")
    st.caption(
        "Statistics computed across **annual composites** — captures inter-annual variability "
        "(year-to-year shifts in the annual mean). Maps render via GEE tiles."
    )

    tab_mean, tab_min, tab_max, tab_std = st.tabs(
        ["📊 Mean", "⬇️ Minimum", "⬆️ Maximum", "〰️ Std Deviation"]
    )

    _MAP_UNAVAILABLE = "Map unavailable — GEE memory limit exceeded. Use **Drive Export** below."

    with tab_mean:
        st.caption("Long-term **mean** soil moisture — typical wet/dry spatial patterns.")
        ok_mean = _render_map_with_layer(
            cached["mean"], VIS_PARAMS["soil_moisture"], study_area,
            "SM Mean", caption="Mean Soil Moisture (cm³/cm³)",
        )
    with tab_min:
        st.caption("Pixel-wise **minimum annual mean** — driest year at each location.")
        if ok_mean:
            ok_min = _render_map_with_layer(
                cached["min"], VIS_PARAMS["soil_moisture"], study_area,
                "SM Minimum", caption="Minimum Annual Mean SM (cm³/cm³)",
            )
        else:
            st.info(_MAP_UNAVAILABLE)
            ok_min = False
    with tab_max:
        st.caption("Pixel-wise **maximum annual mean** — wettest year at each location.")
        if ok_mean:
            ok_max = _render_map_with_layer(
                cached["max"], VIS_PARAMS["soil_moisture"], study_area,
                "SM Maximum", caption="Maximum Annual Mean SM (cm³/cm³)",
            )
        else:
            st.info(_MAP_UNAVAILABLE)
            ok_max = False
    with tab_std:
        st.caption(
            "**Std dev across annual means** — inter-annual variability. "
            "High = unstable year-to-year SM."
        )
        if ok_mean:
            ok_std = _render_map_with_layer(
                cached["std_dev"], VIS_PARAMS["soil_moisture_stddev"], study_area,
                "SM Std Dev", caption="Std Dev of Annual Means (cm³/cm³)",
            )
        else:
            st.info(_MAP_UNAVAILABLE)
            ok_std = False

    any_map_failed = not (ok_mean and ok_min and ok_max and ok_std)
    st.markdown("---")
    with st.expander("💾 Export a statistical image as GeoTIFF", expanded=any_map_failed):
        stat_choice = st.selectbox(
            "Statistic to export:",
            ["Mean", "Minimum", "Maximum", "Std Deviation"],
            key="serves_lt_export_choice",
        )
        img_map = {
            "Mean":          cached["mean"],
            "Minimum":       cached["min"],
            "Maximum":       cached["max"],
            "Std Deviation": cached["std_dev"],
        }
        _render_export_section(
            img_map[stat_choice], study_area, params,
            label=f"{stat_choice}_{cached['period_label']}",
        )


def _render_temporal_cv_results(cached: dict, study_area, params: dict):
    """Mean / std dev / CV across monthly composites — total temporal variability."""
    n_months = cached["total_months"]
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Monthly Composites", str(n_months))
    sc2.metric("Year Range",         f"{params['lt_start_year']} – {params['lt_end_year']}")
    sc3.metric("Method",             "Temporal CV")

    st.markdown("#### 🗺️ Temporal Variability Maps")
    st.caption(
        "Statistics computed across **all monthly composites** — captures total temporal "
        "variability (seasonal amplitude + inter-annual shifts). "
        "CV normalises for baseline wetness so arid and humid pixels are directly comparable."
    )

    tab_mean, tab_std, tab_cv = st.tabs(
        ["📊 Long-term Mean", "〰️ Temporal Std Dev", "📉 CV (%)"]
    )

    _MAP_UNAVAILABLE = "Map unavailable — GEE memory limit exceeded. Use **Drive Export** below."

    with tab_mean:
        st.caption(
            "Long-term mean SM across all monthly composites — "
            "equivalent to the multi-year climatological average."
        )
        ok_mean = _render_map_with_layer(
            cached["mean"], VIS_PARAMS["soil_moisture"], study_area,
            "LT Mean SM", caption="Long-term Mean Soil Moisture (cm³/cm³)",
        )

    with tab_std:
        st.caption(
            "**Total temporal std dev** across all monthly composites — "
            "combines seasonal amplitude and inter-annual variability. "
            "High values mark areas with strong wet/dry cycling."
        )
        if ok_mean:
            ok_std = _render_map_with_layer(
                cached["std_dev"], VIS_PARAMS["soil_moisture_stddev"], study_area,
                "Temporal Std Dev", caption="Temporal Std Dev (cm³/cm³)",
            )
        else:
            st.info(_MAP_UNAVAILABLE)
            ok_std = False

    with tab_cv:
        st.caption(
            "**Coefficient of Variation CV = σ/μ × 100 %** — relative fluctuation "
            "independent of baseline wetness. Semi-arid regions typically show high CV "
            "(> 30 %) while humid forests show low CV (< 10 %). "
            "High CV = soil moisture is highly responsive to precipitation variability."
        )
        if ok_mean:
            ok_cv = _render_map_with_layer(
                cached["cv"], VIS_PARAMS["soil_moisture_cv"], study_area,
                "SM CV (%)", caption="Temporal CV of Soil Moisture (%)",
            )
        else:
            st.info(_MAP_UNAVAILABLE)
            ok_cv = False

    any_map_failed = not (ok_mean and ok_std and ok_cv)
    st.markdown("---")
    with st.expander("💾 Export a variability image as GeoTIFF", expanded=any_map_failed):
        stat_choice = st.selectbox(
            "Layer to export:",
            ["Long-term Mean", "Temporal Std Dev", "CV (%)"],
            key="serves_lt_cv_export_choice",
        )
        img_map = {
            "Long-term Mean":  cached["mean"],
            "Temporal Std Dev": cached["std_dev"],
            "CV (%)":          cached["cv"],
        }
        _render_export_section(
            img_map[stat_choice], study_area, params,
            label=f"{stat_choice.replace(' ', '_')}_{cached['period_label']}",
        )


def _render_static_results(cached: dict, study_area, params: dict):
    """Render map + stats for single / period results."""
    image = cached["image"]
    ndvi  = cached.get("ndvi")

    # ── Map FIRST — fails fast via tile cache; export accessible immediately ──
    st.markdown("#### 🗺️ Soil Moisture Map")
    map_ok = _render_map_with_layer(
        image, VIS_PARAMS["soil_moisture"], study_area,
        "Soil Moisture",
        ndvi_image=ndvi,
        caption="Soil Moisture (cm³/cm³)",
    )

    # ── Statistics — only when map succeeded (same GEE computation) ──────────
    # Cached in `cached` dict so widget interactions never re-call GEE.
    if map_ok:
        st.markdown("#### 📊 Statistics")
        if "stats" not in cached:
            with st.spinner("Computing statistics…"):
                try:
                    cached["stats"] = get_image_statistics(
                        image, study_area,
                        scale=params.get("scale", 500),
                        band="soil_moisture",
                    )
                except Exception:
                    cached["stats"] = {}
        stats    = cached["stats"]
        mean_val = stats.get("soil_moisture_mean") or stats.get("soil_moisture")
        min_val  = stats.get("soil_moisture_min")
        max_val  = stats.get("soil_moisture_max")
        std_val  = stats.get("soil_moisture_stdDev")
        if any(v is not None for v in [mean_val, min_val, max_val, std_val]):
            mc1, mc2, mc3, mc4 = st.columns(4)
            if mean_val is not None:
                mc1.metric("Mean SM (cm³/cm³)", f"{mean_val:.4f}")
            if min_val is not None:
                mc2.metric("Min SM (cm³/cm³)", f"{min_val:.4f}")
            if max_val is not None:
                mc3.metric("Max SM (cm³/cm³)", f"{max_val:.4f}")
            if std_val is not None:
                mc4.metric("Std Dev", f"{std_val:.4f}")
        else:
            st.info("Statistics could not be computed for this region/scale.")

    # ── Export — always accessible, auto-opens when map fails ────────────────
    st.markdown("---")
    full_image = cached.get("full_image", image)
    with st.expander("💾 Export GeoTIFF", expanded=not map_ok):
        _render_export_section(full_image, study_area, params, label=cached["period_label"])


def _render_time_series_results(cached: dict, study_area, params: dict):
    """Render time series chart + preview map for time_series mode."""
    ts_data = st.session_state.get("serves_ts_data", {})
    dates   = ts_data.get("dates", [])
    means   = ts_data.get("means", [])

    # ── Time Series Chart ───────────────────────────────────────────────
    if dates and means:
        st.markdown("#### 📈 Soil Moisture Time Series")
        _render_time_series_chart(dates, means)

    # ── Preview map (first image) ────────────────────────────────────────
    st.markdown("#### 🗺️ Preview — First Time Step")
    _render_map_with_layer(
        cached["image"], VIS_PARAMS["soil_moisture"], study_area,
        "Soil Moisture (first step)",
        caption="Soil Moisture (cm³/cm³)",
    )

    # ── CSV Export of time series ────────────────────────────────────────
    if dates and means:
        st.markdown("---")
        st.markdown("#### 💾 Export Time Series CSV")
        valid_pairs = [(d, m) for d, m in zip(dates, means) if m is not None]
        if valid_pairs:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["date", "mean_soil_moisture_cm3_cm3"])
            writer.writerows(valid_pairs)
            st.download_button(
                label="📥 Download Time Series CSV",
                data=buf.getvalue().encode(),
                file_name=f"serves_timeseries_{params.get('start_date','')}_"
                          f"{params.get('end_date','')}.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
                key="serves_ts_csv_dl",
            )

    # ── GeoTIFF export of first image ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Export GeoTIFF (first time step)")
    _render_export_section(
        cached["image"], study_area, params,
        label=f"first step of {cached['period_label']}"
    )


# ===========================================================================
# TILE URL CACHE
# ===========================================================================

_TILE_FAILED = "__FAILED__"   # sentinel stored in cache when getMapId() fails


def _get_tile_url(image, vis_params: dict, cache_key: str) -> str:
    """
    Register an ee.Image with GEE once and cache the tile URL by cache_key.
    Returns the cached URL instantly on every subsequent Streamlit re-render.

    Failures are also cached (sentinel _TILE_FAILED) so a failing getMapId()
    is never retried on re-renders — avoids repeated GEE calls when the user
    changes any widget (e.g., export mode dropdown) after a memory-limit error.

    Cache is keyed on serves_params_key + layer name and invalidated whenever
    analysis parameters change or results are cleared.
    """
    tile_cache = st.session_state.setdefault("serves_tile_cache", {})
    if cache_key not in tile_cache:
        try:
            map_id = image.getMapId(vis_params)
            tile_cache[cache_key] = map_id["tile_fetcher"].url_format
        except Exception as exc:
            tile_cache[cache_key] = _TILE_FAILED  # cache the failure — don't retry
            raise RuntimeError(str(exc)) from exc
    url = tile_cache[cache_key]
    if url == _TILE_FAILED:
        raise RuntimeError(
            "Map preview unavailable (cached failure). "
            "Use Drive Export below — batch export has higher memory limits."
        )
    return url


# ===========================================================================
# MAP RENDERING
# ===========================================================================

def _render_map_with_layer(image, vis_params: dict, region, layer_name: str,
                            ndvi_image=None, region_name: str = None,
                            caption: str = "Soil Moisture (cm³/cm³)",
                            sample_scale: int = 500) -> bool:
    """
    Build a Folium map with GEE tile layer, colorbar, and pixel inspector.
    Saves/restores pan+zoom so click reruns don't reset the viewport.
    Returns True if the GEE layer rendered successfully.
    """
    # ── Compute initial centre & zoom ────────────────────────────────────
    if region_name and region_name in PREDEFINED_REGION_COORDS:
        coords = PREDEFINED_REGION_COORDS[region_name]
        init_lat  = (coords[1] + coords[3]) / 2
        init_lon  = (coords[0] + coords[2]) / 2
        init_zoom = 3
    else:
        cached_centroid = st.session_state.get("serves_centroid")
        if cached_centroid:
            init_lon, init_lat = cached_centroid[0], cached_centroid[1]
        else:
            try:
                centroid = region.centroid(maxError=1).coordinates().getInfo()
                init_lon, init_lat = centroid[0], centroid[1]
                st.session_state["serves_centroid"] = centroid
            except Exception:
                init_lat, init_lon = 20, 0
        area_km2  = st.session_state.get("serves_area_km2", 10000)
        init_zoom = 9 if area_km2 < 1000 else (7 if area_km2 < 50000 else 4)

    # ── Restore saved pan/zoom so click reruns keep the user's viewport ──
    _map_state_key = f"serves_map_state_{layer_name}"
    saved     = st.session_state.get(_map_state_key, {})
    center_lat = saved.get("lat", init_lat)
    center_lon = saved.get("lon", init_lon)
    zoom       = saved.get("zoom", init_zoom)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Satellite basemap
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # ── SERVES tile layer (URL cached — never re-calls getMapId on re-render) ──
    _params_key = st.session_state.get("serves_params_key", "")
    _tile_key   = f"{_params_key}::{layer_name}"
    layer_ok    = False
    try:
        tile_url = _get_tile_url(image, vis_params, _tile_key)
        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine",
            name=layer_name,
            overlay=True,
            control=True,
        ).add_to(m)
        layer_ok = True
    except Exception as exc:
        err_str = str(exc).lower()
        if "memory" in err_str or "limit" in err_str or "quota" in err_str:
            st.warning(
                "🚫 **Map preview unavailable** — GEE memory limit exceeded for this "
                "computation. Your results are complete. "
                "**💾 Use Drive Export (below) to download** — batch export runs on "
                "GEE's higher-limit infrastructure and will succeed."
            )
        else:
            st.warning(f"Could not render map layer: {exc}")

    # Optional NDVI overlay
    if ndvi_image is not None and layer_ok:
        try:
            ndvi_url = _get_tile_url(ndvi_image, VIS_PARAMS["ndvi"], f"{_tile_key}::ndvi")
            folium.TileLayer(
                tiles=ndvi_url,
                attr="Google Earth Engine — NDVI",
                name="NDVI",
                overlay=True,
                control=True,
                show=False,
            ).add_to(m)
        except Exception:
            pass

    # Colorbar
    LinearColormap(
        colors=vis_params["palette"],
        vmin=vis_params["min"],
        vmax=vis_params["max"],
        caption=caption,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # ── Render — unique key per layer so multi-tab maps don't share state ──
    _folium_key = "folium_" + "".join(c if c.isalnum() else "_" for c in layer_name)
    map_data = st_folium(
        m,
        width=700,
        height=480,
        returned_objects=["last_clicked"],   # only clicks trigger reruns
        key=_folium_key,
    )

    # Save pan/zoom from this render so the next click-rerun restores the viewport
    if map_data:
        _c = map_data.get("center")   # {"lat": ..., "lng": ...}
        _z = map_data.get("zoom")
        if _c and _z:
            st.session_state[_map_state_key] = {
                "lat": _c["lat"], "lon": _c["lng"], "zoom": int(_z),
            }

    # ── Pixel inspector ──────────────────────────────────────────────────
    if layer_ok and map_data:
        clicked = map_data.get("last_clicked")
        if clicked:
            clat, clng = clicked["lat"], clicked["lng"]
            _sample_cache = st.session_state.setdefault("serves_sample_cache", {})
            _sk = f"{_params_key}::{layer_name}::{clat:.4f}::{clng:.4f}"
            if _sk not in _sample_cache:
                with st.spinner("Sampling pixel…"):
                    try:
                        pt = ee.Geometry.Point([clng, clat])
                        val_dict = (
                            image
                            .sample(pt, scale=sample_scale)
                            .first()
                            .toDictionary()
                            .getInfo()
                        )
                        val = next(
                            (v for v in val_dict.values() if isinstance(v, (int, float))),
                            None,
                        )
                        _sample_cache[_sk] = val
                    except Exception:
                        _sample_cache[_sk] = None

            val = _sample_cache.get(_sk)
            if val is not None:
                st.caption(
                    f"📍 **{clat:.4f}°, {clng:.4f}°**  →  **{val:.4f} cm³/cm³**"
                )
            else:
                st.caption(
                    f"📍 **{clat:.4f}°, {clng:.4f}°**  →  No data at this location "
                    f"(pixel may be outside the study area or masked)"
                )
        else:
            st.caption("🖱️ Click anywhere on the map to inspect the pixel value.")

    return layer_ok



# ===========================================================================
# TIME SERIES CHART
# ===========================================================================

def _render_time_series_chart(dates: list, means: list):
    """Plot soil moisture time series with Plotly."""
    # Filter out None values
    pairs  = [(d, m) for d, m in zip(dates, means) if d is not None and m is not None]
    if not pairs:
        st.warning("No valid data points in time series.")
        return

    x_vals, y_vals = zip(*pairs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines+markers",
        line=dict(color="#228B22", width=2),
        marker=dict(size=6, color="#228B22"),
        name="Mean Soil Moisture",
        hovertemplate="<b>%{x}</b><br>SM: %{y:.4f} cm³/cm³<extra></extra>",
    ))

    # Add WP and FC reference lines if available
    fc_val = DEFAULT_FIELD_CAPACITY
    wp_val = DEFAULT_WILTING_POINT
    fig.add_hline(y=fc_val, line_dash="dot", line_color="#1f77b4",
                  annotation_text=f"FC={fc_val}", annotation_position="top right")
    fig.add_hline(y=wp_val, line_dash="dot", line_color="#d62728",
                  annotation_text=f"WP={wp_val}", annotation_position="bottom right")

    fig.update_layout(
        title="Soil Moisture Time Series",
        xaxis_title="Date",
        yaxis_title="Mean Soil Moisture (cm³/cm³)",
        yaxis=dict(range=[0.0, 0.55]),
        height=380,
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats below chart
    valid_means = [m for m in means if m is not None]
    if valid_means:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Steps",    len(valid_means))
        sc2.metric("Mean SM",  f"{sum(valid_means)/len(valid_means):.4f}")
        sc3.metric("Min SM",   f"{min(valid_means):.4f}")
        sc4.metric("Max SM",   f"{max(valid_means):.4f}")


# ===========================================================================
# EXPORT SECTION
# ===========================================================================

def _render_export_section(image, region, params: dict, label: str = ""):
    """Render smart download options and execute export."""
    helper = DownloadHelper()
    export_pref = helper.render_smart_download_options(export_format="GeoTIFF")

    safe_label = label.replace(" ", "_").replace("→", "to").replace("–", "-")
    filename   = f"serves_sm_{safe_label}_{params.get('scale', 250)}m"
    # strip characters that are invalid in filenames
    import re
    filename = re.sub(r"[^\w\-]", "_", filename)[:80]

    if st.button("💾 Export Soil Moisture GeoTIFF", type="primary",
                 width="stretch", key="serves_export_btn"):
        with st.spinner("Exporting…"):
            result = helper.execute_smart_download(
                image=image,
                filename=filename,
                region=region,
                scale=params.get("scale", 250),
                export_preference=export_pref,
                crs="EPSG:4326",
            )
            if not result.get("success"):
                st.error(f"Export failed: {result.get('message', 'Unknown error')}")


# ===========================================================================
# ABOUT EXPANDER
# ===========================================================================

def _render_about_expander():
    with st.expander("ℹ️ About SERVES v2.0", expanded=False):
        st.markdown(f"""
        ### SERVES v{VERSION} — Soil Moisture Estimation

        **Algorithm:**
        The ET fraction is derived from NDVI using the linear relationship
        `ET_fraction = 1.33 × NDVI − 0.049`, then scaled between wilting point (WP)
        and field capacity (FC) to estimate volumetric soil moisture (cm³/cm³):

        > **SM = ET_fraction × (FC − WP) + WP**

        **Data Sources:**

        | Layer | Dataset |
        |---|---|
        | NDVI (Landsat) | Landsat 8/9 C02 T1 L2 (30 m) |
        | NDVI (Sentinel-2) | COPERNICUS/S2_SR_HARMONIZED (10 m) |
        | NDVI (MODIS) | MODIS/061/MOD13A2 (1 km, 16-day) |
        | Field Capacity | ISRIC SoilGrids250m v2.0 wv0033 |
        | Wilting Point | ISRIC SoilGrids250m v2.0 wv1500 |
        | Water bodies | JRC Global Surface Water v1.4 |
        | Vegetation | ESA WorldCover v200 (2021) |

        **v2.0 Enhancements over original SERVES:**
        - Default uniform soil parameters (FC=0.35, WP=0.09 cm³/cm³)
        - Water bodies assigned to field capacity
        - Negative NDVI pixels assigned to field capacity
        - Multi-year monthly / seasonal / annual composites
        - Regional climatology (continental and global scale via MODIS)
        - Full integration into GeoClimate Intelligence Platform

        **Credits:**
        Original SERVES model — Dr. Nawa Raj Pradhan, ERDC/CHL, U.S. Army Corps of Engineers.
        v2.0 — Saurav Bhattarai, Dr. Rocky Talchabhadel, Dr. Nawa Raj Pradhan, Jackson State University.
        """)


# ===========================================================================
# HELPERS
# ===========================================================================

def _build_options(params: dict) -> dict:
    """Convert sidebar params into the options dict consumed by serves_calculator functions."""
    return {
        "satellite":               params.get("satellite", "landsat"),
        "composite_method":        params.get("composite_method", "median"),
        "search_window":           params.get("search_window", 16),
        "soil_parameter_mode":     params.get("soil_parameter_mode", "soilgrids"),
        "soil_depth":              params.get("soil_depth", "b30"),
        "field_capacity":          params.get("field_capacity"),
        "wilting_point":           params.get("wilting_point"),
        "assign_water_to_fc":      params.get("assign_water_to_fc", True),
        "assign_negative_ndvi_to_fc": params.get("assign_negative_ndvi_to_fc", True),
        "mask_water":              params.get("mask_water", False),
        "mask_non_veg":            params.get("mask_non_veg", False),
    }


def _build_period_label(params: dict) -> str:
    """Build a short human-readable label for the current parameters."""
    mode = params["mode"]
    if mode == "single":
        return params.get("target_date", "")
    if mode == "monthly":
        m = get_month_name(params.get("month", 7))
        if params.get("multi_year"):
            return f"{m} {params.get('start_year')}–{params.get('end_year')}"
        return f"{m} {params.get('year', '')}"
    if mode == "seasonal":
        s = SEASON_LABELS.get(params.get("season", "summer"), "Summer")
        if params.get("multi_year"):
            return f"{s} {params.get('start_year')}–{params.get('end_year')}"
        return f"{s} {params.get('year', '')}"
    if mode == "annual":
        if params.get("multi_year"):
            return f"Annual {params.get('start_year')}–{params.get('end_year')}"
        return f"Year {params.get('year', '')}"
    if mode == "custom_range":
        return f"{params.get('start_date')} → {params.get('end_date')}"
    if mode == "time_series":
        return f"{params.get('start_date')} → {params.get('end_date')}"
    if mode == "regional":
        r  = PREDEFINED_REGION_LABELS.get(params.get("region", "europe"), "Custom Area")
        pt = params.get("reg_period_type", "annual")
        sy = params.get("start_year", "")
        ey = params.get("end_year", "")
        if pt == "monthly":
            sub = get_month_name(params.get("month", 1))
        elif pt == "seasonal":
            sub = SEASON_LABELS.get(params.get("reg_season", "summer"), "Season")
        else:
            sub = "Annual"
        return f"{r} · {sub} {sy}–{ey}"
    if mode == "longterm_stats":
        sy      = params.get("lt_start_year", "")
        ey      = params.get("lt_end_year", "")
        subtype = params.get("lt_stats_subtype", "interannual")
        if subtype == "temporal_cv":
            return f"Temporal CV · Monthly · {sy}–{ey}"
        pt = params.get("lt_period_type", "annual")
        if pt == "monthly":
            sub = get_month_name(params.get("lt_month", 1))
        elif pt == "seasonal":
            sub = SEASON_LABELS.get(params.get("lt_season", "summer"), "Season")
        else:
            sub = "Annual"
        return f"Inter-annual Stats · {sub} · {sy}–{ey}"
    return mode
