"""
GLOBAL-SAM Interface Module — Snow Depth & Snow Density
Authentication is handled by the main app — no separate auth needed here.

Analysis modes (same layout as SERVES — mode, period, output and settings in the sidebar):
  1. Single Date           — closest clear-sky day, or a statistic over ±N days
  2. Monthly Composite     — one month, or the same month across several years
  3. Seasonal Composite    — snow season or calendar season, single or multi-year
  4. Annual Composite      — one year or several years
  5. Custom Date Range     — any start / end date
  6. Time Series           — monthly / 16-day / weekly AOI series, step maps, bulk export
  7. Regional Climatology  — multi-year climatology over a predefined region or own AOI
  8. Long-term Statistics  — inter-annual statistics or temporal CV
  9. Algorithm Comparison  — validate against a reference GeoTIFF

Every period mode maps Snow Depth or Snow Density as a Mean, Median, Maximum or Minimum.

Key design decisions:
- Uses shared GeometrySelectionWidget (same as other modules)
- Results and GEE tile URLs are cached in session state and cleared whenever a
  sidebar parameter changes (SERVES pattern)
- ZIP results persisted in session state so re-download survives navigation
"""

import calendar
import csv
import datetime
import hashlib
import io
import json
import os
import re
import sys

import ee
import folium
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from branca.colormap import LinearColormap
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Make Snow_Depth_ERDC importable by adding its directory to sys.path
# ---------------------------------------------------------------------------
_snow_depth_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Snow_Depth_ERDC")
)
if _snow_depth_dir not in sys.path:
    sys.path.insert(0, _snow_depth_dir)

from snow_depth_app import (
    SnowDepthCalculator,
    get_seasonal_snow_density,
    sinusoidal_gamma_function,
    output_file_stem,
    step_file_tag,
    get_season_months,
    snow_time_steps,
    extract_snow_time_series,
    process_period_steps,
    extract_bounds_from_tiff,
    run_comparison,
    create_density_settings_ui,
    display_comparison_results,
    run_snow_depth_period_stats,
    run_snow_depth_temporal_variability,
)

from geoclimate_fetcher.core import GeometrySelectionWidget
from app_components.download_component import DownloadHelper
from app_components.ee_tile_cache import get_ee_tile_url
from geoclimate_fetcher.serves_calculator import (
    PREDEFINED_REGION_COORDS,
    PREDEFINED_REGION_LABELS,
    get_predefined_region,
)


# ---------------------------------------------------------------------------
# Snow Depth visualization constants  (mirrors SERVES VIS_PARAMS pattern)
# ---------------------------------------------------------------------------
_SD_VIS_PARAMS = {
    "min": 0.0,
    "max": 2.0,
    "palette": [
        "#FFFFFF",   # 0 m  — bare ground / no snow
        "#DEEBF7",   # 0.4 m — trace
        "#9ECAE1",   # 0.8 m — shallow
        "#3182BD",   # 1.2 m — moderate
        "#08519C",   # 1.6 m — deep
        "#08306B",   # 2.0 m+ — very deep
    ],
}
_SD_VIS_PARAMS_STDDEV = {
    "min": 0.0,
    "max": 0.5,
    "palette": ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"],
}
_SD_VIS_PARAMS_CV = {
    "min": 0.0,
    # CV in %. Monthly composites include snow-free months, so snow-depth CV is
    # typically ~110–250 % (persistent snowpack → low end, ephemeral snow → high end).
    "max": 250.0,
    "palette": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
}
_SD_VIS_PARAMS_DENSITY = {
    "min": 100.0,
    "max": 500.0,
    # light = fresh, low-density snow → dark = settled / wet, dense snow
    "palette": ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
}
_SD_VIS_PARAMS_DENSITY_STDDEV = {
    "min": 0.0,
    "max": 100.0,
    "palette": _SD_VIS_PARAMS_STDDEV["palette"],
}
_SD_VIS_PARAMS_DENSITY_CV = {
    "min": 0.0,
    "max": 100.0,   # density is masked where there is no snow, so CV stays well below depth CV
    "palette": _SD_VIS_PARAMS_CV["palette"],
}

# Products returned by SnowDepthCalculator (band name → display config)
_SD_PRODUCTS = {
    "snow_depth": {
        "label": "Snow Depth", "unit": "m", "fmt": ".3f",
        "vis": _SD_VIS_PARAMS, "std_vis": _SD_VIS_PARAMS_STDDEV, "cv_vis": _SD_VIS_PARAMS_CV,
        "cover": "Area with data",
    },
    "snow_density": {
        "label": "Snow Density", "unit": "kg/m³", "fmt": ".0f",
        "vis": _SD_VIS_PARAMS_DENSITY, "std_vis": _SD_VIS_PARAMS_DENSITY_STDDEV,
        "cv_vis": _SD_VIS_PARAMS_DENSITY_CV,
        "cover": "Snow-covered area",
    },
}

# Statistic within the selected period (see SnowDepthCalculator._reduce_segments)
_SD_STATISTICS = {"mean": "Mean", "median": "Median", "max": "Maximum", "min": "Minimum"}
# Single Date adds a per-pixel nearest-clear-sky-day composite
_SD_SINGLE_STATISTICS = {"closest": "Closest clear-sky day", **_SD_STATISTICS}

_SD_STATISTIC_HELP = (
    "**Mean** — the GLOBAL-SAM monthly algorithm (mean MODIS NDSI × mean GLDAS SWE) for "
    "each calendar month in the period (or the part of it inside the period), averaged "
    "weighted by days. A single whole month is the original monthly algorithm.\n\n"
    "**Median / Maximum / Minimum** — from daily estimates (each day's MODIS NDSI × that "
    "day's GLDAS SWE) over all clear-sky days in the period."
)
_SD_CLOSEST_HELP = (
    "**Closest clear-sky day** — each pixel takes the day nearest the target date on which "
    "MODIS saw the ground (cloudy target days are gap-filled from nearby days).\n\n"
    + _SD_STATISTIC_HELP
)

_SD_MODES = {
    "single":         "📅 Single Date",
    "monthly":        "📆 Monthly Composite",
    "seasonal":       "🌨️ Seasonal Composite",
    "annual":         "📊 Annual Composite",
    "custom_range":   "📋 Custom Date Range",
    "time_series":    "📈 Time Series & Bulk Export",
    "regional":       "🌍 Regional Climatology",
    "longterm_stats": "📉 Long-term Statistics",
    "comparison":     "🔍 Algorithm Comparison",
}
# Modes whose result is one map over one period
_SD_STATIC_MODES = {"single", "monthly", "seasonal", "annual", "custom_range"}

_SD_SEASONS = {
    "snow_season": "Snow Season (Oct–Apr NH / Apr–Oct SH)",
    "winter":      "Winter (Dec–Feb NH / Jun–Aug SH)",
    "spring":      "Spring (Mar–May NH / Sep–Nov SH)",
    "summer":      "Summer (Jun–Aug NH / Dec–Feb SH)",
    "fall":        "Fall (Sep–Nov NH / Mar–May SH)",
}
_SD_SEASON_SHORT = {
    "snow_season": "snow-season", "winter": "winter", "spring": "spring",
    "summer": "summer", "fall": "fall",
}

_SD_PERIOD_TYPES = {
    "annual":   "Full Year (all 12 months)",
    "monthly":  "Specific Month",
    "seasonal": "Season / Snow Season",
}

_SD_INTERVALS = {"monthly": "Monthly", "16day": "16-day", "weekly": "Weekly"}

_SD_SUMMARY_STATS = {
    "mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum",
    "std_dev": "Std Deviation",
}

_SD_FIRST_DATE = datetime.date(2000, 2, 24)   # first MODIS MOD10A1 day
# Median / Maximum / Minimum read every daily image — warn above this many days
_SD_SLOW_DAYS = 400

_SD_TILE_FAILED_MSG = (
    "Map preview unavailable (GEE memory limit). "
    "Use **💾 Export** below — batch export has higher memory limits."
)
_SD_MAP_UNAVAILABLE = (
    "Map unavailable — GEE memory limit exceeded. "
    "Use **💾 Export** below — batch export has higher limits."
)


def _sd_product_caption(product: str) -> str:
    """Map / colorbar caption, e.g. 'Snow Density (kg/m³)'."""
    cfg = _SD_PRODUCTS[product]
    return f"{cfg['label']} ({cfg['unit']})"


def _month_name(month: int, fmt: str = "%B") -> str:
    return datetime.date(2000, month, 1).strftime(fmt)


def _statistic_name(statistic: str) -> str:
    return _SD_SINGLE_STATISTICS[statistic]


# ---------------------------------------------------------------------------
# Sidebar auto-expand helper
# ---------------------------------------------------------------------------
def _auto_expand_sidebar():
    """Open the sidebar the first time this module is visited in a session."""
    if not st.session_state.get("sd_sidebar_opened"):
        components.html(
            """
            <script>
                (function () {
                    function tryExpand(attempts) {
                        // stExpandSidebarButton: Streamlit ≥ 1.4x; collapsedControl: older releases
                        var btn = window.parent.document.querySelector(
                            '[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"]'
                        );
                        if (btn) {
                            btn.click();
                        } else if (attempts > 0) {
                            setTimeout(function () { tryExpand(attempts - 1); }, 250);
                        }
                    }
                    setTimeout(function () { tryExpand(6); }, 400);
                })();
            </script>
            """,
            height=0,
        )
        st.session_state.sd_sidebar_opened = True


# ===========================================================================
# SIDEBAR
# ===========================================================================

def _render_sidebar():
    """Render all GLOBAL-SAM parameters in the sidebar.

    Returns (params, density_method, density_params).
    """
    st.sidebar.header("❄️ GLOBAL-SAM Parameters")

    # ── Analysis Mode ──────────────────────────────────────────────────────
    st.sidebar.subheader("🔬 Analysis Mode")
    mode_labels = list(_SD_MODES.values())
    mode_keys   = list(_SD_MODES.keys())
    sel_mode    = st.sidebar.selectbox(
        "Mode:", mode_labels, index=mode_keys.index("monthly"), key="sd_mode_select",
    )
    mode   = mode_keys[mode_labels.index(sel_mode)]
    params = {"mode": mode}

    # ── Mode-specific date parameters ──────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Date / Period")
    params.update(_render_date_params(mode))

    # ── Output: product + statistic ────────────────────────────────────────
    if mode != "comparison":
        _render_output_params(params)

    # ── Snow density (applies to every mode) ───────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏔️ Snow Density Settings")
    with st.sidebar:
        density_method, density_params = create_density_settings_ui("snow_depth")

    # ── Resolution ─────────────────────────────────────────────────────────
    if mode != "comparison":
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Export / Stats")
        regional = mode == "regional"
        params["scale"] = st.sidebar.number_input(
            "Resolution (m):", min_value=100, max_value=10000,
            value=5000 if regional else 500, step=100,
            key="sd_scale_regional" if regional else "sd_scale",
            help="Pixel resolution for statistics and GeoTIFF export (maps render at screen "
                 "resolution). 500 m = native MODIS. Use a coarser value for large regions.",
        )

    return params, density_method, density_params


def _year_options() -> list:
    return list(range(datetime.date.today().year, 1999, -1))


def _render_year_choice(p: dict, key: str, default_year: int = 2023,
                        default_range: tuple = (2019, 2023)) -> None:
    """'Multi-year composite' checkbox + Year, or Start / End Year (SERVES pattern)."""
    y_opts = _year_options()
    p["multi_year"] = st.sidebar.checkbox(
        "Multi-year composite", value=False, key=f"sd_{key}_multiyear",
        help="Combine the same period from several years — the statistic is taken over "
             "all of their days together (e.g. Maximum = deepest snow in any of the years).",
    )
    if p["multi_year"]:
        p["start_year"] = st.sidebar.selectbox(
            "Start Year:", y_opts, index=y_opts.index(default_range[0]), key=f"sd_sy_{key}"
        )
        p["end_year"] = st.sidebar.selectbox(
            "End Year:", y_opts, index=y_opts.index(default_range[1]), key=f"sd_ey_{key}"
        )
        p["year"] = None
    else:
        p["year"] = st.sidebar.selectbox(
            "Year:", y_opts, index=y_opts.index(default_year), key=f"sd_year_{key}"
        )
        p["start_year"] = p["end_year"] = None


def _render_season_choice(p: dict, key: str) -> None:
    """Season selectbox + hemisphere radio."""
    season_labels = list(_SD_SEASONS.values())
    season_keys   = list(_SD_SEASONS.keys())
    sel_season    = st.sidebar.selectbox("Season:", season_labels, index=0, key=f"sd_season_{key}")
    p["season"]   = season_keys[season_labels.index(sel_season)]
    p["hemisphere"] = "north" if st.sidebar.radio(
        "Hemisphere:", ["Northern", "Southern"], index=0, horizontal=True, key=f"sd_hemi_{key}",
    ) == "Northern" else "south"


def _render_period_within_year(p: dict, key: str) -> None:
    """Period within each year (Regional Climatology / Long-term Statistics)."""
    pt_labels = list(_SD_PERIOD_TYPES.values())
    pt_keys   = list(_SD_PERIOD_TYPES.keys())
    sel_pt    = st.sidebar.selectbox(
        "Period within each year:", pt_labels, index=0, key=f"sd_{key}_period_type"
    )
    p["period_type"] = pt_keys[pt_labels.index(sel_pt)]
    p["month"]       = 1
    p["season"]      = "snow_season"
    p["hemisphere"]  = "north"
    if p["period_type"] == "monthly":
        p["month"] = st.sidebar.selectbox(
            "Month:", range(1, 13), format_func=_month_name, index=0, key=f"sd_{key}_month",
        )
    elif p["period_type"] == "seasonal":
        _render_season_choice(p, key)


def _render_date_params(mode: str) -> dict:
    """Render mode-specific date widgets. Returns partial params dict."""
    p     = {}
    today = datetime.date.today()
    y_opts = _year_options()

    if mode == "single":
        p["target_date"] = st.sidebar.date_input(
            "Target Date:", value=datetime.date(2023, 1, 15),
            min_value=_SD_FIRST_DATE, max_value=today, key="sd_target_date",
        )
        p["window"] = st.sidebar.slider(
            "Search window (±days):", 0, 16, 5, key="sd_window",
            help="Days on each side of the target date that may be used. MODIS cannot see "
                 "through cloud, so a wider window fills more cloud gaps.",
        )

    elif mode == "monthly":
        p["month"] = st.sidebar.selectbox(
            "Month:", range(1, 13), format_func=_month_name, index=0, key="sd_month",
        )
        _render_year_choice(p, "monthly")

    elif mode == "seasonal":
        _render_season_choice(p, "seasonal")
        _render_year_choice(p, "seasonal", default_year=2022, default_range=(2018, 2022))
        example = p["year"] or p["start_year"]
        st.sidebar.caption(
            f"🗓️ {_SD_SEASONS[p['season']].split(' (')[0]} {example} = "
            f"{_span_text(get_season_months(example, p['season'], p['hemisphere']))}"
        )

    elif mode == "annual":
        _render_year_choice(p, "annual")

    elif mode == "custom_range":
        p["start_date"] = st.sidebar.date_input(
            "Start Date:", value=datetime.date(2022, 10, 1),
            min_value=_SD_FIRST_DATE, max_value=today, key="sd_start_custom",
        )
        p["end_date"] = st.sidebar.date_input(
            "End Date:", value=datetime.date(2023, 4, 30),
            min_value=_SD_FIRST_DATE, max_value=today, key="sd_end_custom",
        )
        if p["start_date"] > p["end_date"]:
            st.sidebar.error("Start date must not be after end date.")

    elif mode == "time_series":
        p["start_date"] = st.sidebar.date_input(
            "Start Date:", value=datetime.date(2022, 10, 1),
            min_value=_SD_FIRST_DATE, max_value=today, key="sd_start_ts",
        )
        p["end_date"] = st.sidebar.date_input(
            "End Date:", value=datetime.date(2023, 9, 30),
            min_value=_SD_FIRST_DATE, max_value=today, key="sd_end_ts",
        )
        int_labels = list(_SD_INTERVALS.values())
        int_keys   = list(_SD_INTERVALS.keys())
        sel_int    = st.sidebar.selectbox("Interval:", int_labels, index=0, key="sd_interval")
        p["interval"] = int_keys[int_labels.index(sel_int)]
        if p["start_date"] > p["end_date"]:
            st.sidebar.error("Start date must not be after end date.")
        else:
            n_steps = len(snow_time_steps(p["start_date"], p["end_date"], p["interval"]))
            st.sidebar.info(f"📅 **Time steps:** {n_steps}")
            if n_steps > 60:
                st.sidebar.caption("⚠️ Long series (> 60 steps) can be slow. "
                                   "Consider a monthly interval or a shorter range.")

    elif mode == "regional":
        aoi_src_labels = ["🗺️ Predefined Region", "✏️ My Area (Draw / Upload)"]
        aoi_src_keys   = ["predefined", "custom"]
        sel_aoi_src    = st.sidebar.radio(
            "AOI Source:", aoi_src_labels, index=0, key="sd_reg_aoi_source"
        )
        p["regional_aoi_source"] = aoi_src_keys[aoi_src_labels.index(sel_aoi_src)]
        if p["regional_aoi_source"] == "predefined":
            reg_labels = list(PREDEFINED_REGION_LABELS.values())
            reg_keys   = list(PREDEFINED_REGION_LABELS.keys())
            sel_reg    = st.sidebar.selectbox(
                "Region:", reg_labels, index=reg_keys.index("europe"), key="sd_region"
            )
            p["region"] = reg_keys[reg_labels.index(sel_reg)]
        else:
            p["region"] = None  # geometry comes from the AOI widget in the main panel

        _render_period_within_year(p, "reg")
        p["start_year"] = st.sidebar.selectbox(
            "Start Year:", y_opts, index=y_opts.index(2019), key="sd_reg_sy"
        )
        p["end_year"] = st.sidebar.selectbox(
            "End Year:", y_opts, index=y_opts.index(2023), key="sd_reg_ey"
        )
        sum_labels = list(_SD_SUMMARY_STATS.values())
        sum_keys   = list(_SD_SUMMARY_STATS.keys())
        sel_sum    = st.sidebar.selectbox(
            "Summary statistic across years:", sum_labels, index=sum_keys.index("median"),
            key="sd_reg_summary",
        )
        p["summary_statistic"] = sum_keys[sum_labels.index(sel_sum)]

    elif mode == "longterm_stats":
        subtype_labels = [
            "📊 Inter-annual  (one composite per year)",
            "〰️ Temporal Variability / CV  (all monthly composites)",
        ]
        subtype_keys = ["interannual", "temporal_cv"]
        sel_sub = st.sidebar.radio(
            "Statistics type:", subtype_labels, index=0, key="sd_lt_subtype",
        )
        p["lt_subtype"] = subtype_keys[subtype_labels.index(sel_sub)]

        p["start_year"] = st.sidebar.selectbox(
            "Start Year:", y_opts, index=y_opts.index(2015), key="sd_lt_start_year"
        )
        p["end_year"] = st.sidebar.selectbox(
            "End Year:", y_opts, index=y_opts.index(2023), key="sd_lt_end_year"
        )
        if p["lt_subtype"] == "interannual":
            _render_period_within_year(p, "lt")
            st.sidebar.caption("⚠️ One composite per year. Tiles load on-demand via GEE.")
        else:
            n_months = (p["end_year"] - p["start_year"] + 1) * 12
            st.sidebar.caption(
                f"⚠️ Builds {n_months} monthly-mean composites "
                f"({p['start_year']}–{p['end_year']}). All GEE ops are lazy — tiles render on-demand."
            )

    elif mode == "comparison":
        p["comparison_year"] = st.sidebar.number_input(
            "Year", min_value=2000, max_value=today.year, value=2020,
            key="sd_comp_year", help="Year for the comparison data",
        )
        p["comparison_month"] = st.sidebar.selectbox(
            "Month", options=list(range(1, 13)), format_func=_month_name, index=0,
            key="sd_comp_month", help="Month for the comparison data",
        )
        p["comp_scale"] = st.sidebar.selectbox(
            "Resolution (meters)", options=[250, 500, 1000], index=1, key="sd_comp_scale",
            help="Resolution for comparison (lower = more accurate but slower)",
        )

    if mode not in ("comparison", "longterm_stats", "regional"):
        st.sidebar.caption(
            "Data: MODIS snow cover from 24 Feb 2000. GLDAS-2.1 SWE lags ~1–2 months, "
            "so the most recent weeks are masked."
        )
    return p


def _render_output_params(params: dict) -> None:
    """Product + statistic selectors (adds 'product' and 'statistic' to params)."""
    mode = params["mode"]
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Output")
    params["product"] = st.sidebar.radio(
        "Product:", list(_SD_PRODUCTS), format_func=_sd_product_caption,
        horizontal=True, key="sd_product",
    )

    if mode == "longterm_stats" and params["lt_subtype"] == "temporal_cv":
        params["statistic"] = "mean"
        st.sidebar.caption("ℹ️ Temporal CV is computed from monthly-mean composites.")
        return

    if mode == "single":
        params["statistic"] = st.sidebar.selectbox(
            "Composite over the window:", list(_SD_SINGLE_STATISTICS),
            format_func=_SD_SINGLE_STATISTICS.get, key="sd_statistic_single",
            help=_SD_CLOSEST_HELP,
        )
        return

    label = {
        "regional":       "Statistic within each year's period:",
        "longterm_stats": "Statistic within each year's period:",
        "time_series":    "Statistic within each time step:",
    }.get(mode, "Statistic over the period:")
    params["statistic"] = st.sidebar.selectbox(
        label, list(_SD_STATISTICS), format_func=_SD_STATISTICS.get,
        key="sd_statistic", help=_SD_STATISTIC_HELP,
    )
    if params["statistic"] != "mean":
        days = _period_days(params)
        if days and days > _SD_SLOW_DAYS:
            st.sidebar.warning(
                f"⏱️ {_SD_STATISTICS[params['statistic']]} reads every daily image — "
                f"about {days:,} days here. Maps may be slow or hit GEE memory limits; "
                "Mean is much faster, and Google Drive export has higher limits."
            )
        else:
            st.sidebar.caption(
                "⏱️ Median / Maximum / Minimum use every daily image in the period — "
                "slower than Mean, especially over long periods or large areas."
            )


# ===========================================================================
# PERIOD HELPERS
# ===========================================================================

def _selected_years(params: dict) -> list:
    if params.get("multi_year"):
        return list(range(params["start_year"], params["end_year"] + 1))
    return [params["year"]]


def _selected_months(params: dict) -> list:
    """(year, month) pairs for the Monthly / Seasonal / Annual composite modes."""
    mode   = params["mode"]
    months = []
    for year in _selected_years(params):
        if mode == "monthly":
            months.append((year, params["month"]))
        elif mode == "seasonal":
            months.extend(get_season_months(year, params["season"], params["hemisphere"]))
        else:
            months.extend((year, m) for m in range(1, 13))
    return months


def _selected_range(params: dict) -> tuple:
    """[start, end) dates for the Single Date / Custom Date Range modes (end exclusive)."""
    if params["mode"] == "single":
        window = datetime.timedelta(days=params["window"])
        return params["target_date"] - window, params["target_date"] + window + datetime.timedelta(days=1)
    return params["start_date"], params["end_date"] + datetime.timedelta(days=1)


def _months_days(months: list) -> int:
    return sum(calendar.monthrange(y, m)[1] for y, m in months)


def _period_days(params: dict):
    """Approximate number of days the selected statistic reads (for slow-query warnings)."""
    mode = params["mode"]
    if mode == "single":
        return 2 * params["window"] + 1
    if mode in ("custom_range", "time_series"):
        return max((params["end_date"] - params["start_date"]).days + 1, 0)
    if mode in ("monthly", "seasonal", "annual"):
        return _months_days(_selected_months(params))
    if mode in ("regional", "longterm_stats"):
        per_year = {"monthly": 31, "annual": 365}.get(params.get("period_type"), 0)
        if not per_year:
            per_year = 212 if params.get("season") == "snow_season" else 92
        return per_year * max(params["end_year"] - params["start_year"] + 1, 0)
    return None


def _span_text(months: list) -> str:
    """'Jan 2023' or 'Oct 2022 – Apr 2023' for a run of (year, month) pairs."""
    first = datetime.date(*months[0], 1)
    last  = datetime.date(*months[-1], 1)
    if first == last:
        return first.strftime("%b %Y")
    return f"{first:%b %Y} – {last:%b %Y}"


def _period_summary(params: dict) -> str:
    """Which dates the result covers, e.g. 'Oct 2022 – Apr 2023 · 7 months, 212 days'."""
    if params["mode"] in ("single", "custom_range"):
        start, end = _selected_range(params)
        days = (end - start).days
        return f"{start:%d %b %Y} – {end - datetime.timedelta(days=1):%d %b %Y} · {days} day{'s' * (days != 1)}"
    months = _selected_months(params)
    days   = _months_days(months)
    if params.get("multi_year"):
        per_year = len(months) // len(_selected_years(params))
        return (f"{_span_text(months[:per_year])} … {_span_text(months[-per_year:])} · "
                f"{len(months)} months, {days:,} days")
    return f"{_span_text(months)} · {len(months)} month{'s' * (len(months) != 1)}, {days} days"


def _season_name(season: str, hemisphere: str) -> str:
    name = _SD_SEASONS[season].split(" (")[0]
    return f"{name} ({'NH' if hemisphere == 'north' else 'SH'})"


def _base_period_label(params: dict) -> str:
    """Period part of the run label (no product / statistic)."""
    mode = params["mode"]
    if mode == "single":
        return f"{params['target_date']} ±{params['window']} d"
    if mode == "monthly":
        m = _month_name(params["month"])
        if params["multi_year"]:
            return f"{m} {params['start_year']}–{params['end_year']}"
        return f"{m} {params['year']}"
    if mode == "seasonal":
        s = _season_name(params["season"], params["hemisphere"])
        if params["multi_year"]:
            return f"{s} {params['start_year']}–{params['end_year']}"
        return f"{s} {params['year']}"
    if mode == "annual":
        if params["multi_year"]:
            return f"Annual {params['start_year']}–{params['end_year']}"
        return f"Year {params['year']}"
    if mode in ("custom_range", "time_series"):
        label = f"{params['start_date']} → {params['end_date']}"
        if mode == "time_series":
            label += f" · {_SD_INTERVALS[params['interval']]}"
        return label
    if mode == "longterm_stats" and params["lt_subtype"] == "temporal_cv":
        return f"Temporal CV · Monthly · {params['start_year']}–{params['end_year']}"
    composite = _composite_name(params["period_type"], params["month"], params["season"],
                                params["statistic"])
    if params["period_type"] == "seasonal":
        composite += f" ({'NH' if params['hemisphere'] == 'north' else 'SH'})"
    return f"Inter-annual · {composite[0].upper() + composite[1:]} · {params['start_year']}–{params['end_year']}"


def _build_run_label(params: dict) -> str:
    """Human-readable label of the current parameters, e.g. 'Maximum Snow Depth · January 2023'."""
    product = _SD_PRODUCTS[params["product"]]["label"]
    if params["mode"] == "longterm_stats":
        return f"{product} · {_base_period_label(params)}"
    if params["statistic"] == "closest":
        return (f"{product} · Closest clear-sky day to {params['target_date']} "
                f"(±{params['window']} d)")
    return f"{_statistic_name(params['statistic'])} {product} · {_base_period_label(params)}"


def _file_tag(params: dict) -> str:
    """Period tag for export file names, e.g. '2023_01', 'snow_season_2019-2023'."""
    mode = params["mode"]
    if mode == "single":
        return f"{params['target_date']:%Y%m%d}_pm{params['window']}d"
    if mode == "custom_range":
        return f"{params['start_date']:%Y%m%d}_{params['end_date']:%Y%m%d}"
    years = (f"{params['start_year']}-{params['end_year']}" if params["multi_year"]
             else str(params["year"]))
    if mode == "monthly":
        return f"{years}_{params['month']:02d}"
    if mode == "seasonal":
        hemi = "" if params["hemisphere"] == "north" else "_sh"
        return f"{params['season']}{hemi}_{years}"
    return years


def _composite_name(period_type: str, month: int, season: str, statistic: str) -> str:
    """What each year is reduced to, e.g. 'annual maximum', 'January mean', 'winter median'."""
    stat = _SD_STATISTICS[statistic].lower()
    if period_type == "monthly":
        return f"{_month_name(month)} {stat}"
    if period_type in ("seasonal", "snow_season"):
        return f"{_SD_SEASON_SHORT[season]} {stat}"
    return f"annual {stat}"


def _validate(params: dict):
    """Error message for inconsistent parameters, or None."""
    mode = params["mode"]
    if mode in ("custom_range", "time_series") and params["start_date"] > params["end_date"]:
        return "❌ Start date must not be after end date."
    if params.get("multi_year") and params["start_year"] > params["end_year"]:
        return "❌ Start Year must not be after End Year."
    if mode == "regional" and params["start_year"] > params["end_year"]:
        return "❌ Start Year must not be after End Year."
    if mode == "longterm_stats" and params["start_year"] >= params["end_year"]:
        return "❌ Start Year must be before End Year."
    return None


# ===========================================================================
# GEOMETRY SELECTION
# ===========================================================================

# Continent keys (shown first) vs sub-continental keys for the region picker
_SD_CONTINENT_KEYS = [
    "globe", "europe", "asia", "north_america",
    "south_america", "africa", "australia",
]
_SD_SUB_KEYS = [k for k in PREDEFINED_REGION_LABELS if k not in _SD_CONTINENT_KEYS]


def _clear_results() -> None:
    """Drop cached results, tile URLs, pixel samples and statistics."""
    for k in ("sd_results", "sd_tile_cache", "sd_sample_cache", "sd_stats_cache"):
        st.session_state.pop(k, None)


def _render_predefined_region_selector() -> None:
    """Selectbox + Apply button for continental / sub-continental regions.

    On confirm, writes snow_depth_geometry / centroid / area_km2 into session
    state without making any GEE network calls, then calls st.rerun().
    """
    import math

    st.markdown(
        "Pick any continental or regional boundary as your study area. "
        "Snow depth data are available globally — choose the region that matches "
        "your research interest."
    )

    group = st.radio(
        "Category:",
        ["🌍 Continents", "🗺️ Sub-continental Regions"],
        horizontal=True,
        key="sd_reg_group",
    )
    keys_in_group = _SD_CONTINENT_KEYS if group == "🌍 Continents" else _SD_SUB_KEYS
    labels_in_group = [PREDEFINED_REGION_LABELS[k] for k in keys_in_group]

    sel_label = st.selectbox("Region:", labels_in_group, key="sd_reg_selectbox")
    sel_key = keys_in_group[labels_in_group.index(sel_label)]

    west, south, east, north = PREDEFINED_REGION_COORDS[sel_key]
    lat_mid = (south + north) / 2
    lon_mid = (west + east) / 2
    approx_area = abs((east - west) * (north - south)) * (111.32 ** 2) * math.cos(math.radians(lat_mid))

    st.caption(
        f"Bounds: {west}° W/E, {south}°–{north}° lat  |  "
        f"~{approx_area:,.0f} km²"
    )

    if st.button(f"🌍 Use {sel_label}", type="primary", key="sd_use_predefined_region"):
        geom = get_predefined_region(sel_key)
        st.session_state["snow_depth_geometry"] = geom
        st.session_state["snow_depth_centroid"] = [lon_mid, lat_mid]
        st.session_state["snow_depth_area_km2"] = round(approx_area, 2)
        _clear_results()
        st.rerun()


def _render_geometry_selection():
    """Render AOI selection: tabs for Custom AOI or Predefined Region.

    Returns True when a geometry is ready for analysis.
    """
    if st.session_state.get("snow_depth_geometry") is not None:
        st.success("✅ Area of Interest is ready for analysis")
        area_km2 = st.session_state.get("snow_depth_area_km2")
        if area_km2 is None:
            try:
                area_km2 = (
                    st.session_state.snow_depth_geometry.area().divide(1e6).getInfo()
                )
                st.session_state["snow_depth_area_km2"] = round(area_km2, 2)
            except Exception:
                pass
        if area_km2:
            st.metric("Selected Area", f"{area_km2:,.2f} km²")
        else:
            st.info("Geometry ready for analysis")

        if st.button("🗑️ Reset Area", key="sd_reset_geometry"):
            for key in [
                "snow_depth_geometry",
                "snow_depth_geometry_complete",
                "snow_depth_area_km2",
                "snow_depth_method_selection",
                "snow_depth_centroid",
                # ZIP cache
                "sd_zip_data", "sd_zip_filename", "sd_zip_files",
            ]:
                st.session_state.pop(key, None)
            _clear_results()
            st.rerun()

        return True

    # ── Not yet set — offer two entry paths ─────────────────────────────────
    tab_custom, tab_predefined = st.tabs(["📍 Custom AOI", "🌍 Predefined Region"])

    with tab_predefined:
        _render_predefined_region_selector()

    with tab_custom:
        def on_geometry_selected(geometry):
            st.session_state["snow_depth_geometry"] = geometry
            try:
                area_km2 = geometry.area().divide(1e6).getInfo()
                st.session_state["snow_depth_area_km2"] = round(area_km2, 2)
            except Exception:
                pass
            try:
                centroid = geometry.centroid(maxError=1).coordinates().getInfo()
                st.session_state["snow_depth_centroid"] = centroid  # [lon, lat]
            except Exception:
                pass

        geometry_widget = GeometrySelectionWidget(
            session_prefix="snow_depth_",
            title="🗺️ Define Study Area",
        )
        if geometry_widget.render_complete_interface(
            on_geometry_selected=on_geometry_selected
        ):
            st.rerun()
            return True

    return False


# ---------------------------------------------------------------------------
# Persistent ZIP download banner
# ---------------------------------------------------------------------------
def _render_persistent_download():
    """Show a re-download banner whenever a ZIP result exists in session state."""
    if "sd_zip_data" not in st.session_state:
        return

    st.success("📦 Your previously generated GeoTIFFs are ready to download again.")

    dl_col, clear_col = st.columns([4, 1])
    with dl_col:
        st.download_button(
            label="📥 Re-Download GeoTIFF ZIP",
            data=st.session_state["sd_zip_data"],
            file_name=st.session_state["sd_zip_filename"],
            mime="application/zip",
            width="stretch",
            key="sd_persistent_download_btn",
        )
    with clear_col:
        if st.button("✖ Clear", key="sd_clear_zip", help="Remove cached results"):
            for key in ["sd_zip_data", "sd_zip_filename", "sd_zip_files"]:
                st.session_state.pop(key, None)
            st.rerun()

    if st.session_state.get("sd_zip_files"):
        with st.expander("📋 Files in this ZIP", expanded=False):
            for name in st.session_state["sd_zip_files"]:
                st.text(name)

    st.divider()


# ===========================================================================
# VISUALIZATION HELPERS  (SERVES-style: tile cache, map, statistics, export)
# ===========================================================================

def _params_key(params: dict, density_params: dict) -> str:
    """MD5 of all parameters + the AOI area — any change invalidates results and caches."""
    raw = json.dumps(
        {
            "params":  params,
            "density": density_params,
            "area":    st.session_state.get("snow_depth_area_km2"),
        },
        sort_keys=True, default=str,
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _sd_build_calculator(scale: int, density_params: dict) -> SnowDepthCalculator:
    """Instantiate a SnowDepthCalculator from sidebar density params."""
    if density_params["method"] == "original":
        return SnowDepthCalculator(
            scale=scale,
            use_seasonal_density=density_params.get("use_seasonal_density", True),
            custom_density_value=density_params.get("custom_density_value"),
            custom_monthly_densities=density_params.get("custom_monthly_densities"),
            density_method="original",
        )
    return SnowDepthCalculator(
        scale=scale,
        density_method=density_params["method"],
        rho_min=density_params["rho_min"],
        rho_max=density_params["rho_max"],
        gamma_params=density_params["gamma_params"],
    )


def _render_snow_map(
    image: ee.Image,
    vis_params: dict,
    aoi,
    layer_key: str,
    cache_prefix: str = "sd",
    caption: str = "Snow Depth (m)",
    value_unit: str = "m",
    value_suffix: str = "",
    sample_scale: int = 500,
    value_fmt: str = ".3f",
    center: tuple = None,
    zoom: int = None,
    tile_key: str = None,
) -> bool:
    """
    Build a Folium map with a GEE tile layer, colorbar, and pixel inspector.
    Saves/restores pan+zoom so selector reruns don't reset the viewport.
    center=(lat, lon) / zoom override the AOI-based initial view (predefined regions).
    Returns True if the GEE layer rendered successfully.
    """
    # ── Centre & zoom ────────────────────────────────────────────────────
    if center:
        init_lat, init_lon = center
        init_zoom = zoom or 4
    else:
        cached_centroid = st.session_state.get("snow_depth_centroid")
        if cached_centroid:
            init_lon, init_lat = cached_centroid[0], cached_centroid[1]
        else:
            try:
                centroid = aoi.centroid(maxError=1).coordinates().getInfo()
                init_lon, init_lat = centroid[0], centroid[1]
                st.session_state["snow_depth_centroid"] = centroid
            except Exception:
                init_lat, init_lon = 45.0, -100.0

        area_km2  = st.session_state.get("snow_depth_area_km2", 10_000)
        init_zoom = next(
            (z for limit, z in ((1_000, 9), (10_000, 8), (50_000, 7), (500_000, 5))
             if area_km2 < limit),
            4,
        )

    # ── Restore saved viewport ───────────────────────────────────────────
    _map_state_key = f"{cache_prefix}_map_state_{layer_key}"
    saved   = st.session_state.get(_map_state_key, {})
    ctr_lat = saved.get("lat", init_lat)
    ctr_lon = saved.get("lon", init_lon)
    zoom    = saved.get("zoom", init_zoom)

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Satellite basemap toggle
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # ── GEE tile layer (URL cached — never re-calls getMapId on re-render) ──
    layer_ok = False
    try:
        tile_url = get_ee_tile_url(image, vis_params, f"{cache_prefix}_tile_cache",
                                   tile_key or layer_key, failed_msg=_SD_TILE_FAILED_MSG)
        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine",
            name=caption,
            overlay=True,
            control=True,
        ).add_to(m)
        layer_ok = True
    except Exception as exc:
        err_str = str(exc).lower()
        if "memory" in err_str or "limit" in err_str or "quota" in err_str:
            st.warning(
                "🚫 **Map preview unavailable** — GEE memory limit exceeded for this "
                "computation. Use **💾 Export** below — batch export runs on GEE's "
                "higher-limit infrastructure and will succeed."
            )
        else:
            st.warning(f"Could not render map layer: {exc}")

    # ── Colorbar ─────────────────────────────────────────────────────────
    LinearColormap(
        colors=vis_params["palette"],
        vmin=vis_params["min"],
        vmax=vis_params["max"],
        caption=caption,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # ── Render — unique key per layer_key so result switches don't share state ──
    _folium_key = f"{cache_prefix}_folium_" + "".join(
        c if c.isalnum() else "_" for c in layer_key
    )
    map_data = st_folium(
        m,
        width=700,
        height=480,
        returned_objects=["last_clicked"],
        key=_folium_key,
    )

    # ── Save viewport ────────────────────────────────────────────────────
    if map_data:
        _c = map_data.get("center")
        _z = map_data.get("zoom")
        if _c and _z:
            st.session_state[_map_state_key] = {
                "lat": _c["lat"], "lon": _c["lng"], "zoom": int(_z)
            }

    # ── Pixel inspector ──────────────────────────────────────────────────
    if layer_ok and map_data:
        clicked = map_data.get("last_clicked")
        if clicked:
            clat, clng = clicked["lat"], clicked["lng"]
            _sample_cache = st.session_state.setdefault(f"{cache_prefix}_sample_cache", {})
            _sk = f"{layer_key}::{clat:.4f}::{clng:.4f}"
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
                    f"📍 **{clat:.4f}°, {clng:.4f}°**  →  **{val:{value_fmt}} {value_unit}**{value_suffix}"
                )
            else:
                st.caption(
                    f"📍 **{clat:.4f}°, {clng:.4f}°**  →  No data at this location "
                    "(pixel outside AOI or fully masked)"
                )
        else:
            st.caption("🖱️ Click anywhere on the map to inspect the pixel value.")

    return layer_ok


def _render_product_map(image: ee.Image, aoi, layer_key: str, product: str, scale: int,
                        caption: str = None, vis: dict = None, unit: str = None,
                        fmt: str = None, tile_key: str = None, **view) -> bool:
    """_render_snow_map with the product's colours, units and number format.

    tile_key (default layer_key) keys the cached tile URL — include the colour range in
    it so a new colour scale re-registers the tiles while the viewport is kept.
    """
    cfg = _SD_PRODUCTS[product]
    return _render_snow_map(
        image, vis or cfg["vis"], aoi, layer_key, tile_key=tile_key,
        caption=caption or _sd_product_caption(product),
        value_unit=unit if unit is not None else cfg["unit"],
        value_fmt=fmt or cfg["fmt"],
        value_suffix="" if unit is not None else f" {cfg['label'].lower()}",
        sample_scale=max(scale, 500),
        **view,
    )


def _colour_scale(product: str, key: str, statistic: str = "mean") -> dict:
    """'Colour scale max' input above a map; returns the product's vis params with that max.

    Maximum over a long period often exceeds the default 2 m depth scale in mountains,
    so it starts higher there. Values above the max show in the darkest colour.
    """
    cfg     = _SD_PRODUCTS[product]
    vis     = dict(cfg["vis"])
    default = vis["max"] * (2.5 if statistic == "max" and product == "snow_depth" else 1)
    col, _ = st.columns([1, 3])
    vis["max"] = col.number_input(
        f"🎨 Colour scale max ({cfg['unit']}):",
        min_value=float(vis["min"]) + (0.1 if product == "snow_depth" else 10.0),
        value=float(default), step=0.5 if product == "snow_depth" else 50.0,
        key=f"sd_vmax_{key}_{product}_{statistic}",
        help="Upper end of the colorbar. Pixels above it show in the darkest colour — "
             "click the map to read exact values. Snow-free pixels are near-white (depth) "
             "or transparent (density is undefined without snow).",
    )
    return vis


def _render_sd_statistics(image: ee.Image, aoi, layer_key: str, scale: int,
                          product: str = "snow_depth") -> None:
    """Mean / min / max / std dev and data cover of a map, as metrics. Cached per layer_key."""
    stats_cache = st.session_state.setdefault("sd_stats_cache", {})
    if layer_key not in stats_cache:
        with st.spinner("Computing statistics…"):
            try:
                cover = image.mask().gt(0).multiply(100).rename("cover")
                raw = ee.Image.cat([image.rename("value"), cover]).reduceRegion(
                    reducer=(
                        ee.Reducer.mean()
                        .combine(ee.Reducer.minMax(), "", True)
                        .combine(ee.Reducer.stdDev(), "", True)
                    ),
                    geometry=aoi,
                    scale=scale,
                    maxPixels=1e9,
                    bestEffort=True,
                ).getInfo()
                stats_cache[layer_key] = raw
            except Exception:
                stats_cache[layer_key] = {}

    raw = stats_cache.get(layer_key, {})
    # GEE names the keys as "{band}_mean", "{band}_min", "{band}_max", "{band}_stdDev"
    mean_val  = raw.get("value_mean")
    min_val   = raw.get("value_min")
    max_val   = raw.get("value_max")
    std_val   = raw.get("value_stdDev")
    cover_val = raw.get("cover_mean")

    cfg  = _SD_PRODUCTS[product]
    name = cfg["label"].split()[-1]          # "Depth" / "Density"
    unit, fmt = cfg["unit"], cfg["fmt"]
    if any(v is not None for v in [mean_val, min_val, max_val, std_val]):
        st.caption("Spatial statistics of the map above across your area of interest.")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        if mean_val  is not None: mc1.metric(f"Mean {name} ({unit})", f"{mean_val:{fmt}}")
        if min_val   is not None: mc2.metric(f"Min {name} ({unit})",  f"{min_val:{fmt}}")
        if max_val   is not None: mc3.metric(f"Max {name} ({unit})",  f"{max_val:{fmt}}")
        if std_val   is not None: mc4.metric(f"Std Dev ({unit})",     f"{std_val:{fmt}}")
        if cover_val is not None:
            mc5.metric(cfg["cover"], f"{cover_val:.0f} %",
                       help="Share of the area with a value. Snow Depth: pixels MODIS could "
                            "observe (not cloud-covered all period). Snow Density: snow-covered "
                            "pixels (density is undefined without snow).")
    else:
        st.info(
            "No statistics available — the area may have no snow or no clear-sky "
            "observations in this period, or the scale is too coarse for the AOI."
        )


def _render_sd_export(image: ee.Image, aoi, scale: int, filename: str,
                      button_label: str, key_suffix: str) -> None:
    """Smart GeoTIFF export (Auto / Local / Drive) of one image."""
    helper      = DownloadHelper()
    export_pref = helper.render_smart_download_options(
        export_format="GeoTIFF",
        key=f"sd_export_mode_{key_suffix}",
    )
    filename = re.sub(r"[^\w\-]", "_", filename)[:100]

    if st.button(
        f"📥 Export {button_label} GeoTIFF",
        type="primary", width="stretch",
        key=f"sd_export_btn_{key_suffix}",
    ):
        with st.spinner(f"Exporting {button_label}…"):
            result = helper.execute_smart_download(
                image=image, filename=filename, region=aoi,
                scale=scale, export_preference=export_pref, crs="EPSG:4326",
            )
            if not result.get("success"):
                st.error(f"❌ Export failed: {result.get('message', 'Unknown error')}")
                st.info(
                    "Tip: for very large regions try a coarser scale, "
                    "or use **Google Drive** mode above — it handles any size."
                )


# ===========================================================================
# ANALYSIS SECTION  (AOI-based modes)
# ===========================================================================

def _render_analysis_section(params: dict, density_params: dict) -> None:
    """Run / Clear buttons and results for all AOI-based modes."""
    aoi = st.session_state["snow_depth_geometry"]

    if params["mode"] == "time_series":
        _render_persistent_download()

    col_run, col_clr = st.columns([3, 1])
    with col_run:
        run_clicked = st.button(
            f"▶ Run GLOBAL-SAM — {_build_run_label(params)}",
            type="primary", width="stretch", key="sd_run_btn",
        )
    with col_clr:
        if st.button("🗑️ Clear Results", key="sd_clear_btn"):
            _clear_results()
            st.rerun()

    if run_clicked:
        _execute_analysis(params, density_params, aoi)

    results = st.session_state.get("sd_results")
    if results:
        st.success(f"✅ Analysis complete: **{results['label']}**")
        if results["type"] == "time_series":
            _render_time_series_results(results, density_params, aoi)
        elif results["type"] == "longterm":
            _render_longterm_results(results, aoi)
        else:
            _render_static_results(results, aoi)


def _execute_analysis(params: dict, density_params: dict, aoi) -> None:
    """Build the (lazy) GEE result for the current mode and cache it in session state."""
    error = _validate(params)
    if error:
        st.error(error)
        return

    mode       = params["mode"]
    product    = params["product"]
    statistic  = params["statistic"]
    calculator = _sd_build_calculator(params["scale"], density_params)
    label      = _build_run_label(params)
    base = {"label": label, "product": product, "statistic": statistic, "scale": params["scale"]}

    try:
        if mode in _SD_STATIC_MODES:
            # All lazy — no GEE network call until the map / statistics render
            if mode == "single" and statistic == "closest":
                image = calculator.calculate_closest_products(
                    aoi, params["target_date"], params["window"]
                )
            elif mode in ("single", "custom_range"):
                image = calculator.calculate_range_products(aoi, *_selected_range(params), statistic)
            else:
                image = calculator.calculate_products(aoi, _selected_months(params), statistic)
            st.session_state["sd_results"] = {
                **base,
                "type":     "static",
                "image":    image.select(product),
                "covers":   _period_summary(params),
                "filename": f"{output_file_stem(product, statistic)}_{_file_tag(params)}_{params['scale']}m",
                "closest_window": params["window"] if statistic == "closest" else None,
            }

        elif mode == "time_series":
            steps = snow_time_steps(params["start_date"], params["end_date"], params["interval"])
            progress = st.progress(0.0, text=f"Computing {len(steps)} time steps on GEE…")

            def _progress(done, total):
                progress.progress(done / total, text=f"Computed {done} of {total} time steps…")

            rows = extract_snow_time_series(
                calculator, aoi, steps, statistic, product, params["scale"],
                batch_size=12 if statistic == "mean" else 4, progress=_progress,
            )
            st.session_state["sd_results"] = {
                **base,
                "type":     "time_series",
                "steps":    steps,
                "rows":     rows,
                "interval": params["interval"],
            }

        elif mode == "longterm_stats":
            with st.spinner(f"❄️ Building {_SD_PRODUCTS[product]['label'].lower()} statistics on GEE…"):
                if params["lt_subtype"] == "temporal_cv":
                    result = run_snow_depth_temporal_variability(
                        calculator, aoi, params["start_year"], params["end_year"], product=product,
                    )
                else:
                    result = run_snow_depth_period_stats(
                        calculator, aoi, params["start_year"], params["end_year"],
                        period_type=params["period_type"], month=params["month"],
                        hemisphere=params["hemisphere"], statistic=statistic,
                        product=product, season=params["season"],
                    )
            st.session_state["sd_results"] = {
                **base, "type": "longterm", "subtype": params["lt_subtype"], "result": result,
            }

        st.rerun()

    except Exception as exc:
        st.error(f"❌ Analysis failed: {exc}")
        st.info(
            "Tips: reduce the AOI, shorten the period, use a coarser resolution, or choose "
            "**Mean** (much lighter than Median / Maximum / Minimum). For very large areas "
            "switch to Google Drive export."
        )


# ===========================================================================
# RESULTS — one map over one period
# ===========================================================================

def _render_static_results(results: dict, aoi) -> None:
    """Map + statistics + GeoTIFF export for Single Date / Monthly / Seasonal / Annual / Range."""
    product   = results["product"]
    statistic = results["statistic"]
    cfg       = _SD_PRODUCTS[product]
    scale     = results["scale"]
    image     = results["image"]
    layer_key = f"{st.session_state.get('sd_params_key', '')}::static"

    st.caption(f"📅 Covers **{results['covers']}**")
    if statistic == "closest":
        st.caption(
            f"Each pixel shows the clear-sky day nearest the target date (up to "
            f"±{results['closest_window']} days away); pixels cloudy on every day stay empty."
        )
    elif statistic != "mean":
        st.caption(f"{_SD_STATISTICS[statistic]} of the daily estimates over all clear-sky "
                   "days in the period, pixel by pixel.")

    st.markdown(f"#### 🗺️ {results['label']}")
    vis    = _colour_scale(product, "static", statistic)
    map_ok = _render_product_map(image, aoi, layer_key, product, scale, vis=vis,
                                 tile_key=f"{layer_key}::{vis['max']}")

    if map_ok:
        st.markdown("#### 📊 Statistics")
        _render_sd_statistics(image, aoi, layer_key, scale, product)

    st.markdown("---")
    with st.expander("💾 Export GeoTIFF", expanded=not map_ok):
        st.caption(f"Single-band Float32 GeoTIFF of {_sd_product_caption(product).lower()} "
                   f"at {scale} m, EPSG:4326.")
        _render_sd_export(image, aoi, scale, results["filename"], cfg["label"], "static")


# ===========================================================================
# RESULTS — time series
# ===========================================================================

def _render_time_series_chart(rows: list, product: str) -> None:
    """AOI-mean product per step (line) with snow-covered area (bars, right axis)."""
    cfg   = _SD_PRODUCTS[product]
    valid = [r for r in rows if r["mean"] is not None]
    if not valid:
        st.warning("No valid data in any time step — the area may have no snow or no "
                   "clear-sky observations in this period.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=[r["start"] for r in rows],
        y=[r["snow_cover_pct"] for r in rows],
        name="Snow-covered area (%)",
        marker_color="rgba(158, 202, 225, 0.45)",
        hovertemplate="<b>%{x}</b><br>Snow-covered: %{y:.0f} %<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=[r["start"] for r in valid],
        y=[r["mean"] for r in valid],
        customdata=[r["data_pct"] or 0 for r in valid],
        mode="lines+markers",
        line=dict(color="#08519C", width=2),
        marker=dict(size=6, color="#08519C"),
        name=f"AOI mean {cfg['label'].lower()}",
        hovertemplate=(f"<b>%{{x}}</b><br>Mean: %{{y:{cfg['fmt']}}} {cfg['unit']}"
                       "<br>Data cover: %{customdata:.0f} %<extra></extra>"),
    ), secondary_y=False)
    fig.update_yaxes(title_text=f"Mean {_sd_product_caption(product)}",
                     rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Snow-covered area (%)", range=[0, 100], tickmode="linear",
                     dtick=20, showgrid=False, secondary_y=True)
    fig.update_layout(
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", y=1.12, x=0),
        bargap=0.15,
    )
    st.plotly_chart(fig, width="stretch")


def _render_time_series_results(results: dict, density_params: dict, aoi) -> None:
    """Chart, table/CSV, per-step map browser and bulk export for Time Series mode."""
    rows      = results["rows"]
    product   = results["product"]
    statistic = results["statistic"]
    cfg       = _SD_PRODUCTS[product]
    scale     = results["scale"]
    pk        = st.session_state.get("sd_params_key", "")
    stat_name = _SD_STATISTICS[statistic]

    st.markdown(f"#### 📈 {stat_name} {cfg['label']} — AOI Time Series")
    st.caption(
        f"Each point is the AOI mean of the **{stat_name.lower()} {cfg['label'].lower()}** "
        f"over one {_SD_INTERVALS[results['interval']].lower()} step. Bars: share of the "
        "observed area with snow depth ≥ 1 cm."
    )
    _render_time_series_chart(rows, product)

    valid = [r for r in rows if r["mean"] is not None]
    if valid:
        peak = max(valid, key=lambda r: r["mean"])
        sca  = [r["snow_cover_pct"] for r in valid if r["snow_cover_pct"] is not None]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Steps with data", f"{len(valid)} / {len(rows)}")
        mc2.metric(f"Mean ({cfg['unit']})", f"{sum(r['mean'] for r in valid) / len(valid):{cfg['fmt']}}")
        mc3.metric(f"Peak {peak['label']} ({cfg['unit']})", f"{peak['mean']:{cfg['fmt']}}")
        if sca:
            mc4.metric("Mean snow-covered area", f"{sum(sca) / len(sca):.0f} %")
        low = [r for r in valid if (r["data_pct"] or 0) < 80]
        if low:
            st.caption(
                f"⚠️ {len(low)} step(s) have estimates for less than 80 % of the area "
                "(persistent cloud) — their AOI means are less representative."
            )

    # ── Table + CSV ──────────────────────────────────────────────────────
    unit_tag = cfg["unit"].replace("/", "_per_").replace("³", "3")
    header = ["step", "start", "end", f"mean_{product}_{unit_tag}", f"min_{product}_{unit_tag}",
              f"max_{product}_{unit_tag}", "snow_covered_pct", "data_cover_pct"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for r in rows:
        writer.writerow([r["label"], r["start"], r["end"], r["mean"], r["min"], r["max"],
                         r["snow_cover_pct"], r["data_pct"]])
    with st.expander("📋 Time series table", expanded=False):
        st.dataframe(
            [{"Step": r["label"], "Start": r["start"], "End": r["end"],
              f"Mean ({cfg['unit']})": r["mean"], f"Min ({cfg['unit']})": r["min"],
              f"Max ({cfg['unit']})": r["max"], "Snow-covered (%)": r["snow_cover_pct"],
              "Data cover (%)": r["data_pct"]} for r in rows],
            hide_index=True, width="stretch",
        )
    st.download_button(
        "📥 Download Time Series CSV",
        data=buf.getvalue().encode(),
        file_name=(f"{output_file_stem(product, statistic)}_timeseries_{results['interval']}_"
                   f"{rows[0]['start']}_{rows[-1]['end']}.csv"),
        mime="text/csv", type="primary", width="stretch", key="sd_ts_csv_dl",
    )

    # ── Map of one step ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗺️ Map of a Time Step")
    labels  = [step[0] for step in results["steps"]]
    default = max(valid, key=lambda r: r["mean"])["label"] if valid else labels[0]
    sel = st.select_slider(
        "Time step:", options=labels, value=default, key=f"sd_ts_step_{pk}",
        help="Defaults to the step with the highest AOI mean. Viewed steps are cached.",
    )
    label, step_start, step_end = results["steps"][labels.index(sel)]
    calculator = _sd_build_calculator(scale, density_params)
    step_img = calculator.calculate_range_products(aoi, step_start, step_end, statistic).select(product)
    layer_key = f"{pk}::ts::{label}"
    period = f"{step_start:%d %b %Y} – {step_end - datetime.timedelta(days=1):%d %b %Y}"
    st.markdown(f"##### ❄️ {stat_name} {cfg['label']} — {period}")
    vis    = _colour_scale(product, "ts", statistic)
    map_ok = _render_product_map(step_img, aoi, layer_key, product, scale, vis=vis,
                                 tile_key=f"{layer_key}::{vis['max']}")
    if map_ok:
        _render_sd_statistics(step_img, aoi, layer_key, scale, product)
    with st.expander(f"💾 Export this step only — {period}", expanded=False):
        _render_sd_export(
            step_img, aoi, scale,
            f"{output_file_stem(product, statistic)}_{step_file_tag(label)}_{scale}m",
            f"{label}", "ts_step",
        )

    # ── Every step ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("💾 Export — Download All Time Steps as GeoTIFFs", expanded=False):
        _render_bulk_export(results, density_params, aoi)


def _render_bulk_export(results: dict, density_params: dict, aoi) -> None:
    """One GeoTIFF per time step — ZIP to the browser, or one Google Drive task per step."""
    steps     = results["steps"]
    product   = results["product"]
    statistic = results["statistic"]
    scale     = results["scale"]
    file_stem = output_file_stem(product, statistic)

    st.markdown(
        "Generate one GeoTIFF per time step. **Local / Auto** packs them into a ZIP "
        "and downloads to your browser. **Google Drive** submits each step as a "
        "separate GEE export task — handles any AOI size without memory limits."
    )
    st.caption(
        f"Exporting **{_SD_STATISTICS[statistic].lower()} {_sd_product_caption(product).lower()}** "
        f"per {_SD_INTERVALS[results['interval']].lower()} step — change it in the sidebar. "
        f"Files: `{file_stem}_{step_file_tag(steps[0][0])}.tif`, …"
    )

    zip_helper      = DownloadHelper()
    zip_export_pref = zip_helper.render_smart_download_options(
        export_format="GeoTIFF",
        key="sd_export_mode_zip",
    )

    proc_col1, proc_col2 = st.columns([2, 1])
    with proc_col2:
        st.metric("Steps to export", len(steps))
        area_km2 = st.session_state.get("snow_depth_area_km2")
        if area_km2:
            st.metric("Area", f"{area_km2:,.2f} km²")

    with proc_col1:
        btn_label = (
            "☁️ Submit to Google Drive"
            if zip_export_pref == "drive"
            else "🚀 Generate & Download GeoTIFFs (ZIP)"
        )
        if not st.button(btn_label, type="primary", width="stretch", key="sd_generate_btn"):
            return

    calculator = _sd_build_calculator(scale, density_params)

    if zip_export_pref == "drive":
        # ── Google Drive: one GEE export task per step ───────────────────
        tasks_started = 0
        tasks_failed  = 0
        drive_folder  = "GeoClimate_SnowDepth"
        with st.spinner(f"Submitting {len(steps)} GEE export tasks to Drive…"):
            for label, step_start, step_end in steps:
                name = f"{file_stem}_{step_file_tag(label)}_{scale}m"
                try:
                    img = calculator.calculate_range_products(
                        aoi, step_start, step_end, statistic
                    ).select(product)
                    task = ee.batch.Export.image.toDrive(
                        image=img,
                        description=name,
                        folder=drive_folder,
                        fileNamePrefix=name,
                        region=aoi,
                        scale=scale,
                        crs="EPSG:4326",
                        maxPixels=1e13,
                    )
                    task.start()
                    tasks_started += 1
                except Exception as exc:
                    tasks_failed += 1
                    st.warning(f"⚠️ {label}: {exc}")

        if tasks_started:
            st.success(
                f"✅ {tasks_started} export tasks submitted to Google Drive "
                f"folder **'{drive_folder}'**."
            )
            if tasks_failed:
                st.warning(f"⚠️ {tasks_failed} steps failed to submit.")
            st.info("Files appear in your Drive once each task completes "
                    "(typically a few minutes per step).")
            st.markdown("🔗 [Monitor GEE Tasks](https://code.earthengine.google.com/tasks)")
        else:
            st.error("❌ No tasks could be submitted. Check GEE authentication.")
        return

    # ── Auto / Local: chunked download → ZIP → browser ───────────────────
    progress_bar = st.progress(0)
    status_text  = st.empty()
    try:
        with st.spinner(f"Processing {_sd_product_caption(product).lower()} time steps…"):
            zip_data = process_period_steps(
                calculator, aoi, steps, progress_bar, status_text,
                product=product, statistic=statistic,
            )
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")

        filename = (f"{file_stem}_{results['interval']}_{steps[0][1]:%Y%m%d}_"
                    f"{steps[-1][2] - datetime.timedelta(days=1):%Y%m%d}_{scale}m.zip")
        files = [f"{file_stem}_{step_file_tag(label)}.tif" for label, _, _ in steps]
        st.session_state["sd_zip_data"]     = zip_data
        st.session_state["sd_zip_filename"] = filename
        st.session_state["sd_zip_files"]    = files

        st.success(f"✅ Generated GeoTIFFs for {len(steps)} time steps.")
        st.download_button(
            label="📥 Download GeoTIFFs (ZIP)",
            data=zip_data,
            file_name=filename,
            mime="application/zip",
            width="stretch",
            key="sd_immediate_download_btn",
        )
        with st.expander("📋 Files included in the ZIP"):
            for name in files:
                st.text(name)

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.info(
            "Try reducing the area size, date range, or increasing the "
            "resolution scale. For very large regions switch to "
            "**Google Drive** mode above."
        )


# ===========================================================================
# RESULTS — long-term statistics
# ===========================================================================

def _render_longterm_results(results: dict, aoi) -> None:
    """Dispatch long-term statistics to the inter-annual or temporal-CV renderer."""
    if results["subtype"] == "temporal_cv":
        _render_temporal_cv_snow_results(results, aoi)
    else:
        _render_interannual_snow_results(results, aoi)


def _render_interannual_snow_results(results: dict, aoi) -> None:
    """Display Mean / Median / Min / Max / Std Dev tabs for inter-annual statistics."""
    result     = results["result"]
    product    = results["product"]
    scale      = results["scale"]
    params_key = st.session_state.get("sd_params_key", "lt")
    cfg        = _SD_PRODUCTS[product]
    prod       = cfg["label"]
    composite  = _composite_name(
        result["period_type"], result.get("month", 1), result.get("season", "snow_season"),
        result.get("statistic", "mean"),
    )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Valid Years",        str(result["year_count"]))
    sc2.metric("Year Range",         f"{result['start_year']} – {result['end_year']}")
    sc3.metric("Per-year Composite", composite[0].upper() + composite[1:])

    st.markdown("#### 🗺️ Pixel-wise Statistical Maps")
    st.caption(
        f"Each year is first reduced to its **{composite} {prod.lower()}**; the tabs then "
        f"summarise those {result['year_count']} yearly values pixel by pixel. "
        "Maps render via live GEE tile fetching."
    )
    if result.get("statistic", "mean") == "mean":
        st.caption(
            "💡 For peak or low snow conditions, choose **Maximum** or **Minimum** under "
            "*Statistic within each year's period* in the sidebar."
        )

    # (result key, tab title, across-years word, tab caption)
    layers = [
        ("mean",    "📊 Mean",    "Mean",    "Long-term **mean** of the yearly values — typical conditions."),
        ("median",  "➗ Median",  "Median",  "**Median** of the yearly values — typical year, robust to outlier years."),
        ("min",     "⬇️ Minimum", "Minimum", "Lowest yearly value at each pixel."),
        ("max",     "⬆️ Maximum", "Maximum", "Highest yearly value at each pixel."),
        ("std_dev", "〰️ Std Dev", "Std Dev", "**Std dev** of the yearly values — inter-annual variability. "
                                            "High = unstable year to year."),
    ]
    vis  = _colour_scale(product, "lt", result.get("statistic", "mean"))
    tabs = st.tabs([title for _, title, _, _ in layers])
    ok_mean = False
    for tab, (key, _, across, caption) in zip(tabs, layers):
        with tab:
            st.caption(caption)
            if key != "mean" and not ok_mean:
                st.info(_SD_MAP_UNAVAILABLE)
                continue
            layer_vis = cfg["std_vis"] if key == "std_dev" else vis
            ok = _render_product_map(
                result[key], aoi, f"{params_key}::lt::{key}", product, scale,
                caption=f"{across} of {composite} {_sd_product_caption(product)}",
                vis=layer_vis, tile_key=f"{params_key}::lt::{key}::{layer_vis['max']}",
            )
            if key == "mean":
                ok_mean = ok

    st.markdown("---")
    with st.expander("💾 Export a statistical image as GeoTIFF", expanded=not ok_mean):
        export_labels = {across: key for key, _, across, _ in layers}
        stat_choice = st.selectbox(
            "Statistic to export:", list(export_labels), key="sd_lt_ia_export_choice",
        )
        _render_sd_export(
            result[export_labels[stat_choice]], aoi, scale,
            f"{product}_{stat_choice}_of_{composite}_{result['start_year']}-{result['end_year']}_{scale}m",
            f"{stat_choice} of {composite}", f"ia_{export_labels[stat_choice]}",
        )


def _render_temporal_cv_snow_results(results: dict, aoi) -> None:
    """Display Long-term Mean / Temporal Std Dev / CV tabs for temporal variability."""
    result     = results["result"]
    product    = results["product"]
    scale      = results["scale"]
    params_key = st.session_state.get("sd_params_key", "lt")
    cfg        = _SD_PRODUCTS[product]
    prod       = cfg["label"]

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Monthly Composites", str(result["total_months"]))
    sc2.metric("Year Range",         f"{result['start_year']} – {result['end_year']}")
    sc3.metric("Method",             "Temporal CV")

    st.markdown("#### 🗺️ Temporal Variability Maps")
    st.caption(
        "Statistics computed across **all monthly-mean composites** — captures total temporal "
        "variability (seasonal amplitude + inter-annual shifts). "
        f"CV normalises for the baseline {prod.lower()} so pixels are directly comparable."
    )

    tab_mean, tab_std, tab_cv = st.tabs(
        ["📊 Long-term Mean", "〰️ Temporal Std Dev", "📉 CV (%)"]
    )

    with tab_mean:
        st.caption(
            f"Long-term mean {prod.lower()} across all monthly composites — "
            "multi-year climatological average."
        )
        vis = _colour_scale(product, "cv")
        ok_mean = _render_product_map(
            result["mean"], aoi, f"{params_key}::cv::mean", product, scale,
            caption=f"Long-term Mean {_sd_product_caption(product)}",
            vis=vis, tile_key=f"{params_key}::cv::mean::{vis['max']}",
        )

    with tab_std:
        st.caption(
            "**Total temporal std dev** across all monthly composites — "
            "combines seasonal amplitude and inter-annual variability."
        )
        if ok_mean:
            _render_product_map(
                result["std_dev"], aoi, f"{params_key}::cv::std", product, scale,
                caption=f"Temporal Std Dev ({cfg['unit']})", vis=cfg["std_vis"],
            )
        else:
            st.info(_SD_MAP_UNAVAILABLE)

    with tab_cv:
        if product == "snow_depth":
            st.caption(
                "**CV = σ/μ × 100 %** — relative fluctuation independent of baseline depth. "
                "Because snow-free months are included, values of ~100–250 % are typical: "
                "lower CV (≈ 100–150 %) = persistent seasonal snowpack; "
                "higher CV (> 200 %) = ephemeral, short-lived snow cover."
            )
        else:
            st.caption(
                "**CV = σ/μ × 100 %** of snow density. Density is undefined without snow, so "
                "only snow-covered months contribute — CV reflects how much the snowpack "
                "densifies through the season (typically ~20–60 %)."
            )
        if ok_mean:
            _render_product_map(
                result["cv"], aoi, f"{params_key}::cv::cv", product, scale,
                caption=f"Temporal CV of {prod} (%)", vis=cfg["cv_vis"], unit="%", fmt=".1f",
            )
        else:
            st.info(_SD_MAP_UNAVAILABLE)

    st.markdown("---")
    with st.expander("💾 Export a variability image as GeoTIFF", expanded=not ok_mean):
        stat_choice = st.selectbox(
            "Layer to export:",
            ["Long-term Mean", "Temporal Std Dev", "CV (%)"],
            key="sd_lt_cv_export_choice",
        )
        img_map = {
            "Long-term Mean":   result["mean"],
            "Temporal Std Dev": result["std_dev"],
            "CV (%)":           result["cv"],
        }
        _render_sd_export(
            img_map[stat_choice], aoi, scale,
            f"{product}_{stat_choice}_monthly_{result['start_year']}-{result['end_year']}_{scale}m",
            stat_choice, f"cv_{stat_choice[:3].lower()}",
        )


# ===========================================================================
# REGIONAL CLIMATOLOGY MODE
# ===========================================================================

def _render_regional_mode(params: dict, density_params: dict) -> None:
    """Run button and results for Regional Climatology (predefined region or own AOI)."""
    predefined = params["regional_aoi_source"] == "predefined"
    product    = params["product"]
    cfg        = _SD_PRODUCTS[product]
    composite  = _composite_name(params["period_type"], params["month"], params["season"],
                                 params["statistic"])
    if params["period_type"] == "seasonal":
        composite += f" ({'NH' if params['hemisphere'] == 'north' else 'SH'})"
    summary    = _SD_SUMMARY_STATS[params["summary_statistic"]]

    if predefined:
        region_key   = params["region"]
        region_label = PREDEFINED_REGION_LABELS[region_key]
        aoi          = get_predefined_region(region_key)
        west, south, east, north = PREDEFINED_REGION_COORDS[region_key]
        view = {"center": ((south + north) / 2, (west + east) / 2),
                "zoom": 2 if region_key == "globe" else (3 if east - west > 40 else 4)}
        st.markdown("### 🌍 Regional Climatology")
        st.info(
            "Multi-year snow depth / density climatology over a predefined region — no area "
            "drawing needed. Keep the resolution coarse (1–5 km) for continental regions and "
            "use **Google Drive** export for GeoTIFFs."
        )
    else:
        region_key, region_label = None, "Custom Area"
        aoi  = st.session_state["snow_depth_geometry"]
        view = {}

    period_label = (f"{region_label} · {summary} of {composite} {cfg['label'].lower()} · "
                    f"{params['start_year']}–{params['end_year']}")

    col_run, col_clr = st.columns([3, 1])
    with col_run:
        run_clicked = st.button(
            f"▶ Run Regional Climatology — {period_label}",
            type="primary", width="stretch", key="sd_reg_run_btn",
        )
    with col_clr:
        if st.button("🗑️ Clear", key="sd_reg_clear"):
            _clear_results()
            st.rerun()

    if run_clicked:
        error = _validate(params)
        if error:
            st.error(error)
        else:
            calculator = _sd_build_calculator(params["scale"], density_params)
            with st.spinner(f"🌍 Building regional climatology for {region_label}…"):
                try:
                    result = run_snow_depth_period_stats(
                        calculator, aoi, params["start_year"], params["end_year"],
                        period_type=params["period_type"], month=params["month"],
                        hemisphere=params["hemisphere"], statistic=params["statistic"],
                        product=product, season=params["season"],
                    )
                    st.session_state["sd_results"] = {
                        "type":         "regional",
                        "image":        result[params["summary_statistic"]],
                        "label":        period_label,
                        "region_label": region_label,
                        "year_count":   result["year_count"],
                        "summary":      params["summary_statistic"],
                        "composite":    composite,
                        "product":      product,
                        "scale":        params["scale"],
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Regional analysis failed: {exc}")

    results = st.session_state.get("sd_results")
    if not results or results["type"] != "regional":
        return

    st.success(f"✅ Regional Climatology complete: **{results['label']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Region",    results["region_label"])
    c2.metric("Years",     str(results["year_count"]))
    c3.metric("Summary",   _SD_SUMMARY_STATS[results["summary"]])

    across = _SD_SUMMARY_STATS[results["summary"]]
    st.markdown(f"#### 🗺️ {cfg['label']} Climatology")
    st.caption(
        f"Each year is reduced to its **{results['composite']} {cfg['label'].lower()}**; the map "
        f"shows the **{across.lower()}** of those {results['year_count']} yearly values."
    )
    layer_key = f"{st.session_state.get('sd_params_key', '')}::regional"
    if results["summary"] == "std_dev":
        vis = cfg["std_vis"]
    else:
        vis = _colour_scale(product, "regional", "max" if results["summary"] == "max" else "mean")
    map_ok = _render_product_map(
        results["image"], aoi, layer_key, product, results["scale"],
        caption=f"{across} of {results['composite']} {_sd_product_caption(product)}",
        vis=vis, tile_key=f"{layer_key}::{vis['max']}", **view,
    )

    st.markdown("---")
    with st.expander("💾 Export GeoTIFF", expanded=not map_ok):
        _render_sd_export(
            results["image"], aoi, results["scale"],
            f"{product}_{results['summary']}_of_{results['composite']}_{region_key or 'aoi'}_"
            f"{params['start_year']}-{params['end_year']}_{results['scale']}m",
            f"{across} {cfg['label']}", "regional",
        )


# ---------------------------------------------------------------------------
# Algorithm Comparison mode
# ---------------------------------------------------------------------------
def _render_algorithm_comparison(params: dict, density_method, density_params):
    """Render the Algorithm Comparison mode UI."""
    st.subheader("🔍 Algorithm Comparison Mode")

    comparison_year  = params["comparison_year"]
    comparison_month = params["comparison_month"]
    comp_scale       = params["comp_scale"]

    st.markdown("### Upload Reference Data")

    uploaded_tiff = st.file_uploader(
        "Upload Reference Snow Depth TIFF File",
        type=["tif", "tiff"],
        help="Upload a GeoTIFF file containing reference snow depth data for comparison",
        key="sd_ref_tiff",
    )

    if uploaded_tiff is None:
        st.info("📤 Please upload a reference TIFF file to begin comparison")
        return

    bounds = extract_bounds_from_tiff(uploaded_tiff)
    if bounds is None:
        return

    st.success("✅ Reference TIFF file uploaded successfully!")
    st.info(
        f"**Bounds:** {bounds[0]:.4f}, {bounds[1]:.4f}, "
        f"{bounds[2]:.4f}, {bounds[3]:.4f}"
    )

    if density_params["method"] == "original":
        density_value = get_seasonal_snow_density(
            comparison_month,
            density_params.get("use_seasonal_density", True),
            density_params.get("custom_density_value"),
            density_params.get("custom_monthly_densities"),
        )
        if density_params.get("custom_density_value") is not None:
            density_type = "Custom Constant"
        elif density_params.get("custom_monthly_densities") is not None:
            density_type = "Custom Monthly"
        elif density_params.get("use_seasonal_density", True):
            density_type = "Default Seasonal"
        else:
            density_type = "Default Constant"
        density_info = f"{density_type} — {density_value:.0f} kg/m³"

    elif density_params["method"] == "enhanced_method1":
        gamma_val = density_params["gamma_params"].get(
            f"gamma_{comparison_month}", 1.0
        )
        density_info = (
            f"Enhanced Method 1 — γ = {gamma_val:.3f}, "
            f"ρ_min = {density_params['rho_min']:.0f}, "
            f"ρ_max = {density_params['rho_max']:.0f} kg/m³"
        )

    elif density_params["method"] == "enhanced_method2":
        alpha = density_params["gamma_params"].get("alpha", 1.0)
        beta  = density_params["gamma_params"].get("beta", 0.0)
        phi   = density_params["gamma_params"].get("phi", 1.0)
        gamma_val = sinusoidal_gamma_function(comparison_month, alpha, beta, phi)
        density_info = (
            f"Enhanced Method 2 — γ(t) = {gamma_val:.3f}, "
            f"ρ_min = {density_params['rho_min']:.0f}, "
            f"ρ_max = {density_params['rho_max']:.0f} kg/m³"
        )
    else:
        density_info = "Unknown density method"

    st.info(
        f"🏔️ Density setting: {density_info} "
        f"for {_month_name(comparison_month)}"
    )

    st.markdown("---")
    if st.button(
        "🔍 Run Comparison",
        type="primary",
        width="stretch",
        key="sd_run_comparison",
    ):
        run_comparison(
            uploaded_tiff,
            bounds,
            comparison_year,
            comparison_month,
            comp_scale,
            density_method,
            density_params,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def render_snow_depth():
    """Render the GLOBAL-SAM Snow Depth & Snow Density interface.

    GEE is already initialised by the main app, so this function goes
    straight to the analysis UI — no auth gate.
    """
    st.markdown("## ❄️ GLOBAL-SAM — Snow Depth & Snow Density")
    st.markdown(
        "Snow depth and snow density anywhere on Earth from MODIS snow cover and GLDAS snow "
        "water equivalent on Google Earth Engine — for a single date, a month, a season, a "
        "year or any date range, as a **mean, median, maximum or minimum** — with time series, "
        "long-term statistics, GeoTIFF export and algorithm comparison."
    )
    st.divider()

    _auto_expand_sidebar()
    params, density_method, density_params = _render_sidebar()

    # Auto-clear results whenever any sidebar parameter (or the AOI) changes
    params_key = _params_key(params, density_params)
    if st.session_state.get("sd_params_key") != params_key:
        _clear_results()
        st.session_state["sd_params_key"] = params_key

    mode = params["mode"]
    if mode == "comparison":
        _render_algorithm_comparison(params, density_method, density_params)
        display_comparison_results()
    elif mode == "regional" and params["regional_aoi_source"] == "predefined":
        _render_regional_mode(params, density_params)
    elif _render_geometry_selection():
        st.divider()
        if mode == "regional":
            _render_regional_mode(params, density_params)
        else:
            _render_analysis_section(params, density_params)

    _render_about_expander()


def _render_about_expander():
    with st.expander("ℹ️ About GLOBAL-SAM"):
        st.markdown("""
        ### ❄️ GLOBAL-SAM — Snow Depth & Snow Density

        **Data Sources:**
        - **MODIS Snow Cover** (MOD10A1, 500 m, daily) — Normalized Difference Snow Index
        - **GLDAS-2.1 Snow Water Equivalent** (0.25°, 3-hourly) — NASA Global Land Data Assimilation System

        **Algorithm:**
        1. Calculate Fractional Snow Cover (FSC) from MODIS NDSI
        2. Estimate snow density from FSC and the selected density method (per month)
        3. Downscale GLDAS SWE using FSC
        4. Snow Depth = Downscaled SWE ÷ Snow Density

        ## 📦 Products & Statistics

        Every map and download can show **Snow Depth (m)** or **Snow Density (kg/m³)**
        (density is masked where there is no snow), reduced over the chosen period with:

        | Statistic | How it is computed |
        |---|---|
        | **Mean** | Monthly algorithm (mean MODIS NDSI × mean GLDAS SWE) for each calendar month — or the part of it inside the period — averaged weighted by days |
        | **Median / Maximum / Minimum** | Daily estimates — each day's MODIS NDSI × that day's GLDAS SWE — reduced over all clear-sky days in the period |
        | **Closest clear-sky day** *(Single Date)* | Per pixel, the daily estimate from the clear-sky day nearest the target date |

        ## 🔬 Analysis Modes

        | Mode | What it gives |
        |---|---|
        | 📅 **Single Date** | Snow on a target date — the closest clear-sky day, or a statistic over ±N days |
        | 📆 **Monthly Composite** | One month, or the same month across several years |
        | 🌨️ **Seasonal Composite** | Snow season (Oct–Apr NH / Apr–Oct SH) or a calendar season, single or multi-year |
        | 📊 **Annual Composite** | One calendar year, or several years together |
        | 📋 **Custom Date Range** | Any start and end date |
        | 📈 **Time Series & Bulk Export** | Monthly / 16-day / weekly AOI means with snow-covered area, CSV, a map of any step, and one GeoTIFF per step (ZIP or Google Drive) |
        | 🌍 **Regional Climatology** | Multi-year climatology over a predefined region (or your own area) — mean / median / min / max / std dev across years |
        | 📉 **Long-term Statistics** | Inter-annual maps (mean, median, min, max, std dev of yearly composites) or temporal variability (mean, std dev, CV of all monthly composites) |
        | 🔍 **Algorithm Comparison** | Validate against a reference GeoTIFF — R², RMSE, MAE, bias, correlation, plots and difference map |

        **Multi-year composites** take the statistic over all days of the selected period in
        every year together — e.g. *Maximum + January 2019–2023* = the deepest January snow in
        any of those years.

        All maps are live GEE tiles with a pixel inspector (click to read a value) and AOI
        statistics; results can be exported as GeoTIFF (Auto / Local / Google Drive).

        ## 🏔️ Snow Density Methods

        | Method | Description |
        |---|---|
        | Original | Seasonal look-up table (180–420 kg/m³) |
        | Enhanced Method 1 | Inverse-fit γ per month using literature densities |
        | Enhanced Method 2 | Sinusoidal γ fit for smooth seasonal variation |

        **Notes:** MODIS cannot see through cloud — cloud-covered pixels are filled from other
        days in the period, and stay empty if cloudy on every day. GLDAS-2.1 lags real time by
        ~1–2 months, so the most recent weeks are masked. Large AOIs are automatically chunked
        during local export.
        """)
