"""
Snow Depth Interface Module
Handles the complete interface for the Snow Depth Analysis & Comparison tool.
Authentication is handled by the main app — no separate auth needed here.

Key design decisions:
- Uses shared GeometrySelectionWidget (same as other modules)
- ZIP results persisted in session state so re-download survives navigation
- Sidebar auto-expands on first visit (all parameters live there)
- Fast GEE-tile preview (SERVES-style) comes before the slow ZIP export pipeline
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import datetime
import hashlib
import json
import folium
import ee
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

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
    generate_month_list,
    process_monthly_data,
    extract_bounds_from_tiff,
    run_comparison,
    create_density_settings_ui,
    display_comparison_results,
    create_chunks,
    download_chunk,
    merge_chunks,
)

from geoclimate_fetcher.core import GeometrySelectionWidget
from app_components.download_component import DownloadHelper
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
_SD_TILE_FAILED = "__FAILED__"   # sentinel cached on getMapId() failure


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
                        var btn = window.parent.document.querySelector(
                            '[data-testid="collapsedControl"]'
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


# ---------------------------------------------------------------------------
# Geometry selection
# ---------------------------------------------------------------------------

# Continent keys (shown first) vs sub-continental keys for the region picker
_SD_CONTINENT_KEYS = [
    "globe", "europe", "asia", "north_america",
    "south_america", "africa", "australia",
]
_SD_SUB_KEYS = [k for k in PREDEFINED_REGION_LABELS if k not in _SD_CONTINENT_KEYS]


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
        # Invalidate all visualization caches
        for k in ("sd_tile_cache", "sd_sample_cache", "sd_stats_cache", "sd_params_key"):
            st.session_state.pop(k, None)
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
                "sd_zip_data", "sd_zip_filename", "sd_zip_months_list",
                # Visualization caches
                "sd_tile_cache", "sd_sample_cache", "sd_stats_cache",
                "sd_params_key",
            ]:
                st.session_state.pop(key, None)
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

    st.success("📦 Your previously generated snow depth data is ready to download again.")

    dl_col, clear_col = st.columns([4, 1])
    with dl_col:
        st.download_button(
            label="📥 Re-Download Monthly Snow Depth ZIP",
            data=st.session_state["sd_zip_data"],
            file_name=st.session_state["sd_zip_filename"],
            mime="application/zip",
            width="stretch",
            key="sd_persistent_download_btn",
        )
    with clear_col:
        if st.button("✖ Clear", key="sd_clear_results", help="Remove cached results"):
            for key in ["sd_zip_data", "sd_zip_filename", "sd_zip_months_list"]:
                st.session_state.pop(key, None)
            st.rerun()

    if st.session_state.get("sd_zip_months_list"):
        with st.expander("📋 Files in this ZIP", expanded=False):
            for yr, mo in st.session_state["sd_zip_months_list"]:
                st.text(f"snow_depth_{datetime.date(yr, mo, 1).strftime('%Y_%m')}.tif")

    st.divider()


# ===========================================================================
# VISUALIZATION HELPERS  (SERVES-style: tile cache, map, statistics)
# ===========================================================================

def _sd_params_key(scale: int, density_params: dict) -> str:
    """MD5 of scale + area_km2 + density params — changes → invalidate all caches."""
    area_km2 = st.session_state.get("snow_depth_area_km2", 0.0)
    # Serialize density_params safely (no complex objects)
    safe = {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in density_params.items()
        if k != "custom_monthly_densities"  # handled separately below
    }
    cmd = density_params.get("custom_monthly_densities")
    if cmd:
        safe["custom_monthly_densities"] = json.dumps(
            {str(k): v for k, v in cmd.items()}, sort_keys=True
        )
    raw = f"{scale}_{area_km2:.2f}_{json.dumps(safe, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _sd_get_tile_url(image: ee.Image, cache_key: str) -> str:
    """
    Register an ee.Image with GEE once and cache the tile URL.
    Failures are also cached so getMapId() is never retried on re-renders.
    """
    cache = st.session_state.setdefault("sd_tile_cache", {})
    if cache_key not in cache:
        try:
            map_id = image.getMapId(_SD_VIS_PARAMS)
            cache[cache_key] = map_id["tile_fetcher"].url_format
        except Exception as exc:
            cache[cache_key] = _SD_TILE_FAILED
            raise RuntimeError(str(exc)) from exc
    url = cache[cache_key]
    if url == _SD_TILE_FAILED:
        raise RuntimeError(
            "Map preview unavailable (cached failure). "
            "Use **💾 Export** below — batch export has higher memory limits."
        )
    return url


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


def _render_sd_single_month_export(
    snow_img: ee.Image,
    scale: int,
    aoi,
    layer_key: str,
    year: int,
    month: int,
    period_str: str,
) -> None:
    """Smart export (Auto / Local / Drive) for the currently previewed month."""
    st.markdown(f"#### 💾 Export This Month Only — {period_str}")

    helper = DownloadHelper()
    export_pref = helper.render_smart_download_options(
        export_format="GeoTIFF",
        key=f"sd_export_mode_single_{layer_key}",
    )

    filename = f"snow_depth_{year}_{month:02d}_{scale}m"

    if st.button(
        f"📥 Export {period_str} GeoTIFF",
        type="primary",
        key=f"sd_single_gen_{layer_key}",
    ):
        with st.spinner(f"Exporting snow depth for {period_str}…"):
            result = helper.execute_smart_download(
                image=snow_img,
                filename=filename,
                region=aoi,
                scale=scale,
                export_preference=export_pref,
                crs="EPSG:4326",
            )
            if not result.get("success"):
                st.error(f"❌ Export failed: {result.get('message', 'Unknown error')}")
                st.info(
                    "Tip: for very large regions try a coarser scale, "
                    "or use **Google Drive** mode above — it handles any size."
                )


def _render_sd_map(
    image: ee.Image,
    aoi,
    params_key: str,
    layer_key: str,
    sample_scale: int = 500,
) -> bool:
    """
    Build a Folium map with GEE snow-depth tile layer, colorbar, and pixel inspector.
    Saves/restores pan+zoom so selector reruns don't reset the viewport.
    Returns True if the GEE layer rendered successfully.
    """
    # ── Centre & zoom ────────────────────────────────────────────────────
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
    init_zoom = 9 if area_km2 < 1_000 else (7 if area_km2 < 50_000 else 4)

    # ── Restore saved viewport ───────────────────────────────────────────
    _map_state_key = f"sd_map_state_{layer_key}"
    saved   = st.session_state.get(_map_state_key, {})
    ctr_lat = saved.get("lat", init_lat)
    ctr_lon = saved.get("lon", init_lon)
    zoom    = saved.get("zoom", init_zoom)

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Satellite basemap toggle
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # ── Snow depth tile layer (URL cached — never re-calls getMapId on re-render) ──
    layer_ok = False
    try:
        tile_url = _sd_get_tile_url(image, layer_key)
        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine",
            name="Snow Depth (m)",
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
        colors=_SD_VIS_PARAMS["palette"],
        vmin=_SD_VIS_PARAMS["min"],
        vmax=_SD_VIS_PARAMS["max"],
        caption="Snow Depth (m)",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # ── Render — unique key per layer_key so month-switches don't share state ──
    _folium_key = "sd_folium_" + "".join(
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
            _sample_cache = st.session_state.setdefault("sd_sample_cache", {})
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
                    f"📍 **{clat:.4f}°, {clng:.4f}°**  →  **{val:.3f} m** snow depth"
                )
            else:
                st.caption(
                    f"📍 **{clat:.4f}°, {clng:.4f}°**  →  No data at this location "
                    "(pixel outside AOI or fully masked)"
                )
        else:
            st.caption("🖱️ Click anywhere on the map to inspect the pixel value.")

    return layer_ok


def _render_sd_statistics(
    image: ee.Image,
    aoi,
    layer_key: str,
    scale: int,
) -> None:
    """Compute mean / min / max / std dev and display as metrics. Cached per layer_key."""
    stats_cache = st.session_state.setdefault("sd_stats_cache", {})
    if layer_key not in stats_cache:
        with st.spinner("Computing statistics…"):
            try:
                raw = image.reduceRegion(
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
    mean_val = raw.get("snow_depth_mean")
    min_val  = raw.get("snow_depth_min")
    max_val  = raw.get("snow_depth_max")
    std_val  = raw.get("snow_depth_stdDev")

    if any(v is not None for v in [mean_val, min_val, max_val, std_val]):
        mc1, mc2, mc3, mc4 = st.columns(4)
        if mean_val is not None: mc1.metric("Mean Depth (m)", f"{mean_val:.3f}")
        if min_val  is not None: mc2.metric("Min Depth (m)",  f"{min_val:.3f}")
        if max_val  is not None: mc3.metric("Max Depth (m)",  f"{max_val:.3f}")
        if std_val  is not None: mc4.metric("Std Dev (m)",    f"{std_val:.3f}")
    else:
        st.info(
            "No statistics available — the area may have no snow cover this month, "
            "or the scale is too coarse for the AOI."
        )


def _render_sd_preview_section(
    months_list: list,
    scale: int,
    density_params: dict,
) -> None:
    """
    Interactive preview panel — year/month toggles, Folium map with GEE tile layer,
    colorbar, pixel inspector, and statistics.
    Tile URLs are cached per (params, year, month) so switching between already-viewed
    months is instant without pressing Preview again.
    """
    st.markdown("### 🗺️ Snow Depth Preview")

    if not months_list:
        st.warning("No months available in the selected date range.")
        return

    # ── Invalidate all caches when density params or scale changes ───────
    params_key = _sd_params_key(scale, density_params)
    if st.session_state.get("sd_params_key") != params_key:
        st.session_state.pop("sd_tile_cache", None)
        st.session_state.pop("sd_sample_cache", None)
        st.session_state.pop("sd_stats_cache", None)
        st.session_state["sd_params_key"] = params_key

    # ── Year / Month selectors (the "toggle years" feature) ──────────────
    unique_years = sorted({yr for yr, _ in months_list}, reverse=True)

    c_year, c_month, c_btn = st.columns([2, 2, 1])
    with c_year:
        sel_year = st.selectbox(
            "Year:", unique_years,
            key="sd_preview_year",
        )
    with c_month:
        avail_months = sorted({mo for yr, mo in months_list if yr == sel_year})
        sel_month = st.selectbox(
            "Month:", avail_months,
            format_func=lambda m: datetime.date(2000, m, 1).strftime("%B"),
            key="sd_preview_month",
        )
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)   # align button vertically
        preview_clicked = st.button(
            "▶ Preview", type="primary", width="stretch", key="sd_preview_btn"
        )

    # ── Check if this month is already in the tile cache ─────────────────
    layer_key = f"{params_key}::{sel_year}_{sel_month:02d}"
    tile_cache = st.session_state.get("sd_tile_cache", {})
    already_cached = (
        layer_key in tile_cache and tile_cache[layer_key] != _SD_TILE_FAILED
    )

    period_str = datetime.date(sel_year, sel_month, 1).strftime("%B %Y")

    if already_cached or preview_clicked:
        st.markdown(f"#### ❄️ Snow Depth — {period_str}")
        try:
            calculator = _sd_build_calculator(scale, density_params)
            # calculate_monthly_snow_depth is lazy — no GEE network call here
            snow_img = (
                calculator
                .calculate_monthly_snow_depth(
                    st.session_state["snow_depth_geometry"], sel_year, sel_month
                )
                .rename("snow_depth")
            )

            map_ok = _render_sd_map(
                snow_img,
                st.session_state["snow_depth_geometry"],
                params_key,
                layer_key,
                sample_scale=max(scale, 500),
            )

            if map_ok:
                st.markdown("#### 📊 Statistics")
                _render_sd_statistics(
                    snow_img,
                    st.session_state["snow_depth_geometry"],
                    layer_key,
                    scale,
                )

            st.markdown("---")
            _render_sd_single_month_export(
                snow_img,
                scale,
                st.session_state["snow_depth_geometry"],
                layer_key,
                sel_year,
                sel_month,
                period_str,
            )

        except Exception as exc:
            st.error(f"❌ Preview failed: {exc}")
            st.info(
                "Tips: try a smaller AOI, coarser resolution, or a different month. "
                "Large areas with complex geometries may exceed GEE's interactive limits."
            )
    else:
        st.info(
            f"Select a **Year** and **Month** above, then click **▶ Preview** to visualize "
            f"snow depth for **{period_str}**. "
            "Once a month is previewed, switching back to it is instant (tile URL is cached)."
        )

    # Legend / palette note
    st.caption(
        "🎨 Colorbar range: 0 m (white) → 2 m (dark blue). "
        "Areas with no MODIS snow cover appear transparent / near-white."
    )


# ---------------------------------------------------------------------------
# Monthly Analysis mode
# ---------------------------------------------------------------------------
def _render_monthly_analysis(density_method, density_params):
    """Render the Monthly Analysis mode UI."""
    st.subheader("📊 Monthly Snow Depth Analysis")

    # ── Sidebar parameters ───────────────────────────────────────────────
    st.sidebar.header("📊 Monthly Analysis Parameters")

    sb_col1, sb_col2 = st.sidebar.columns(2)
    with sb_col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2020, 1, 1),
            max_value=datetime.date.today(),
            key="sd_start_date",
        )
    with sb_col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.date(2023, 12, 31),
            max_value=datetime.date.today(),
            key="sd_end_date",
        )

    scale = st.sidebar.number_input(
        "Resolution (meters)",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        key="sd_scale",
        help="Pixel resolution for preview and export. Lower = higher detail but slower.",
    )

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        st.stop()

    months_list = generate_month_list(start_date, end_date)
    st.sidebar.info(f"📅 **Total months:** {len(months_list)}")
    st.sidebar.markdown(
        f"**Range:** {start_date.strftime('%Y-%m')} → {end_date.strftime('%Y-%m')}"
    )

    # ── Persistent re-download banner ────────────────────────────────────
    _render_persistent_download()

    # ── AOI selection ────────────────────────────────────────────────────
    geometry_ready = _render_geometry_selection()
    if not geometry_ready:
        return

    st.divider()

    # ── PREVIEW SECTION ──────────────────────────────────────────────────
    _render_sd_preview_section(months_list, scale, density_params)

    st.divider()

    # ── EXPORT SECTION (collapsed by default — visualization comes first) ──
    with st.expander("💾 Export — Download All Monthly GeoTIFFs", expanded=False):
        st.markdown(
            "Generate one GeoTIFF per month. **Local / Auto** packs them into a ZIP "
            "and downloads to your browser. **Google Drive** submits each month as a "
            "separate GEE export task — handles any AOI size without memory limits."
        )

        # Smart download options (mirrors SERVES export section)
        zip_helper   = DownloadHelper()
        zip_export_pref = zip_helper.render_smart_download_options(
            export_format="GeoTIFF",
            key="sd_export_mode_zip",
        )

        proc_col1, proc_col2 = st.columns([2, 1])
        with proc_col2:
            st.metric("Months to export", len(months_list))
            area_km2 = st.session_state.get("snow_depth_area_km2")
            if area_km2:
                st.metric("Area", f"{area_km2:,.2f} km²")

        with proc_col1:
            btn_label = (
                "☁️ Submit to Google Drive"
                if zip_export_pref == "drive"
                else "🚀 Generate & Download Monthly GeoTIFFs (ZIP)"
            )
            if st.button(
                btn_label,
                type="primary",
                width="stretch",
                key="sd_generate_btn",
            ):
                calculator = _sd_build_calculator(scale, density_params)
                aoi_geom   = st.session_state["snow_depth_geometry"]

                if zip_export_pref == "drive":
                    # ── Google Drive: one GEE export task per month ──────────
                    tasks_started = 0
                    tasks_failed  = 0
                    drive_folder  = "GeoClimate_SnowDepth"
                    with st.spinner(
                        f"Submitting {len(months_list)} GEE export tasks to Drive…"
                    ):
                        for yr, mo in months_list:
                            try:
                                img = (
                                    calculator
                                    .calculate_monthly_snow_depth(aoi_geom, yr, mo)
                                    .rename("snow_depth")
                                )
                                task = ee.batch.Export.image.toDrive(
                                    image=img,
                                    description=f"snow_depth_{yr}_{mo:02d}_{scale}m",
                                    folder=drive_folder,
                                    fileNamePrefix=f"snow_depth_{yr}_{mo:02d}_{scale}m",
                                    region=aoi_geom,
                                    scale=scale,
                                    crs="EPSG:4326",
                                    maxPixels=1e13,
                                )
                                task.start()
                                tasks_started += 1
                            except Exception as exc:
                                tasks_failed += 1
                                st.warning(f"⚠️ {yr}/{mo:02d}: {exc}")

                    if tasks_started:
                        st.success(
                            f"✅ {tasks_started} export tasks submitted to Google Drive "
                            f"folder **'{drive_folder}'**."
                        )
                        if tasks_failed:
                            st.warning(f"⚠️ {tasks_failed} months failed to submit.")
                        st.info(
                            "Files appear in your Drive once each task completes "
                            "(typically a few minutes per month)."
                        )
                        st.markdown(
                            "🔗 [Monitor GEE Tasks](https://code.earthengine.google.com/tasks)"
                        )
                    else:
                        st.error("❌ No tasks could be submitted. Check GEE authentication.")

                else:
                    # ── Auto / Local: chunked download → ZIP → browser ───────
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    try:
                        with st.spinner("Processing monthly snow depth data…"):
                            zip_data = process_monthly_data(
                                calculator, aoi_geom, months_list, progress_bar, status_text
                            )

                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")

                        filename = (
                            f"snow_depth_monthly_{start_date.strftime('%Y%m')}_"
                            f"{end_date.strftime('%Y%m')}_{scale}m.zip"
                        )

                        st.session_state["sd_zip_data"]        = zip_data
                        st.session_state["sd_zip_filename"]    = filename
                        st.session_state["sd_zip_months_list"] = list(months_list)

                        st.success(
                            f"✅ Generated {len(months_list)} monthly GeoTIFF files."
                        )
                        st.download_button(
                            label="📥 Download Monthly Snow Depth Data (ZIP)",
                            data=zip_data,
                            file_name=filename,
                            mime="application/zip",
                            width="stretch",
                            key="sd_immediate_download_btn",
                        )

                        with st.expander("📋 Files included in the ZIP"):
                            for yr, mo in months_list:
                                st.text(
                                    f"snow_depth_{datetime.date(yr, mo, 1).strftime('%Y_%m')}.tif"
                                )

                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        st.info(
                            "Try reducing the area size, date range, or increasing the "
                            "resolution scale. For very large regions switch to "
                            "**Google Drive** mode above."
                        )


# ---------------------------------------------------------------------------
# Algorithm Comparison mode
# ---------------------------------------------------------------------------
def _render_algorithm_comparison(density_method, density_params):
    """Render the Algorithm Comparison mode UI."""
    st.subheader("🔍 Algorithm Comparison Mode")

    st.sidebar.header("🔍 Comparison Parameters")

    comparison_year = st.sidebar.number_input(
        "Year",
        min_value=2000,
        max_value=2024,
        value=2020,
        key="sd_comp_year",
        help="Year for the comparison data",
    )

    comparison_month = st.sidebar.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: datetime.date(2000, x, 1).strftime("%B"),
        index=0,
        key="sd_comp_month",
        help="Month for the comparison data",
    )

    comp_scale = st.sidebar.selectbox(
        "Resolution (meters)",
        options=[250, 500, 1000],
        index=1,
        key="sd_comp_scale",
        help="Resolution for comparison (lower = more accurate but slower)",
    )

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
        f"for {datetime.date(2000, comparison_month, 1).strftime('%B')}"
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
    """Render the Snow Depth Analysis & Comparison interface.

    GEE is already initialised by the main app, so this function goes
    straight to the analysis UI — no auth gate.
    """
    st.markdown("## ❄️ GLOBAL-SAM — Snow Depth Analysis & Comparison Tool")
    st.markdown(
        "Calculate monthly snow depth using MODIS and GLDAS data from "
        "Google Earth Engine, with interactive map preview and algorithm comparison."
    )
    st.divider()

    _auto_expand_sidebar()

    # ── Sidebar: density settings (shared by both modes) ──────────────────
    st.sidebar.header("🌍 Snow Depth Parameters")
    st.sidebar.subheader("🏔️ Snow Density Settings")

    with st.sidebar:
        density_method, density_params = create_density_settings_ui("snow_depth")

    st.sidebar.markdown("---")

    # ── Mode selector ──────────────────────────────────────────────────────
    mode = st.radio(
        "Choose mode:",
        ["📊 Monthly Analysis", "🔍 Algorithm Comparison"],
        horizontal=True,
        key="sd_mode",
        help=(
            "Monthly Analysis: preview snow depth maps by month/year, then export GeoTIFFs. "
            "Algorithm Comparison: validate against a reference TIFF."
        ),
    )

    if mode == "📊 Monthly Analysis":
        _render_monthly_analysis(density_method, density_params)
    else:
        _render_algorithm_comparison(density_method, density_params)

    display_comparison_results()

    # ── About ──────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool provides **monthly snow depth analysis** and **algorithm comparison** capabilities.

        ## 📊 Monthly Analysis Mode

        **New: GEE Tile Preview**

        Before downloading, visualize snow depth directly in the browser via live GEE tile rendering.
        Select any year/month within your date range and click **▶ Preview**:
        - Interactive Folium map with OpenStreetMap and Google Satellite basemaps
        - Colorbar: white (0 m) → dark blue (≥ 2 m)
        - Pixel inspector — click anywhere to read the exact depth value
        - Pan/zoom is preserved when switching months
        - Statistics (mean, min, max, std dev) computed for your AOI

        **Data Sources:**
        - **MODIS Snow Cover** (MOD10A1) — Normalized Difference Snow Index
        - **GLDAS Snow Water Equivalent** — NASA Global Land Data Assimilation System

        **Algorithm (per month):**
        1. Calculate Fractional Snow Cover (FSC) from MODIS NDSI
        2. Estimate snow density based on FSC and selected density method
        3. Downscale GLDAS SWE using FSC
        4. Snow Depth = Downscaled SWE ÷ Snow Density

        **Output:** one GeoTIFF per month (`snow_depth_YYYY_MM.tif`), bundled in a ZIP.

        ## 🔍 Comparison Mode

        **Features:**
        - Upload a reference GeoTIFF for validation
        - Automatic spatial alignment and resampling
        - Metrics: R², RMSE, MAE, bias, correlation, coverage
        - Interactive scatter plots, heatmaps, histograms, and difference map
        - Downloadable JSON results

        ## 🏔️ Snow Density Methods

        | Method | Description |
        |---|---|
        | Original | Seasonal look-up table (180–420 kg/m³) |
        | Enhanced Method 1 | Inverse-fit γ per month using literature densities |
        | Enhanced Method 2 | Sinusoidal γ fit for smooth seasonal variation |

        **Notes:** Large AOIs are automatically chunked during export. Preview uses GEE
        tile rendering which is much faster (seconds vs. minutes) and works well for
        MODIS-resolution (500 m) data.
        """)
