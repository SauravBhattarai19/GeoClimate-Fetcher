"""
Snow Depth Interface Module
Handles the complete interface for the Snow Depth Analysis & Comparison tool.
Authentication is handled by the main app — no separate auth needed here.

Key design decisions:
- Uses shared GeometrySelectionWidget (same as other modules)
- ZIP results persisted in session state so re-download survives navigation
- Sidebar auto-expands on first visit (all parameters live there)
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import datetime

# ---------------------------------------------------------------------------
# Make Snow_Depth_ERDC importable by adding its directory to sys.path
# ---------------------------------------------------------------------------
_snow_depth_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Snow_Depth_ERDC")
)
if _snow_depth_dir not in sys.path:
    sys.path.insert(0, _snow_depth_dir)

# Import computation + UI helpers from the standalone snow depth module.
# We intentionally do NOT import or call `main()` / `initialize_earth_engine()`
# because GEE authentication is already handled by the main GeoClimate app.
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
)

# Shared geometry selection widget (same as Data Explorer, Hydrology, etc.)
from geoclimate_fetcher.core import GeometrySelectionWidget


# ---------------------------------------------------------------------------
# Sidebar auto-expand helper
# ---------------------------------------------------------------------------
def _auto_expand_sidebar():
    """Open the sidebar the first time this module is visited in a session.

    Uses a JS snippet (via components.html iframe) to click the Streamlit
    sidebar toggle button.  The sd_sidebar_opened flag prevents re-triggering
    on every rerun or after the user manually closes it.
    """
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
                    // Give Streamlit a moment to finish rendering before we click
                    setTimeout(function () { tryExpand(6); }, 400);
                })();
            </script>
            """,
            height=0,
        )
        st.session_state.sd_sidebar_opened = True


# ---------------------------------------------------------------------------
# Geometry selection (mirrors hydrology_analyzer pattern)
# ---------------------------------------------------------------------------
def _render_geometry_selection():
    """Render AOI selection using the shared GeometrySelectionWidget.

    Returns True when a geometry is ready for analysis.
    """
    if st.session_state.get("snow_depth_geometry") is not None:
        # Geometry already chosen — show summary + reset option
        st.success("✅ Area of Interest is ready for analysis")
        try:
            area_km2 = (
                st.session_state.snow_depth_geometry.area().divide(1e6).getInfo()
            )
            st.metric("Selected Area", f"{area_km2:.2f} km²")
        except Exception:
            st.info("Geometry ready for analysis")

        if st.button("🗑️ Reset Area", key="sd_reset_geometry"):
            for key in [
                "snow_depth_geometry",
                "snow_depth_geometry_complete",
                "snow_depth_area_km2",
                "snow_depth_method_selection",
            ]:
                st.session_state.pop(key, None)
            # Clear any cached ZIP when AOI changes
            for key in ["sd_zip_data", "sd_zip_filename", "sd_zip_months_list"]:
                st.session_state.pop(key, None)
            st.rerun()

        return True

    # No geometry yet — render the full selection widget
    def on_geometry_selected(geometry):
        st.session_state["snow_depth_geometry"] = geometry
        try:
            area_km2 = geometry.area().divide(1e6).getInfo()
            st.session_state["snow_depth_area_km2"] = round(area_km2, 2)
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
    """Show a re-download banner whenever a ZIP result exists in session state.

    This survives navigation to other modules and back.
    """
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


# ---------------------------------------------------------------------------
# Monthly Analysis mode
# ---------------------------------------------------------------------------
def _render_monthly_analysis(density_method, density_params):
    """Render the Monthly Analysis mode UI."""
    st.subheader("📊 Monthly Snow Depth Analysis")

    # Sidebar: date range & resolution
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
        help="Pixel resolution in meters. Lower values = higher resolution but longer processing time.",
    )

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        st.stop()

    months_list = generate_month_list(start_date, end_date)
    st.sidebar.info(f"📅 **Total months to process:** {len(months_list)}")
    st.sidebar.markdown(
        f"**Date range:** {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}"
    )

    # Persistent re-download banner (survives navigation away and back)
    _render_persistent_download()

    # AOI selection via shared geometry widget
    geometry_ready = _render_geometry_selection()

    if not geometry_ready:
        return  # Waiting for user to select an area

    # Geometry is confirmed — show processing panel
    st.markdown("---")
    st.subheader("⚙️ Processing")

    proc_col1, proc_col2 = st.columns([2, 1])
    with proc_col2:
        st.metric("Months to process", len(months_list))
        area_km2 = st.session_state.get("snow_depth_area_km2")
        if area_km2:
            st.metric("Area", f"{area_km2:.2f} km²")

    with proc_col1:
        if st.button(
            "🚀 Generate Monthly Snow Depth",
            type="primary",
            width="stretch",
            key="sd_generate_btn",
        ):
            aoi_geom = st.session_state["snow_depth_geometry"]

            if density_params["method"] == "original":
                calculator = SnowDepthCalculator(
                    scale=scale,
                    use_seasonal_density=density_params["use_seasonal_density"],
                    custom_density_value=density_params["custom_density_value"],
                    custom_monthly_densities=density_params["custom_monthly_densities"],
                    density_method="original",
                )
            else:
                calculator = SnowDepthCalculator(
                    scale=scale,
                    density_method=density_params["method"],
                    rho_min=density_params["rho_min"],
                    rho_max=density_params["rho_max"],
                    gamma_params=density_params["gamma_params"],
                )

            progress_bar = st.progress(0)
            status_text = st.empty()

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

                # ── Persist in session state so re-download survives navigation ──
                st.session_state["sd_zip_data"] = zip_data
                st.session_state["sd_zip_filename"] = filename
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
                    "Try reducing the area size, date range, or increasing the resolution scale."
                )


# ---------------------------------------------------------------------------
# Algorithm Comparison mode
# ---------------------------------------------------------------------------
def _render_algorithm_comparison(density_method, density_params):
    """Render the Algorithm Comparison mode UI."""
    st.subheader("🔍 Algorithm Comparison Mode")

    # Sidebar: comparison parameters
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
        return  # extract_bounds_from_tiff already shows an error

    st.success("✅ Reference TIFF file uploaded successfully!")
    st.info(
        f"**Bounds:** {bounds[0]:.4f}, {bounds[1]:.4f}, "
        f"{bounds[2]:.4f}, {bounds[3]:.4f}"
    )

    # Show current density info for the selected month
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
        beta = density_params["gamma_params"].get("beta", 0.0)
        phi = density_params["gamma_params"].get("phi", 1.0)
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
        "Google Earth Engine, with algorithm comparison capabilities."
    )
    st.divider()

    # Open sidebar on first visit so users immediately see the controls
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
            "Monthly Analysis: generate time-series GeoTIFFs for a date range. "
            "Algorithm Comparison: validate against a reference TIFF."
        ),
    )

    if mode == "📊 Monthly Analysis":
        _render_monthly_analysis(density_method, density_params)
    else:
        _render_algorithm_comparison(density_method, density_params)

    # Comparison results are always displayed when present in session state
    display_comparison_results()

    # ── About ──────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool provides **monthly snow depth analysis** and **algorithm comparison** capabilities.

        ## 📊 Monthly Analysis Mode

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

        **Notes:** Large AOIs are automatically chunked to avoid GEE memory limits.
        Processing time scales with area, date range, and resolution.
        """)
