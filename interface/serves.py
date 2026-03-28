"""
SERVES Interface Module
Landing page for the SERVES (Satellite-based Environmental and Remote-sensing
Visualization & Estimation System) model developed by Dr. Nawa Raj Pradhan
at the U.S. Army Engineer Research and Development Center (ERDC).

The full interactive application runs as a Google Earth Engine App.
This page describes the model and launches it externally.
"""

import streamlit as st

SERVES_URL = "https://ee-sauravbhattarai1999.projects.earthengine.app/view/serves"


def render_serves():
    """Render the SERVES module landing and launch page."""

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("## 🛰️ SERVES")
    st.markdown(
        "**Satellite-based Environmental and Remote-sensing Visualization & "
        "Estimation System** — developed by Dr. Nawa Raj Pradhan, "
        "U.S. Army Engineer Research and Development Center (ERDC), CRREL."
    )
    st.divider()

    # ── Launch banner ─────────────────────────────────────────────────────
    st.info(
        "✅ **The Google Earth Engine App version of SERVES has been created!**  \n"
        "Click the button below to open the fully interactive SERVES application "
        "hosted on Google Earth Engine. No additional login is required beyond "
        "your existing Earth Engine access.",
        icon="🌍",
    )

    st.link_button(
        "🚀 Open SERVES in Google Earth Engine",
        url=SERVES_URL,
        type="primary",
    )

    st.divider()

    # ── About SERVES ──────────────────────────────────────────────────────
    st.subheader("📖 About SERVES")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        SERVES is a satellite-driven geospatial modeling framework designed to
        estimate and visualize key land-surface and hydrological variables at
        regional to continental scales using Google Earth Engine's cloud
        computing infrastructure.

        The model leverages multi-source satellite imagery — including optical,
        microwave, and thermal sensors — to derive spatially continuous fields
        of surface conditions that are critical for water-resource management,
        cold-region hydrology, and military terrain analysis.

        **Core capabilities:**
        - Satellite-based soil moisture retrieval using vegetation indices
          and microwave backscatter
        - Snow cover and Snow Water Equivalent (SWE) estimation
        - Land-surface temperature and evapotranspiration mapping
        - Integration of outputs with physics-based distributed watershed
          models (e.g., GSSHA)
        - Near-real-time monitoring over data-sparse or remote regions
        """)

    with col_right:
        st.markdown("""
        **Developed by:**
        Dr. Nawa Raj Pradhan
        Research Hydraulic Engineer
        Cold Regions Research and Engineering Laboratory (CRREL)
        U.S. Army Engineer Research and Development Center (ERDC)
        Vicksburg, MS

        ---

        **Key applications:**
        - Initialization of hydrological and land-surface models
        - Flood forecasting and early-warning support
        - Cold-region terrain assessment for military operations
        - Drought monitoring and agricultural water management
        - Validation of numerical weather prediction (NWP) soil-moisture fields

        ---

        **Platform:** Google Earth Engine
        **Access:** Public GEE App (no code required)
        **Scale:** Regional to global
        """)

    st.divider()

    # ── Data sources ──────────────────────────────────────────────────────
    st.subheader("🛰️ Satellite Data Sources")

    ds_col1, ds_col2, ds_col3 = st.columns(3)

    with ds_col1:
        st.markdown("""
        **Optical / Multispectral**
        - MODIS (Terra & Aqua)
        - Landsat 8 / 9
        - Sentinel-2 MSI
        """)

    with ds_col2:
        st.markdown("""
        **Passive Microwave**
        - AMSR2 (GCOM-W1)
        - SMAP L-band
        - SSM/I – SSMI/S
        """)

    with ds_col3:
        st.markdown("""
        **Auxiliary / Reanalysis**
        - GLDAS land-surface model
        - SRTM / NASADEM terrain
        - ERA5 / MERRA-2 climate
        """)

    st.divider()

    # ── Relationship to this platform ─────────────────────────────────────
    st.subheader("🔗 Relationship to GeoClimate Intelligence Platform")
    st.markdown("""
    SERVES and the GeoClimate Intelligence Platform share a common research
    lineage — both are developed under the ERDC / Jackson State University
    collaboration led by Dr. Nawa Raj Pradhan and Dr. Rocky Talchabhadel.

    While the GeoClimate platform provides **general-purpose** access to 50+
    climate and Earth-observation datasets with custom geometry, date range,
    and export controls, SERVES provides a **specialized, pre-configured**
    Google Earth Engine application focused specifically on the satellite
    retrieval algorithms developed and validated by the ERDC team.

    The two tools complement each other:
    | GeoClimate Platform | SERVES GEE App |
    |---|---|
    | Flexible dataset exploration | Pre-built ERDC retrieval algorithms |
    | Custom AOI & date range | Optimized for ERDC use-cases |
    | Multi-format export (GeoTIFF, CSV) | Interactive web visualization |
    | Snow Depth Analyzer module | Integrated SWE & soil moisture outputs |
    """)

    st.divider()

    # ── Launch again at the bottom ────────────────────────────────────────
    st.markdown("### Ready to explore SERVES?")
    st.link_button(
        "🚀 Launch SERVES GEE Application",
        url=SERVES_URL,
        type="primary",
    )

    st.caption(
        "SERVES is hosted on Google Earth Engine. Clicking the button opens "
        "the application in a new browser tab. For technical inquiries, "
        "contact the ERDC Cold Regions Research and Engineering Laboratory."
    )
