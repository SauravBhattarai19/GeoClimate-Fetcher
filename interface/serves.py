"""
SERVES Interface Module
Landing page for SERVES v2.0 – Enhanced Edition

Full name: Soil-moisture Estimation of Root zone through Vegetation index-based
           Evapotranspiration fraction and Soil properties

Developed by Saurav Bhattarai, Dr. Rocky Talchabhadel, and Dr. Nawa Raj Pradhan
Jackson State University, Department of Civil and Environmental Engineering

Original SERVES model by Dr. Nawa Raj Pradhan
U.S. Army Engineer Research and Development Center (ERDC),
Coastal and Hydraulics Laboratory (CHL)

The full interactive application runs as a Google Earth Engine App.
This page describes the model and launches it externally.
"""

import streamlit as st

SERVES_URL = "https://ee-sauravbhattarai1999.projects.earthengine.app/view/serves"


def render_serves():
    """Render the SERVES module landing and launch page."""

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("## 🛰️ SERVES v2.0 — Enhanced Edition")
    st.markdown(
        "**Soil-moisture Estimation of Root zone through Vegetation index-based "
        "Evapotranspiration fraction and Soil properties** — developed at "
        "**Jackson State University** in collaboration with the "
        "U.S. Army Engineer Research and Development Center (ERDC)."
    )
    st.divider()

    # ── Launch banner ─────────────────────────────────────────────────────
    st.info(
        "✅ **SERVES v2.0 is live as a Google Earth Engine App!**  \n"
        "Click the button below to open the fully interactive application "
        "hosted on Google Earth Engine. No additional login is required beyond "
        "your existing Earth Engine access.",
        icon="🌍",
    )

    st.link_button(
        "🚀 Open SERVES v2.0 in Google Earth Engine",
        url=SERVES_URL,
        type="primary",
    )

    st.divider()

    # ── About SERVES ──────────────────────────────────────────────────────
    st.subheader("📖 About SERVES v2.0")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        SERVES is a satellite-driven soil moisture estimation framework that uses
        the NDVI-based Evapotranspiration fraction approach to retrieve root-zone
        soil moisture continuously across regional to continental scales using
        Google Earth Engine's cloud computing infrastructure.

        **Core algorithm:**
        ET fraction is derived from NDVI using the linear relationship
        *ET_fraction = 1.33 × NDVI − 0.049*, then scaled between the wilting
        point and field capacity to estimate volumetric soil moisture (cm³/cm³).

        **Version 2.0 enhancements:**
        - Default soil parameters option (FC = 0.35, WP = 0.09)
        - Custom soil parameter inputs
        - Water bodies assigned to field capacity
        - Negative NDVI pixels assigned to field capacity
        - Regional climatology mode (global & continental scale)
        - Multi-year min / max / median composites
        - Enhanced export for large areas (Google Drive + direct download)
        """)

    with col_right:
        st.markdown("""
        **Developed by:**
        Saurav Bhattarai, Dr. Rocky Talchabhadel,
        and Dr. Nawa Raj Pradhan
        Department of Civil and Environmental Engineering
        Jackson State University

        ---

        **Original SERVES model:**
        Dr. Nawa Raj Pradhan
        Research Hydraulic Engineer
        Coastal and Hydraulics Laboratory (CHL)
        U.S. Army Engineer Research and Development Center (ERDC)

        ---

        **Analysis modes available:**
        - Single Date
        - Monthly Average
        - Seasonal Composite (multi-year)
        - Annual Average (multi-year)
        - Custom Date Range
        - Time Series (monthly / bi-weekly / weekly)
        - 🌍 Regional Climatology (global & continental)
        """)

    st.divider()

    # ── Satellite & Auxiliary Data Sources ────────────────────────────────
    st.subheader("🛰️ Satellite & Auxiliary Data Sources")

    ds_col1, ds_col2, ds_col3 = st.columns(3)

    with ds_col1:
        st.markdown("""
        **Optical / Multispectral (NDVI)**
        - Landsat 8 / 9 C02 T1 L2 (30 m)
        - Sentinel-2 SR Harmonized (10 m)
        - MODIS MOD13A2 (1 km / 16-day)
        """)

    with ds_col2:
        st.markdown("""
        **Soil Properties**
        - ISRIC SoilGrids250m v2.0
          (field capacity & wilting point,
          spatially variable, 6 depth layers
          from 0 cm to 200 cm)
        - Uniform defaults
          (FC = 0.35, WP = 0.09)
        """)

    with ds_col3:
        st.markdown("""
        **Quality & Auxiliary Masks**
        - JRC Global Surface Water v1.4
          (water body detection)
        - ESA WorldCover v200 (2021)
          (vegetation classification)
        """)

    st.divider()

    # ── Relationship to this platform ─────────────────────────────────────
    st.subheader("🔗 Relationship to GeoClimate Intelligence Platform")
    st.markdown("""
    SERVES and the GeoClimate Intelligence Platform share a common research
    lineage — both developed under the ERDC / Jackson State University
    collaboration led by Dr. Nawa Raj Pradhan and Dr. Rocky Talchabhadel.

    The GeoClimate platform provides access to **33+ Earth observation datasets**
    catalogued via the **STAC API** (SpatioTemporal Asset Catalog), giving users
    flexible access to all datasets available in Google Earth Engine with custom
    geometry, date range, and multi-format export.

    | GeoClimate Platform | SERVES v2.0 GEE App |
    |---|---|
    | 33+ GEE datasets via STAC API | Pre-configured NDVI → soil moisture pipeline |
    | Custom AOI & date range | 7 analysis modes (single to regional climatology) |
    | Multi-format export (GeoTIFF, CSV) | Multi-year min / max / median composites |
    | Interactive dataset catalog | Spatially variable or uniform soil parameters |
    | Snow Depth & Hydrology modules | Regional climatology (global & continental) |
    """)

    st.divider()

    # ── Launch again at the bottom ────────────────────────────────────────
    st.markdown("### Ready to estimate soil moisture with SERVES?")
    st.link_button(
        "🚀 Launch SERVES v2.0 GEE Application",
        url=SERVES_URL,
        type="primary",
    )

    st.caption(
        "SERVES v2.0 is hosted on Google Earth Engine. Clicking the button opens "
        "the application in a new browser tab. For technical inquiries, contact "
        "Saurav Bhattarai (saurav.bhattarai.1999@gmail.com) or the ERDC "
        "Coastal and Hydraulics Laboratory."
    )
