"""
Raster Calculator Interface Module
Allows users to perform band math on Earth Engine datasets
with STAC-based discovery, expression building, and export.
"""

import streamlit as st
import ee
import logging
from typing import Dict, List
from datetime import datetime, timedelta

from geoclimate_fetcher.core import (
    GeometryHandler,
    GeometrySelectionWidget,
)
from geoclimate_fetcher.core.stac_client import STACClient
from geoclimate_fetcher.core.band_math import (
    INDEX_PRESETS,
    extract_band_references,
    validate_expression,
    apply_expression,
    apply_expression_to_collection,
    aggregate_collection,
)
from app_components.download_component import DownloadHelper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common dataset shortcuts (most-used image collections with optical bands)
# ---------------------------------------------------------------------------
POPULAR_DATASETS = {
    "Landsat 9 SR": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "bands": {
            "SR_B1": "Coastal Aerosol",
            "SR_B2": "Blue",
            "SR_B3": "Green",
            "SR_B4": "Red",
            "SR_B5": "NIR",
            "SR_B6": "SWIR 1",
            "SR_B7": "SWIR 2",
            "ST_B10": "Thermal",
        },
        "scale": 30,
    },
    "Landsat 8 SR": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "bands": {
            "SR_B1": "Coastal Aerosol",
            "SR_B2": "Blue",
            "SR_B3": "Green",
            "SR_B4": "Red",
            "SR_B5": "NIR",
            "SR_B6": "SWIR 1",
            "SR_B7": "SWIR 2",
            "ST_B10": "Thermal",
        },
        "scale": 30,
    },
    "Sentinel-2 SR": {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": {
            "B1": "Coastal Aerosol",
            "B2": "Blue",
            "B3": "Green",
            "B4": "Red",
            "B5": "Red Edge 1",
            "B6": "Red Edge 2",
            "B7": "Red Edge 3",
            "B8": "NIR",
            "B8A": "Red Edge 4",
            "B9": "Water Vapour",
            "B11": "SWIR 1",
            "B12": "SWIR 2",
        },
        "scale": 10,
    },
    "Sentinel-2 TOA": {
        "id": "COPERNICUS/S2_HARMONIZED",
        "bands": {
            "B1": "Coastal Aerosol",
            "B2": "Blue",
            "B3": "Green",
            "B4": "Red",
            "B5": "Red Edge 1",
            "B6": "Red Edge 2",
            "B7": "Red Edge 3",
            "B8": "NIR",
            "B8A": "Red Edge 4",
            "B9": "Water Vapour",
            "B10": "Cirrus",
            "B11": "SWIR 1",
            "B12": "SWIR 2",
        },
        "scale": 10,
    },
    "MODIS Terra SR (Daily)": {
        "id": "MODIS/061/MOD09GA",
        "bands": {
            "sur_refl_b01": "Red (620-670nm)",
            "sur_refl_b02": "NIR (841-876nm)",
            "sur_refl_b03": "Blue (459-479nm)",
            "sur_refl_b04": "Green (545-565nm)",
            "sur_refl_b05": "SWIR 1 (1230-1250nm)",
            "sur_refl_b06": "SWIR 2 (1628-1652nm)",
            "sur_refl_b07": "SWIR 3 (2105-2155nm)",
        },
        "scale": 500,
    },
    "MODIS NDVI (16-day)": {
        "id": "MODIS/061/MOD13Q1",
        "bands": {
            "NDVI": "NDVI",
            "EVI": "EVI",
            "sur_refl_b01": "Red",
            "sur_refl_b02": "NIR",
            "sur_refl_b03": "Blue",
            "sur_refl_b07": "MIR",
        },
        "scale": 250,
    },
    "Custom (Enter EE ID)": {
        "id": "",
        "bands": {},
        "scale": 30,
    },
}


def _init_session_state():
    """Initialize session state variables for the raster calculator."""
    defaults = {
        'rc_geometry_complete': False,
        'rc_geometry': None,
        'rc_dataset_id': None,
        'rc_bands_loaded': False,
        'rc_band_list': [],
        'rc_expression': '',
        'rc_band_mapping': {},
        'rc_result_image': None,
        'rc_download_complete': False,
        'rc_download_results': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_area_selection():
    """Step 1: Area of interest selection via interactive map."""
    st.markdown("### Step 1: Select Area of Interest")

    def on_geometry_selected(geometry):
        st.session_state.rc_geometry_complete = True
        st.session_state.rc_geometry = geometry
        try:
            st.success("✅ Area of interest selected.")
        except Exception:
            pass

    geometry_widget = GeometrySelectionWidget(
        session_prefix="rc_",
        title="🗺️ Area of Interest",
    )

    result = geometry_widget.render_complete_interface(on_geometry_selected=on_geometry_selected)
    if result:
        existing_geom = st.session_state.get('rc_geometry')
        if existing_geom and not st.session_state.get('rc_geometry_complete', False):
            on_geometry_selected(existing_geom)
        st.rerun()

    return st.session_state.get('rc_geometry_complete', False)


def _render_dataset_selection():
    """Step 2: Pick a dataset (popular shortcut or custom EE ID)."""
    st.markdown("### Step 2: Select Dataset")

    dataset_choice = st.selectbox(
        "Choose a dataset",
        list(POPULAR_DATASETS.keys()),
        help="Pick a commonly used dataset or enter a custom Earth Engine collection ID."
    )

    ds_info = POPULAR_DATASETS[dataset_choice]

    if dataset_choice == "Custom (Enter EE ID)":
        custom_id = st.text_input(
            "Earth Engine Collection ID",
            placeholder="e.g. LANDSAT/LC08/C02/T1_L2",
            help="Full Earth Engine asset path for an ImageCollection."
        )
        if not custom_id:
            st.info("Enter a valid Earth Engine ImageCollection ID to continue.")
            return None, None, None
        ds_id = custom_id
        scale = st.number_input("Scale (meters)", value=30, min_value=1, max_value=10000)
    else:
        ds_id = ds_info["id"]
        scale = ds_info["scale"]
        st.caption(f"**Collection:** `{ds_id}` | **Resolution:** {scale}m")

    return ds_id, ds_info, scale


def _load_bands(ds_id: str, ds_info: dict, geometry: ee.Geometry):
    """Load band names from the dataset, using preset info or EE metadata."""
    # If we have preset bands and it's not custom, use those
    if ds_info and ds_info.get("bands"):
        band_names = list(ds_info["bands"].keys())
        band_descriptions = ds_info["bands"]
        return band_names, band_descriptions

    # Otherwise query EE for band info
    with st.spinner("Detecting bands from Earth Engine..."):
        try:
            col = ee.ImageCollection(ds_id).filterBounds(geometry).limit(1)
            first = col.first()
            band_names = first.bandNames().getInfo()
            band_descriptions = {b: b for b in band_names}
            return band_names, band_descriptions
        except Exception as e:
            st.error(f"Could not load bands: {e}")
            return [], {}


def _render_date_selection():
    """Step 3: Date range and aggregation."""
    st.markdown("### Step 3: Time Period & Aggregation")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            value=datetime.now() - timedelta(days=365),
            help="Start of the analysis period."
        )
    with col2:
        end_date = st.date_input(
            "End date",
            value=datetime.now() - timedelta(days=1),
            help="End of the analysis period."
        )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return None, None, None

    aggregation = st.selectbox(
        "Temporal aggregation",
        ["median", "mean", "min", "max", "sum"],
        help=(
            "How to combine multiple images in the time range into one. "
            "'median' is most common for optical data (removes clouds)."
        )
    )

    return str(start_date), str(end_date), aggregation


def _render_expression_builder(band_names: List, band_descriptions: dict):
    """Step 4: Build band math expression with mapping UI."""
    st.markdown("### Step 4: Band Math Expression")

    # Show available bands
    with st.expander("Available bands in this dataset", expanded=False):
        for bname in band_names:
            desc = band_descriptions.get(bname, "")
            if desc and desc != bname:
                st.markdown(f"- **`{bname}`** — {desc}")
            else:
                st.markdown(f"- **`{bname}`**")

    # Preset selector
    st.markdown("**Quick presets** (or write your own expression below)")
    preset_cols = st.columns(4)
    for i, (name, info) in enumerate(INDEX_PRESETS.items()):
        with preset_cols[i % 4]:
            if st.button(name, key=f"preset_{name}",
                         help=info["description"]):
                st.session_state.rc_expression = info["expression"]
                # Auto-populate band mapping from preset
                st.session_state.rc_band_mapping = {}
                st.rerun()

    # Expression input
    expression = st.text_input(
        "Expression",
        value=st.session_state.rc_expression,
        placeholder="e.g. (NIR - RED) / (NIR + RED)",
        help=(
            "Use band variable names and standard math operators: + - * / ** ( ). "
            "Map each variable to a dataset band below."
        )
    )
    st.session_state.rc_expression = expression

    if not expression:
        st.info("Enter an expression or select a preset above.")
        return None, None

    # Extract variables from expression
    variables = extract_band_references(expression)

    if not variables:
        st.warning("No band variables detected in expression.")
        return None, None

    # Band mapping UI
    st.markdown("**Map variables to dataset bands:**")
    band_mapping = {}
    mapping_cols = st.columns(min(len(variables), 4))

    for i, var in enumerate(variables):
        with mapping_cols[i % len(mapping_cols)]:
            # Try to auto-match variable name to a band
            default_idx = 0
            # Check if the variable name exactly matches a band
            if var in band_names:
                default_idx = band_names.index(var)
            else:
                # Try common aliases
                alias_map = _get_alias_map(band_names, band_descriptions)
                if var.upper() in alias_map:
                    matched = alias_map[var.upper()]
                    if matched in band_names:
                        default_idx = band_names.index(matched)

            selected = st.selectbox(
                f"`{var}` =",
                band_names,
                index=default_idx,
                key=f"band_map_{var}"
            )
            band_mapping[var] = selected

    return expression, band_mapping


def _get_alias_map(band_names: list, band_descriptions: dict) -> dict:
    """
    Build a mapping of common aliases (RED, NIR, etc.) to actual band names.
    Uses band descriptions to infer which band matches which alias.
    """
    alias_map = {}
    desc_lower = {b: band_descriptions.get(b, "").lower() for b in band_names}

    alias_keywords = {
        "RED": ["red"],
        "GREEN": ["green"],
        "BLUE": ["blue"],
        "NIR": ["nir", "near-infrared", "near infrared"],
        "SWIR1": ["swir 1", "swir1", "short-wave infrared 1"],
        "SWIR2": ["swir 2", "swir2", "short-wave infrared 2"],
    }

    for alias, keywords in alias_keywords.items():
        for bname in band_names:
            desc = desc_lower[bname]
            # Exact match on description
            if any(kw == desc for kw in keywords):
                alias_map[alias] = bname
                break
            # Partial match
            if any(kw in desc for kw in keywords):
                alias_map[alias] = bname
                break

    return alias_map


def _render_compute_and_export(
    ds_id: str,
    expression: str,
    band_mapping: dict,
    start_date: str,
    end_date: str,
    aggregation: str,
    geometry: ee.Geometry,
    scale: int,
):
    """Step 5: Compute the expression and export results."""
    st.markdown("### Step 5: Compute & Export")

    # Validate expression
    available_bands = list(set(band_mapping.values()))
    is_valid, err_msg = validate_expression(expression, available_bands, band_mapping)

    if not is_valid:
        st.error(f"Expression error: {err_msg}")
        return

    st.success("Expression is valid.")

    # Show summary
    with st.expander("Computation summary", expanded=True):
        st.markdown(f"- **Dataset:** `{ds_id}`")
        st.markdown(f"- **Period:** {start_date} to {end_date}")
        st.markdown(f"- **Aggregation:** {aggregation}")
        st.markdown(f"- **Expression:** `{expression}`")
        st.markdown(f"- **Scale:** {scale}m")
        mapping_str = ", ".join(f"{v}={b}" for v, b in band_mapping.items())
        st.markdown(f"- **Band mapping:** {mapping_str}")

    output_name = st.text_input("Output band name", value="result",
                                help="Name for the computed band in the output file.")

    col_compute, col_preview = st.columns(2)

    with col_compute:
        compute_clicked = st.button("Compute", type="primary",
                                    use_container_width=True)

    # Compute
    if compute_clicked or st.session_state.rc_result_image is not None:
        if compute_clicked:
            with st.spinner("Computing on Earth Engine..."):
                try:
                    result_image = _run_computation(
                        ds_id, expression, band_mapping,
                        start_date, end_date, aggregation,
                        geometry, output_name
                    )
                    st.session_state.rc_result_image = result_image
                    st.success("Computation complete!")
                except Exception as e:
                    st.error(f"Computation failed: {e}")
                    logger.error("Raster calculator computation failed: %s", e)
                    return

        result_image = st.session_state.rc_result_image

        st.markdown("---")

        # Preview — behind a button to avoid re-fetching on every widget interaction
        with st.expander("🗺️ Preview on Map (Optional)", expanded=True):
            if st.button("🗺️ Show Map Preview", type="secondary",
                         key="rc_show_preview_btn", use_container_width=True):
                with st.spinner("Generating map preview..."):
                    _render_preview(result_image, geometry, output_name)

        st.markdown("---")

        # Export
        st.markdown("### 💾 Download / Export")
        _render_export(result_image, geometry, scale, output_name)


def _run_computation(ds_id, expression, band_mapping,
                     start_date, end_date, aggregation,
                     geometry, output_name):
    """Run the band math computation on Earth Engine."""
    # Load collection
    collection = (
        ee.ImageCollection(ds_id)
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .select(list(set(band_mapping.values())))
    )

    # Check image count
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(
            f"No images found for {ds_id} between {start_date} and {end_date} "
            "in the selected area."
        )

    st.info(f"Found {count} images. Applying expression and {aggregation} aggregation...")

    # Apply expression to each image, then aggregate
    computed = apply_expression_to_collection(
        collection, expression, band_mapping, output_name
    )
    result = aggregate_collection(computed, aggregation)

    # Clip to geometry
    result = result.clip(geometry)

    return result


def _render_preview(result_image: ee.Image, geometry: ee.Geometry, band_name: str):
    """Render an interactive Folium map preview of the result, matching GeoData Explorer style."""
    import folium
    import streamlit.components.v1 as components

    palette = ['#d73027', '#f46d43', '#fdae61', '#fee08b',
               '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']

    # Compute 5th–95th percentile for a sensible colour stretch
    vmin, vmax = -1.0, 1.0
    try:
        stats = result_image.reduceRegion(
            reducer=ee.Reducer.percentile([5, 95]),
            geometry=geometry,
            scale=200,
            maxPixels=1e8,
            bestEffort=True,
        ).getInfo()
        p5 = stats.get(f"{band_name}_p5")
        p95 = stats.get(f"{band_name}_p95")
        if p5 is not None and p95 is not None and p5 != p95:
            vmin, vmax = round(p5, 4), round(p95, 4)
    except Exception:
        pass

    try:
        bounds = geometry.bounds().getInfo()['coordinates'][0]
        lons = [c[0] for c in bounds]
        lats = [c[1] for c in bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        thumb_url = result_image.select(band_name).getThumbURL({
            'min': vmin,
            'max': vmax,
            'palette': palette,
            'region': geometry.bounds(),
            'dimensions': 768,
            'format': 'png',
        })

        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
        folium.raster_layers.ImageOverlay(
            image=thumb_url,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.85,
            name=band_name,
            interactive=False,
            cross_origin=False,
        ).add_to(m)
        folium.LayerControl().add_to(m)

        components.html(m.get_root().render(), height=480)
        st.caption(f"**{band_name}** — colour stretch: {vmin} to {vmax} (5th–95th percentile)")

    except Exception as e:
        st.warning(f"Could not generate map preview: {e}")


def _render_export(result_image: ee.Image, geometry: ee.Geometry,
                   scale: int, output_name: str):
    """Render export options.

    Results are persisted in session state so the download button stays available
    across reruns. Drive export is hidden when using Quick Access (shared service
    account) because exports would land in the platform's Drive, not the user's.
    """
    from geoclimate_fetcher.core import GEEExporter
    from datetime import datetime as _dt

    # ── Persistent results panel ────────────────────────────────────────────
    if st.session_state.get('rc_download_complete') and st.session_state.get('rc_download_results'):
        _render_download_results(output_name)
        return

    # ── Export configuration ─────────────────────────────────────────────────
    using_quick_access = st.session_state.get('auth_mode') == 'quick_access'

    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox(
            "Format",
            ["GeoTIFF", "CSV (Zonal Stats)"],
            key="rc_export_format",
        )
    with col2:
        filename = st.text_input(
            "Filename",
            value=f"raster_calc_{output_name}",
            key="rc_export_filename",
        )

    if export_format == "GeoTIFF":
        if using_quick_access:
            export_preference = "local"
            st.caption(
                "ℹ️ Quick Access: local download only. "
                "Use your own GEE account to enable Google Drive export."
            )
        else:
            export_preference = st.radio(
                "Export method",
                ["auto", "local", "drive"],
                horizontal=True,
                key="rc_export_method",
                help="Auto tries local first and falls back to Drive for large files.",
            )

        if st.button("🚀 Export GeoTIFF", type="primary", use_container_width=True,
                     key="rc_export_btn_tiff"):
            with st.spinner("Exporting..."):
                exporter = GEEExporter()
                result = exporter.smart_export_with_fallback(
                    image=result_image.toFloat(),
                    filename=filename,
                    region=geometry,
                    scale=float(scale),
                    export_preference=export_preference,
                )

            if result.get('success') and result.get('export_method') == 'local':
                size_mb = result.get('actual_size_mb', 0) or 0
                st.session_state.rc_download_results = {
                    'file_data': result['file_data'],
                    'filename': f"{filename}.tif",
                    'mime_type': 'image/tiff',
                    'export_format': 'GeoTIFF',
                    'file_size_mb': size_mb,
                    'timestamp': _dt.now().strftime('%H:%M:%S'),
                }
                st.session_state.rc_download_complete = True
                st.rerun()
            elif result.get('success') and result.get('export_method') == 'drive':
                folder = result.get('drive_folder', 'GeoClimate_Exports')
                drive_url = result.get('drive_url', 'https://drive.google.com/drive/')
                st.success(f"📤 Export submitted to Google Drive — folder: **{folder}**")
                st.markdown(f"[📁 Open Google Drive]({drive_url})")
            else:
                st.error(f"Export failed: {result.get('message', 'Unknown error')}")
                logger.error("Raster calculator export failed: %s", result.get('message'))

    elif export_format == "CSV (Zonal Stats)":
        if st.button("🚀 Compute & Download CSV", type="primary", use_container_width=True,
                     key="rc_export_btn_csv"):
            with st.spinner("Computing zonal statistics..."):
                csv_data = _export_zonal_csv(result_image, geometry, scale, filename, output_name)
            if csv_data:
                st.session_state.rc_download_results = {
                    'file_data': csv_data.encode('utf-8'),
                    'filename': f"{filename}_zonal_stats.csv",
                    'mime_type': 'text/csv',
                    'export_format': 'CSV',
                    'file_size_mb': len(csv_data) / (1024 * 1024),
                    'timestamp': _dt.now().strftime('%H:%M:%S'),
                }
                st.session_state.rc_download_complete = True
                st.rerun()


def _render_download_results(output_name: str):
    """Persistent results panel — mirrors geodata explorer's _render_download_results_interface."""
    results = st.session_state.rc_download_results

    st.success("✅ Export completed successfully!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 Format", results['export_format'])
    with col2:
        st.metric("💾 File Size", f"{results['file_size_mb']:.2f} MB")
    with col3:
        st.metric("⏰ Exported at", results['timestamp'])

    st.markdown("---")
    st.markdown("### 📥 Download")

    st.download_button(
        label=f"📥 Download {results['export_format']} ({results['file_size_mb']:.2f} MB)",
        data=results['file_data'],
        file_name=results['filename'],
        mime=results['mime_type'],
        type="primary",
        use_container_width=True,
        key="rc_redownload_btn",
    )

    st.markdown("---")
    st.markdown("### 🔄 What would you like to do next?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Export Again / Change Format", use_container_width=True,
                     key="rc_reset_export_btn"):
            st.session_state.rc_download_complete = False
            st.session_state.rc_download_results = None
            st.rerun()
    with col2:
        if st.button("🆕 New Calculation", use_container_width=True,
                     key="rc_new_calc_btn"):
            for k in ['rc_download_complete', 'rc_download_results',
                      'rc_result_image', 'rc_geometry_complete',
                      'rc_geometry', 'rc_dataset_id', 'rc_bands_loaded',
                      'rc_band_list', 'rc_expression', 'rc_band_mapping']:
                st.session_state[k] = None if 'geometry' in k or 'image' in k or 'id' in k else (
                    False if isinstance(st.session_state.get(k), bool) else
                    [] if isinstance(st.session_state.get(k), list) else
                    '' if isinstance(st.session_state.get(k), str) else None
                )
            st.session_state.rc_geometry_complete = False
            st.session_state.rc_download_complete = False
            st.session_state.rc_download_results = None
            st.session_state.rc_result_image = None
            st.rerun()


def _export_zonal_csv(result_image, geometry, scale, filename, band_name) -> str | None:
    """Compute zonal statistics and return CSV string, or None on failure."""
    import pandas as pd

    stats = result_image.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.median(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    if stats:
        df = pd.DataFrame([stats])
        st.dataframe(df)
        return df.to_csv(index=False)

    st.warning("No statistics could be computed for this region.")
    return None


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_raster_calculator():
    """Render the complete Raster Calculator interface."""

    st.markdown('<h1 class="main-title">🧮 Raster Calculator</h1>',
                unsafe_allow_html=True)
    st.markdown("### Band math on Earth Engine datasets")
    st.caption(
        "Compute spectral indices (NDVI, NDWI, EVI, ...) or custom band math "
        "expressions on any Earth Engine ImageCollection."
    )

    _init_session_state()

    # --- Step 1: Area of Interest ---
    area_ok = _render_area_selection()
    if not area_ok:
        st.info("Draw or upload an area of interest to continue.")
        return

    geometry = st.session_state.rc_geometry

    st.markdown("---")

    # --- Step 2: Dataset ---
    ds_result = _render_dataset_selection()
    if ds_result is None or ds_result[0] is None:
        return
    ds_id, ds_info, scale = ds_result

    # Load bands
    band_names, band_descriptions = _load_bands(ds_id, ds_info, geometry)
    if not band_names:
        st.warning("No bands available for this dataset.")
        return

    st.markdown("---")

    # --- Step 3: Date Range ---
    date_result = _render_date_selection()
    if date_result is None or date_result[0] is None:
        return
    start_date, end_date, aggregation = date_result

    st.markdown("---")

    # --- Step 4: Expression ---
    expr_result = _render_expression_builder(band_names, band_descriptions)
    if expr_result is None or expr_result[0] is None:
        return
    expression, band_mapping = expr_result

    st.markdown("---")

    # --- Step 5: Compute & Export ---
    _render_compute_and_export(
        ds_id, expression, band_mapping,
        start_date, end_date, aggregation,
        geometry, scale,
    )
