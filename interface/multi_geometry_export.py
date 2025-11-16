"""
Multi-Geometry Data Export Interface Module
Handles exporting time series data for multiple geometries within a single file (e.g., counties within a state)
"""

import streamlit as st
import pandas as pd
import json
import time
import os
import ee
import zipfile
import tempfile
import io
from pathlib import Path
from datetime import datetime, timedelta

# Import core components
from geoclimate_fetcher.core import (
    MetadataCatalog,
    GeometryHandler,
    GEEExporter
)


def render_multi_geometry_export():
    """Render the complete Multi-Geometry Data Export interface"""

    # App title and header
    st.markdown('<h1 class="main-title">🗺️ Multi-Geometry Data Export</h1>', unsafe_allow_html=True)
    st.markdown("### Export time series data for multiple geometries (e.g., counties, districts, watersheds)")

    # Initialize session state specific to this module
    _initialize_session_state()

    # Initialize core objects
    try:
        with st.spinner("🔄 Initializing Multi-Geometry Export components..."):
            exporter = GEEExporter()
        st.success("✅ Multi-Geometry Export initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing: {str(e)}")
        exporter = None

    # Progress indicator
    _show_progress_indicator()

    # Workflow steps
    if not st.session_state.mg_geometry_uploaded:
        _render_geometry_upload()
    elif not st.session_state.mg_identifier_selected:
        _render_identifier_selection()
    elif not st.session_state.mg_dataset_selected:
        _render_dataset_selection()
    elif not st.session_state.mg_bands_selected:
        _render_band_selection()
    elif not st.session_state.mg_dates_selected:
        _render_date_selection()
    elif not st.session_state.mg_reducer_selected:
        _render_reducer_selection()
    else:
        _render_export_interface(exporter)


def _initialize_session_state():
    """Initialize all session state variables for multi-geometry export"""
    if 'mg_geometry_uploaded' not in st.session_state:
        st.session_state.mg_geometry_uploaded = False
    if 'mg_identifier_selected' not in st.session_state:
        st.session_state.mg_identifier_selected = False
    if 'mg_dataset_selected' not in st.session_state:
        st.session_state.mg_dataset_selected = False
    if 'mg_bands_selected' not in st.session_state:
        st.session_state.mg_bands_selected = False
    if 'mg_dates_selected' not in st.session_state:
        st.session_state.mg_dates_selected = False
    if 'mg_reducer_selected' not in st.session_state:
        st.session_state.mg_reducer_selected = False
    if 'mg_geometries' not in st.session_state:
        st.session_state.mg_geometries = None
    if 'mg_identifier_field' not in st.session_state:
        st.session_state.mg_identifier_field = None
    if 'mg_selected_dataset' not in st.session_state:
        st.session_state.mg_selected_dataset = None
    if 'mg_selected_bands' not in st.session_state:
        st.session_state.mg_selected_bands = []
    if 'mg_start_date' not in st.session_state:
        st.session_state.mg_start_date = None
    if 'mg_end_date' not in st.session_state:
        st.session_state.mg_end_date = None
    if 'mg_start_year' not in st.session_state:
        st.session_state.mg_start_year = None
    if 'mg_end_year' not in st.session_state:
        st.session_state.mg_end_year = None
    if 'mg_reducer_type' not in st.session_state:
        st.session_state.mg_reducer_type = 'mean'
    if 'mg_geojson_data' not in st.session_state:
        st.session_state.mg_geojson_data = None
    if 'mg_simplify_tolerance' not in st.session_state:
        st.session_state.mg_simplify_tolerance = 100  # Default 100 meters


def _show_progress_indicator():
    """Display visual progress indicator"""
    steps = [
        ("📁", "Upload", st.session_state.mg_geometry_uploaded),
        ("🔑", "Identifier", st.session_state.mg_identifier_selected),
        ("📊", "Dataset", st.session_state.mg_dataset_selected),
        ("🎛️", "Bands", st.session_state.mg_bands_selected),
        ("📅", "Dates", st.session_state.mg_dates_selected),
        ("🧮", "Reducer", st.session_state.mg_reducer_selected),
        ("💾", "Export", False)
    ]

    cols = st.columns(len(steps))

    for i, (icon, name, complete) in enumerate(steps):
        with cols[i]:
            if complete:
                st.markdown(f"""
                <div class="step-item step-complete">
                    <div class="step-number">✓</div>
                    <div>{icon} {name}</div>
                </div>
                """, unsafe_allow_html=True)
            elif i == len([s for s in steps[:i] if s[2]]):
                st.markdown(f"""
                <div class="step-item step-current">
                    <div class="step-number">{i+1}</div>
                    <div>{icon} {name}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="step-item step-pending">
                    <div class="step-number">{i+1}</div>
                    <div>{icon} {name}</div>
                </div>
                """, unsafe_allow_html=True)


def _render_geometry_upload():
    """Step 1: Upload GeoJSON/Shapefile with multiple geometries"""
    st.markdown('<div class="step-header"><h2>📁 Step 1: Upload Multi-Geometry File</h2></div>', unsafe_allow_html=True)

    st.info("""
    **Upload a GeoJSON or Shapefile containing multiple geometries**
    - Example: A state with counties, a country with districts, or a watershed with sub-basins
    - Each geometry should have a unique identifier (e.g., FIPS code, district ID, name)
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a GeoJSON or Shapefile (ZIP)",
        type=['geojson', 'json', 'zip'],
        help="Upload a GeoJSON file or a zipped Shapefile containing multiple geometries"
    )

    if uploaded_file:
        try:
            with st.spinner("🔄 Processing uploaded file..."):
                geojson_data = _process_uploaded_file(uploaded_file)

            if geojson_data:
                # Validate the GeoJSON
                features = geojson_data.get('features', [])
                num_geometries = len(features)

                if num_geometries == 0:
                    st.error("❌ No geometries found in the uploaded file")
                    return

                st.success(f"✅ Successfully loaded **{num_geometries}** geometries!")

                # Show preview
                st.markdown("### Preview of Geometries:")

                # Extract properties from first few features
                if features:
                    properties_list = [f.get('properties', {}) for f in features[:5]]
                    preview_df = pd.DataFrame(properties_list)
                    st.dataframe(preview_df)

                    if num_geometries > 5:
                        st.info(f"Showing first 5 of {num_geometries} geometries")

                # Calculate original GeoJSON size
                original_size = len(json.dumps(geojson_data))
                original_size_mb = original_size / (1024 * 1024)

                # Geometry Simplification Settings
                st.markdown("### ⚙️ Geometry Simplification (Required for Large Files)")

                if original_size_mb > 5:
                    st.error(f"""
                    ⚠️ **Large File Detected: {original_size_mb:.2f} MB**

                    Earth Engine has a 10MB payload limit. Your file is {original_size_mb:.2f} MB.
                    **You MUST simplify geometries** to reduce the file size before processing.
                    """)
                else:
                    st.info(f"📁 Current file size: {original_size_mb:.2f} MB (EE limit: 10MB)")

                st.warning("""
                **Important:** Geometry simplification reduces vertices while preserving shape.
                - Reduces payload size to stay under Earth Engine's 10MB limit
                - Improves computation performance significantly
                - Higher tolerance = more simplification = smaller file
                """)

                simplify_tolerance = st.number_input(
                    "Simplification Tolerance (meters)",
                    min_value=0,
                    max_value=50000,
                    value=1000 if original_size_mb > 5 else 100,
                    step=100,
                    help="Higher values = more simplification. Recommended: 1000-5000m for large files, 100-500m for smaller ones. 0 = no simplification (may fail for large files)."
                )

                # Apply local simplification and show size reduction
                if simplify_tolerance > 0:
                    with st.spinner("Applying local geometry simplification..."):
                        simplified_geojson, new_size = _simplify_geojson_locally(geojson_data, simplify_tolerance)
                        new_size_mb = new_size / (1024 * 1024)
                        reduction_pct = ((original_size - new_size) / original_size) * 100

                    st.success(f"""
                    ✅ **Simplification Applied:**
                    - Original size: {original_size_mb:.2f} MB
                    - New size: {new_size_mb:.2f} MB
                    - Reduction: {reduction_pct:.1f}%
                    """)

                    if new_size_mb > 10:
                        st.error(f"❌ File still too large ({new_size_mb:.2f} MB > 10MB). Increase simplification tolerance.")
                    elif new_size_mb > 8:
                        st.warning(f"⚠️ File is close to limit ({new_size_mb:.2f} MB). Consider increasing tolerance.")
                    else:
                        st.success(f"✅ File size OK: {new_size_mb:.2f} MB (under 10MB limit)")

                    # Use simplified version
                    geojson_to_store = simplified_geojson
                else:
                    st.warning("⚠️ No simplification applied. May fail for complex geometries.")
                    geojson_to_store = geojson_data
                    if original_size_mb > 10:
                        st.error("❌ File exceeds 10MB limit. Simplification is required!")

                # Store the data
                st.session_state.mg_geojson_data = geojson_to_store
                st.session_state.mg_geometries = geojson_to_store.get('features', features)
                st.session_state.mg_simplify_tolerance = simplify_tolerance

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"📊 **Total geometries:** {num_geometries}")
                with col2:
                    if st.button("Continue →", type="primary"):
                        st.session_state.mg_geometry_uploaded = True
                        st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid GeoJSON or zipped Shapefile")


def _process_uploaded_file(uploaded_file):
    """Process uploaded GeoJSON or Shapefile"""
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.zip'):
        # Handle zipped shapefile
        return _process_shapefile_zip(uploaded_file)
    else:
        # Handle GeoJSON
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return json.loads(content)


def _simplify_geojson_locally(geojson_data, tolerance_meters):
    """
    Simplify GeoJSON geometries locally using shapely to reduce payload size.

    Args:
        geojson_data: GeoJSON dict with features
        tolerance_meters: Simplification tolerance in meters

    Returns:
        Tuple of (simplified_geojson, new_size_bytes)
    """
    import geopandas as gpd
    from shapely.geometry import shape, mapping

    try:
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

        # Set CRS if not set (assume WGS84)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

        # Convert tolerance from meters to degrees (approximate)
        # At equator: 1 degree ≈ 111,320 meters
        # This is approximate but works for simplification purposes
        tolerance_degrees = tolerance_meters / 111320.0

        # Simplify geometries
        gdf['geometry'] = gdf['geometry'].simplify(tolerance_degrees, preserve_topology=True)

        # Convert back to GeoJSON
        simplified_geojson = json.loads(gdf.to_json())

        # Calculate new size
        new_size = len(json.dumps(simplified_geojson))

        return simplified_geojson, new_size

    except Exception as e:
        st.warning(f"⚠️ Local simplification failed: {str(e)}. Using original geometries.")
        # Return original if simplification fails
        return geojson_data, len(json.dumps(geojson_data))


def _process_shapefile_zip(uploaded_file):
    """Process a zipped shapefile and convert to GeoJSON"""
    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the zip file
        zip_path = os.path.join(tmpdir, 'shapefile.zip')
        with open(zip_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Extract the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find the .shp file
        shp_files = list(Path(tmpdir).glob('**/*.shp'))
        if not shp_files:
            raise ValueError("No .shp file found in the uploaded ZIP")

        # Read with geopandas
        gdf = gpd.read_file(shp_files[0])

        # Convert to GeoJSON
        return json.loads(gdf.to_json())


def _render_identifier_selection():
    """Step 2: Select the unique identifier field"""
    st.markdown('<div class="step-header"><h2>🔑 Step 2: Select Unique Identifier</h2></div>', unsafe_allow_html=True)

    if st.button("← Back to File Upload"):
        st.session_state.mg_geometry_uploaded = False
        st.rerun()

    st.info("""
    **Select the field that uniquely identifies each geometry**
    - This will be used to label the data in the exported CSV
    - Examples: FIPS, STATE_ID, NAME, GEOID, etc.
    """)

    # Get available fields from the first feature
    if st.session_state.mg_geometries:
        first_feature = st.session_state.mg_geometries[0]
        available_fields = list(first_feature.get('properties', {}).keys())

        if not available_fields:
            st.error("❌ No properties found in the geometries")
            return

        # Show sample values for each field
        st.markdown("### Available Fields:")

        sample_data = {}
        for field in available_fields:
            sample_values = []
            for feature in st.session_state.mg_geometries[:3]:
                val = feature.get('properties', {}).get(field, 'N/A')
                sample_values.append(str(val))
            sample_data[field] = sample_values

        sample_df = pd.DataFrame(sample_data).T
        sample_df.columns = ['Sample 1', 'Sample 2', 'Sample 3']
        sample_df.index.name = 'Field Name'
        st.dataframe(sample_df)

        # Select identifier field
        selected_field = st.selectbox(
            "Choose the unique identifier field:",
            available_fields,
            help="Select the field that uniquely identifies each geometry"
        )

        # Validate uniqueness
        if selected_field:
            all_values = [f.get('properties', {}).get(selected_field) for f in st.session_state.mg_geometries]
            unique_values = set(all_values)

            if len(unique_values) != len(all_values):
                st.warning(f"⚠️ The field '{selected_field}' has duplicate values. This may cause issues in the exported data.")
            else:
                st.success(f"✅ Field '{selected_field}' has {len(unique_values)} unique values")

            if st.button("Continue to Dataset Selection →", type="primary"):
                st.session_state.mg_identifier_field = selected_field
                st.session_state.mg_identifier_selected = True
                st.rerun()


def _render_dataset_selection():
    """Step 3: Select dataset (only ImageCollections)"""
    st.markdown('<div class="step-header"><h2>📊 Step 3: Select Dataset</h2></div>', unsafe_allow_html=True)

    if st.button("← Back to Identifier Selection"):
        st.session_state.mg_identifier_selected = False
        st.rerun()

    st.info("**Note:** Only ImageCollection datasets are supported for multi-geometry time series export.")

    # Dataset search
    search_term = st.text_input("🔍 Search datasets:", placeholder="Enter keywords (e.g., precipitation, temperature)")

    # Load datasets from CSV
    datasets = _load_image_collection_datasets()

    if not datasets:
        st.error("❌ No datasets found")
        return

    # Filter datasets
    filtered_datasets = datasets
    if search_term:
        search_lower = search_term.lower()
        filtered_datasets = [d for d in datasets if search_lower in d['name'].lower() or search_lower in d.get('description', '').lower()]

    st.info(f"📊 Found {len(filtered_datasets)} ImageCollection datasets")

    # Display datasets
    for dataset in filtered_datasets[:20]:  # Show first 20
        with st.expander(f"📊 {dataset['name']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Earth Engine ID:** `{dataset['ee_id']}`")
                st.markdown(f"**Temporal Resolution:** {dataset.get('temporal_resolution', 'Unknown')}")
                st.markdown(f"**Provider:** {dataset.get('provider', 'Unknown')}")
                st.markdown(f"**Description:** {dataset.get('description', 'No description')}")
            with col2:
                if st.button("Select", key=f"select_{dataset['ee_id']}", type="primary"):
                    st.session_state.mg_selected_dataset = dataset
                    st.session_state.mg_dataset_selected = True
                    st.rerun()


def _load_image_collection_datasets():
    """Load only ImageCollection datasets from CSV"""
    datasets = []

    # Get project root directory
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "geoclimate_fetcher" / "data"
    datasets_csv = data_dir / "Datasets.csv"

    if datasets_csv.exists():
        try:
            df = pd.read_csv(datasets_csv)
            for _, row in df.iterrows():
                snippet_type = str(row.get('Snippet Type', '')).strip()
                # Only include ImageCollections
                if snippet_type == 'ImageCollection':
                    dataset = {
                        'name': row.get('Dataset Name', 'Unknown'),
                        'ee_id': row.get('Earth Engine ID', ''),
                        'snippet_type': snippet_type,
                        'description': row.get('Description', 'No description available'),
                        'temporal_resolution': row.get('Temporal Resolution', 'Unknown'),
                        'provider': row.get('Provider', 'Unknown'),
                        'start_date': row.get('Start Date', ''),
                        'end_date': row.get('End Date', ''),
                        'pixel_size': row.get('Pixel Size (m)', ''),
                        'band_names': row.get('Band Names', ''),
                        'band_units': row.get('Band Units', '')
                    }
                    datasets.append(dataset)
        except Exception as e:
            st.error(f"Error loading datasets: {str(e)}")

    return datasets


def _render_band_selection():
    """Step 4: Select bands/parameters to export"""
    st.markdown('<div class="step-header"><h2>🎛️ Step 4: Select Parameters/Bands</h2></div>', unsafe_allow_html=True)

    if st.button("← Back to Dataset Selection"):
        st.session_state.mg_dataset_selected = False
        st.rerun()

    selected_dataset = st.session_state.mg_selected_dataset
    st.success(f"✅ Selected Dataset: **{selected_dataset.get('name', 'Unknown')}**")

    # Get available bands
    band_names_str = selected_dataset.get('band_names', '')
    band_units_str = selected_dataset.get('band_units', '')

    if band_names_str:
        bands = [b.strip() for b in band_names_str.split(',')]
        units = [u.strip() for u in band_units_str.split(',')] if band_units_str else []

        st.markdown("### Available Parameters/Bands:")

        # Show bands with units
        if len(units) == len(bands):
            band_info = pd.DataFrame({
                'Band Name': bands,
                'Unit': units
            })
            st.dataframe(band_info, use_container_width=True)
        else:
            for band in bands:
                st.markdown(f"• {band}")

        # Multi-select bands
        selected_bands = st.multiselect(
            "Choose parameters to export:",
            bands,
            default=bands[:1] if bands else [],  # Select first band by default
            help="Select the parameters you want to extract for each geometry"
        )

        if selected_bands:
            st.success(f"✅ Selected {len(selected_bands)} parameter(s)")

            st.warning("""
            ⚠️ **Performance Note:**
            - More parameters = longer processing time
            - Consider selecting only essential parameters for large date ranges
            """)

            if st.button("Continue to Date Selection →", type="primary"):
                st.session_state.mg_selected_bands = selected_bands
                st.session_state.mg_bands_selected = True
                st.rerun()
        else:
            st.warning("⚠️ Please select at least one parameter")
    else:
        st.error("❌ No band information available for this dataset")


def _render_date_selection():
    """Step 5: Select date range (flexible - any dates, not just years)"""
    st.markdown('<div class="step-header"><h2>📅 Step 5: Select Date Range</h2></div>', unsafe_allow_html=True)

    if st.button("← Back to Band Selection"):
        st.session_state.mg_bands_selected = False
        st.rerun()

    # Show current selections
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Geometries:** {len(st.session_state.mg_geometries)}")
    with col2:
        st.info(f"**Dataset:** {st.session_state.mg_selected_dataset.get('name', 'Unknown')[:30]}...")
    with col3:
        st.info(f"**Parameters:** {len(st.session_state.mg_selected_bands)}")

    # Parse dataset dates
    selected_dataset = st.session_state.mg_selected_dataset
    start_str = selected_dataset.get('start_date', '')
    end_str = selected_dataset.get('end_date', '')

    try:
        # Try to parse dates (handle different formats)
        if '/' in str(start_str):
            dataset_start = datetime.strptime(str(start_str), '%m/%d/%Y').date()
            dataset_end = datetime.strptime(str(end_str), '%m/%d/%Y').date()
        else:
            dataset_start = datetime.strptime(str(start_str), '%Y-%m-%d').date()
            dataset_end = datetime.strptime(str(end_str), '%Y-%m-%d').date()
    except:
        dataset_start = datetime(2000, 1, 1).date()
        dataset_end = datetime.now().date()

    st.markdown("### Data Availability:")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Start:** {dataset_start}")
    with col2:
        st.info(f"**End:** {dataset_end}")

    # Flexible date selection
    st.markdown("### Select Your Date Range:")
    st.info("""
    **Flexible Date Selection:** You can select any date range - a few days, months, or years.
    The system will automatically chunk long ranges for optimal processing.
    """)

    col1, col2 = st.columns(2)
    with col1:
        user_start_date = st.date_input(
            "Start Date",
            value=max(dataset_start, datetime(datetime.now().year - 1, 1, 1).date()),
            min_value=dataset_start,
            max_value=dataset_end,
            help="Select start date for your data export"
        )
    with col2:
        user_end_date = st.date_input(
            "End Date",
            value=min(dataset_end, datetime.now().date()),
            min_value=user_start_date,
            max_value=dataset_end,
            help="Select end date for your data export"
        )

    # Calculate date range statistics
    date_range_days = (user_end_date - user_start_date).days + 1
    num_geometries = len(st.session_state.mg_geometries)
    num_params = len(st.session_state.mg_selected_bands)

    # Determine chunking strategy
    if date_range_days <= 365:
        num_chunks = 1
        chunk_strategy = "single chunk"
    elif date_range_days <= 730:  # 2 years
        num_chunks = 2
        chunk_strategy = "2 chunks (~1 year each)"
    else:
        num_chunks = max(2, (date_range_days // 365) + 1)
        chunk_strategy = f"{num_chunks} chunks (~1 year each)"

    st.markdown("### Export Strategy:")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        - **Date Range:** {date_range_days} days
        - **Geometries:** {num_geometries}
        - **Parameters:** {num_params}
        """)
    with col2:
        st.info(f"""
        - **Chunking:** {chunk_strategy}
        - **Processing:** One chunk at a time
        - **Smart Export:** Local first, Drive fallback
        """)

    # Complexity warning
    complexity_per_chunk = num_geometries * num_params * (date_range_days / num_chunks / 365)
    if complexity_per_chunk > 100:
        st.warning(f"""
        ⚠️ **High Complexity Detected** (Score: {complexity_per_chunk:.0f} per chunk)
        - Local download likely to fail or timeout
        - System will automatically use Google Drive export
        - Consider reducing: fewer geometries, fewer parameters, or shorter date range
        """)
    else:
        st.success(f"""
        ✅ **Moderate Complexity** (Score: {complexity_per_chunk:.0f} per chunk)
        - Local download may succeed for smaller chunks
        - Automatic fallback to Drive if needed
        """)

    if st.button("Continue to Reducer Selection →", type="primary"):
        st.session_state.mg_start_date = user_start_date
        st.session_state.mg_end_date = user_end_date
        # Store years for backward compatibility
        st.session_state.mg_start_year = user_start_date.year
        st.session_state.mg_end_year = user_end_date.year
        st.session_state.mg_dates_selected = True
        st.rerun()


def _render_reducer_selection():
    """Step 6: Select reducer type (mean/median)"""
    st.markdown('<div class="step-header"><h2>🧮 Step 6: Select Aggregation Method</h2></div>', unsafe_allow_html=True)

    if st.button("← Back to Date Selection"):
        st.session_state.mg_dates_selected = False
        st.rerun()

    st.info("""
    **Select how to aggregate pixel values within each geometry**
    - **Mean:** Average of all pixels (faster, good for general use)
    - **Median:** Middle value (more robust to outliers)
    """)

    reducer_type = st.radio(
        "Choose aggregation method:",
        ['mean', 'median'],
        format_func=lambda x: f"{'📊 Mean (Average)' if x == 'mean' else '📈 Median (Middle Value)'}",
        horizontal=True
    )

    st.markdown(f"**Selected:** {'Mean - calculates the average of all pixel values' if reducer_type == 'mean' else 'Median - uses the middle value, more robust to outliers'}")

    if st.button("Continue to Export →", type="primary"):
        st.session_state.mg_reducer_type = reducer_type
        st.session_state.mg_reducer_selected = True
        st.rerun()


def _render_export_interface(exporter):
    """Step 7: Configure and execute export"""
    st.markdown('<div class="step-header"><h2>💾 Step 7: Export Data</h2></div>', unsafe_allow_html=True)

    if st.button("← Back to Reducer Selection"):
        st.session_state.mg_reducer_selected = False
        st.rerun()

    # Show summary of all selections
    st.markdown("### 📋 Export Summary:")

    # Calculate date chunks for display
    start_date = st.session_state.mg_start_date
    end_date = st.session_state.mg_end_date
    date_chunks = _calculate_date_chunks(start_date, end_date)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **Geometries:** {len(st.session_state.mg_geometries)}
        - **Identifier:** {st.session_state.mg_identifier_field}
        - **Dataset:** {st.session_state.mg_selected_dataset.get('name', 'Unknown')}
        - **Parameters:** {', '.join(st.session_state.mg_selected_bands)}
        """)
    with col2:
        st.markdown(f"""
        - **Date Range:** {start_date} to {end_date}
        - **Chunks:** {len(date_chunks)} (processed separately)
        - **Aggregation:** {st.session_state.mg_reducer_type.capitalize()}
        - **Simplification:** {st.session_state.get('mg_simplify_tolerance', 100)}m
        """)

    # Export preference
    export_preference = st.radio(
        "Export Preference:",
        ['auto', 'drive'],
        format_func=lambda x: "🤖 Smart (Try local first, fallback to Drive)" if x == 'auto' else "☁️ Google Drive Only",
        horizontal=True,
        help="Smart export tries to download locally first, then falls back to Google Drive for large exports"
    )

    # Google Drive folder name (if using Drive)
    drive_folder = st.text_input(
        "Google Drive Folder Name:",
        value=f"MultiGeometry_Export_{datetime.now().strftime('%Y%m%d')}",
        help="Folder name in your Google Drive where files will be saved"
    )

    # Scale/Resolution
    scale = st.number_input(
        "Pixel Resolution (meters):",
        min_value=10,
        max_value=100000,
        value=int(st.session_state.mg_selected_dataset.get('pixel_size', 250)),
        help="Resolution for spatial averaging. Use dataset's native resolution for best results."
    )

    st.markdown("---")

    # Calculate complexity score for user guidance using actual chunks
    num_geometries = len(st.session_state.mg_geometries)
    num_chunks = len(date_chunks)
    num_bands = len(st.session_state.mg_selected_bands)
    total_days = (end_date - start_date).days + 1

    # Complexity per chunk (more accurate than total)
    complexity_per_chunk = num_geometries * num_bands * (total_days / num_chunks / 365)
    total_complexity = num_geometries * num_chunks * num_bands

    # Provide intelligent feedback based on complexity
    if complexity_per_chunk > 100:
        st.error(f"""
        🚨 **High Complexity Per Chunk** (Score: {complexity_per_chunk:.0f})
        - {num_geometries} geometries × {num_bands} bands × {num_chunks} chunks
        - **Local download WILL BE SKIPPED** - going directly to Google Drive
        - This prevents long waits (EE has 5-min internal timeout)
        - Estimated time: {num_chunks * 3}-{num_chunks * 10} minutes
        """)
    elif complexity_per_chunk > 50:
        st.warning(f"""
        ⚠️ **Medium-High Complexity** (Score: {complexity_per_chunk:.0f})
        - {num_geometries} geometries × {num_bands} bands × {num_chunks} chunks
        - Local download will be attempted for **first chunk only**
        - If first chunk fails, remaining chunks go directly to Drive
        - This prevents wasting time on repeated failures
        - Estimated time: {num_chunks * 2}-{num_chunks * 7} minutes
        """)
    elif complexity_per_chunk > 20:
        st.info(f"""
        ℹ️ **Moderate Complexity** (Score: {complexity_per_chunk:.0f})
        - {num_geometries} geometries × {num_bands} bands × {num_chunks} chunks
        - Local download likely to succeed
        - Automatic Drive fallback if local fails
        - Estimated time: {num_chunks * 1}-{num_chunks * 3} minutes
        """)
    else:
        st.success(f"""
        ✅ **Low Complexity** (Score: {complexity_per_chunk:.0f})
        - {num_geometries} geometries × {num_bands} bands × {num_chunks} chunks
        - Local download expected to succeed
        - Fast processing expected
        - Estimated time: {max(1, num_chunks * 0.5):.0f}-{num_chunks * 2} minutes
        """)

    # Warning about processing time
    st.warning(f"""
    ⚠️ **Smart Export Strategy:**
    - **{num_chunks} chunk(s)** will be processed (each ~1 year or less)
    - **Learning from failure:** If local fails on first chunk, skips local for all remaining chunks
    - **No fake timeouts:** Uses pre-flight complexity check instead of waiting
    - Geometry simplification applied: {st.session_state.get('mg_simplify_tolerance', 100)}m
    - Monitor Drive tasks: https://code.earthengine.google.com/tasks
    """)

    # Export button
    if st.button("🚀 Start Export", type="primary", use_container_width=True):
        _execute_multi_geometry_export(exporter, export_preference, drive_folder, scale)

    # Reset button
    st.markdown("---")
    if st.button("🔄 Start Over", use_container_width=True):
        _reset_all_mg_selections()
        st.rerun()


def _execute_multi_geometry_export(exporter, export_preference, drive_folder, scale):
    """Execute the multi-geometry export with smart chunking and failure learning"""

    st.markdown("### 🔄 Processing Export...")

    # Get selections with safety checks
    geojson_data = st.session_state.mg_geojson_data
    identifier_field = st.session_state.mg_identifier_field
    dataset_id = st.session_state.mg_selected_dataset.get('ee_id') if st.session_state.mg_selected_dataset else None
    parameters = st.session_state.mg_selected_bands
    start_date = st.session_state.mg_start_date
    end_date = st.session_state.mg_end_date
    reducer_type = st.session_state.mg_reducer_type
    simplify_tolerance = st.session_state.get('mg_simplify_tolerance', 100)

    # Validate required session state variables
    if start_date is None or end_date is None:
        st.error("❌ Date range not properly set. Please go back and select date range.")
        if st.button("🔄 Reset and Start Over"):
            _reset_all_mg_selections()
            st.rerun()
        return

    if not dataset_id:
        st.error("❌ Dataset not properly selected. Please go back and select a dataset.")
        return

    try:
        # Convert GeoJSON to Earth Engine FeatureCollection
        with st.spinner("Converting geometries to Earth Engine format..."):
            # Check payload size before sending
            payload_size = len(json.dumps(geojson_data))
            payload_size_mb = payload_size / (1024 * 1024)

            if payload_size_mb > 10:
                st.error(f"""
                ❌ **Payload Too Large: {payload_size_mb:.2f} MB**

                Earth Engine has a 10MB limit. Please go back to Step 1 and:
                1. Increase the simplification tolerance (try 2000-5000m)
                2. Or upload a smaller file with fewer/simpler geometries

                Current file is {payload_size_mb:.2f} MB (limit: 10MB)
                """)
                return

            st.info(f"📤 Uploading {payload_size_mb:.2f} MB to Earth Engine...")

            try:
                ee_fc = ee.FeatureCollection(geojson_data)
                # Verify the feature collection
                fc_size = ee_fc.size().getInfo()
                st.success(f"✅ Created Earth Engine FeatureCollection with {fc_size} features")
            except Exception as e:
                error_msg = str(e)
                if "payload size exceeds" in error_msg.lower() or "10485760" in error_msg:
                    st.error(f"""
                    ❌ **Payload Size Error**

                    The GeoJSON file is too large for Earth Engine ({payload_size_mb:.2f} MB).

                    **Solutions:**
                    1. Go back to Step 1 and increase simplification tolerance (try 2000-10000m)
                    2. Upload a file with fewer geometries
                    3. Split your data into smaller batches

                    Current simplification: {simplify_tolerance}m
                    Try increasing to: {max(simplify_tolerance * 2, 5000)}m
                    """)
                else:
                    st.error(f"❌ Failed to create FeatureCollection: {error_msg}")
                return

        # Calculate date chunks
        date_chunks = _calculate_date_chunks(start_date, end_date)
        total_chunks = len(date_chunks)

        # Process chunk by chunk with learning
        all_results = []
        task_ids = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate complexity per chunk to determine strategy
        num_geometries = len(st.session_state.mg_geometries)
        num_bands = len(parameters)
        total_days = (end_date - start_date).days + 1
        complexity_per_chunk = num_geometries * num_bands * (total_days / total_chunks / 365)

        # SMART STRATEGY: Determine if we should skip local entirely based on pre-check
        local_failed = False  # Track if local export has failed (for learning)

        if complexity_per_chunk > 100:
            st.warning(f"🚨 High complexity ({complexity_per_chunk:.0f}). Skipping local downloads entirely → Google Drive only.")
            force_drive = True
        elif complexity_per_chunk > 50:
            st.info(f"ℹ️ Medium-high complexity ({complexity_per_chunk:.0f}). Will try local for first chunk only. If it fails, remaining chunks go to Drive.")
            force_drive = False
        else:
            st.success(f"✅ Moderate complexity ({complexity_per_chunk:.0f}). Local downloads should work.")
            force_drive = False

        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            chunk_label = f"{chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
            status_text.text(f"Processing chunk {chunk_idx + 1}/{total_chunks}: {chunk_label}")
            progress_bar.progress(chunk_idx / total_chunks)

            # LEARNING: Determine effective preference for this chunk
            if force_drive:
                effective_preference = 'drive'
            elif local_failed:
                # Local failed on previous chunk, skip local for this and all remaining
                st.info(f"⏭️ Chunk {chunk_idx + 1}: Skipping local (learned from previous failure)")
                effective_preference = 'drive'
            else:
                effective_preference = export_preference

            # Export this chunk
            chunk_result = _export_chunk_data(
                ee_fc, dataset_id, parameters, chunk_start, chunk_end,
                identifier_field, reducer_type, scale,
                effective_preference, drive_folder, exporter
            )

            if chunk_result:
                all_results.append(chunk_result)
                if 'task_id' in chunk_result:
                    task_ids.append(chunk_result['task_id'])

                # LEARNING: If this chunk failed locally, don't try local for remaining chunks
                if chunk_result.get('local_failed', False):
                    local_failed = True
                    st.warning(f"📝 **Learning:** Local download failed. Skipping local for remaining {total_chunks - chunk_idx - 1} chunk(s).")

            time.sleep(0.5)  # Small delay between chunks

        progress_bar.progress(1.0)
        status_text.text("✅ Export completed!")

        # Show results
        st.markdown("### 📊 Export Results:")

        # Separate local and drive results
        local_results = [r for r in all_results if r.get('method') == 'local']
        drive_results = [r for r in all_results if r.get('method') == 'drive']

        if local_results:
            st.success(f"✅ **Local Downloads:** {len(local_results)} file(s)")
            for idx, result in enumerate(local_results):
                if result.get('data'):
                    # Create download button for local data
                    chunk_label = result.get('chunk_label', f'chunk_{idx}')
                    st.download_button(
                        label=f"📥 Download {result.get('filename', 'data.csv')}",
                        data=result['data'],
                        file_name=result.get('filename', 'data.csv'),
                        mime='text/csv',
                        key=f"download_{chunk_label}_{idx}"
                    )

        if drive_results:
            st.info(f"☁️ **Google Drive Exports:** {len(drive_results)} task(s) submitted")
            st.markdown(f"""
            **Tasks submitted to Google Drive folder:** `{drive_folder}`

            Monitor your tasks: [Earth Engine Tasks](https://code.earthengine.google.com/tasks)

            Access your files: [Google Drive](https://drive.google.com/drive/folders/)
            """)

            for result in drive_results:
                chunk_label = result.get('chunk_label', 'Unknown')
                st.markdown(f"• {chunk_label}: Task ID `{result.get('task_id', 'Unknown')}`")

        if not local_results and not drive_results:
            st.error("❌ No exports were successful. Please check the error messages above.")

    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def _export_chunk_data(ee_fc, dataset_id, parameters, chunk_start, chunk_end, identifier_field, reducer_type, scale, export_preference, drive_folder, exporter):
    """
    Export data for a single date chunk with NO fake timeouts.
    Returns result dict with 'local_failed' flag for learning.
    """
    chunk_label = f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"

    try:
        # Access the dataset
        dataset = ee.ImageCollection(dataset_id)

        # Set time range for the chunk
        ee_start = ee.Date.fromYMD(chunk_start.year, chunk_start.month, chunk_start.day)
        ee_end = ee.Date.fromYMD(chunk_end.year, chunk_end.month, chunk_end.day).advance(1, 'day')

        # Filter dataset for the chunk
        filtered_collection = dataset.filterDate(ee_start, ee_end)

        # Select parameters
        if len(parameters) == 1:
            filtered_collection = filtered_collection.select(parameters[0])
        else:
            filtered_collection = filtered_collection.select(parameters)

        # Define reducer
        if reducer_type == 'mean':
            reducer = ee.Reducer.mean()
        else:
            reducer = ee.Reducer.median()

        # Set output names for reducer
        if len(parameters) == 1:
            reducer = reducer.setOutputs(parameters)

        # Define the mapping function
        def reduce_regions(image):
            timestamp = image.date().format('YYYY-MM-dd')
            reduced = image.reduceRegions(
                collection=ee_fc,
                reducer=reducer,
                scale=scale
            )

            def add_timestamp(feature):
                return feature.set('timestamp', timestamp)

            return reduced.map(add_timestamp)

        # Map over the collection and flatten
        with st.spinner(f"Reducing regions for {chunk_label}..."):
            feature_collection = filtered_collection.map(reduce_regions).flatten()

        # Set up selectors (columns to export)
        selectors = parameters + ['timestamp', identifier_field]

        # TRY LOCAL EXPORT (only if preference allows)
        if export_preference == 'auto':
            try:
                with st.spinner(f"Attempting local download for {chunk_label}..."):
                    # Direct getInfo call - NO FAKE TIMEOUT
                    # This will either succeed quickly or fail with EE's own timeout
                    data_dict = feature_collection.getInfo()
                    data_list = data_dict['features']

                    # Convert to DataFrame
                    rows = []
                    for feature in data_list:
                        props = feature.get('properties', {})
                        row = {key: props.get(key) for key in selectors if key in props}
                        rows.append(row)

                    df = pd.DataFrame(rows)

                    # Reorder columns
                    cols = [identifier_field, 'timestamp'] + parameters
                    df = df[[c for c in cols if c in df.columns]]

                    # Convert to CSV
                    csv_data = df.to_csv(index=False)

                    filename = f"multigeometry_{chunk_label}_{reducer_type}.csv"

                    st.success(f"✅ {chunk_label}: Local download successful ({len(df)} records)")

                    return {
                        'method': 'local',
                        'chunk_label': chunk_label,
                        'filename': filename,
                        'data': csv_data,
                        'records': len(df),
                        'local_failed': False  # Success!
                    }

            except Exception as e:
                error_msg = str(e)
                st.warning(f"⚠️ {chunk_label}: Local download failed ({error_msg}). Submitting to Google Drive...")
                # Return will include local_failed=True after Drive submission
                local_failed_flag = True
        else:
            local_failed_flag = False  # Didn't try local

        # EXPORT TO GOOGLE DRIVE
        with st.spinner(f"Submitting {chunk_label} to Google Drive..."):
            task = ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=f'multigeometry_{chunk_label}_{reducer_type}',
                folder=drive_folder,
                fileNamePrefix=f'multigeometry_{chunk_label}_{reducer_type}',
                fileFormat='CSV',
                selectors=selectors
            )

            task.start()
            task_id = task.id

            st.info(f"☁️ {chunk_label}: Submitted to Google Drive (Task ID: {task_id})")

            return {
                'method': 'drive',
                'chunk_label': chunk_label,
                'task_id': task_id,
                'folder': drive_folder,
                'local_failed': local_failed_flag if export_preference == 'auto' else False
            }

    except Exception as e:
        st.error(f"❌ {chunk_label}: Export failed - {str(e)}")
        return None


def _export_year_data(ee_fc, dataset_id, parameters, year, identifier_field, reducer_type, scale, export_preference, drive_folder, exporter, complexity_score=0):
    """Export data for a single year with intelligent timeout handling (DEPRECATED - use _export_chunk_data)"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    # Determine timeout based on complexity (60s default, shorter for very complex)
    if complexity_score > 500:
        local_timeout = 60  # 60 seconds for medium-large datasets
    else:
        local_timeout = 120  # 2 minutes for smaller datasets

    try:
        # Access the dataset
        dataset = ee.ImageCollection(dataset_id)

        # Set time range for the year
        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = ee.Date.fromYMD(year + 1, 1, 1)

        # Filter dataset for the year
        filtered_collection = dataset.filterDate(start_date, end_date)

        # Select parameters
        if len(parameters) == 1:
            filtered_collection = filtered_collection.select(parameters[0])
        else:
            filtered_collection = filtered_collection.select(parameters)

        # Define reducer
        if reducer_type == 'mean':
            reducer = ee.Reducer.mean()
        else:
            reducer = ee.Reducer.median()

        # Set output names for reducer
        if len(parameters) == 1:
            reducer = reducer.setOutputs(parameters)

        # Define the mapping function
        def reduce_regions(image):
            # Get timestamp from image
            timestamp = image.date().format('YYYY-MM-dd')

            # Reduce regions
            reduced = image.reduceRegions(
                collection=ee_fc,
                reducer=reducer,
                scale=scale
            )

            # Add timestamp to each feature
            def add_timestamp(feature):
                return feature.set('timestamp', timestamp)

            return reduced.map(add_timestamp)

        # Map over the collection and flatten
        with st.spinner(f"Reducing regions for year {year}..."):
            feature_collection = filtered_collection.map(reduce_regions).flatten()

        # Set up selectors (columns to export)
        selectors = parameters + ['timestamp', identifier_field]

        # Try to determine size and decide export method
        if export_preference == 'auto':
            # Try local first with timeout
            try:
                with st.spinner(f"Attempting local download for year {year} (timeout: {local_timeout}s)..."):
                    # Use ThreadPoolExecutor for timeout control
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(feature_collection.getInfo)
                        try:
                            # Wait with timeout
                            result = future.result(timeout=local_timeout)
                            data_list = result['features']
                        except FuturesTimeoutError:
                            raise Exception(f"Local download timed out after {local_timeout} seconds")

                    # Convert to DataFrame
                    rows = []
                    for feature in data_list:
                        props = feature.get('properties', {})
                        row = {key: props.get(key) for key in selectors if key in props}
                        rows.append(row)

                    df = pd.DataFrame(rows)

                    # Reorder columns
                    cols = [identifier_field, 'timestamp'] + parameters
                    df = df[[c for c in cols if c in df.columns]]

                    # Convert to CSV
                    csv_data = df.to_csv(index=False)

                    filename = f"multigeometry_{year}_{reducer_type}.csv"

                    st.success(f"✅ Year {year}: Local download successful ({len(df)} records)")

                    return {
                        'method': 'local',
                        'year': year,
                        'filename': filename,
                        'data': csv_data,
                        'records': len(df)
                    }

            except Exception as e:
                error_msg = str(e)
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    st.warning(f"⏱️ Year {year}: Local download timed out after {local_timeout}s. Falling back to Google Drive...")
                else:
                    st.warning(f"⚠️ Year {year}: Local download failed ({error_msg}). Falling back to Google Drive...")
                # Fall through to Drive export

        # Export to Google Drive
        with st.spinner(f"Submitting year {year} to Google Drive..."):
            # Create the export task
            task = ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=f'multigeometry_{year}_{reducer_type}',
                folder=drive_folder,
                fileNamePrefix=f'multigeometry_{year}_{reducer_type}',
                fileFormat='CSV',
                selectors=selectors
            )

            task.start()
            task_id = task.id

            st.info(f"☁️ Year {year}: Submitted to Google Drive (Task ID: {task_id})")

            return {
                'method': 'drive',
                'year': year,
                'task_id': task_id,
                'folder': drive_folder
            }

    except Exception as e:
        st.error(f"❌ Year {year}: Export failed - {str(e)}")
        return None


def _calculate_date_chunks(start_date, end_date):
    """
    Split a date range into manageable chunks for processing.

    Strategy:
    - If range <= 365 days: single chunk
    - If range > 365 days: split into ~yearly chunks

    Returns list of (chunk_start, chunk_end) tuples
    """
    from datetime import date, timedelta

    # Ensure we're working with date objects
    if hasattr(start_date, 'date'):
        start_date = start_date.date()
    if hasattr(end_date, 'date'):
        end_date = end_date.date()

    total_days = (end_date - start_date).days + 1

    if total_days <= 365:
        # Single chunk for short ranges
        return [(start_date, end_date)]

    # Split into yearly chunks
    chunks = []
    chunk_start = start_date

    while chunk_start <= end_date:
        # Calculate chunk end (approximately 1 year later or end_date)
        chunk_end = min(
            chunk_start + timedelta(days=364),  # ~1 year
            end_date
        )
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)

    return chunks


def _reset_all_mg_selections():
    """Reset all multi-geometry export selections"""
    st.session_state.mg_geometry_uploaded = False
    st.session_state.mg_identifier_selected = False
    st.session_state.mg_dataset_selected = False
    st.session_state.mg_bands_selected = False
    st.session_state.mg_dates_selected = False
    st.session_state.mg_reducer_selected = False
    st.session_state.mg_geometries = None
    st.session_state.mg_identifier_field = None
    st.session_state.mg_selected_dataset = None
    st.session_state.mg_selected_bands = []
    st.session_state.mg_start_date = None
    st.session_state.mg_end_date = None
    st.session_state.mg_reducer_type = 'mean'
    st.session_state.mg_geojson_data = None
    st.session_state.mg_simplify_tolerance = 100
    if 'mg_start_year' in st.session_state:
        del st.session_state.mg_start_year
    if 'mg_end_year' in st.session_state:
        del st.session_state.mg_end_year
