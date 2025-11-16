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
    if 'mg_reducer_type' not in st.session_state:
        st.session_state.mg_reducer_type = 'mean'
    if 'mg_geojson_data' not in st.session_state:
        st.session_state.mg_geojson_data = None


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

                # Store the data
                st.session_state.mg_geojson_data = geojson_data
                st.session_state.mg_geometries = features

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
    """Step 5: Select date range"""
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

    # Year-based selection for easier handling
    st.markdown("### Select Year Range:")
    st.warning("""
    ⚠️ **Important:** Large date ranges with many geometries can exceed Google Earth Engine limits.
    - Recommended: Start with 1-2 years for testing
    - The export will process data year-by-year to handle limits
    """)

    min_year = dataset_start.year
    max_year = min(dataset_end.year, datetime.now().year)

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=min_year, max_value=max_year, value=max_year - 1)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=max_year, value=max_year)

    # Calculate expected data volume
    num_years = end_year - start_year + 1
    num_geometries = len(st.session_state.mg_geometries)
    num_params = len(st.session_state.mg_selected_bands)

    st.markdown("### Expected Data Volume:")
    st.info(f"""
    - **Years:** {num_years}
    - **Geometries:** {num_geometries}
    - **Parameters:** {num_params}
    - **Estimated tasks:** {num_years} (one per year)
    """)

    if num_years > 5:
        st.warning("⚠️ Large date range detected. Consider reducing to avoid long processing times.")

    if st.button("Continue to Reducer Selection →", type="primary"):
        st.session_state.mg_start_date = datetime(start_year, 1, 1).date()
        st.session_state.mg_end_date = datetime(end_year, 12, 31).date()
        st.session_state.mg_start_year = start_year
        st.session_state.mg_end_year = end_year
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
        - **Start Year:** {st.session_state.mg_start_year}
        - **End Year:** {st.session_state.mg_end_year}
        - **Aggregation:** {st.session_state.mg_reducer_type.capitalize()}
        - **Export Method:** Smart (Local first, Drive fallback)
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

    # Warning about processing time
    st.warning("""
    ⚠️ **Important Notes:**
    - Processing may take several minutes depending on data volume
    - Each year is processed as a separate task to handle GEE limits
    - Local download works for smaller exports; larger ones go to Google Drive
    - Monitor progress in the Earth Engine Tasks panel: https://code.earthengine.google.com/tasks
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
    """Execute the multi-geometry export"""

    st.markdown("### 🔄 Processing Export...")

    # Get selections
    geojson_data = st.session_state.mg_geojson_data
    identifier_field = st.session_state.mg_identifier_field
    dataset_id = st.session_state.mg_selected_dataset.get('ee_id')
    parameters = st.session_state.mg_selected_bands
    start_year = st.session_state.mg_start_year
    end_year = st.session_state.mg_end_year
    reducer_type = st.session_state.mg_reducer_type

    try:
        # Convert GeoJSON to Earth Engine FeatureCollection
        with st.spinner("Converting geometries to Earth Engine format..."):
            ee_fc = ee.FeatureCollection(geojson_data)

            # Verify the feature collection
            fc_size = ee_fc.size().getInfo()
            st.success(f"✅ Created Earth Engine FeatureCollection with {fc_size} features")

        # Process year by year
        all_results = []
        task_ids = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_years = end_year - start_year + 1

        for year_idx, year in enumerate(range(start_year, end_year + 1)):
            status_text.text(f"Processing year {year} ({year_idx + 1}/{total_years})...")
            progress_bar.progress((year_idx) / total_years)

            # Try local export first for each year
            year_result = _export_year_data(
                ee_fc, dataset_id, parameters, year,
                identifier_field, reducer_type, scale,
                export_preference, drive_folder, exporter
            )

            if year_result:
                all_results.append(year_result)
                if 'task_id' in year_result:
                    task_ids.append(year_result['task_id'])

            time.sleep(1)  # Small delay between years

        progress_bar.progress(1.0)
        status_text.text("✅ Export completed!")

        # Show results
        st.markdown("### 📊 Export Results:")

        # Separate local and drive results
        local_results = [r for r in all_results if r.get('method') == 'local']
        drive_results = [r for r in all_results if r.get('method') == 'drive']

        if local_results:
            st.success(f"✅ **Local Downloads:** {len(local_results)} file(s)")
            for result in local_results:
                if result.get('data'):
                    # Create download button for local data
                    st.download_button(
                        label=f"📥 Download {result.get('filename', 'data.csv')}",
                        data=result['data'],
                        file_name=result.get('filename', 'data.csv'),
                        mime='text/csv',
                        key=f"download_{result.get('year', 'unknown')}"
                    )

        if drive_results:
            st.info(f"☁️ **Google Drive Exports:** {len(drive_results)} task(s) submitted")
            st.markdown(f"""
            **Tasks submitted to Google Drive folder:** `{drive_folder}`

            Monitor your tasks: [Earth Engine Tasks](https://code.earthengine.google.com/tasks)

            Access your files: [Google Drive](https://drive.google.com/drive/folders/)
            """)

            for result in drive_results:
                st.markdown(f"• Year {result.get('year')}: Task ID `{result.get('task_id', 'Unknown')}`")

        if not local_results and not drive_results:
            st.error("❌ No exports were successful. Please check the error messages above.")

    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def _export_year_data(ee_fc, dataset_id, parameters, year, identifier_field, reducer_type, scale, export_preference, drive_folder, exporter):
    """Export data for a single year"""

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
            # Try local first
            try:
                with st.spinner(f"Attempting local download for year {year}..."):
                    # Get the data as a list
                    data_list = feature_collection.getInfo()['features']

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
                st.warning(f"⚠️ Year {year}: Local download failed ({str(e)}). Falling back to Google Drive...")
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
    if 'mg_start_year' in st.session_state:
        del st.session_state.mg_start_year
    if 'mg_end_year' in st.session_state:
        del st.session_state.mg_end_year
