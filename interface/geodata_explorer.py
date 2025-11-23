"""
GeoData Explorer Interface Module
Handles the complete interface for the GeoData Explorer tool
"""

import streamlit as st
import pandas as pd
import json
import time
import os
import ee
import numpy as np
import geemap.foliumap as geemap
from pathlib import Path
from datetime import datetime, timedelta

# Import core components
from geoclimate_fetcher.core import (
    authenticate,
    MetadataCatalog,
    GeometryHandler,
    GEEExporter,
    ImageCollectionFetcher,
    StaticRasterFetcher,
    GeometrySelectionWidget
)

# Import post-download integration
from app_components.post_download_integration import (
    get_download_handler,
    register_csv_download,
    register_tiff_download,
    render_post_download_integration
)

# Import smart download components
from app_components.download_component import DownloadHelper


def render_geodata_explorer():
    """Render the complete GeoData Explorer interface"""

    # App title and header
    st.markdown('<h1 class="main-title">🔍 GeoData Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Download and visualize Earth Engine climate datasets")
    
    # Initialize session state for post-download persistence
    if 'post_download_active' not in st.session_state:
        st.session_state.post_download_active = False
    if 'post_download_results' not in st.session_state:
        st.session_state.post_download_results = []

    # Initialize core objects with proper error handling
    try:
        with st.spinner("🔄 Initializing Data Explorer components..."):
            metadata_catalog = MetadataCatalog()
            exporter = GEEExporter()
        st.success("✅ Data Explorer initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing Data Explorer: {str(e)}")
        st.info("💡 You can still proceed with manual dataset selection")
        metadata_catalog = None
        exporter = None
    
    # Progress indicator
    def show_progress_indicator():
        """Display a visual progress indicator showing current step"""
        steps = [
            ("🗺️", "Area", st.session_state.geometry_complete),
            ("📊", "Dataset", st.session_state.dataset_selected),
            ("🎛️", "Bands", st.session_state.bands_selected),
            ("📅", "Dates", st.session_state.dates_selected),
            ("💾", "Download", False)  # Never complete until download finishes
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
                elif i == len([s for s in steps[:i] if s[2]]):  # Current step
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

    # Show post-download integration if active (takes priority over workflow)
    if st.session_state.post_download_active and st.session_state.post_download_results:
        st.markdown("### 🎉 Download Complete!")
        st.success("✅ Your data has been successfully downloaded!")

        # Show download summary
        total_files = len(st.session_state.post_download_results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Files Downloaded", total_files)
        with col2:
            latest_result = st.session_state.post_download_results[-1] if st.session_state.post_download_results else {}
            file_type = latest_result.get('data_type', 'Unknown').upper()
            st.metric("📊 File Type", file_type)
        with col3:
            st.metric("⏰ Status", "Ready to Visualize")

        # Render persistent post-download integration
        render_post_download_integration("geodata_explorer", st.session_state.post_download_results)

        # Add option to start new download
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Download", type="secondary", use_container_width=True):
                # Clear post-download state and reset workflow
                st.session_state.post_download_active = False
                st.session_state.post_download_results = []
                _reset_all_selections()
                st.rerun()
        with col2:
            if st.button("❌ Dismiss Results", type="secondary", use_container_width=True):
                # Just clear post-download state, keep workflow state
                st.session_state.post_download_active = False
                st.session_state.post_download_results = []
                st.rerun()

        # Stop here - don't show the workflow steps when post-download is active
        return

    # Show progress if any step is started
    if any([st.session_state.auth_complete, st.session_state.geometry_complete,
            st.session_state.dataset_selected, st.session_state.bands_selected,
            st.session_state.dates_selected]):
        st.markdown('<div class="progress-steps">', unsafe_allow_html=True)
        show_progress_indicator()
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 1: Area of Interest Selection
    if not st.session_state.geometry_complete:
        def on_geometry_selected(geometry):
            """Callback when geometry is selected"""
            st.session_state.geometry_handler._current_geometry = geometry
            st.session_state.geometry_handler._current_geometry_name = "selected_aoi"
            st.session_state.geometry_complete = True
            st.success("✅ Area of interest selected successfully!")
        
        # Use the unified geometry selection widget
        geometry_widget = GeometrySelectionWidget(
            session_prefix="geodata_",
            title="📍 Step 1: Select Area of Interest"
        )
        
        if geometry_widget.render_complete_interface(on_geometry_selected=on_geometry_selected):
            st.rerun()
    
    # Step 2: Dataset Selection
    elif not st.session_state.dataset_selected:
        _render_dataset_selection()
    
    # Step 3: Band Selection
    elif not st.session_state.bands_selected:
        _render_band_selection()
    
    # Step 4: Date Range Selection
    elif not st.session_state.dates_selected:
        _render_date_selection()

    # Step 5: Interactive Preview and Download
    else:
        # Show geemap preview first
        _render_geemap_preview()

        # Show download interface below preview
        st.markdown("---")
        _render_download_interface()


def _render_dataset_selection():
    """Render dataset selection interface"""
    from app_utils import go_back_to_step, get_bands_for_dataset
    
    st.markdown('<div class="step-header"><h2>📊 Step 2: Select Dataset</h2></div>', unsafe_allow_html=True)
    
    # Add back button
    if st.button("← Back to Area of Interest"):
        go_back_to_step("geometry")
    
    # Dataset search and filtering
    search_term = st.text_input("🔍 Search datasets:", placeholder="Enter keywords (e.g., precipitation, temperature, MODIS)")
    
    # Category filters
    category_filter = st.multiselect(
        "Filter by categories:",
        ["Climate", "Weather", "Land Cover", "Vegetation", "Ocean", "Atmosphere", "Terrain", "Land", "Other"],
        default=[]
    )
    
    # Time period filter
    time_filter = st.selectbox(
        "Filter by temporal resolution:",
        ["All", "Daily", "hourly", "Monthly", "Yearly", "Annual", "Static", "Other"]
    )
    
    # Load datasets - try multiple methods
    datasets = []
    
    with st.spinner("🔄 Loading dataset catalog..."):
        # Method 1: Load from geoclimate_fetcher data directory
        # Get project root directory first
        current_dir = Path(__file__).parent.parent  # Go up from interface/ to project root
        data_dir = current_dir / "geoclimate_fetcher" / "data"
        datasets_csv = data_dir / "Datasets.csv"
        
        if datasets_csv.exists():
            try:
                df = pd.read_csv(datasets_csv)
                # Transform CSV columns to expected format
                datasets = []
                for _, row in df.iterrows():
                    dataset = {
                        'name': row.get('Dataset Name', 'Unknown'),
                        'ee_id': row.get('Earth Engine ID', ''),
                        'snippet_type': row.get('Snippet Type', 'Image').strip() if pd.notna(row.get('Snippet Type')) else 'Image',  # Critical for processing logic
                        'description': row.get('Description', 'No description available'),
                        'category': _extract_category(row.get('Dataset Name', '')),
                        'temporal_resolution': row.get('Temporal Resolution', 'Unknown'),
                        'provider': row.get('Provider', 'Unknown'),
                        'start_date': row.get('Start Date', ''),
                        'end_date': row.get('End Date', ''),
                        'pixel_size': row.get('Pixel Size (m)', ''),
                        'band_names': row.get('Band Names', ''),
                        'band_units': row.get('Band Units', '')
                    }
                    datasets.append(dataset)
                st.success(f"📂 Loaded {len(datasets)} datasets from catalog")
            except Exception as e:
                st.warning(f"⚠️ Could not load catalog: {str(e)}")
        else:
            st.warning("⚠️ Dataset catalog not found. Using fallback datasets.")
    
    # Fallback datasets if catalog loading fails
    if not datasets:
        datasets = _get_fallback_datasets()
        st.info(f"📋 Using {len(datasets)} fallback datasets")
    
    # Apply filters
    filtered_datasets = _apply_filters(datasets, search_term, category_filter, time_filter)
    
    if not filtered_datasets:
        st.warning("⚠️ No datasets match your filters. Try adjusting your search criteria.")
        return
    
    st.info(f"📊 Found {len(filtered_datasets)} datasets matching your criteria")
    
    # Display datasets
    _display_dataset_options(filtered_datasets)


def _render_band_selection():
    """Render band selection interface"""
    from app_utils import go_back_to_step, get_bands_for_dataset
    
    st.markdown('<div class="step-header"><h2>🎛️ Step 3: Select Bands</h2></div>', unsafe_allow_html=True)
    
    # Add back button
    if st.button("← Back to Dataset Selection"):
        go_back_to_step("dataset")
    
    # Show selected dataset info
    selected_dataset = st.session_state.selected_dataset
    st.success(f"✅ Selected Dataset: **{selected_dataset.get('name', 'Unknown')}**")
    
    # Get available bands
    bands = get_bands_for_dataset(selected_dataset.get('name', ''))
    
    if bands:
        st.markdown("### Available Bands:")
        
        # Band selection interface
        selected_bands = st.multiselect(
            "Choose bands to download:",
            bands,
            default=bands[:3] if len(bands) >= 3 else bands,  # Select first 3 by default
            help="Select the spectral bands or variables you want to download"
        )
        
        if selected_bands:
            st.success(f"✅ Selected {len(selected_bands)} band(s)")
            
            # Show band details if available
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Selected Bands:**")
                for band in selected_bands:
                    st.markdown(f"• {band}")
            
            with col2:
                if 'description' in selected_dataset:
                    st.markdown("**Dataset Description:**")
                    st.markdown(selected_dataset['description'])
            
            if st.button("Continue to Date Selection", type="primary"):
                st.session_state.selected_bands = selected_bands
                st.session_state.bands_selected = True
                st.rerun()
        else:
            st.warning("⚠️ Please select at least one band to continue")
    else:
        st.warning("⚠️ No bands found for this dataset. This might be a single-band dataset.")
        if st.button("Continue with Default Band", type="primary"):
            st.session_state.selected_bands = ["default"]
            st.session_state.bands_selected = True
            st.rerun()


def _render_date_selection():
    """Render date range selection interface"""
    from app_utils import go_back_to_step

    st.markdown('<div class="step-header"><h2>📅 Step 4: Select Date Range</h2></div>', unsafe_allow_html=True)

    # Add back button
    if st.button("← Back to Band Selection"):
        go_back_to_step("bands")

    # Show current selections
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Dataset: **{st.session_state.selected_dataset.get('name', 'Unknown')}**")
    with col2:
        st.success(f"✅ Bands: **{len(st.session_state.selected_bands)} selected**")

    # Get dataset availability dates
    selected_dataset = st.session_state.selected_dataset
    dataset_start_date, dataset_end_date = _parse_dataset_dates(
        selected_dataset.get('start_date', ''),
        selected_dataset.get('end_date', '')
    )

    if not dataset_start_date or not dataset_end_date:
        st.error("❌ Unable to determine data availability dates for this dataset")
        st.info("Please contact support or try a different dataset")
        return

    # Show data availability
    st.markdown("### 📊 Data Availability:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Start:** {dataset_start_date}")
    with col2:
        st.info(f"**End:** {dataset_end_date}")
    with col3:
        total_days = (dataset_end_date - dataset_start_date).days
        st.info(f"**Duration:** {total_days} days")

    st.markdown("### Select Time Period:")

    # Calculate preset options based on data availability
    today = datetime.now().date()
    max_end_date = min(dataset_end_date, today)

    preset_options = []
    preset_dates = {}

    # Last 30 days (if data allows)
    if (max_end_date - timedelta(days=30)) >= dataset_start_date:
        preset_options.append("Last 30 days")
        preset_dates["Last 30 days"] = (max_end_date - timedelta(days=30), max_end_date)

    # Last 3 months (if data allows)
    if (max_end_date - timedelta(days=90)) >= dataset_start_date:
        preset_options.append("Last 3 months")
        preset_dates["Last 3 months"] = (max_end_date - timedelta(days=90), max_end_date)

    # Last year (if data allows)
    if (max_end_date - timedelta(days=365)) >= dataset_start_date:
        preset_options.append("Last year")
        preset_dates["Last year"] = (max_end_date - timedelta(days=365), max_end_date)

    # Full available period
    preset_options.append("Full available period")
    preset_dates["Full available period"] = (dataset_start_date, dataset_end_date)

    # Custom range is always available
    preset_options.append("Custom range")

    # Date range selection
    date_option = st.radio(
        "Choose date range:",
        preset_options,
        horizontal=True
    )

    if date_option == "Custom range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max(dataset_start_date, max_end_date - timedelta(days=30)),
                min_value=dataset_start_date,
                max_value=dataset_end_date,
                help=f"Available from {dataset_start_date} to {dataset_end_date}"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_end_date,
                min_value=dataset_start_date,
                max_value=dataset_end_date,
                help=f"Available from {dataset_start_date} to {dataset_end_date}"
            )
    else:
        start_date, end_date = preset_dates[date_option]

    # Validate date range
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        return

    # Validate against dataset availability
    if start_date < dataset_start_date:
        st.error(f"❌ Start date cannot be before {dataset_start_date} (dataset start)")
        return

    if end_date > dataset_end_date:
        st.error(f"❌ End date cannot be after {dataset_end_date} (dataset end)")
        return

    # Show selected range
    days_diff = (end_date - start_date).days
    st.success(f"📅 Selected range: **{start_date}** to **{end_date}** ({days_diff} days)")

    # Show data coverage
    total_available_days = (dataset_end_date - dataset_start_date).days
    coverage_percent = (days_diff / total_available_days) * 100
    st.info(f"📊 This covers {coverage_percent:.1f}% of the available data period")

    # Warning for large date ranges
    if days_diff > 365:
        st.warning("⚠️ Large date ranges may take longer to process and result in larger files")
    elif days_diff > 1825:  # 5 years
        st.warning("🚨 Very large date range! Consider splitting into smaller periods for better performance")

    if st.button("Continue to Download", type="primary"):
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.dates_selected = True
        st.rerun()


def _parse_dataset_dates(start_date_str, end_date_str):
    """Parse dataset start and end dates from CSV format"""
    try:
        # Handle common date formats from the CSV
        date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y']

        start_date = None
        end_date = None

        # Parse start date
        if start_date_str and start_date_str.strip():
            for date_format in date_formats:
                try:
                    start_date = datetime.strptime(start_date_str.strip(), date_format).date()
                    break
                except ValueError:
                    continue

        # Parse end date
        if end_date_str and end_date_str.strip():
            for date_format in date_formats:
                try:
                    end_date = datetime.strptime(end_date_str.strip(), date_format).date()
                    break
                except ValueError:
                    continue

        return start_date, end_date

    except Exception as e:
        print(f"Error parsing dates: {e}")
        return None, None


# =======================================================================================
# GEEMAP PREVIEW FUNCTIONS - Adaptive visualization based on temporal resolution
# =======================================================================================

# Band visualization configurations - auto-detect appropriate palettes
BAND_VIS_CONFIGS = {
    # Temperature bands (Celsius/Kelvin)
    'temperature': {
        'keywords': ['temp', 'tmax', 'tmin', 'tmean', 'tavg', 'sst', 'lst', 'air_temperature',
                     'skin_temperature', 'surface_temperature', 't2m', 'temperature_2m'],
        'palette': ['blue', 'cyan', 'yellow', 'orange', 'red'],
        'unit': '°C',
        'description': 'Temperature'
    },
    # Precipitation bands (mm)
    'precipitation': {
        'keywords': ['prec', 'prcp', 'rain', 'precipitation', 'tp', 'total_precipitation',
                     'rainfall', 'precip', 'pr'],
        'palette': ['white', 'lightblue', 'blue', 'darkblue', 'purple'],
        'unit': 'mm',
        'description': 'Precipitation'
    },
    # NDVI and vegetation indices
    'vegetation': {
        'keywords': ['ndvi', 'evi', 'evi2', 'savi', 'lai', 'fpar', 'vegetation', 'greenness',
                     'normalized_difference', 'modis_ndvi', 'ndvi_', 'evi_'],
        'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen'],
        'unit': 'index',
        'description': 'Vegetation Index'
    },
    # Soil moisture
    'soil_moisture': {
        'keywords': ['soil', 'moisture', 'sm', 'volumetric', 'swvl', 'soil_moisture'],
        'palette': ['red', 'orange', 'yellow', 'lightblue', 'blue'],
        'unit': 'm³/m³',
        'description': 'Soil Moisture'
    },
    # Snow cover
    'snow': {
        'keywords': ['snow', 'snowfall', 'snow_cover', 'swe', 'snow_depth', 'snw', 'sd'],
        'palette': ['brown', 'white', 'cyan', 'blue'],
        'unit': 'mm',
        'description': 'Snow'
    },
    # Solar radiation
    'radiation': {
        'keywords': ['rad', 'srad', 'radiation', 'shortwave', 'longwave', 'solar', 'ssrd',
                     'surface_solar', 'downward'],
        'palette': ['black', 'purple', 'red', 'orange', 'yellow', 'white'],
        'unit': 'W/m²',
        'description': 'Radiation'
    },
    # Humidity
    'humidity': {
        'keywords': ['humid', 'rh', 'relative_humidity', 'specific_humidity', 'vap', 'vapor'],
        'palette': ['red', 'yellow', 'green', 'cyan', 'blue'],
        'unit': '%',
        'description': 'Humidity'
    },
    # Wind speed
    'wind': {
        'keywords': ['wind', 'u10', 'v10', 'wind_speed', 'windspeed', 'ws'],
        'palette': ['white', 'lightblue', 'blue', 'purple', 'red'],
        'unit': 'm/s',
        'description': 'Wind Speed'
    },
    # Elevation/DEM
    'elevation': {
        'keywords': ['elevation', 'dem', 'altitude', 'height', 'srtm', 'aster'],
        'palette': ['green', 'yellow', 'brown', 'white'],
        'unit': 'm',
        'description': 'Elevation'
    },
    # Default fallback
    'default': {
        'keywords': [],
        'palette': ['blue', 'cyan', 'yellow', 'orange', 'red'],
        'unit': 'value',
        'description': 'Data'
    }
}


def _get_vis_params_for_band(band_name, image, geometry):
    """
    Automatically determine appropriate visualization parameters for a band

    Args:
        band_name: Name of the band
        image: Earth Engine image to analyze
        geometry: Region to calculate statistics

    Returns:
        Dictionary with palette, unit, description
    """
    band_lower = band_name.lower()

    # Find matching configuration
    config = None
    for key, cfg in BAND_VIS_CONFIGS.items():
        if key == 'default':
            continue
        if any(keyword in band_lower for keyword in cfg['keywords']):
            config = cfg
            break

    # Use default if no match
    if not config:
        config = BAND_VIS_CONFIGS['default']

    return {
        'palette': config['palette'],
        'unit': config['unit'],
        'description': config['description']
    }


def _determine_preview_strategy(collection, temporal_resolution, num_images, days_span):
    """
    Determine the best preview strategy based on temporal resolution and image count

    Args:
        collection: Earth Engine ImageCollection
        temporal_resolution: String like 'Hourly', 'Daily', 'Monthly', 'Yearly', 'Static'
        num_images: Number of images in collection
        days_span: Number of days in date range

    Returns:
        Dictionary with strategy details:
        {
            'type': 'aggregate' | 'sample' | 'direct',
            'method': specific method to use,
            'max_layers': maximum layers to display,
            'description': user-facing description
        }
    """
    # Static data - show directly
    if temporal_resolution == 'Static' or num_images == 1:
        return {
            'type': 'direct',
            'method': 'single_image',
            'max_layers': 1,
            'description': 'Displaying static image'
        }

    # Determine temporal category
    temporal_lower = temporal_resolution.lower()

    # SUB-DAILY: Hourly, 30-minute, 3-hourly data
    if any(x in temporal_lower for x in ['hour', 'minute']):
        if num_images <= 100:
            return {
                'type': 'direct',
                'method': 'all_images',
                'max_layers': num_images,
                'description': f'Displaying all {num_images} images with layer toggles'
            }
        else:
            # Offer aggregation options
            return {
                'type': 'aggregate',
                'method': 'to_daily',  # Aggregate sub-daily to daily
                'max_layers': min(days_span, 100),
                'description': f'Aggregating {num_images} hourly images to daily composites'
            }

    # DAILY data
    elif 'daily' in temporal_lower or 'day' in temporal_lower:
        if num_images <= 100:
            return {
                'type': 'direct',
                'method': 'all_images',
                'max_layers': num_images,
                'description': f'Displaying all {num_images} daily images'
            }
        elif num_images <= 400:
            # Aggregate to weekly
            return {
                'type': 'aggregate',
                'method': 'to_weekly',
                'max_layers': min(num_images // 7, 100),
                'description': f'Aggregating to weekly composites ({num_images} daily images)'
            }
        else:
            # Aggregate to monthly
            return {
                'type': 'aggregate',
                'method': 'to_monthly',
                'max_layers': min(num_images // 30, 100),
                'description': f'Aggregating to monthly composites ({num_images} daily images)'
            }

    # MONTHLY data
    elif 'month' in temporal_lower:
        if num_images <= 100:
            return {
                'type': 'direct',
                'method': 'all_images',
                'max_layers': num_images,
                'description': f'Displaying all {num_images} monthly images'
            }
        else:
            # Cannot aggregate monthly to daily - use sampling
            return {
                'type': 'sample',
                'method': 'evenly_spaced',
                'max_layers': 100,
                'description': f'Sampling 100 images from {num_images} monthly images'
            }

    # YEARLY or other coarse data
    elif 'year' in temporal_lower:
        return {
            'type': 'direct',
            'method': 'all_images',
            'max_layers': min(num_images, 100),
            'description': f'Displaying {min(num_images, 100)} yearly images'
        }

    # Default: if we have <= 100 images, show all; otherwise sample
    else:
        if num_images <= 100:
            return {
                'type': 'direct',
                'method': 'all_images',
                'max_layers': num_images,
                'description': f'Displaying all {num_images} images'
            }
        else:
            return {
                'type': 'sample',
                'method': 'evenly_spaced',
                'max_layers': 100,
                'description': f'Sampling 100 images from {num_images} total'
            }


def _aggregate_to_daily(collection):
    """Aggregate sub-daily data (hourly, 30-min) to daily means"""
    def aggregate_day(date):
        date = ee.Date(date)
        daily = collection.filterDate(date, date.advance(1, 'day')).mean()
        return daily.set('system:time_start', date.millis())

    # Get unique dates
    dates = collection.aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).format('YYYY-MM-dd')
    ).distinct()

    return ee.ImageCollection(dates.map(aggregate_day))


def _aggregate_to_weekly(collection):
    """Aggregate daily data to weekly means"""
    def aggregate_week(date):
        date = ee.Date(date)
        weekly = collection.filterDate(date, date.advance(7, 'day')).mean()
        return weekly.set('system:time_start', date.millis())

    # Get start of each week
    start_date = ee.Date(collection.first().get('system:time_start'))
    end_date = ee.Date(collection.sort('system:time_start', False).first().get('system:time_start'))

    days_diff = end_date.difference(start_date, 'day')
    weeks = ee.List.sequence(0, days_diff, 7)

    weekly_dates = weeks.map(lambda d: start_date.advance(d, 'day'))

    return ee.ImageCollection(weekly_dates.map(aggregate_week))


def _aggregate_to_monthly(collection):
    """Aggregate daily data to monthly means"""
    def aggregate_month(date):
        date = ee.Date(date)
        year = date.get('year')
        month = date.get('month')

        monthly = collection.filter(
            ee.Filter.calendarRange(year, year, 'year')
        ).filter(
            ee.Filter.calendarRange(month, month, 'month')
        ).mean()

        return monthly.set({
            'system:time_start': date.millis(),
            'year': year,
            'month': month
        })

    # Get unique year-month combinations
    def get_month_string(img):
        date = ee.Date(img.get('system:time_start'))
        return date.format('YYYY-MM')

    months = collection.map(lambda img: ee.Feature(None, {'month': get_month_string(img)})) \
        .distinct('month') \
        .aggregate_array('month')

    monthly_dates = months.map(lambda m: ee.Date.parse('YYYY-MM', m))

    return ee.ImageCollection(monthly_dates.map(aggregate_month))


def _aggregate_to_yearly(collection):
    """Aggregate monthly data to yearly means"""
    def aggregate_year(year):
        year = ee.Number(year)
        yearly = collection.filter(ee.Filter.calendarRange(year, year, 'year')).mean()
        return yearly.set({
            'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
            'year': year
        })

    # Get unique years
    years = collection.map(lambda img: ee.Feature(None, {
        'year': ee.Date(img.get('system:time_start')).get('year')
    })).distinct('year').aggregate_array('year')

    return ee.ImageCollection(years.map(aggregate_year))


def _sample_evenly_spaced(collection, target_count=100):
    """Sample images evenly across the time range"""
    total_count = collection.size().getInfo()

    if total_count <= target_count:
        return collection

    # Calculate sampling interval
    interval = total_count // target_count

    # Get list of all images
    img_list = collection.toList(total_count)

    # Sample at intervals
    indices = ee.List.sequence(0, total_count - 1, interval).slice(0, target_count)
    sampled = indices.map(lambda i: img_list.get(i))

    return ee.ImageCollection(sampled)


def _render_geemap_preview():
    """
    Render interactive geemap preview with adaptive strategy
    Called after date selection, before download interface
    """
    st.markdown("---")
    st.markdown("### 🗺️ Interactive Map Preview")
    st.info("💡 Preview your data before downloading. Use layer controls to toggle between time periods.")

    # Ensure Earth Engine is initialized
    try:
        if not ee.data._initialized:
            ee.Initialize()
    except:
        # For newer EE versions, just try to initialize
        try:
            ee.Initialize()
        except Exception as init_error:
            st.error(f"❌ Earth Engine initialization failed: {str(init_error)}")
            st.info("💡 Please check your Earth Engine authentication.")
            return

    # Get all selections from session state
    geometry = st.session_state.geometry_handler.current_geometry
    selected_dataset = st.session_state.selected_dataset
    selected_bands = st.session_state.selected_bands
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    temporal_resolution = selected_dataset.get('temporal_resolution', 'Daily')

    # Check if static or temporal dataset
    is_static = temporal_resolution == 'Static'

    try:
        # Fetch the image or image collection
        with st.spinner("🔄 Loading data from Earth Engine..."):
            if is_static:
                # Use StaticRasterFetcher
                fetcher = StaticRasterFetcher(
                    ee_id=selected_dataset['ee_id'],
                    bands=selected_bands,
                    geometry=geometry
                )
                # CRITICAL: Clip to user's selected geometry!
                image = fetcher.image.clip(geometry)
                collection = None
                num_images = 1
            else:
                # Use ImageCollectionFetcher
                fetcher = ImageCollectionFetcher(
                    ee_id=selected_dataset['ee_id'],
                    bands=selected_bands,
                    geometry=geometry
                )
                # Apply date filter
                fetcher = fetcher.filter_dates(
                    start_date=start_date,
                    end_date=end_date
                )
                # CRITICAL: Clip entire collection to user's selected geometry!
                collection = fetcher.collection.map(lambda img: img.clip(geometry))
                num_images = collection.size().getInfo()
                image = None

        # Calculate days span
        days_span = (end_date - start_date).days

        # Show data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Temporal Resolution", temporal_resolution)
        with col2:
            if is_static:
                st.metric("🖼️ Images", "1 (Static)")
            else:
                st.metric("🖼️ Total Images", num_images)
        with col3:
            st.metric("📅 Date Range", f"{days_span} days")

        # Determine preview strategy
        if is_static:
            strategy = _determine_preview_strategy(None, 'Static', 1, 0)
        else:
            strategy = _determine_preview_strategy(collection, temporal_resolution, num_images, days_span)

        # Show strategy to user
        st.info(f"📋 **Preview Strategy:** {strategy['description']}")

        # Apply strategy
        if not is_static:
            if strategy['type'] == 'aggregate':
                with st.spinner(f"⚙️ Aggregating data using {strategy['method']}..."):
                    if strategy['method'] == 'to_daily':
                        collection = _aggregate_to_daily(collection)
                    elif strategy['method'] == 'to_weekly':
                        collection = _aggregate_to_weekly(collection)
                    elif strategy['method'] == 'to_monthly':
                        collection = _aggregate_to_monthly(collection)
                    elif strategy['method'] == 'to_yearly':
                        collection = _aggregate_to_yearly(collection)

                    # Update image count after aggregation
                    num_images = collection.size().getInfo()
                    st.success(f"✅ Aggregated to {num_images} composite images")

            elif strategy['type'] == 'sample':
                with st.spinner(f"⚙️ Sampling {strategy['max_layers']} images..."):
                    collection = _sample_evenly_spaced(collection, strategy['max_layers'])
                    num_images = collection.size().getInfo()
                    st.success(f"✅ Sampled {num_images} representative images")

        # Create geemap Map with error handling
        try:
            Map = geemap.Map()
        except AttributeError as e:
            # Handle geemap version compatibility issues
            st.error(f"❌ geemap Map creation failed: {str(e)}")
            st.warning("⚠️ This may be a version compatibility issue. Try updating geemap: `pip install --upgrade geemap`")
            return
        except Exception as e:
            st.error(f"❌ Error creating map: {str(e)}")
            return

        # Center map on geometry
        try:
            centroid = geometry.centroid().getInfo()['coordinates']
            Map.setCenter(centroid[0], centroid[1], 8)
        except:
            Map.setCenter(0, 0, 2)

        # Add geometry outline
        Map.addLayer(geometry, {'color': 'red'}, 'Study Area', True, 0.5)

        # Select band to visualize
        if len(selected_bands) > 1:
            selected_band = st.selectbox(
                "Select band to visualize:",
                selected_bands,
                help="Choose which band to display on the map"
            )
        else:
            selected_band = selected_bands[0]
            st.info(f"**Visualizing band:** {selected_band}")

        # Get visualization parameters
        if is_static:
            vis_config = _get_vis_params_for_band(selected_band, image, geometry)

            # Calculate stats for static image
            stats = image.select(selected_band).reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=geometry,
                scale=1000,
                maxPixels=1e8
            ).getInfo()

            vmin = stats.get(f'{selected_band}_p5', 0)
            vmax = stats.get(f'{selected_band}_p95', 100)

            vis_params = {
                'bands': [selected_band],
                'min': vmin,
                'max': vmax,
                'palette': vis_config['palette']
            }

            # Add static layer
            Map.addLayer(image, vis_params, f"{selected_band}", True)

            # Add colorbar
            colorbar_vis = {'min': vmin, 'max': vmax, 'palette': vis_config['palette']}
            Map.add_colorbar(
                vis_params=colorbar_vis,
                label=f"{selected_band} ({vis_config['unit']})",
                position='bottomright'
            )

            st.markdown(f"**📊 Value Range:** {vmin:.2f} to {vmax:.2f} {vis_config['unit']} (5th-95th percentile)")

        else:
            # TEMPORAL COLLECTION
            # Get first image for band config
            first_image = ee.Image(collection.first())
            vis_config = _get_vis_params_for_band(selected_band, first_image, geometry)

            # Calculate percentile-based range across ALL images for colorbar
            with st.spinner("📊 Calculating value range for colorbar..."):
                all_min_values = []
                all_max_values = []

                # Sample up to 20 images for statistics (performance optimization)
                sample_size = min(num_images, 20)
                sample_collection = _sample_evenly_spaced(collection, sample_size)
                sample_list = sample_collection.toList(sample_size)

                for i in range(sample_size):
                    img = ee.Image(sample_list.get(i))
                    stats = img.select(selected_band).reduceRegion(
                        reducer=ee.Reducer.minMax(),
                        geometry=geometry,
                        scale=5000,
                        maxPixels=1e8
                    ).getInfo()

                    vmin = stats.get(f'{selected_band}_min', 0)
                    vmax = stats.get(f'{selected_band}_max', 100)
                    all_min_values.append(vmin)
                    all_max_values.append(vmax)

                # Calculate percentile range
                overall_min = float(np.percentile(all_min_values, 5))
                overall_max = float(np.percentile(all_max_values, 95))

            # Add all images as layers
            show_all = st.checkbox("Show all layers at once", value=False,
                                   help="Warning: This may make the map slower")

            img_list = collection.toList(num_images)

            with st.spinner(f"🗺️ Adding {num_images} layers to map..."):
                for i in range(num_images):
                    img = ee.Image(img_list.get(i))

                    # Get date for layer name
                    try:
                        date_millis = img.get('system:time_start').getInfo()
                        date_obj = datetime.fromtimestamp(date_millis / 1000)
                        date_str = date_obj.strftime('%Y-%m-%d')
                    except:
                        date_str = f"Image {i+1}"

                    # Use consistent vis params
                    vis_params = {
                        'bands': [selected_band],
                        'min': overall_min,
                        'max': overall_max,
                        'palette': vis_config['palette']
                    }

                    # Show only last layer by default (unless show_all is checked)
                    show_layer = show_all or (i == num_images - 1)

                    Map.addLayer(img, vis_params, f"{selected_band} - {date_str}", show_layer)

            # Add colorbar
            colorbar_vis = {'min': overall_min, 'max': overall_max, 'palette': vis_config['palette']}
            Map.add_colorbar(
                vis_params=colorbar_vis,
                label=f"{selected_band} ({vis_config['unit']})",
                position='bottomright'
            )

            st.markdown(f"**📊 Colorbar Range:** {overall_min:.2f} to {overall_max:.2f} {vis_config['unit']} (5th-95th percentile)")

        # Display the map
        Map.to_streamlit(height=600)

        st.success("✅ Preview loaded! Use the layer control (top-right) to toggle between dates.")

    except Exception as e:
        st.error(f"❌ Error creating preview: {str(e)}")
        st.info("💡 You can still proceed to download the data.")
        import traceback
        st.code(traceback.format_exc())


def _render_download_interface():
    """Render download and export interface below the preview"""
    from app_utils import go_back_to_step

    # Check if download is already complete and show persistent results
    if st.session_state.get('download_complete', False) and st.session_state.get('download_results'):
        _render_download_results_interface()
        return

    # Header
    st.markdown("### 💾 Download Data")
    st.info("💡 Download the data for offline analysis or further processing in your preferred format.")

    # Show summary of selections
    st.markdown("#### 📋 Download Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Dataset:**")
        st.info(st.session_state.selected_dataset.get('name', 'Unknown'))
        
        st.markdown("**🎛️ Bands:**")
        for band in st.session_state.selected_bands:
            st.info(f"• {band}")
    
    with col2:
        st.markdown("**🗺️ Area:**")
        try:
            area = st.session_state.geometry_handler.get_geometry_area()
            st.info(f"{area:.2f} km²")
        except:
            st.info("Custom area selected")
        
        st.markdown("**📅 Time Period:**")
        days = (st.session_state.end_date - st.session_state.start_date).days
        st.info(f"{st.session_state.start_date} to {st.session_state.end_date}")
        st.info(f"({days} days)")
    
    with col3:
        st.markdown("**📁 Export Options:**")
        
        # Export format selection
        export_format = st.selectbox(
            "Choose format:",
            ["CSV", "GeoTIFF", "NetCDF"],
            help="• CSV: Time series with area-averaged values (daily/hourly data with date column)\n"
                 "• GeoTIFF: Individual raster files with time info (≤5 images) or temporal composite (>5 images)\n"
                 "• NetCDF: Multi-dimensional dataset with proper time axis (requires xarray)"
        )

        # Add performance hints based on dataset and format selection
        if export_format == 'CSV':
            temporal_resolution = st.session_state.selected_dataset.get('temporal_resolution', 'Daily')
            days = (st.session_state.end_date - st.session_state.start_date).days

            # Estimate total records for performance prediction
            if temporal_resolution == '30-minute':
                estimated_records = days * 48  # 48 records per day
            elif temporal_resolution == 'Hourly':
                estimated_records = days * 24
            elif temporal_resolution == '3-hourly':
                estimated_records = days * 8
            elif temporal_resolution == 'Daily':
                estimated_records = days
            elif temporal_resolution == '2-Day':
                estimated_records = days // 2
            elif temporal_resolution == '6-Day':
                estimated_records = days // 6
            elif temporal_resolution == '8-Day':
                estimated_records = days // 8
            elif temporal_resolution == '16-Day':
                estimated_records = days // 16
            elif temporal_resolution == 'Monthly':
                estimated_records = days // 30
            else:
                estimated_records = days

            # Performance estimates
            if estimated_records < 1000:
                performance_hint = "🚀 **Very Fast** (~10-30 seconds)"
                hint_color = "success"
            elif estimated_records < 5000:
                performance_hint = "⚡ **Fast** (~30-90 seconds)"
                hint_color = "success"
            elif estimated_records < 20000:
                performance_hint = "⏱️ **Moderate** (~2-5 minutes)"
                hint_color = "info"
            elif estimated_records < 50000:
                performance_hint = "🐌 **Slow** (~5-15 minutes)"
                hint_color = "warning"
            else:
                performance_hint = "🐌 **Very Slow** (>15 minutes)"
                hint_color = "warning"

            # Performance hint removed as requested

            # Optimization tips for slow downloads
            if estimated_records > 20000:
                st.info("💡 **Optimization Tips**: CSV downloads use 10km scale automatically for maximum performance. Consider shorter date ranges for sub-daily data.")

        # Get dataset pixel size for all formats
        dataset_pixel_size = st.session_state.selected_dataset.get('pixel_size')

        # Scale selection for both GeoTIFF and CSV formats
        st.markdown("---")
        if export_format == 'GeoTIFF':
            st.markdown("**🎯 Spatial Resolution (GeoTIFF):**")
        else:  # CSV
            st.markdown("**🎯 Spatial Resolution (CSV Area Averaging):**")

        # Get native resolution
        native_resolution = None
        if dataset_pixel_size:
            try:
                native_resolution = int(float(dataset_pixel_size))
            except (ValueError, TypeError):
                native_resolution = 100  # fallback

        # Format-specific guidance
        if export_format == 'CSV':
            st.info("💡 **CSV Performance Guide**: Higher scales (coarser resolution) process faster. For area-averaged CSV data, spatial detail has minimal impact unless analyzing very small areas.")

        # Scale input method selection
        scale_method = st.radio(
            "Scale selection method:",
            ["Preset values", "Custom value"],
            help="Choose preset values for common scales or enter your own custom scale"
        )

        if scale_method == "Preset values":
            # Create scale options with format-specific defaults
            if export_format == 'CSV':
                # CSV options emphasize performance
                base_options = [1000, 5000, 10000, 25000, 50000]
                recommended_default = 10000
            else:
                # GeoTIFF options emphasize spatial detail
                base_options = [30, 100, 250, 500, 1000, 5000, 10000]
                recommended_default = 100

            if native_resolution and native_resolution not in base_options:
                # Add native resolution to options and sort
                scale_options = sorted(base_options + [native_resolution])
            else:
                scale_options = base_options

            # Determine default index
            if export_format == 'CSV':
                # For CSV, default to 10km for performance
                default_index = scale_options.index(recommended_default) if recommended_default in scale_options else 2
                help_text = f"10km recommended for CSV (fast processing). Native resolution: {native_resolution}m" if native_resolution else "10km recommended for CSV (fast processing)"
            else:
                # For GeoTIFF, prefer native resolution
                if native_resolution and native_resolution in scale_options:
                    default_index = scale_options.index(native_resolution)
                    help_text = f"Dataset native resolution is {native_resolution}m (recommended)"
                else:
                    default_index = scale_options.index(100) if 100 in scale_options else 1
                    help_text = "100m is a good balance between detail and processing speed"

            scale = st.selectbox(
                "Spatial resolution:",
                scale_options,
                index=default_index,
                help=help_text,
                format_func=lambda x: f"{x}m ({'Fast' if x >= 10000 else 'Medium' if x >= 1000 else 'Detailed'})"
            )

        else:  # Custom value
            col1, col2 = st.columns([3, 1])
            with col1:
                if export_format == 'CSV':
                    default_custom = 10000
                    help_text = f"Enter scale in meters. For CSV: Higher values = faster processing. Native resolution: {native_resolution}m" if native_resolution else "Enter scale in meters. For CSV: Higher values = faster processing"
                else:
                    default_custom = native_resolution if native_resolution else 100
                    help_text = f"Enter scale in meters. Native resolution: {native_resolution}m" if native_resolution else "Enter scale in meters. Smaller values = higher detail but slower processing"

                scale = st.number_input(
                    "Custom scale (meters):",
                    min_value=10,
                    max_value=100000,
                    value=default_custom,
                    step=10,
                    help=help_text
                )
            with col2:
                if export_format == 'CSV':
                    if st.button("🚀 Fast (10km)", help="Set to 10km for fastest CSV processing"):
                        scale = 10000
                        st.rerun()
                else:
                    if native_resolution:
                        if st.button("Use Native", help=f"Set to dataset's native resolution ({native_resolution}m)"):
                            scale = native_resolution
                            st.rerun()

        # Format-specific performance and accuracy guidance
        if export_format == 'CSV':
            if scale >= 10000:
                st.success(f"🚀 **Fast Processing**: {scale}m scale optimized for CSV area averaging")
            elif scale >= 1000:
                st.warning(f"⚡ **Medium Speed**: {scale}m scale - moderate processing time")
            else:
                st.error(f"🐌 **Slow Processing**: {scale}m scale may be slow for large time series. Consider 10km+ for CSV area averaging")

            # Additional guidance for small areas
            st.caption("📍 **Note**: For very small study areas (<1km²), finer resolution may provide more accurate results despite longer processing time.")
        else:
            # GeoTIFF specific guidance about native resolution
            if dataset_pixel_size:
                try:
                    native_res = int(float(dataset_pixel_size))
                    if scale == native_res:
                        st.success(f"✅ Using native dataset resolution ({native_res}m)")
                    elif scale < native_res:
                        st.warning(f"⚠️ Selected resolution ({scale}m) is finer than native resolution ({native_res}m). This won't add detail but may increase processing time.")
                    else:
                        st.info(f"ℹ️ Selected resolution ({scale}m) is coarser than native resolution ({native_res}m). This will reduce file size and processing time.")
                except (ValueError, TypeError):
                    pass
    
    # Smart Download Interface
    st.markdown("---")

    # Initialize download helper
    download_helper = DownloadHelper()

    # Render smart download options with format-aware descriptions
    export_preference = download_helper.render_smart_download_options(
        export_format=export_format
    )

    # Download button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Start Smart Download", type="primary", use_container_width=True):
            # Route CSV downloads through optimized path
            if export_format == 'CSV':
                if export_preference == 'drive':
                    st.info("📊 Using fast CSV Drive export...")
                    _process_csv_drive_export(export_format, scale)
                else:
                    st.info("📊 Using optimized CSV processing...")
                    _process_smart_download(export_format, scale, export_preference)
            else:
                # For GeoTIFF format, use preference-based routing
                if export_preference == 'local':
                    st.info("🔄 Using reliable local download method...")
                    _process_download(export_format, scale)
                elif export_preference == 'auto':
                    # Try the reliable local method first, then fall back to Drive
                    st.info("🤖 Smart Auto: Trying local download first...")
                    try:
                        success = _process_download(export_format, scale)
                        if not success:
                            st.warning("Local download failed. Falling back to Google Drive...")
                            _process_smart_download(export_format, scale, 'drive')
                    except Exception as e:
                        st.warning(f"Local download failed: {str(e)}. Falling back to Google Drive...")
                        _process_smart_download(export_format, scale, 'drive')
                else:  # drive
                    _process_smart_download(export_format, scale, export_preference)
    
    # Reset button
    if st.button("🔄 Start Over", help="Clear all selections and start from the beginning"):
        _reset_all_selections()
        st.rerun()


def _render_download_results_interface():
    """Render persistent download results interface (similar to climate analytics)"""
    from app_utils import go_back_to_step

    results = st.session_state.download_results

    # Show success message
    st.success("✅ Download completed successfully!")
    st.info(results['success_message'])

    # Show download summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📁 File Format", results['export_format'])
        st.metric("📊 Dataset", results['dataset_name'])

    with col2:
        st.metric("💾 File Size", f"{results['file_size_mb']:.1f} MB")
        st.metric("🔍 Resolution", f"{results['scale']}m")

    with col3:
        st.metric("⏰ Downloaded", results['download_timestamp'])
        st.metric("📋 Status", "Ready to Use")

    st.markdown("---")

    # Main download button (re-download same file)
    st.markdown("### 📥 Download Options")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label=f"📥 Download {results['export_format']} ({results['file_size_mb']:.1f} MB)",
            data=results['file_data'],
            file_name=results['filename'],
            mime=results['mime_type'],
            type="primary",
            use_container_width=True,
            help="Download the same file again"
        )

    with col2:
        if st.button("📋 Download Different Format", use_container_width=True):
            # Clear download complete state to show format options
            st.session_state.download_complete = False
            st.session_state.download_results = None
            st.rerun()

    st.markdown("---")

    # Action buttons
    st.markdown("### 🔄 Next Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Go to Visualization", use_container_width=True, type="primary"):
            # Prepare data for visualization with error handling
            try:
                _launch_visualization_from_results()
            except Exception as e:
                # On error, go to visualizer without data (fallback to upload interface)
                st.warning(f"⚠️ Could not transfer data to visualizer: {str(e)}")
                st.info("💡 Redirecting to visualization module for manual upload...")
                st.session_state.direct_visualization_data = None
                st.session_state.app_mode = "data_visualizer"
                st.rerun()

    with col2:
        if st.button("🔄 Download Different Format/Resolution", use_container_width=True):
            # Go back to download configuration but keep selections
            st.session_state.download_complete = False
            st.session_state.download_results = None
            st.rerun()

    with col3:
        if st.button("📅 Change Date Range", use_container_width=True):
            # Go back to date selection
            st.session_state.download_complete = False
            st.session_state.download_results = None
            st.session_state.dates_selected = False
            st.rerun()

    with col4:
        if st.button("🆕 Start New Download", use_container_width=True):
            # Reset everything for new download
            st.session_state.download_complete = False
            st.session_state.download_results = None
            _reset_all_selections()
            st.rerun()


def _prepare_data_for_visualization(df, metadata=None):
    """Process DataFrame for visualization compatibility with all required metadata"""
    try:
        from app_components.data_processors import data_detector
        import pandas as pd
        import numpy as np

        # Initialize data detector
        detector = data_detector()

        # Generate column suggestions
        column_suggestions = detector.get_column_suggestions(df)

        # Create quality report
        quality_report = detector.validate_data_quality(df)

        # Create summary statistics
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
            'has_missing_values': df.isnull().any().any(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'date_columns': len(column_suggestions.get('date_columns', [])),
            'categorical_columns': len(column_suggestions.get('categorical_columns', []))
        }

        # Combine with provided metadata
        if metadata is None:
            metadata = {}

        return {
            'type': 'csv',
            'data': df,
            'column_suggestions': column_suggestions,
            'quality_report': quality_report,
            'summary': summary,
            'metadata': metadata
        }

    except Exception as e:
        # Fallback: create minimal required structure
        return {
            'type': 'csv',
            'data': df,
            'column_suggestions': {
                'date_columns': [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()],
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'text_columns': []
            },
            'quality_report': {
                'quality_score': 85,
                'issues': [],
                'suggestions': []
            },
            'summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            },
            'metadata': metadata or {}
        }


def _launch_early_visualization(result, dataset, export_format, scale):
    """Launch data visualizer with processing results immediately after completion"""
    try:
        # Prepare data for visualization based on format
        visualization_data = []

        if export_format == 'CSV':
            # For CSV data, convert file_data back to DataFrame and prepare with full metadata
            try:
                import pandas as pd
                import io
                from datetime import datetime

                # Convert file_data to DataFrame
                if isinstance(result['file_data'], bytes):
                    csv_string = result['file_data'].decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_string))
                elif isinstance(result['file_data'], str):
                    df = pd.read_csv(io.StringIO(result['file_data']))
                else:
                    # Assume it's already a DataFrame
                    df = result['file_data']

                # Prepare metadata with rich band information
                source_metadata = {
                    'dataset': dataset.get('name', 'Unknown'),
                    'export_format': export_format,
                    'resolution': f"{scale}m",
                    'size_mb': len(result['file_data']) / (1024 * 1024) if isinstance(result['file_data'], bytes) else 0,
                    'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'geodata_explorer_early',
                    'selected_bands': getattr(st.session_state, 'selected_bands', []),
                    'band_names': dataset.get('band_names', ''),
                    'band_units': dataset.get('band_units', ''),
                    'dataset_type': dataset.get('snippet_type', 'Unknown')
                }

                # Use helper function to prepare complete data structure
                prepared_data = _prepare_data_for_visualization(df, source_metadata)

                # Generate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                filename = f"{dataset_name}_{timestamp}.csv"

                visualization_data.append({
                    'file_name': filename,
                    'data': prepared_data['data'],
                    'data_type': 'csv',
                    'transfer_method': 'data_object',
                    'column_suggestions': prepared_data['column_suggestions'],
                    'quality_report': prepared_data['quality_report'],
                    'summary': prepared_data['summary'],
                    'metadata': prepared_data['metadata']
                })
            except Exception as e:
                st.error(f"❌ Error preparing CSV data for visualization: {str(e)}")
                raise e

        elif export_format in ['GeoTIFF', 'NetCDF']:
            # For spatial data, create a temporary file
            try:
                import tempfile
                from datetime import datetime

                # Check if the result indicates a ZIP file (multiple GeoTIFF files)
                if 'ZIP archive' in result.get('message', ''):
                    file_extension = '.zip'
                    data_type = 'zip'
                elif export_format == 'GeoTIFF':
                    file_extension = '.tif'
                    data_type = 'tiff'
                else:  # NetCDF
                    file_extension = '.nc'
                    data_type = 'netcdf'

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(result['file_data'])
                    temp_path = temp_file.name

                # Generate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                filename = f"{dataset_name}_{timestamp}{file_extension}"

                visualization_data.append({
                    'file_name': filename,
                    'data_type': data_type,
                    'file_path': temp_path,
                    'transfer_method': 'temp_file',
                    'metadata': {
                        'dataset': dataset.get('name', 'Unknown'),
                        'export_format': export_format,
                        'resolution': f"{scale}m",
                        'size_mb': len(result['file_data']) / (1024 * 1024),
                        'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'geodata_explorer_early',
                        'selected_bands': getattr(st.session_state, 'selected_bands', []),
                        'band_names': dataset.get('band_names', ''),
                        'band_units': dataset.get('band_units', ''),
                        'dataset_type': dataset.get('snippet_type', 'Unknown')
                    }
                })
            except Exception as e:
                st.error(f"❌ Error preparing {export_format} data for visualization: {str(e)}")
                raise e

        # Set up direct visualization data
        st.session_state.direct_visualization_data = {
            'results': visualization_data,
            'source_module': 'geodata_explorer'
        }

        # Switch to data visualizer
        st.session_state.app_mode = "data_visualizer"
        st.rerun()

    except Exception as e:
        # If any error occurs, show warning and redirect to upload interface
        st.warning(f"⚠️ Error transferring data to visualization: {str(e)}")
        st.info("🔄 Redirecting to Data Visualizer upload interface. Please upload your downloaded file manually.")

        # Clear any failed direct visualization data
        if 'direct_visualization_data' in st.session_state:
            del st.session_state.direct_visualization_data

        # Redirect to data visualizer with clean state
        st.session_state.app_mode = "data_visualizer"
        st.rerun()


def _launch_visualization_from_results():
    """Launch data visualizer with current download results"""
    try:
        results = st.session_state.download_results

        # Prepare data for visualization based on format
        visualization_data = []

        if results['export_format'] == 'CSV':
            # For CSV data, convert file_data back to DataFrame and prepare with full metadata
            try:
                import pandas as pd
                import io

                # Convert file_data to DataFrame
                if isinstance(results['file_data'], bytes):
                    csv_string = results['file_data'].decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_string))
                elif isinstance(results['file_data'], str):
                    df = pd.read_csv(io.StringIO(results['file_data']))
                else:
                    # Assume it's already a DataFrame
                    df = results['file_data']

                # Prepare metadata
                source_metadata = {
                    'dataset': results['dataset_name'],
                    'export_format': results['export_format'],
                    'resolution': f"{results['scale']}m",
                    'size_mb': results['file_size_mb'],
                    'download_timestamp': results['download_timestamp'],
                    'source': 'geodata_explorer',
                    'selected_bands': results.get('selected_bands', getattr(st.session_state, 'selected_bands', [])),
                    'band_names': results.get('band_names', ''),
                    'band_units': results.get('band_units', ''),
                    'dataset_type': results.get('dataset_type', 'Unknown')
                }

                # Use helper function to prepare complete data structure
                prepared_data = _prepare_data_for_visualization(df, source_metadata)

                visualization_data.append({
                    'file_name': results['filename'],
                    'data': prepared_data['data'],
                    'data_type': 'csv',
                    'transfer_method': 'data_object',
                    'column_suggestions': prepared_data['column_suggestions'],
                    'quality_report': prepared_data['quality_report'],
                    'summary': prepared_data['summary'],
                    'metadata': prepared_data['metadata']
                })
            except Exception as e:
                st.error(f"❌ Error preparing CSV data for visualization: {str(e)}")
                raise e

        elif results['export_format'] in ['GeoTIFF', 'NetCDF']:
            # For spatial data, create a temporary file
            try:
                import tempfile
                import os

                # Check if the filename indicates a ZIP file (multiple GeoTIFF files)
                if results['filename'].lower().endswith('.zip'):
                    # Handle ZIP file containing multiple GeoTIFF files
                    file_extension = '.zip'
                    data_type = 'zip'
                else:
                    # Handle single file
                    file_extension = '.tif' if results['export_format'] == 'GeoTIFF' else '.nc'
                    data_type = 'tiff' if results['export_format'] == 'GeoTIFF' else 'netcdf'

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(results['file_data'])
                    temp_path = temp_file.name

                visualization_data.append({
                    'file_name': results['filename'],
                    'data_type': data_type,
                    'file_path': temp_path,
                    'transfer_method': 'temp_file',
                    'metadata': {
                        'dataset': results['dataset_name'],
                        'export_format': results['export_format'],
                        'resolution': f"{results['scale']}m",
                        'size_mb': results['file_size_mb'],
                        'download_timestamp': results['download_timestamp'],
                        'source': 'geodata_explorer',
                        'selected_bands': results.get('selected_bands', getattr(st.session_state, 'selected_bands', [])),
                        'band_names': results.get('band_names', ''),
                        'band_units': results.get('band_units', ''),
                        'dataset_type': results.get('dataset_type', 'Unknown')
                    }
                })
            except Exception as e:
                st.error(f"❌ Error preparing {results['export_format']} data for visualization: {str(e)}")
                raise e

        # Set up direct visualization data
        st.session_state.direct_visualization_data = {
            'results': visualization_data,
            'source_module': 'geodata_explorer'
        }

        # Switch to data visualizer
        st.session_state.app_mode = "data_visualizer"
        st.rerun()

    except Exception as e:
        # If any error occurs, show warning and redirect to upload interface
        st.warning(f"⚠️ Error transferring data to visualization: {str(e)}")
        st.info("🔄 Redirecting to Data Visualizer upload interface. Please upload your downloaded file manually.")

        # Clear any failed direct visualization data
        if 'direct_visualization_data' in st.session_state:
            del st.session_state.direct_visualization_data

        # Redirect to data visualizer with clean state
        st.session_state.app_mode = "data_visualizer"
        st.rerun()


def _process_download(export_format, scale):
    """Process the download request with real GEE data processing"""
    try:
        # Get all user selections
        dataset = st.session_state.selected_dataset
        bands = st.session_state.selected_bands
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date

        # Get geometry from geometry handler
        try:
            geometry = st.session_state.geometry_handler.current_geometry
            if geometry is None:
                st.error("❌ No geometry selected. Please go back and select an area of interest.")
                return False
        except Exception as e:
            st.error(f"❌ Error getting geometry: {str(e)}")
            return False

        # Validate required data
        if not dataset or not dataset.get('ee_id'):
            st.error("❌ No dataset selected or invalid dataset information.")
            return False

        if not bands:
            st.error("❌ No bands selected.")
            return False

        ee_id = dataset.get('ee_id')
        snippet_type = dataset.get('snippet_type', 'Image')

        st.info(f"""
        **Processing Request:**
        - Dataset: {dataset.get('name', 'Unknown')}
        - Type: {snippet_type}
        - Format: {export_format}
        - Resolution: {scale}m
        - Bands: {', '.join(bands)}
        - Date Range: {start_date} to {end_date}
        - Area: {geometry.area().divide(1000000).getInfo():.2f} km²
        """)

        with st.spinner("🔄 Processing Earth Engine data... This may take a few minutes."):
            # Use the simplified, proven download approach
            from app_utils import download_ee_data_simple

            # Process based on dataset type
            if snippet_type == 'ImageCollection':
                st.info("📊 Processing ImageCollection - this dataset contains multiple images over time.")
            else:
                st.info("🗺️ Processing static Image - this dataset contains a single image.")

            # Call the simplified download function
            result = download_ee_data_simple(
                dataset=dataset,
                bands=bands,
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                export_format=export_format,
                scale=scale
            )

            if result['success']:
                st.success("✅ Data processing completed successfully!")

                # Display result message
                st.info(result['message'])

                # Add early visualization option
                st.markdown("---")
                st.markdown("### 🎯 Quick Actions")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📊 Visualize Data", type="primary", use_container_width=True):
                        # Prepare data for immediate visualization
                        try:
                            _launch_early_visualization(result, dataset, export_format, scale)
                        except Exception as e:
                            st.warning(f"⚠️ Could not transfer data to visualizer: {str(e)}")
                            st.info("💡 You can still download the file and upload it manually to the visualizer.")

                with col2:
                    st.markdown("**Or download first:**")

                # Create download button
                file_data = result['file_data']

                # Determine file extension and MIME type
                if export_format == 'CSV':
                    file_extension = '.csv'
                    mime_type = "text/csv"
                elif export_format == 'NetCDF':
                    file_extension = '.nc'
                    mime_type = "application/x-netcdf"
                elif export_format == 'GeoTIFF':
                    # Check if it's a ZIP file (multiple GeoTIFFs)
                    if 'ZIP archive' in result['message']:
                        file_extension = '.zip'
                        mime_type = "application/zip"
                    else:
                        file_extension = '.tif'
                        mime_type = "image/tiff"
                else:
                    file_extension = '.tif'
                    mime_type = "image/tiff"

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                filename = f"{dataset_name}_{timestamp}{file_extension}"

                file_size_mb = len(file_data) / (1024 * 1024)

                # Store download results for persistent interface with rich band information
                download_results = {
                    'file_data': file_data,
                    'filename': filename,
                    'mime_type': mime_type,
                    'file_size_mb': file_size_mb,
                    'export_format': export_format,
                    'dataset_name': dataset.get('name', 'Unknown'),
                    'success_message': result['message'],
                    'scale': scale,
                    'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'selected_bands': getattr(st.session_state, 'selected_bands', []),
                    'band_names': dataset.get('band_names', ''),
                    'band_units': dataset.get('band_units', ''),
                    'dataset_type': dataset.get('snippet_type', 'Unknown')
                }

                st.session_state.download_complete = True
                st.session_state.download_results = download_results

                st.download_button(
                    label=f"📥 Download {export_format} ({file_size_mb:.1f} MB)",
                    data=file_data,
                    file_name=filename,
                    mime=mime_type,
                    type="primary",
                    help=f"Click to download your {export_format} file"
                )
                return True

            else:
                st.error("❌ Data processing failed.")
                st.error(result['message'])
                return False

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return False


def _process_image_collection_download(ee_id, bands, geometry, start_date, end_date,
                                      export_format, scale, exporter, download_helper):
    """Process ImageCollection download with real GEE data"""
    try:
        from geoclimate_fetcher.core import ImageCollectionFetcher
        import tempfile
        import os
        from datetime import datetime

        # Create fetcher (similar to HydrologyAnalyzer approach)
        fetcher = ImageCollectionFetcher(
            ee_id=ee_id,
            bands=bands,
            geometry=geometry
        )

        # Apply date filters
        if start_date and end_date:
            fetcher = fetcher.filter_dates(start_date=start_date, end_date=end_date)

        # Generate filename
        dataset_name = st.session_state.selected_dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Process based on format
        if export_format == 'CSV':
            # Get temporal resolution for smart processing
            temporal_resolution = dataset.get('temporal_resolution', 'Daily')

            # Show processing info
            st.info("📊 Extracting time series data...")

            # Get native scale for optimization
            native_scale = None
            if dataset_pixel_size:
                try:
                    native_scale = float(dataset_pixel_size)
                except (ValueError, TypeError):
                    native_scale = None

            # Get time series data with optimizations
            df = fetcher.get_time_series_average(
                export_format='CSV',
                user_scale=scale,
                dataset_native_scale=native_scale
            )

            if df is not None and not df.empty:
                # Create CSV data
                csv_data = df.to_csv(index=False)
                filename = f"{dataset_name}_{timestamp}.csv"

                # Create download button with post-download integration
                st.success(f"✅ Time series data extracted ({len(df)} records)")

                # Register the download
                download_handler = get_download_handler("geodata_explorer")
                result_id = register_csv_download(
                    "geodata_explorer",
                    filename,
                    df,
                    metadata={
                        'dataset': dataset_name,
                        'extraction_type': 'time_series',
                        'records': len(df),
                        'size_kb': len(csv_data) / 1024
                    }
                )

                # Show download button
                download_clicked = st.download_button(
                    label=f"📥 Download CSV ({len(csv_data)/1024:.1f} KB)",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    type="primary"
                )

                # Set consistent download complete state when download is successful
                if download_clicked:
                    # Store download results for persistent interface
                    download_results = {
                        'file_data': csv_data.encode(),
                        'filename': filename,
                        'mime_type': "text/csv",
                        'file_size_mb': len(csv_data) / (1024 * 1024),
                        'export_format': 'CSV',
                        'dataset_name': dataset_name,
                        'success_message': f"Time series data extracted ({len(df)} records)",
                        'scale': 1000,  # Default scale for time series
                        'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    st.session_state.download_complete = True
                    st.session_state.download_results = download_results
                return True
            else:
                st.error("❌ No time series data extracted. Try a different date range or area.")
                return False

        elif export_format == 'NetCDF':
            st.info("🌐 Creating NetCDF dataset...")

            try:
                # Get gridded data
                ds = fetcher.get_gridded_data(scale=scale)

                if ds is not None:
                    # Create temporary file for NetCDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as temp_file:
                        temp_path = temp_file.name

                    # Export to NetCDF
                    netcdf_path = exporter.export_gridded_data_to_netcdf(ds, temp_path)

                    if os.path.exists(netcdf_path):
                        # Read file for download
                        with open(netcdf_path, 'rb') as f:
                            netcdf_data = f.read()

                        filename = f"{dataset_name}_{timestamp}.nc"
                        file_size_mb = len(netcdf_data) / (1024 * 1024)

                        st.success(f"✅ NetCDF dataset created ({file_size_mb:.1f} MB)")
                        st.download_button(
                            label=f"📥 Download NetCDF ({file_size_mb:.1f} MB)",
                            data=netcdf_data,
                            file_name=filename,
                            mime="application/x-netcdf",
                            type="primary"
                        )

                        # Clean up temporary file
                        os.unlink(netcdf_path)
                        return True
                    else:
                        st.error("❌ Failed to create NetCDF file.")
                        return False
                else:
                    st.error("❌ No gridded data available. Try a smaller area or different date range.")
                    return False

            except Exception as e:
                st.error(f"❌ NetCDF processing failed: {str(e)}")
                return False

        elif export_format == 'GeoTIFF':
            st.warning("🔄 GeoTIFF export for ImageCollections is complex. Consider using CSV or NetCDF format.")
            st.info("💡 For individual GeoTIFF files, select a single-image dataset instead.")
            return False

        return False

    except Exception as e:
        st.error(f"❌ ImageCollection processing error: {str(e)}")
        return False


def _process_static_image_download(ee_id, bands, geometry, export_format, scale, exporter, download_helper):
    """Process static Image download with real GEE data"""
    try:
        from geoclimate_fetcher.core import StaticRasterFetcher
        import tempfile
        import os
        from datetime import datetime

        # Create fetcher
        fetcher = StaticRasterFetcher(
            ee_id=ee_id,
            bands=bands,
            geometry=geometry
        )

        # Generate filename
        dataset_name = st.session_state.selected_dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Process based on format
        if export_format == 'CSV':
            st.info("📊 Extracting zonal statistics...")

            # Get zonal statistics
            stats = fetcher.get_zonal_statistics()

            if stats:
                # Convert to DataFrame
                import pandas as pd
                rows = []
                for band, band_stats in stats.items():
                    row = {'band': band}
                    row.update(band_stats)
                    rows.append(row)

                df = pd.DataFrame(rows)
                csv_data = df.to_csv(index=False)
                filename = f"{dataset_name}_stats_{timestamp}.csv"

                st.success(f"✅ Zonal statistics extracted ({len(rows)} bands)")

                # Register the CSV download
                stats_df = pd.DataFrame(rows)
                download_handler = get_download_handler("geodata_explorer")
                result_id = register_csv_download(
                    "geodata_explorer",
                    filename,
                    stats_df,
                    metadata={
                        'dataset': dataset_name,
                        'extraction_type': 'zonal_statistics',
                        'bands': len(rows),
                        'size_kb': len(csv_data) / 1024
                    }
                )

                # Show download button
                download_clicked = st.download_button(
                    label=f"📥 Download Statistics CSV ({len(csv_data)/1024:.1f} KB)",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    type="primary"
                )

                # Set consistent download complete state when download is successful
                if download_clicked:
                    # Store download results for persistent interface
                    download_results = {
                        'file_data': csv_data.encode(),
                        'filename': filename,
                        'mime_type': "text/csv",
                        'file_size_mb': len(csv_data) / (1024 * 1024),
                        'export_format': 'CSV',
                        'dataset_name': dataset_name,
                        'success_message': f"Zonal statistics extracted ({len(rows)} bands)",
                        'scale': 1000,  # Default scale for statistics
                        'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    st.session_state.download_complete = True
                    st.session_state.download_results = download_results
                return True
            else:
                st.error("❌ No statistics extracted.")
                return False

        elif export_format in ['GeoTIFF', 'NetCDF']:
            st.info(f"🗺️ Exporting {export_format}...")

            try:
                # Create temporary file
                ext = '.tif' if export_format == 'GeoTIFF' else '.nc'
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                    temp_path = temp_file.name

                # Export image to local file with EPSG:4326 projection
                result_path = exporter.export_image_to_local(
                    image=fetcher.image,
                    output_path=temp_path,
                    region=geometry,
                    scale=scale,
                    crs='EPSG:4326'  # Ensure EPSG:4326 projection for TIFF exports
                )

                if os.path.exists(result_path):
                    # Read file for download
                    with open(result_path, 'rb') as f:
                        file_data = f.read()

                    filename = f"{dataset_name}_{timestamp}{ext}"
                    file_size_mb = len(file_data) / (1024 * 1024)

                    # Check file size for download method
                    if file_size_mb > 50:
                        st.warning(f"⚠️ File is large ({file_size_mb:.1f} MB). This may take time to download.")
                        st.info("💡 Consider reducing the area size or resolution for faster downloads.")

                    mime_type = "image/tiff" if export_format == 'GeoTIFF' else "application/x-netcdf"

                    st.success(f"✅ {export_format} export completed ({file_size_mb:.1f} MB)")

                    # Register the download for TIFF files
                    download_handler = get_download_handler("geodata_explorer")
                    result_id = None

                    if export_format == 'GeoTIFF':
                        # For TIFF files, we need to save temporarily to get metadata
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                            tmp_file.write(file_data)
                            temp_path = tmp_file.name

                        # Extract basic metadata for TIFF
                        try:
                            import rasterio
                            with rasterio.open(temp_path) as src:
                                tiff_metadata = {
                                    'width': src.width,
                                    'height': src.height,
                                    'bands': src.count,
                                    'crs': str(src.crs) if src.crs else 'Unknown',
                                    'dtype': str(src.dtypes[0]) if src.count > 0 else 'unknown',
                                    'size_mb': file_size_mb,
                                    'dataset': dataset_name,
                                    'export_format': export_format
                                }
                        except Exception as e:
                            tiff_metadata = {
                                'size_mb': file_size_mb,
                                'dataset': dataset_name,
                                'export_format': export_format,
                                'error': str(e)
                            }

                        result_id = register_tiff_download(
                            "geodata_explorer",
                            filename,
                            temp_path,
                            metadata=tiff_metadata
                        )

                    # Show download button
                    download_clicked = st.download_button(
                        label=f"📥 Download {export_format} ({file_size_mb:.1f} MB)",
                        data=file_data,
                        file_name=filename,
                        mime=mime_type,
                        type="primary"
                    )

                    # Set consistent download complete state when TIFF download is successful
                    if download_clicked and result_id:
                        # Store download results for persistent interface
                        download_results = {
                            'file_data': file_data,
                            'filename': filename,
                            'mime_type': mime_type,
                            'file_size_mb': file_size_mb,
                            'export_format': export_format,
                            'dataset_name': dataset_name,
                            'success_message': f"{export_format} export completed ({file_size_mb:.1f} MB)",
                            'scale': scale,
                            'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        st.session_state.download_complete = True
                        st.session_state.download_results = download_results

                    # Clean up temporary file
                    os.unlink(result_path)
                    return True
                else:
                    st.error("❌ Export failed - no output file created.")
                    return False

            except Exception as e:
                error_msg = str(e)
                if "too large" in error_msg.lower():
                    st.error("❌ Area too large for direct export.")
                    st.info("💡 Try:")
                    st.info("- Reducing the area size")
                    st.info("- Using a coarser resolution (higher scale value)")
                    st.info("- Using CSV format for statistical summaries")
                else:
                    st.error(f"❌ Export error: {error_msg}")
                return False

        return False

    except Exception as e:
        st.error(f"❌ Static image processing error: {str(e)}")
        return False


def _process_smart_download(export_format, scale, export_preference):
    """Process smart download using the enhanced GEEExporter with fallback"""
    try:
        # Get all user selections
        dataset = st.session_state.selected_dataset
        bands = st.session_state.selected_bands
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date

        # Get geometry from geometry handler
        try:
            geometry = st.session_state.geometry_handler.current_geometry
        except Exception as e:
            st.error(f"❌ Geometry error: {str(e)}")
            return

        # Create download helper
        download_helper = DownloadHelper()

        # Show processing status
        st.markdown("### 🔄 Processing Your Request")
        st.info("⏳ Processing your Earth Engine data request...")

        with st.spinner("🌍 Fetching Earth Engine data..."):
            # Handle different dataset types
            if dataset.get('snippet_type') == 'ImageCollection':
                # Get ImageCollection
                fetcher = ImageCollectionFetcher(
                    ee_id=dataset['ee_id'],
                    bands=bands,
                    geometry=geometry
                )

                # Apply date filtering
                fetcher = fetcher.filter_dates(start_date, end_date)

                # Handle CSV format with optimized path
                if export_format == 'CSV':
                    st.info("📊 Using optimized CSV processing for area-averaged time series...")

                    # Use the optimized get_time_series_average method
                    try:
                        # Get dataset native scale for optimization
                        dataset_native_scale = dataset.get('pixel_size')
                        if dataset_native_scale:
                            try:
                                dataset_native_scale = float(dataset_native_scale)
                            except (ValueError, TypeError):
                                dataset_native_scale = None

                        # Get optimized time series data with automatic chunking if needed
                        temporal_resolution = dataset.get('temporal_resolution', 'Daily')

                        # Check if we need chunking based on date range and temporal resolution
                        days_diff = (end_date - start_date).days + 1

                        # Estimate if collection will exceed 5000 elements
                        if temporal_resolution == 'Hourly' and days_diff > 200:
                            # Use chunked method for large hourly collections
                            df = fetcher.get_time_series_average_chunked(
                                chunk_months=3,
                                export_format='CSV',
                                user_scale=scale,
                                temporal_resolution=temporal_resolution,
                                dataset_native_scale=dataset_native_scale
                            )
                        elif temporal_resolution == '30-minute' and days_diff > 100:
                            # Use chunked method for large sub-hourly collections
                            df = fetcher.get_time_series_average_chunked(
                                chunk_months=1,
                                export_format='CSV',
                                user_scale=scale,
                                temporal_resolution=temporal_resolution,
                                dataset_native_scale=dataset_native_scale
                            )
                        elif temporal_resolution == 'Daily' and days_diff > 3650:
                            # Use chunked method for very large daily collections (>10 years)
                            df = fetcher.get_time_series_average_chunked(
                                chunk_months=12,
                                export_format='CSV',
                                user_scale=scale,
                                temporal_resolution=temporal_resolution,
                                dataset_native_scale=dataset_native_scale
                            )
                        else:
                            # Use optimized single-pass method
                            df = fetcher.get_time_series_average(
                                export_format='CSV',
                                user_scale=scale,
                                dataset_native_scale=dataset_native_scale
                            )

                        if df is None or df.empty:
                            st.error("❌ No data returned from optimized processing")
                            st.warning("⚠️ Falling back to standard download method...")
                            return _process_download(export_format, scale)

                        # Generate filename
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                        filename = f"{dataset_name}_timeseries_{timestamp}.csv"

                        # Convert DataFrame to CSV
                        csv_data = df.to_csv(index=False)
                        file_size_mb = len(csv_data.encode('utf-8')) / (1024 * 1024)

                        # Set up download results
                        download_results = {
                            'file_data': csv_data.encode('utf-8'),
                            'filename': filename,
                            'mime_type': "text/csv",
                            'file_size_mb': file_size_mb,
                            'export_format': 'CSV',
                            'dataset_name': dataset.get('name', 'Unknown'),
                            'success_message': f"Optimized CSV download ({file_size_mb:.1f} MB)",
                            'scale': scale,
                            'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        st.session_state.download_complete = True
                        st.session_state.download_results = download_results

                        st.success(f"✅ CSV data processed successfully! ({file_size_mb:.1f} MB)")
                        st.info(f"📊 Generated {len(df)} time series records")

                        # Early visualization option for CSV
                        st.markdown("---")
                        st.markdown("### 🎯 Quick Actions")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📊 Visualize Data", type="primary", use_container_width=True):
                                try:
                                    _launch_early_visualization(download_results, dataset, export_format, scale)
                                except Exception as viz_error:
                                    st.error(f"❌ Visualization error: {str(viz_error)}")
                        with col2:
                            st.download_button(
                                label="💾 Download CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                type="secondary",
                                use_container_width=True
                            )
                        return

                    except Exception as csv_error:
                        st.error(f"❌ CSV processing failed: {str(csv_error)}")
                        if "Collection query aborted after accumulating over 5000 elements" in str(csv_error):
                            st.warning("⚠️ Large collection detected. Trying chunked processing...")
                            try:
                                # Force chunked processing for large collections
                                df = fetcher.get_time_series_average_chunked(
                                    chunk_months=6,
                                    export_format='CSV',
                                    user_scale=scale,
                                    temporal_resolution=dataset.get('temporal_resolution', 'Daily'),
                                    dataset_native_scale=dataset_native_scale
                                )
                                if df is not None and not df.empty:
                                    # Process chunked result same as above
                                    from datetime import datetime
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                                    filename = f"{dataset_name}_timeseries_{timestamp}.csv"
                                    csv_data = df.to_csv(index=False)
                                    file_size_mb = len(csv_data.encode('utf-8')) / (1024 * 1024)

                                    download_results = {
                                        'file_data': csv_data.encode('utf-8'),
                                        'filename': filename,
                                        'mime_type': "text/csv",
                                        'file_size_mb': file_size_mb,
                                        'export_format': 'CSV',
                                        'dataset_name': dataset.get('name', 'Unknown'),
                                        'success_message': f"Chunked CSV download ({file_size_mb:.1f} MB)",
                                        'scale': scale,
                                        'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    }

                                    st.session_state.download_complete = True
                                    st.session_state.download_results = download_results
                                    st.success(f"✅ Chunked CSV processing successful! ({len(df)} records)")
                                    return
                            except Exception as chunk_error:
                                st.error(f"❌ Chunked processing also failed: {str(chunk_error)}")

                        st.warning("⚠️ Falling back to standard download method...")
                        return _process_download(export_format, scale)

                else:
                    # For GeoTIFF format, create a median composite for smart download
                    composite_image = fetcher.collection.median()

                    # Generate filename
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                    filename = f"{dataset_name}_composite_{timestamp}"

                    # Execute smart download
                    result = download_helper.execute_smart_download(
                        image=composite_image,
                        filename=filename,
                        region=geometry,
                        scale=scale,
                        export_preference=export_preference
                    )

            else:
                # For single images (StaticRaster)
                # Create the image directly since StaticRasterFetcher requires constructor params
                image = ee.Image(dataset['ee_id'])

                if bands:
                    image = image.select(bands)

                # Generate filename
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dataset_name = dataset.get('name', 'data').replace(' ', '_').replace('/', '_')
                filename = f"{dataset_name}_{timestamp}"

                # Execute smart download
                result = download_helper.execute_smart_download(
                    image=image,
                    filename=filename,
                    region=geometry,
                    scale=scale,
                    export_preference=export_preference
                )

            # Handle results based on export method
            if result['success']:
                if result['export_method'] == 'local':
                    # Local download succeeded - register the download properly
                    download_handler = get_download_handler("geodata_explorer")

                    # Create temporary file with the downloaded data
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                        temp_file.write(result['file_data'])
                        temp_path = temp_file.name

                    # Register the TIFF download
                    result_id = register_tiff_download(
                        "geodata_explorer",
                        f"{result['filename']}.tif",
                        temp_path,
                        {
                            'file_size_mb': result.get('actual_size_mb', 0),
                            'export_method': 'smart_local',
                            'processing_details': f"Smart download - local export ({result.get('actual_size_mb', 0):.1f} MB)"
                        }
                    )

                    # Set consistent download complete state
                    download_results = {
                        'file_data': result['file_data'],
                        'filename': f"{result['filename']}.tif",
                        'mime_type': "image/tiff",
                        'file_size_mb': result.get('actual_size_mb', 0),
                        'export_format': 'GeoTIFF',
                        'dataset_name': dataset.get('name', 'Unknown'),
                        'success_message': f"Smart download - local export ({result.get('actual_size_mb', 0):.1f} MB)",
                        'scale': scale,
                        'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    st.session_state.download_complete = True
                    st.session_state.download_results = download_results

                    # Add early visualization option for smart downloads
                    st.markdown("---")
                    st.markdown("### 🎯 Quick Actions")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("📊 Visualize Smart Download", type="primary", use_container_width=True):
                            # Prepare data for immediate visualization
                            try:
                                _launch_visualization_from_results()
                            except Exception as e:
                                st.warning(f"⚠️ Could not transfer data to visualizer: {str(e)}")
                                st.info("💡 Redirecting to visualizer for manual upload...")
                                st.session_state.direct_visualization_data = None
                                st.session_state.app_mode = "data_visualizer"
                                st.rerun()

                    with col2:
                        st.markdown("**Or check downloads section:**")

                elif result['export_method'] == 'drive':
                    # Drive export submitted - show task information
                    st.success("🎉 Export completed successfully!")
                    st.info("📤 Your data has been submitted to Google Drive for processing.")

                    # Show task monitoring information
                    st.markdown("### 📊 Export Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Export Method", "Google Drive")
                        # Don't show estimated size since it's unreliable
                        st.metric("Processing", "In Progress")
                    with col2:
                        st.metric("Drive Folder", result.get('drive_folder', 'N/A'))
                        if result.get('task_id'):
                            st.code(f"Task ID: {result['task_id']}")

                    # Show monitoring links
                    st.markdown("### 🔗 Monitoring Links")
                    if result.get('task_url'):
                        st.markdown(f"[⚙️ Monitor Task Progress]({result['task_url']})")
                    if result.get('drive_url'):
                        st.markdown(f"[📁 Open Google Drive]({result['drive_url']})")

                    # Show next steps
                    st.markdown("### 📋 Next Steps")
                    st.markdown("""
                    1. 📱 Monitor task progress using the links above
                    2. 📁 Check your Google Drive folder once tasks complete
                    3. ⏱️ Large exports may take several minutes to complete
                    """)

            else:
                # Only show error if the result was actually unsuccessful
                result_msg = result.get('message', '') or 'Unknown error occurred'
                st.error(f"❌ Smart download failed: {result_msg}")
                if 'error' in result:
                    error_msg = result.get('error', '') or 'No error details available'
                    st.error(f"Details: {error_msg}")

    except Exception as e:
        st.error(f"❌ Smart download failed: {str(e)}")
        st.info("💡 You can try the standard download option or contact support.")


def _reset_all_selections():
    """Reset all session state variables"""
    keys_to_reset = [
        'geometry_complete', 'dataset_selected', 'bands_selected', 'dates_selected',
        'selected_dataset', 'selected_bands', 'start_date', 'end_date',
        'download_complete', 'download_results'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def _extract_category(dataset_name):
    """Extract category from dataset name"""
    name_lower = dataset_name.lower()

    if any(term in name_lower for term in ['precipitation', 'rain', 'chirps', 'imerg', 'gpm']):
        return 'Climate'
    elif any(term in name_lower for term in ['temperature', 'temp', 'era5', 'daymet', 'terraclimate']):
        return 'Weather'
    elif any(term in name_lower for term in ['land cover', 'landcover', 'nlcd', 'worldcover', 'dynamic world', 'modis']):
        return 'Land Cover'
    elif any(term in name_lower for term in ['vegetation', 'ndvi', 'evi', 'lai', 'fpar', 'gpp']):
        return 'Vegetation'
    elif any(term in name_lower for term in ['sea surface', 'ocean', 'oisst']):
        return 'Ocean'
    elif any(term in name_lower for term in ['atmosphere', 'pressure', 'wind', 'humidity']):
        return 'Atmosphere'
    elif any(term in name_lower for term in ['elevation', 'dem', 'srtm', 'nasadem', 'topography']):
        return 'Terrain'
    elif any(term in name_lower for term in ['soil', 'moisture', 'smap']):
        return 'Land'
    else:
        return 'Other'


def _get_fallback_datasets():
    """Get fallback datasets if catalog loading fails"""
    return [
        {
            'name': 'CHIRPS Daily Precipitation',
            'ee_id': 'UCSB-CHG/CHIRPS/DAILY',
            'description': 'Climate Hazards Group InfraRed Precipitation with Station data',
            'category': 'Climate',
            'temporal_resolution': 'Daily'
        },
        {
            'name': 'MODIS Terra Surface Temperature',
            'ee_id': 'MODIS/006/MOD11A1',
            'description': 'MODIS/Terra Land Surface Temperature/Emissivity Daily L3 Global',
            'category': 'Climate',
            'temporal_resolution': 'Daily'
        },
        {
            'name': 'ERA5 Reanalysis',
            'ee_id': 'ECMWF/ERA5/DAILY',
            'description': 'ERA5 Daily aggregated reanalysis data',
            'category': 'Weather',
            'temporal_resolution': 'Daily'
        }
    ]


def _apply_filters(datasets, search_term, category_filter, time_filter):
    """Apply search and filter criteria to datasets"""
    filtered = datasets
    
    # Apply search filter
    if search_term:
        filtered = [d for d in filtered if search_term.lower() in d.get('name', '').lower() 
                   or search_term.lower() in d.get('description', '').lower()]
    
    # Apply category filter
    if category_filter:
        filtered = [d for d in filtered if d.get('category') in category_filter]
    
    # Apply time filter
    if time_filter != "All":
        filtered = [d for d in filtered if _matches_temporal_filter(d.get('temporal_resolution', ''), time_filter)]
    
    return filtered


def _matches_temporal_filter(temporal_resolution, filter_value):
    """Check if temporal resolution matches the filter"""
    if not temporal_resolution or not filter_value:
        return False

    temp_res_lower = temporal_resolution.lower()
    filter_lower = filter_value.lower()

    if filter_lower == "daily":
        return any(term in temp_res_lower for term in ["daily", "day"])
    elif filter_lower == "hourly":
        return any(term in temp_res_lower for term in ["hourly", "hour", "3-hourly", "6 hours"])
    elif filter_lower == "monthly":
        return any(term in temp_res_lower for term in ["monthly", "month"])
    elif filter_lower == "yearly":
        return any(term in temp_res_lower for term in ["yearly", "year"])
    elif filter_lower == "annual":
        return any(term in temp_res_lower for term in ["annual", "year"])
    elif filter_lower == "static":
        return any(term in temp_res_lower for term in ["static", "composite"])
    elif filter_lower == "other":
        return not any(term in temp_res_lower for term in ["daily", "hourly", "monthly", "yearly", "annual", "static"])
    else:
        return filter_lower in temp_res_lower


def _process_csv_drive_export(export_format, scale):
    """Process CSV export to Google Drive using Earth Engine Tasks"""
    try:
        # Get all user selections
        dataset = st.session_state.selected_dataset
        bands = st.session_state.selected_bands
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date

        # Get geometry from geometry handler
        try:
            geometry = st.session_state.geometry_handler.current_geometry
        except Exception as e:
            st.error(f"❌ Geometry error: {str(e)}")
            return

        st.info("📤 **Google Drive CSV Export**: Using Earth Engine Tasks for large collections...")

        # For CSV to Google Drive, we use Earth Engine's table export (FeatureCollection)
        # which can handle large collections better than local processing

        # Create collection and apply filters
        collection = ee.ImageCollection(dataset['ee_id'])
        collection = collection.filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        collection = collection.filterBounds(geometry)

        if bands:
            collection = collection.select(bands)

        # Check collection size
        collection_size = collection.size().getInfo()
        st.info(f"🔄 Collection contains {collection_size} images. Preparing Google Drive export...")

        if collection_size == 0:
            st.error("❌ No images found for the specified date range and region")
            return

        # Create feature collection with time series data
        def image_to_feature(image):
            """Convert image to feature with area-averaged values"""
            # Get the date
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')

            # Calculate area-averaged values
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9
            )

            # Create clean properties with only date and band values
            clean_props = ee.Dictionary({'date': date})

            # Add only the band values (exclude system properties)
            for band in bands:
                clean_props = clean_props.set(band, stats.get(band))

            # Create feature with null geometry and clean properties
            return ee.Feature(None, clean_props)

        # Map over collection to create features
        features = collection.map(image_to_feature)

        # Generate filename with clean characters
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = dataset.get('name', 'data')

        # Clean dataset name for filename (allow only safe characters)
        import re
        clean_dataset_name = re.sub(r'[^a-zA-Z0-9._-]', '_', dataset_name)
        filename = f"{clean_dataset_name}_timeseries_{timestamp}"

        # Clean description for Earth Engine (max 100 chars, only allowed characters)
        clean_description = re.sub(r'[^a-zA-Z0-9._,:;-]', '_', f"CSV_Export_{clean_dataset_name}")
        if len(clean_description) > 100:
            clean_description = clean_description[:97] + "..."

        # Export to Google Drive using Earth Engine Task
        task = ee.batch.Export.table.toDrive(
            collection=features,
            description=clean_description,
            folder="GeoClimate_Exports",
            fileNamePrefix=filename,
            fileFormat='CSV'
        )

        # Start the task
        task.start()

        # Show task information
        st.success("✅ **Google Drive export task started!**")

        task_id = task.id
        task_url = f"https://code.earthengine.google.com/tasks"

        st.markdown("### 📋 Export Details")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Task ID**: {task_id}")
            st.info(f"**Filename**: {filename}.csv")
            st.info(f"**Drive Folder**: GeoClimate_Exports")
            st.info(f"**Images**: {collection_size}")

        with col2:
            st.markdown(f"**🔗 [Monitor Task Progress]({task_url})**")
            st.markdown("**📁 [Open Google Drive](https://drive.google.com/drive/folders/)**")
            st.markdown("**📊 [Earth Engine Console](https://code.earthengine.google.com/)**")

        st.markdown("---")
        st.markdown("### ⏱️ **What happens next:**")
        st.markdown("""
        1. 🔄 **Processing**: Earth Engine processes your collection in the cloud
        2. 📤 **Upload**: Results automatically upload to your Google Drive
        3. 📁 **Organization**: File saved in 'GeoClimate_Exports' folder
        4. ✅ **Completion**: Check Google Drive folder for completed file
        """)

        st.info("💡 **Tip**: Large collections may take 5-30 minutes. Check the task monitor link above for progress.")

        # Store task info in session state for tracking
        if 'drive_export_tasks' not in st.session_state:
            st.session_state.drive_export_tasks = []

        task_info = {
            'task_id': task_id,
            'filename': f"{filename}.csv",
            'folder': 'GeoClimate_Exports',
            'dataset': dataset_name,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'RUNNING',
            'task_url': task_url
        }
        st.session_state.drive_export_tasks.append(task_info)

    except Exception as e:
        st.error(f"❌ CSV Drive export failed: {str(e)}")

        # Check if it's the 5000 elements error and provide specific guidance
        if "Collection query aborted after accumulating over 5000 elements" in str(e):
            st.warning("⚠️ **Large Collection Detected**")
            st.info("💡 **Recommendation**: The collection is too large for local processing. The Google Drive export using Earth Engine Tasks should handle this better. If this error persists, try:")
            st.markdown("""
            - ✅ Use shorter date ranges (e.g., 1-2 years at a time)
            - ✅ Select fewer bands if multiple are selected
            - ✅ Use the 'Auto' download method instead
            """)
        else:
            st.info("💡 You can try the 'Auto' download method instead.")


def _display_dataset_options(datasets):
    """Display dataset selection options"""
    st.markdown("### 📊 Available Datasets:")

    for i, dataset in enumerate(datasets):
        with st.expander(f"📊 {dataset.get('name', 'Unknown Dataset')}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {dataset.get('description', 'No description available')}")
                st.markdown(f"**Category:** {dataset.get('category', 'Unknown')}")
                st.markdown(f"**Type:** {dataset.get('snippet_type', 'Unknown')} ({'Multiple images over time' if dataset.get('snippet_type') == 'ImageCollection' else 'Single static image'})")
                st.markdown(f"**Temporal Resolution:** {dataset.get('temporal_resolution', 'Unknown')}")

                # Show data availability dates if available
                start_date_str = dataset.get('start_date', '')
                end_date_str = dataset.get('end_date', '')
                if start_date_str and end_date_str:
                    start_date, end_date = _parse_dataset_dates(start_date_str, end_date_str)
                    if start_date and end_date:
                        st.markdown(f"**📅 Data Available:** {start_date} to {end_date}")
                        duration_years = (end_date - start_date).days / 365.25
                        st.markdown(f"**📊 Duration:** {duration_years:.1f} years")

                if dataset.get('ee_id'):
                    st.markdown(f"**Earth Engine ID:** `{dataset['ee_id']}`")
                if dataset.get('provider'):
                    st.markdown(f"**Provider:** {dataset.get('provider')}")

            with col2:
                if st.button(f"Select Dataset", key=f"select_dataset_{i}", type="primary"):
                    st.session_state.selected_dataset = dataset
                    st.session_state.dataset_selected = True
                    st.rerun()
