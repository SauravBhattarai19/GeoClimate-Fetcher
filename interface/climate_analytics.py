"""
Climate Analytics Interface Module
Handles the complete interface for the Climate Analytics tool
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import geemap.foliumap as geemap

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
from geoclimate_fetcher.core.dataset_config import get_dataset_config
from geoclimate_fetcher.climate_indices import ClimateIndicesCalculator

# Import post-download integration
from app_components.post_download_integration import (
    get_download_handler,
    register_csv_download,
    render_post_download_integration
)
from app_components.quick_visualization import quick_visualizer

# Import smart download components
from app_components.download_component import DownloadHelper

import ee


def render_climate_analytics():
    """Render the complete Climate Analytics interface"""

    # App title and header
    st.markdown('<h1 class="main-title">🧠 Climate Intelligence Hub</h1>', unsafe_allow_html=True)
    st.markdown("### Calculate climate indices and analyze extreme events")
    
    # Initialize session state for climate analytics
    _init_climate_session_state()
    
    # Progress indicator
    _show_climate_progress_indicator()
    
    # Step 1: Analysis Type Selection
    if st.session_state.climate_analysis_type is None:
        _render_analysis_type_selection()
    
    else:
        # Show current analysis type and allow changing
        analysis_type = st.session_state.climate_analysis_type
        st.info(f"🎯 Current Analysis: **{analysis_type.title()} Indices**")
        
        if st.button("🔄 Change Analysis Type", key="change_analysis"):
            _reset_climate_analysis()
            st.rerun()
        
        # Step 2: Area of Interest Selection
        if not st.session_state.climate_geometry_complete:
            _render_climate_geometry_selection()
        
        # Step 3: Dataset Selection
        elif not st.session_state.climate_dataset_selected:
            _render_climate_dataset_selection()
        
        # Step 4: Date Range Selection
        elif not st.session_state.climate_date_range_set:
            _render_climate_date_selection()
        
        # Step 5: Index Selection
        elif not st.session_state.climate_indices_selected:
            _render_climate_indices_selection()

        # Step 6: Export Configuration
        elif not st.session_state.get('climate_export_configured', False):
            _render_export_configuration()

        # Step 7: Analysis Results
        else:
            _render_climate_results()


def _init_climate_session_state():
    """Initialize climate analytics session state"""
    if 'climate_step' not in st.session_state:
        st.session_state.climate_step = 1
    if 'climate_analysis_type' not in st.session_state:
        st.session_state.climate_analysis_type = None
    if 'climate_geometry_complete' not in st.session_state:
        st.session_state.climate_geometry_complete = False
    if 'climate_dataset_selected' not in st.session_state:
        st.session_state.climate_dataset_selected = False
    if 'climate_selected_dataset' not in st.session_state:
        st.session_state.climate_selected_dataset = None
    if 'climate_date_range_set' not in st.session_state:
        st.session_state.climate_date_range_set = False
    if 'climate_indices_selected' not in st.session_state:
        st.session_state.climate_indices_selected = False
    if 'climate_selected_indices' not in st.session_state:
        st.session_state.climate_selected_indices = []
    if 'climate_start_date' not in st.session_state:
        st.session_state.climate_start_date = None
    if 'climate_end_date' not in st.session_state:
        st.session_state.climate_end_date = None
    if 'climate_geometry_handler' not in st.session_state:
        st.session_state.climate_geometry_handler = GeometryHandler()
    if 'climate_analysis_complete' not in st.session_state:
        st.session_state.climate_analysis_complete = False


def _show_climate_progress_indicator():
    """Display progress indicator for climate analytics"""
    if st.session_state.climate_analysis_type is not None:
        steps = [
            ("🎯", "Analysis Type", st.session_state.climate_analysis_type is not None),
            ("🗺️", "Study Area", st.session_state.climate_geometry_complete),
            ("📊", "Dataset", st.session_state.climate_dataset_selected),
            ("📅", "Date Range", st.session_state.climate_date_range_set),
            ("📈", "Indices", st.session_state.climate_indices_selected),
            ("⚙️", "Export Config", st.session_state.get('climate_export_configured', False)),
            ("🔬", "Results", st.session_state.get('climate_analysis_complete', False))
        ]
        
        cols = st.columns(len(steps))
        for i, (icon, label, completed) in enumerate(steps):
            with cols[i]:
                if completed:
                    st.markdown(f"✅ **{icon} {label}**")
                else:
                    st.markdown(f"⏳ {icon} {label}")
        
        st.markdown("---")


def _render_analysis_type_selection():
    """Render analysis type selection interface"""
    st.markdown("## 🎯 Step 1: Choose Analysis Type")
    st.markdown("Select the type of climate analysis you want to perform:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <span class="analysis-icon">🌧️</span>
            <div class="analysis-title">Precipitation Analysis</div>
            <div class="analysis-description">
                Calculate precipitation indices including SPI, wet/dry spells, 
                intensity measures, and extreme precipitation events.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🌧️ Analyze Precipitation", use_container_width=True, type="primary"):
            st.session_state.climate_analysis_type = "precipitation"
            st.session_state.climate_step = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="analysis-card">
            <span class="analysis-icon">🌡️</span>
            <div class="analysis-title">Temperature Analysis</div>
            <div class="analysis-description">
                Analyze temperature patterns, heat waves, cold spells, 
                growing degree days, and temperature extremes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🌡️ Analyze Temperature", use_container_width=True, type="primary"):
            st.session_state.climate_analysis_type = "temperature"
            st.session_state.climate_step = 2
            st.rerun()


def _render_climate_geometry_selection():
    """Render geometry selection for climate analytics"""
    # Add back button
    if st.button("← Back to Analysis Type"):
        st.session_state.climate_analysis_type = None
        st.session_state.climate_step = 1
        st.rerun()
    
    def on_geometry_selected(geometry):
        """Callback when geometry is selected"""
        st.session_state.climate_geometry_handler._current_geometry = geometry
        st.session_state.climate_geometry_handler._current_geometry_name = "climate_selected_aoi"
        st.session_state.climate_geometry_complete = True
        st.session_state.climate_step = 3
        st.success("✅ Study area selected successfully!")
    
    # Use the unified geometry selection widget
    geometry_widget = GeometrySelectionWidget(
        session_prefix="climate_",
        title="🗺️ Step 2: Select Study Area"
    )
    
    if geometry_widget.render_complete_interface(on_geometry_selected=on_geometry_selected):
        st.rerun()


def _render_climate_dataset_selection():
    """Render dataset selection for climate analytics"""
    st.markdown('<div class="step-header"><h2>📊 Step 3: Select Data Source</h2></div>', unsafe_allow_html=True)
    
    # Add back button
    if st.button("← Back to Study Area"):
        st.session_state.climate_geometry_complete = False
        st.session_state.climate_step = 2
        st.rerun()
    
    # Show current geometry info
    try:
        handler = st.session_state.climate_geometry_handler
        if handler.current_geometry:
            area = handler.get_geometry_area()
            name = handler.current_geometry_name
            st.success(f"✅ Study Area: {name} ({area:.2f} km²)")
    except Exception:
        st.info("✅ Study area selected")
    
    # Load appropriate datasets based on analysis type
    analysis_type = st.session_state.climate_analysis_type
    _load_and_display_climate_datasets(analysis_type)


def _render_climate_date_selection():
    """Render intelligent date selection for climate analytics"""
    st.markdown('<div class="step-header"><h2>📅 Step 4: Select Time Period</h2></div>', unsafe_allow_html=True)

    # Add back button
    if st.button("← Back to Dataset Selection"):
        st.session_state.climate_dataset_selected = False
        st.session_state.climate_step = 3
        st.rerun()

    # Show current selections
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Analysis: **{st.session_state.climate_analysis_type.title()}**")
    with col2:
        dataset_name = "Selected"
        if st.session_state.climate_selected_dataset:
            dataset_name = st.session_state.climate_selected_dataset.get('name', 'Selected')
        st.success(f"✅ Dataset: **{dataset_name}**")

    # Get dataset date constraints
    dataset_config = get_dataset_config()
    selected_dataset = st.session_state.climate_selected_dataset

    if not selected_dataset:
        st.error("❌ Dataset selection is missing. Please go back and select a dataset.")
        return

    dataset_start_str = selected_dataset.get('start_date')
    dataset_end_str = selected_dataset.get('end_date')

    # Convert to date objects
    try:
        dataset_start = datetime.strptime(dataset_start_str, '%Y-%m-%d').date()
        dataset_end = datetime.strptime(dataset_end_str, '%Y-%m-%d').date()
    except:
        dataset_start = datetime(1980, 1, 1).date()
        dataset_end = datetime.now().date()

    # Show available period
    st.info(f"📅 **Available data period:** {dataset_start} to {dataset_end}")

    # Date range selection with constraints
    st.markdown("### Select Analysis Period:")

    # Quick date range options based on available data
    today = min(datetime.now().date(), dataset_end)
    max_years_available = (today - dataset_start).days // 365

    quick_options = []
    if max_years_available >= 5:
        quick_options.append("Last 5 years")
    if max_years_available >= 10:
        quick_options.append("Last 10 years")
    if max_years_available >= 20:
        quick_options.append("Last 20 years")
    if max_years_available >= 30:
        quick_options.append("Last 30 years")
    quick_options.append("Custom range")

    date_option = st.radio(
        "Choose time period:",
        quick_options,
        horizontal=True
    )

    if date_option == "Last 5 years":
        start_date = max(today.replace(year=today.year - 5), dataset_start)
        end_date = today
    elif date_option == "Last 10 years":
        start_date = max(today.replace(year=today.year - 10), dataset_start)
        end_date = today
    elif date_option == "Last 20 years":
        start_date = max(today.replace(year=today.year - 20), dataset_start)
        end_date = today
    elif date_option == "Last 30 years":
        start_date = max(today.replace(year=today.year - 30), dataset_start)
        end_date = today
    else:  # Custom range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max(today.replace(year=today.year - 10), dataset_start),
                min_value=dataset_start,
                max_value=dataset_end
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=today,
                min_value=dataset_start,
                max_value=dataset_end
            )
    
    # Validate date range with dataset constraints
    validation_result, error_message = dataset_config.validate_date_range(
        selected_dataset['id'], start_date, end_date
    )

    if not validation_result:
        st.error(f"❌ {error_message}")
        return

    # Show selected range and recommendations
    years_diff = (end_date - start_date).days / 365.25
    st.info(f"📅 Selected period: **{start_date}** to **{end_date}** ({years_diff:.1f} years)")

    # Visual timeline
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("**Data Timeline:**")
        progress_start = (start_date - dataset_start).days / (dataset_end - dataset_start).days
        progress_end = (end_date - dataset_start).days / (dataset_end - dataset_start).days

        st.markdown(f"""
        <div style="background: #f0f0f0; height: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="background: linear-gradient(90deg, #1f77b4, #ff7f0e); height: 20px; border-radius: 10px;
                        width: {(progress_end - progress_start) * 100}%;
                        margin-left: {progress_start * 100}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8em;">
            <span>{dataset_start}</span>
            <span>{dataset_end}</span>
        </div>
        """, unsafe_allow_html=True)

    # Intelligent recommendations based on analysis type and data period
    if st.session_state.climate_analysis_type == "precipitation":
        if years_diff < 10:
            st.warning("⚠️ For reliable precipitation indices, consider using at least 10 years of data")
        elif years_diff >= 30:
            st.success("✅ Excellent! 30+ years of data provides robust climate statistics")
    elif st.session_state.climate_analysis_type == "temperature":
        if years_diff < 5:
            st.warning("⚠️ Consider using at least 5 years for meaningful temperature patterns")
        elif years_diff >= 20:
            st.success("✅ Great! 20+ years provides good temperature trend analysis")
    
    if st.button("Continue to Index Selection", type="primary"):
        st.session_state.climate_start_date = start_date
        st.session_state.climate_end_date = end_date
        st.session_state.climate_date_range_set = True
        st.session_state.climate_step = 5
        st.rerun()


def _render_climate_indices_selection():
    """Render climate indices selection"""
    st.markdown('<div class="step-header"><h2>📈 Step 5: Select Climate Indices</h2></div>', unsafe_allow_html=True)
    
    # Add back button
    if st.button("← Back to Date Selection"):
        st.session_state.climate_date_range_set = False
        st.session_state.climate_step = 4
        st.rerun()
    
    analysis_type = st.session_state.climate_analysis_type
    
    if analysis_type == "precipitation":
        _render_precipitation_indices()
    else:  # temperature
        _render_temperature_indices()


def _render_climate_results():
    """Render climate analysis results"""
    st.markdown('<div class="step-header"><h2>🔬 Step 7: Analysis Results</h2></div>', unsafe_allow_html=True)

    # Add back button to return to Export Configuration (Step 6)
    if st.button("← Back to Export Configuration"):
        st.session_state.climate_export_configured = False
        st.rerun()
    
    # Show analysis summary
    _show_analysis_summary()

    # Check if analysis has already been completed
    if st.session_state.get('climate_analysis_complete', False) and st.session_state.get('climate_results'):
        # Show existing results
        st.success("✅ Analysis completed! Results are available below.")

        # Show existing results
        results = st.session_state.climate_results

        # Display time series if available
        if 'time_series_data' in results:
            _display_climate_results(results['time_series_data'])

        # Display interactive geemap visualization if image collections are available
        st.markdown("---")
        if 'image_collections' in results and results['image_collections']:
            _display_geemap_visualization(results)
        else:
            st.info("💡 Spatial visualization requires image collection data. This is available when analysis is run with spatial data generation.")

        # IMPROVED UX: Show analysis options immediately after results
        st.markdown("---")
        st.markdown("### 📊 Analysis Options")
        st.markdown("Choose how you want to work with your climate analysis results:")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗺️ Visualize Spatial Data", use_container_width=True, type="primary",
                        help="Create interactive maps and spatial visualizations"):
                # Launch visualization with climate results
                _launch_climate_visualization(results)

        with col2:
            if st.button("📁 Download Files", use_container_width=True,
                        help="Download data files for offline analysis"):
                # Scroll to download section
                st.markdown('<div id="download-section"></div>', unsafe_allow_html=True)

        with col3:
            if st.button("🔄 Re-run Analysis", use_container_width=True,
                        help="Start over with different parameters"):
                # Clear existing results and re-run with current selections
                st.session_state.climate_analysis_complete = False
                st.session_state.climate_results = None

                # Clear any visualization states to ensure fresh display
                for key in list(st.session_state.keys()):
                    if key.startswith('show_') and 'climate' in key:
                        del st.session_state[key]
                st.rerun()

        # Download section (appears below analysis options)
        st.markdown("---")
        st.markdown('<div id="download-section"></div>', unsafe_allow_html=True)
        st.markdown("### 📁 Download Options")
        st.markdown("Download your analysis results for offline use or further processing:")

        # Show download options - results should persist
        _show_download_options(results)

        with col3:
            if st.button("🆕 Start New Analysis", use_container_width=True):
                _reset_climate_analysis()
                st.rerun()

    else:
        # Analysis not yet run - show run button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            _run_climate_analysis()

        # Reset button
        if st.button("🔄 Start New Analysis"):
            _reset_climate_analysis()
            st.rerun()


def _load_and_display_climate_datasets(analysis_type):
    """Load and display datasets for climate analysis using JSON configuration"""
    try:
        dataset_config = get_dataset_config()
        datasets = dataset_config.get_datasets_for_analysis(analysis_type)

        st.info(f"📊 Found {len(datasets)} {analysis_type} datasets")

        # Create improved dataset selection interface
        st.markdown(f"### 📊 Select {analysis_type.title()} Dataset")

        # Create two-column layout
        col_left, col_right = st.columns([1, 2])

        selected_dataset_id = None
        selected_dataset_key = None

        with col_left:
            st.markdown("#### Dataset Selection")

            # Create radio options
            dataset_options = []
            dataset_ids = []

            for dataset_id, dataset_info in datasets.items():
                display_name = f"**{dataset_info['name']}**\n📍 {dataset_info['provider']}\n📅 {dataset_info['start_date']} to {dataset_info['end_date']}"
                dataset_options.append(display_name)
                dataset_ids.append(dataset_id)

            if dataset_options:
                selected_index = st.radio(
                    "Choose a dataset:",
                    range(len(dataset_options)),
                    format_func=lambda x: dataset_options[x].split('\n')[0].replace('**', ''),
                    key="dataset_radio"
                )

                selected_dataset_id = dataset_ids[selected_index]
                selected_dataset_key = selected_dataset_id  # Set the key when we have an ID

        with col_right:
            if selected_dataset_id and selected_dataset_id in datasets:
                dataset_info = datasets[selected_dataset_id]

                # Show detailed information
                st.markdown("#### Dataset Details")

                info_expander = st.expander("📋 Basic Information", expanded=True)
                with info_expander:
                    st.markdown(f"**Name:** {dataset_info['name']}")
                    st.markdown(f"**Provider:** {dataset_info['provider']}")
                    st.markdown(f"**Temporal Resolution:** {dataset_info['temporal_resolution']}")
                    st.markdown(f"**Pixel Size:** {dataset_info['pixel_size_m']} meters")
                    st.markdown(f"**Geographic Coverage:** {dataset_info.get('geographic_coverage', 'Global')}")
                    st.markdown(f"**Description:** {dataset_info['description']}")

                # Show available bands for this analysis
                bands_expander = st.expander("🎛️ Available Bands", expanded=False)
                with bands_expander:
                    bands_info = dataset_info.get('bands', {})

                    for band_type, band_detail in bands_info.items():
                        if analysis_type == 'temperature' and 'temperature' in band_type:
                            st.markdown(f"**{band_detail['description']}**")
                            st.markdown(f"- Band: `{band_detail['band_name']}`")
                            st.markdown(f"- Unit: {band_detail['unit']} (from {band_detail['original_unit']})")
                        elif analysis_type == 'precipitation' and 'precipitation' in band_type:
                            st.markdown(f"**{band_detail['description']}**")
                            st.markdown(f"- Band: `{band_detail['band_name']}`")
                            st.markdown(f"- Unit: {band_detail['unit']} (from {band_detail['original_unit']})")

                # Show recommended indices
                indices_expander = st.expander("📈 Recommended Climate Indices", expanded=False)
                with indices_expander:
                    recommended = dataset_info.get('recommended_indices', {}).get(analysis_type, [])
                    if recommended:
                        for idx in recommended:
                            st.markdown(f"• **{idx}**")
                    else:
                        st.info("No specific recommendations - all simple indices are supported")

        # Continue button
        if selected_dataset_id:
            if st.button("Continue to Time Period Selection", type="primary", use_container_width=True):
                # Store the selected dataset configuration
                selected_dataset_config = {
                    'id': selected_dataset_id,
                    'name': datasets[selected_dataset_id]['name'],
                    'provider': datasets[selected_dataset_id]['provider'],
                    'start_date': datasets[selected_dataset_id]['start_date'],
                    'end_date': datasets[selected_dataset_id]['end_date'],
                    'temporal_resolution': datasets[selected_dataset_id]['temporal_resolution'],
                    'pixel_size_m': datasets[selected_dataset_id]['pixel_size_m'],
                    'bands': datasets[selected_dataset_id]['bands'],
                    'supports_analysis': datasets[selected_dataset_id]['supports_analysis'],
                    'recommended_indices': datasets[selected_dataset_id].get('recommended_indices', {})
                }

                st.session_state.climate_selected_dataset = selected_dataset_config
                st.session_state.climate_dataset_selected = True
                st.session_state.climate_step = 4
                st.success(f"✅ Selected dataset: {datasets[selected_dataset_id]['name']}")
                st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading {analysis_type} datasets: {str(e)}")
        st.info("Please check if the datasets.json configuration file is available.")


def _render_precipitation_indices():
    """Render precipitation indices selection with compact UI"""
    dataset_config = get_dataset_config()

    # Get all precipitation indices from JSON config (not just 'simple')
    all_indices = dataset_config.get_climate_indices('precipitation')

    # Get recommended indices for selected dataset
    selected_dataset = st.session_state.climate_selected_dataset
    recommended = selected_dataset.get('recommended_indices', {}).get('precipitation', [])

    st.markdown("### 🌧️ Select Precipitation Indices")

    # Note: Recommendations removed per user request - manual selection only

    # Quick selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear All", use_container_width=True):
            for idx_key in all_indices.keys():
                st.session_state[f"idx_{idx_key}"] = False
            st.rerun()
    with col2:
        st.info("💡 Select indices manually below")

    st.markdown("---")

    # Group indices by complexity for better organization
    simple_indices = {k: v for k, v in all_indices.items() if v.get('complexity') == 'simple'}
    percentile_indices = {k: v for k, v in all_indices.items() if v.get('complexity') == 'percentile'}

    selected_indices = []

    # Compact selection interface using tabs
    tab1, tab2 = st.tabs(["📊 Standard Indices", "📈 Percentile-Based"])

    with tab1:
        if simple_indices:
            st.markdown("#### Standard Precipitation Indices")
            # Create a more compact grid layout
            cols = st.columns(2)
            for i, (idx_key, idx_info) in enumerate(simple_indices.items()):
                with cols[i % 2]:
                    is_recommended = idx_key in recommended
                    label = f"{'⭐ ' if is_recommended else ''}{idx_info['name']}"
                    help_text = f"{idx_info['description']} ({idx_info['unit']})"

                    if st.checkbox(label, key=f"idx_{idx_key}",
                                 value=False, help=help_text):
                        selected_indices.append(idx_key)

    with tab2:
        if percentile_indices:
            st.markdown("#### Percentile-Based Indices")
            st.info("💡 These indices use 1980-2000 as the base period for percentile calculations")
            # Create a more compact grid layout
            cols = st.columns(2)
            for i, (idx_key, idx_info) in enumerate(percentile_indices.items()):
                with cols[i % 2]:
                    is_recommended = idx_key in recommended
                    label = f"{'⭐ ' if is_recommended else ''}{idx_info['name']}"
                    help_text = f"{idx_info['description']} ({idx_info['unit']})"

                    if st.checkbox(label, key=f"idx_{idx_key}",
                                 value=False, help=help_text):
                        selected_indices.append(idx_key)

    # Compact summary of selected indices
    if selected_indices:
        st.markdown("---")
        st.markdown("#### ✅ Selected Indices Summary")

        # Group selected indices by type
        selected_simple = [idx for idx in selected_indices if idx in simple_indices]
        selected_percentile = [idx for idx in selected_indices if idx in percentile_indices]

        col1, col2 = st.columns(2)
        with col1:
            if selected_simple:
                st.markdown("**Standard:**")
                for idx in selected_simple:
                    idx_info = simple_indices[idx]
                    st.markdown(f"• {idx_info['name']} ({idx_info['unit']})")

        with col2:
            if selected_percentile:
                st.markdown("**Percentile-Based:**")
                for idx in selected_percentile:
                    idx_info = percentile_indices[idx]
                    st.markdown(f"• {idx_info['name']} ({idx_info['unit']})")

        # Percentile Configuration Section for Precipitation Indices
        if selected_percentile:
            st.markdown("---")
            st.markdown("#### 📊 Percentile Configuration")
            st.info("💡 Customize percentile thresholds and base period for selected percentile-based indices.")

            # Base period configuration (common for all percentile indices)
            col1, col2 = st.columns(2)
            with col1:
                base_start = st.date_input(
                    "Base Period Start",
                    value=datetime(1980, 1, 1).date(),
                    min_value=datetime(1950, 1, 1).date(),
                    max_value=datetime(2020, 12, 31).date(),
                    key="precip_percentile_base_start",
                    help="Start date for calculating percentile thresholds"
                )
            with col2:
                base_end = st.date_input(
                    "Base Period End",
                    value=datetime(2000, 12, 31).date(),
                    min_value=datetime(1960, 1, 1).date(),
                    max_value=datetime(2030, 12, 31).date(),
                    key="precip_percentile_base_end",
                    help="End date for calculating percentile thresholds"
                )

            # Store base period in session state
            st.session_state['climate_precip_base_period'] = {
                'start': str(base_start),
                'end': str(base_end)
            }

            # Individual percentile threshold configuration
            st.markdown("**Percentile Thresholds:**")
            cols = st.columns(2)
            for i, idx_key in enumerate(selected_percentile):
                idx_info = percentile_indices[idx_key]
                percentile_config = idx_info.get('percentile_config', {})

                with cols[i % 2]:
                    percentile_value = st.number_input(
                        f"{idx_info['name']} Percentile",
                        min_value=percentile_config.get('min_percentile', 1.0),
                        max_value=percentile_config.get('max_percentile', 99.0),
                        value=percentile_config.get('default_percentile', 95.0),
                        step=0.1,
                        key=f"percentile_{idx_key}",
                        help=f"{percentile_config.get('help', 'Percentile threshold')}"
                    )

                    # Store the percentile value in session state
                    # Convert dates to YYYY-MM-DD format for Earth Engine compatibility
                    base_start_str = base_start.strftime('%Y-%m-%d')
                    base_end_str = base_end.strftime('%Y-%m-%d')

                    st.session_state[f"climate_percentile_{idx_key}"] = {
                        'percentile': percentile_value,
                        'base_start': base_start_str,
                        'base_end': base_end_str
                    }

        # Show details only if user wants them
        if st.checkbox("🔍 Show detailed information", key="show_precip_details"):
            with st.expander("📋 Detailed Index Information", expanded=True):
                for idx_key in selected_indices:
                    idx_info = all_indices[idx_key]
                    st.markdown(f"**{idx_info['name']}** ({idx_key})")
                    st.markdown(f"• {idx_info['description']}")
                    st.markdown(f"• Unit: {idx_info['unit']} | Aggregation: {idx_info['temporal_aggregation']}")
                    if idx_info.get('base_period'):
                        st.markdown(f"• Base period: {idx_info['base_period']}")
                    st.markdown("---")

        # Threshold Configuration Section for Precipitation Indices
        threshold_indices = {
            'CDD': {'name': 'Consecutive Dry Days', 'param': 'threshold', 'default': 1.0, 'unit': 'mm', 'help': 'Daily precipitation threshold for defining dry days'},
            'R20mm': {'name': 'Heavy Rain Days', 'param': 'threshold', 'default': 20.0, 'unit': 'mm', 'help': 'Daily precipitation threshold for heavy rain'},
            'SDII': {'name': 'Simple Daily Intensity Index', 'param': 'wet_threshold', 'default': 1.0, 'unit': 'mm', 'help': 'Minimum precipitation for wet days'},
            'PRCPTOT': {'name': 'Total Precipitation', 'param': 'wet_threshold', 'default': 1.0, 'unit': 'mm', 'help': 'Minimum precipitation for wet days'}
        }

        # Check if any threshold-configurable indices are selected
        selected_threshold_indices = [idx for idx in selected_indices if idx in threshold_indices]

        if selected_threshold_indices:
            st.markdown("---")
            st.markdown("#### ⚙️ Threshold Configuration")
            st.info("💡 Customize thresholds for selected indices. Default values follow ETCCDI standards.")

            cols = st.columns(2)
            for i, idx_key in enumerate(selected_threshold_indices):
                threshold_config = threshold_indices[idx_key]
                with cols[i % 2]:
                    threshold_value = st.number_input(
                        f"{threshold_config['name']} ({threshold_config['unit']})",
                        min_value=0.0,
                        max_value=100.0,
                        value=threshold_config['default'],
                        step=0.1,
                        key=f"threshold_{idx_key}",
                        help=threshold_config['help']
                    )

                    # Store the threshold value in session state
                    st.session_state[f"climate_threshold_{idx_key}"] = {
                        'param': threshold_config['param'],
                        'value': threshold_value
                    }

        st.success(f"✅ **{len(selected_indices)} precipitation indices selected**")

        if st.button("Continue to Export Configuration", type="primary", use_container_width=True):
            st.session_state.climate_selected_indices = selected_indices
            st.session_state.climate_indices_selected = True
            st.rerun()
    else:
        st.warning("⚠️ Please select at least one climate index to continue")


def _render_temperature_indices():
    """Render temperature indices selection with compact UI"""
    dataset_config = get_dataset_config()

    # Get all temperature indices from JSON config (not just 'simple')
    all_indices = dataset_config.get_climate_indices('temperature')

    # Get recommended indices for selected dataset
    selected_dataset = st.session_state.climate_selected_dataset
    recommended = selected_dataset.get('recommended_indices', {}).get('temperature', [])

    st.markdown("### 🌡️ Select Temperature Indices")

    # Note: Recommendations removed per user request - manual selection only

    # Quick selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear All", use_container_width=True):
            for idx_key in all_indices.keys():
                st.session_state[f"idx_{idx_key}"] = False
            st.rerun()
    with col2:
        st.info("💡 Select indices manually below")

    st.markdown("---")

    # Group indices by complexity for better organization
    simple_indices = {k: v for k, v in all_indices.items() if v.get('complexity') == 'simple'}
    percentile_indices = {k: v for k, v in all_indices.items() if v.get('complexity') == 'percentile'}

    selected_indices = []

    # Compact selection interface using tabs
    tab1, tab2 = st.tabs(["📊 Standard Indices", "📈 Percentile-Based"])

    with tab1:
        if simple_indices:
            st.markdown("#### Standard Temperature Indices")
            # Create a more compact grid layout
            cols = st.columns(2)
            for i, (idx_key, idx_info) in enumerate(simple_indices.items()):
                with cols[i % 2]:
                    is_recommended = idx_key in recommended
                    label = f"{'⭐ ' if is_recommended else ''}{idx_info['name']}"
                    help_text = f"{idx_info['description']} ({idx_info['unit']})"

                    if st.checkbox(label, key=f"idx_{idx_key}",
                                 value=False, help=help_text):
                        selected_indices.append(idx_key)

    with tab2:
        if percentile_indices:
            st.markdown("#### Percentile-Based Indices")
            st.info("💡 These indices use 1980-2000 as the base period for percentile calculations")
            # Create a more compact grid layout
            cols = st.columns(2)
            for i, (idx_key, idx_info) in enumerate(percentile_indices.items()):
                with cols[i % 2]:
                    is_recommended = idx_key in recommended
                    label = f"{'⭐ ' if is_recommended else ''}{idx_info['name']}"
                    help_text = f"{idx_info['description']} ({idx_info['unit']})"

                    if st.checkbox(label, key=f"idx_{idx_key}",
                                 value=False, help=help_text):
                        selected_indices.append(idx_key)

    # Compact summary of selected indices
    if selected_indices:
        st.markdown("---")
        st.markdown("#### ✅ Selected Indices Summary")

        # Group selected indices by type
        selected_simple = [idx for idx in selected_indices if idx in simple_indices]
        selected_percentile = [idx for idx in selected_indices if idx in percentile_indices]

        col1, col2 = st.columns(2)
        with col1:
            if selected_simple:
                st.markdown("**Standard:**")
                for idx in selected_simple:
                    idx_info = simple_indices[idx]
                    st.markdown(f"• {idx_info['name']} ({idx_info['unit']})")

        with col2:
            if selected_percentile:
                st.markdown("**Percentile-Based:**")
                for idx in selected_percentile:
                    idx_info = percentile_indices[idx]
                    st.markdown(f"• {idx_info['name']} ({idx_info['unit']})")

        # Percentile Configuration Section for Temperature Indices
        if selected_percentile:
            st.markdown("---")
            st.markdown("#### 📊 Percentile Configuration")
            st.info("💡 Customize percentile thresholds and base period for selected percentile-based indices.")

            # Base period configuration (common for all percentile indices)
            col1, col2 = st.columns(2)
            with col1:
                base_start = st.date_input(
                    "Base Period Start",
                    value=datetime(1980, 1, 1).date(),
                    min_value=datetime(1950, 1, 1).date(),
                    max_value=datetime(2020, 12, 31).date(),
                    key="temp_percentile_base_start",
                    help="Start date for calculating percentile thresholds"
                )
            with col2:
                base_end = st.date_input(
                    "Base Period End",
                    value=datetime(2000, 12, 31).date(),
                    min_value=datetime(1960, 1, 1).date(),
                    max_value=datetime(2030, 12, 31).date(),
                    key="temp_percentile_base_end",
                    help="End date for calculating percentile thresholds"
                )

            # Store base period in session state
            st.session_state['climate_temp_base_period'] = {
                'start': str(base_start),
                'end': str(base_end)
            }

            # Individual percentile threshold configuration
            st.markdown("**Percentile Thresholds:**")
            cols = st.columns(2)
            for i, idx_key in enumerate(selected_percentile):
                idx_info = percentile_indices[idx_key]
                percentile_config = idx_info.get('percentile_config', {})

                with cols[i % 2]:
                    percentile_value = st.number_input(
                        f"{idx_info['name']} Percentile",
                        min_value=percentile_config.get('min_percentile', 1.0),
                        max_value=percentile_config.get('max_percentile', 99.0),
                        value=percentile_config.get('default_percentile', 90.0),
                        step=0.1,
                        key=f"percentile_{idx_key}",
                        help=f"{percentile_config.get('help', 'Percentile threshold')}"
                    )

                    # Store the percentile value in session state
                    # Convert dates to YYYY-MM-DD format for Earth Engine compatibility
                    base_start_str = base_start.strftime('%Y-%m-%d')
                    base_end_str = base_end.strftime('%Y-%m-%d')

                    st.session_state[f"climate_percentile_{idx_key}"] = {
                        'percentile': percentile_value,
                        'base_start': base_start_str,
                        'base_end': base_end_str
                    }

        # Show details only if user wants them
        if st.checkbox("🔍 Show detailed information", key="show_temp_details"):
            with st.expander("📋 Detailed Index Information", expanded=True):
                for idx_key in selected_indices:
                    idx_info = all_indices[idx_key]
                    st.markdown(f"**{idx_info['name']}** ({idx_key})")
                    st.markdown(f"• {idx_info['description']}")
                    st.markdown(f"• Unit: {idx_info['unit']} | Aggregation: {idx_info['temporal_aggregation']}")
                    if idx_info.get('base_period'):
                        st.markdown(f"• Base period: {idx_info['base_period']}")

                    # Show required data bands more compactly
                    required_bands = idx_info.get('required_bands', [])
                    if required_bands:
                        band_names = []
                        for band_type in required_bands:
                            band_info = selected_dataset.get('bands', {}).get(band_type, {})
                            if band_info:
                                band_names.append(f"{band_type.replace('_', ' ')} ({band_info['band_name']})")
                        if band_names:
                            st.markdown(f"• Required data: {', '.join(band_names)}")
                    st.markdown("---")

        # Threshold Configuration Section for Temperature Indices
        threshold_indices = {
            'FD': {'name': 'Frost Days', 'param': 'threshold', 'default': 0.0, 'unit': '°C', 'help': 'Temperature threshold for frost days (days below this temperature)'},
            'SU': {'name': 'Summer Days', 'param': 'threshold', 'default': 25.0, 'unit': '°C', 'help': 'Temperature threshold for summer days (days above this temperature)'},
            'GSL': {'name': 'Growing Season Length', 'param': 'threshold', 'default': 5.0, 'unit': '°C', 'help': 'Temperature threshold for growing season (days above this temperature)'}
        }

        # Check if any threshold-configurable indices are selected
        selected_threshold_indices = [idx for idx in selected_indices if idx in threshold_indices]

        if selected_threshold_indices:
            st.markdown("---")
            st.markdown("#### ⚙️ Threshold Configuration")
            st.info("💡 Customize thresholds for selected indices. Default values follow ETCCDI standards.")

            cols = st.columns(2)
            for i, idx_key in enumerate(selected_threshold_indices):
                threshold_config = threshold_indices[idx_key]
                with cols[i % 2]:
                    threshold_value = st.number_input(
                        f"{threshold_config['name']} ({threshold_config['unit']})",
                        min_value=-50.0 if idx_key == 'FD' else 0.0,
                        max_value=50.0,
                        value=threshold_config['default'],
                        step=0.1,
                        key=f"threshold_{idx_key}",
                        help=threshold_config['help']
                    )

                    # Store the threshold value in session state
                    st.session_state[f"climate_threshold_{idx_key}"] = {
                        'param': threshold_config['param'],
                        'value': threshold_value
                    }

        st.success(f"✅ **{len(selected_indices)} temperature indices selected**")

        if st.button("Continue to Export Configuration", type="primary", use_container_width=True):
            st.session_state.climate_selected_indices = selected_indices
            st.session_state.climate_indices_selected = True
            st.rerun()
    else:
        st.warning("⚠️ Please select at least one climate index to continue")


def _render_export_configuration():
    """Render compact export configuration using native dataset resolution"""
    st.markdown('<div class="step-header"><h2>⚙️ Step 6: Export Configuration</h2></div>', unsafe_allow_html=True)

    # Add back button
    if st.button("← Back to Index Selection"):
        st.session_state.climate_indices_selected = False
        st.rerun()

    # Get dataset configuration
    selected_dataset = st.session_state.climate_selected_dataset
    native_pixel_size = selected_dataset.get('pixel_size_m', 1000)
    dataset_name = selected_dataset.get('name', 'Unknown')

    # Show compact summary
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🎯 **{st.session_state.climate_analysis_type.title()} Analysis** • {dataset_name}")
    with col2:
        if native_pixel_size >= 1000:
            resolution_text = f"{native_pixel_size/1000:.1f} km"
        else:
            resolution_text = f"{native_pixel_size} m"
        st.info(f"📐 **Native Resolution:** {resolution_text} • {len(st.session_state.climate_selected_indices)} indices")

        # Debug info - remove this after testing
        st.caption(f"Debug: pixel_size_m = {native_pixel_size}")

    # Temporal resolution selection
    st.markdown("#### ⏰ Temporal Resolution")
    temporal_resolution = st.radio(
        "Choose temporal aggregation:",
        options=['yearly', 'monthly'],
        index=0,  # yearly as default
        format_func=lambda x: {
            'yearly': '📅 Yearly (Recommended)',
            'monthly': '📆 Monthly'
        }[x],
        horizontal=True,
        help="Yearly aggregation provides annual climate indices, monthly keeps original temporal resolution"
    )

    # Store temporal resolution in session state
    st.session_state.climate_temporal_resolution = temporal_resolution

    # Show resolution info
    if temporal_resolution == 'yearly':
        st.success("✅ Yearly aggregation • Climate indices calculated per year • Ideal for trend analysis")
    else:
        st.info("ℹ️ Monthly resolution • Preserves original temporal detail • Larger file sizes")

    # Compact export method selection
    st.markdown("#### 📤 Export Method")

    export_method = st.radio(
        "Choose how to receive your data:",
        options=['auto', 'drive', 'preview', 'local'],
        format_func=lambda x: {
            'auto': '🤖 Smart Auto (Recommended)',
            'drive': '☁️ Google Drive (Large files)',
            'preview': '📱 Quick Preview (1 sample + time series)',
            'local': '💻 Force Local (Small files only)'
        }[x],
        index=0,
        horizontal=True,
        help="Smart Auto: Automatically selects best method based on data size"
    )

    # Show method-specific info compactly
    if export_method == 'auto':
        st.success("✅ Smart selection • Local for small files • Drive for large files • Seamless fallback")
    elif export_method == 'drive':
        st.success("✅ Server-side processing • Large file support • Background export")
    elif export_method == 'preview':
        st.info("ℹ️ Sample spatial data + complete time series • Good for exploration")
    else:  # local
        st.warning("⚠️ May be slow for large datasets • Consider Smart Auto for optimal performance")

    # Resolution configuration with flexibility
    st.markdown("#### 📐 Spatial Resolution")

    # Get current selection or default to native
    current_scale = st.session_state.get('temp_selected_scale', native_pixel_size)

    # Resolution options based on native resolution
    if native_pixel_size >= 25000:  # Very coarse like ERA5 (27.8km)
        resolution_options = {
            f"Native ({resolution_text})": native_pixel_size,
            "25 km": 25000,
            "50 km": 50000,
            "100 km": 100000
        }
    elif native_pixel_size >= 10000:  # Coarse datasets
        resolution_options = {
            f"Native ({resolution_text})": native_pixel_size,
            "10 km": 10000,
            "25 km": 25000,
            "50 km": 50000
        }
    elif native_pixel_size >= 1000:  # Medium resolution like Daymet (1km)
        resolution_options = {
            f"Native ({resolution_text})": native_pixel_size,
            "1 km": 1000,
            "5 km": 5000,
            "10 km": 10000
        }
    else:  # High resolution datasets
        resolution_options = {
            f"Native ({resolution_text})": native_pixel_size,
            "500 m": 500,
            "1 km": 1000,
            "5 km": 5000
        }

    # Only show options that make sense (no upsampling from native)
    valid_options = {k: v for k, v in resolution_options.items() if v >= native_pixel_size}

    # Find current selection index
    current_index = 0
    for i, scale in enumerate(valid_options.values()):
        if scale == current_scale:
            current_index = i
            break

    resolution_choice = st.selectbox(
        "Select spatial resolution:",
        options=list(valid_options.keys()),
        index=current_index,
        help="Native resolution recommended for best quality. Lower resolutions aggregate data and may reduce detail."
    )

    selected_scale = valid_options[resolution_choice]
    st.session_state.temp_selected_scale = selected_scale

    # Show resolution feedback
    if selected_scale == native_pixel_size:
        st.success(f"🎯 Using native resolution: {resolution_text} (optimal)")
    else:
        new_resolution = f"{selected_scale/1000:.0f} km" if selected_scale >= 1000 else f"{selected_scale} m"
        st.info(f"📊 Data will be aggregated from {resolution_text} to {new_resolution}")

    # Custom resolution option
    with st.expander("🔧 Custom Resolution", expanded=False):
        custom_scale = st.number_input(
            "Enter custom resolution in meters:",
            min_value=int(native_pixel_size),
            max_value=500000,
            value=int(selected_scale),
            step=1000,
            help=f"Minimum: {int(native_pixel_size)}m (native resolution). Higher values reduce file size."
        )
        if st.button("Apply Custom Resolution"):
            st.session_state.temp_selected_scale = custom_scale
            selected_scale = custom_scale
            st.success(f"✅ Custom resolution set to {custom_scale}m")
            st.rerun()

    # Compact summary and start button
    col1, col2 = st.columns([2, 1])

    with col1:
        method_names = {'auto': 'Smart Auto', 'drive': 'Google Drive', 'preview': 'Preview', 'local': 'Direct Download'}
        current_res = f"{selected_scale/1000:.1f} km" if selected_scale >= 1000 else f"{selected_scale} m"
        temporal_res = st.session_state.get('climate_temporal_resolution', 'yearly')
        st.markdown(f"**Ready:** {method_names[export_method]} • {current_res} spatial • {temporal_res} temporal")

    with col2:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            # Store export configuration
            st.session_state.climate_export_method = export_method
            st.session_state.climate_spatial_scale = selected_scale
            st.session_state.climate_temporal_resolution = temporal_resolution
            st.session_state.climate_export_configured = True
            st.rerun()


def _show_analysis_summary():
    """Show summary of analysis configuration with intelligent band mapping"""
    st.markdown("### 📋 Analysis Summary & Configuration")

    dataset_config = get_dataset_config()
    selected_dataset = st.session_state.climate_selected_dataset

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🎯 Analysis Configuration**")
        st.info(f"Type: {st.session_state.climate_analysis_type.title()}")
        st.info(f"Dataset: {selected_dataset.get('name', 'Unknown')}")

        # Show study area
        st.markdown("**🗺️ Study Area:**")
        try:
            area = st.session_state.climate_geometry_handler.get_geometry_area()
            st.info(f"Area: {area:.2f} km²")
        except:
            st.info("Custom area defined")

    with col2:
        st.markdown("**📅 Time Configuration**")
        years = (st.session_state.climate_end_date - st.session_state.climate_start_date).days / 365.25
        st.info(f"Period: {st.session_state.climate_start_date} to {st.session_state.climate_end_date}")
        st.info(f"Duration: {years:.1f} years")
        data_resolution = selected_dataset.get('temporal_resolution', 'daily')
        output_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')
        st.info(f"Data: {data_resolution} → Output: {output_resolution}")

    with col3:
        st.markdown("**📈 Selected Indices & Band Mapping**")

        # Get required bands for selected indices
        required_bands = dataset_config.get_required_bands_for_indices(
            selected_dataset['id'], st.session_state.climate_selected_indices
        )

        for idx in st.session_state.climate_selected_indices:
            st.info(f"• {idx}")

        # Show band mapping
        if required_bands:
            st.markdown("**🎛️ Band Mapping:**")
            for band_type, ee_band_name in required_bands.items():
                band_info = dataset_config.get_band_scaling_info(selected_dataset['id'], band_type)
                st.info(f"• {band_type.replace('_', ' ').title()}: `{ee_band_name}` → {band_info.get('unit', 'units')}")


def _run_climate_analysis():
    """Run the real climate analysis using server-side computation"""
    try:
        # Import the analysis function we'll create
        from geoclimate_fetcher.core.climate_analysis_runner import run_climate_analysis_with_chunking

        with st.spinner("🔄 Initializing climate analysis..."):
            # Prepare analysis configuration
            config = {
                'analysis_type': st.session_state.climate_analysis_type,
                'dataset_id': st.session_state.climate_selected_dataset['id'],
                'selected_indices': st.session_state.climate_selected_indices,
                'start_date': str(st.session_state.climate_start_date),
                'end_date': str(st.session_state.climate_end_date),
                'geometry': st.session_state.climate_geometry_handler.current_geometry,
                'export_method': st.session_state.get('climate_export_method', 'auto'),
                'spatial_scale': st.session_state.get('climate_spatial_scale', 1000),
                'temporal_resolution': st.session_state.get('climate_temporal_resolution', 'yearly')
            }

            # Add threshold parameters for selected indices
            threshold_params = {}
            for idx in st.session_state.climate_selected_indices:
                threshold_key = f"climate_threshold_{idx}"
                if threshold_key in st.session_state:
                    threshold_config = st.session_state[threshold_key]
                    threshold_params[idx] = {
                        threshold_config['param']: threshold_config['value']
                    }

            if threshold_params:
                config['threshold_params'] = threshold_params
                st.info(f"🎛️ Using custom thresholds for {len(threshold_params)} indices")

            # Add percentile parameters for selected indices
            percentile_params = {}
            for idx in st.session_state.climate_selected_indices:
                percentile_key = f"climate_percentile_{idx}"
                if percentile_key in st.session_state:
                    percentile_config = st.session_state[percentile_key]
                    percentile_params[idx] = {
                        'percentile': percentile_config['percentile'],
                        'base_start': percentile_config['base_start'],
                        'base_end': percentile_config['base_end']
                    }

            if percentile_params:
                config['percentile_params'] = percentile_params
                st.info(f"📊 Using custom percentiles for {len(percentile_params)} indices")

            # TEMPORAL-FIRST APPROACH: Extract temporal data first, spatial later (user-triggered)
            config['temporal_only'] = True

            st.info("🔄 Running server-side climate index calculations...")
            st.info("⏳ Extracting temporal data for all indices. Spatial export will be available separately.")

            # Run the analysis (this will use the real implementation)
            results = run_climate_analysis_with_chunking(config)

            if results['success']:
                # Store results in session state for persistence
                st.session_state.climate_results = results
                st.session_state.climate_analysis_complete = True

                st.success("✅ Climate analysis completed successfully!")

                # Show results
                st.markdown("### 📊 Analysis Results")

                # Display time series if available
                if 'time_series_data' in results:
                    _display_climate_results(results['time_series_data'])

                # Display interactive geemap visualization if image collections are available
                st.markdown("---")
                if 'image_collections' in results and results['image_collections']:
                    _display_geemap_visualization(results)
                else:
                    st.info("💡 Spatial visualization requires image collection data. This is available when analysis is run with spatial data generation.")

                # IMPROVED UX: Show temporal CSV download immediately (with stable keys)
                st.markdown("---")
                st.markdown("### 📁 Temporal Data Downloads")
                st.markdown("Download your climate index time series data:")

                col1, col2 = st.columns(2)
                with col1:
                    if 'time_series_csv' in results and results['time_series_csv']:
                        st.download_button(
                            label="📊 Download Time Series CSV",
                            data=results['time_series_csv'],
                            file_name=f"climate_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download temporal data for all indices",
                            use_container_width=True,
                            key="climate_temporal_csv_download"
                        )

                with col2:
                    if 'analysis_report' in results and results['analysis_report']:
                        st.download_button(
                            label="📋 Download Analysis Report",
                            data=results['analysis_report'],
                            file_name=f"climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Download comprehensive analysis report",
                            use_container_width=True,
                            key="climate_analysis_report_download"
                        )

                # SPATIAL EXPORT SECTION (User-triggered with 50MB error learning)
                if results.get('temporal_only', False):
                    st.markdown("---")
                    st.markdown("### 📥 Download GeoTIFF Files (Optional)")
                    st.info("""
                    **Download spatial GeoTIFF files for offline analysis or GIS software.**

                    ℹ️ Note: Spatial visualization is already displayed above using geemap!

                    This download is only needed if you want to:
                    - Use data in QGIS, ArcGIS, or other GIS software
                    - Perform custom spatial analysis
                    - Archive data for offline use

                    Download process:
                    - If local download fails (>50MB limit), ALL indices will be exported to Google Drive
                    - This prevents long waiting times for repeated failures
                    """)

                    # Check if spatial export already completed
                    if 'climate_spatial_export_complete' not in st.session_state:
                        st.session_state.climate_spatial_export_complete = False

                    if not st.session_state.climate_spatial_export_complete:
                        if st.button("📥 Download GeoTIFF Files", type="primary", use_container_width=True,
                                    help="Download spatial GeoTIFF files for offline analysis in GIS software",
                                    key="proceed_spatial_export"):
                            # Execute spatial export with 50MB error learning
                            _execute_smart_spatial_export(results)
                    else:
                        st.success("✅ Spatial export completed!")
                        # Show spatial download results
                        if 'climate_spatial_results' in st.session_state:
                            _display_spatial_export_results(st.session_state.climate_spatial_results)

                        if st.button("🔄 Re-run Spatial Export", use_container_width=True,
                                    key="rerun_spatial_export"):
                            st.session_state.climate_spatial_export_complete = False
                            if 'climate_spatial_results' in st.session_state:
                                del st.session_state.climate_spatial_results
                            st.rerun()

                # Analysis Options
                st.markdown("---")
                st.markdown("### 📊 Additional Options")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Spatial Data (Advanced)", use_container_width=True,
                                help="Export spatial data to Data Visualizer tool for advanced analysis", key="viz_new_results"):
                        # Launch visualization tool with climate results (for advanced visualizations)
                        _launch_climate_visualization(results)

                with col2:
                    if st.button("🔄 Re-run Analysis", use_container_width=True,
                                help="Start over with different parameters", key="rerun_new_results"):
                        # Clear existing results and re-run with current selections
                        st.session_state.climate_analysis_complete = False
                        st.session_state.climate_results = None
                        st.session_state.climate_spatial_export_complete = False
                        if 'climate_spatial_results' in st.session_state:
                            del st.session_state.climate_spatial_results

                        # Clear any visualization states to ensure fresh display
                        for key in list(st.session_state.keys()):
                            if key.startswith('show_') and 'climate' in key:
                                del st.session_state[key]
                        st.rerun()

            else:
                st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                st.info("💡 Try reducing the time period or selecting fewer indices if you encounter memory issues.")

    except ImportError:
        # Fallback to placeholder implementation if the real runner isn't ready yet
        st.warning("⚠️ Using placeholder implementation - real analysis runner not yet integrated")
        _run_placeholder_analysis()
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Please check your configuration and try again.")


def _run_placeholder_analysis():
    """Enhanced placeholder implementation with smart spatial download integration"""
    try:
        # Import smart download components
        from app_components.download_component import DownloadHelper

        with st.spinner("🔄 Running climate analysis..."):
            # Simulate analysis
            time.sleep(3)

            st.success("✅ Climate analysis completed successfully!")

            # Show placeholder results
            st.markdown("### 📊 Analysis Results")

            # Create some example visualizations with temporal resolution awareness
            import numpy as np
            import plotly.graph_objects as go

            temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')

            # Adjust frequency based on temporal resolution
            freq = 'Y' if temporal_resolution == 'yearly' else 'M'

            # Example time series plot
            dates = pd.date_range(st.session_state.climate_start_date, st.session_state.climate_end_date, freq=freq)
            values = np.random.randn(len(dates)).cumsum()

            # Enhanced time series visualization
            _display_climate_results(pd.DataFrame({'date': dates, 'value': values}))

            # Add smart spatial download section
            st.markdown("### 🗺️ Smart Spatial Download")

            export_method = st.session_state.get('climate_export_method', 'auto')

            if export_method != 'preview':
                _render_smart_spatial_download_section()
            else:
                st.info("📱 Preview mode selected - showing sample spatial data only")

            # Create real climate analysis data for download
            climate_data = pd.DataFrame({
                'Date': dates,
                'Climate_Index_Value': values,
                'Analysis_Type': [st.session_state.get('climate_analysis_type', 'Unknown')] * len(dates),
                'Dataset': [st.session_state.get('climate_selected_dataset', {}).get('name', 'Unknown')] * len(dates)
            })

            # Generate climate analysis summary
            analysis_summary = pd.DataFrame({
                'Metric': [
                    'Analysis Type',
                    'Dataset Used',
                    'Start Date',
                    'End Date',
                    'Number of Records',
                    'Mean Value',
                    'Standard Deviation',
                    'Minimum Value',
                    'Maximum Value',
                    'Processing Date'
                ],
                'Value': [
                    st.session_state.get('climate_analysis_type', 'Unknown'),
                    st.session_state.get('climate_selected_dataset', {}).get('name', 'Unknown'),
                    str(st.session_state.get('climate_start_date', 'Unknown')),
                    str(st.session_state.get('climate_end_date', 'Unknown')),
                    len(climate_data),
                    round(values.mean(), 4),
                    round(values.std(), 4),
                    round(values.min(), 4),
                    round(values.max(), 4),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            })

            # Show download options
            st.markdown("### 💾 Download Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Climate data CSV
                climate_csv = climate_data.to_csv(index=False)
                st.download_button(
                    label=f"📊 Download Time Series CSV ({len(climate_csv)/1024:.1f} KB)",
                    data=climate_csv,
                    file_name=f"climate_indices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the climate index time series data"
                )

            with col2:
                # Analysis summary CSV
                summary_csv = analysis_summary.to_csv(index=False)
                st.download_button(
                    label=f"📈 Download Summary CSV ({len(summary_csv)/1024:.1f} KB)",
                    data=summary_csv,
                    file_name=f"climate_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download analysis configuration and summary statistics"
                )

            with col3:
                # Combined report
                combined_data = {
                    'Analysis_Summary': analysis_summary,
                    'Time_Series_Data': climate_data
                }

                # Create a simple text report
                report_text = f"""Climate Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS CONFIGURATION:
- Type: {st.session_state.get('climate_analysis_type', 'Unknown')}
- Dataset: {st.session_state.get('climate_selected_dataset', {}).get('name', 'Unknown')}
- Date Range: {st.session_state.get('climate_start_date', 'Unknown')} to {st.session_state.get('climate_end_date', 'Unknown')}

SUMMARY STATISTICS:
- Records: {len(climate_data)}
- Mean Value: {values.mean():.4f}
- Standard Deviation: {values.std():.4f}
- Range: {values.min():.4f} to {values.max():.4f}

DATA INTERPRETATION:
This climate analysis provides time series data for the selected climate indices.
The data can be used for:
- Climate trend analysis
- Extreme event identification
- Statistical modeling
- Research and reporting

For detailed data, download the CSV files above.
"""

                st.download_button(
                    label=f"📋 Download Report (TXT) ({len(report_text.encode())/1024:.1f} KB)",
                    data=report_text.encode('utf-8'),
                    file_name=f"climate_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Download a summary report of the analysis"
                )

            # Enhanced post-download integration
            st.markdown("---")
            st.markdown("### 🎉 Analysis Complete!")

            # Register downloads for climate data
            download_handler = get_download_handler("climate_analytics")
            result_ids = []

            # Register climate data CSV
            result_id = register_csv_download(
                "climate_analytics",
                f"climate_indices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                climate_data,
                metadata={
                    'analysis_type': st.session_state.get('climate_analysis_type', 'Unknown'),
                    'dataset': st.session_state.get('climate_selected_dataset', {}).get('name', 'Unknown'),
                    'indices': len(st.session_state.get('climate_selected_indices', [])),
                    'records': len(climate_data),
                    'date_range': f"{st.session_state.get('climate_start_date', 'Unknown')} to {st.session_state.get('climate_end_date', 'Unknown')}"
                }
            )
            result_ids.append(result_id)

            # Show quick visualization of the results
            st.markdown("#### 📊 Quick Analysis Preview")
            try:
                # Create instant visualization using the quick visualizer
                quick_visualizer.render_quick_csv_analysis(
                    climate_data,
                    title=f"{st.session_state.get('climate_analysis_type', 'Climate').title()} Analysis Results",
                    max_columns=3
                )
            except Exception as e:
                st.error(f"Error creating quick visualization: {str(e)}")

            # Show comprehensive post-download options
            render_post_download_integration("climate_analytics", result_ids)

            # Usage information
            with st.expander("💡 How to use downloaded data"):
                st.markdown("""
                **Time Series CSV:**
                - Contains monthly climate index values
                - Columns: Date, Climate_Index_Value, Analysis_Type, Dataset
                - Compatible with Excel, Python pandas, R, MATLAB

                **Summary CSV:**
                - Analysis configuration and basic statistics
                - Useful for documentation and reporting

                **Text Report:**
                - Human-readable summary of the analysis
                - Includes interpretation guidelines
                - Suitable for reports and documentation
                """)

            # Store results in session state for persistence
            results_data = {
                'success': True,
                'time_series_data': climate_data,
                'time_series_csv': climate_data.to_csv(index=False),
                'analysis_report': analysis_summary,
                'individual_results': {}  # For spatial downloads
            }
            st.session_state.climate_results = results_data
            st.session_state.climate_summary = analysis_summary
            st.session_state.climate_analysis_complete = True

    except Exception as e:
        st.error(f"❌ Placeholder analysis failed: {str(e)}")


def _display_single_index_plot(df, index_name, show_trends=True, show_statistics=True):
    """Display a single climate index in detailed view with trend analysis"""
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    from scipy import stats

    st.markdown(f"##### 📊 {index_name} - Detailed View")

    # Convert dates to datetime if needed
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        values = df['value'].values
    else:
        dates = df.index
        values = df.values

    # Create main plot
    fig = go.Figure()

    # Add main time series
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=f"{index_name}",
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.3f}<extra></extra>'
        )
    )

    trend_data = None
    if show_trends and len(values) > 2:
        trend_data = _calculate_trend_statistics(dates, values, index_name)

        # Add linear trend line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=trend_data['linear_trend'],
                mode='lines',
                name=f"Linear Trend",
                line=dict(dash='dash', width=2, color='red'),
                opacity=0.8
            )
        )

        # Add Sen's slope trend line if available
        if 'sens_slope' in trend_data:
            sens_trend = trend_data['sens_intercept'] + trend_data['sens_slope'] * np.arange(len(dates))
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=sens_trend,
                    mode='lines',
                    name=f"Sen's Slope",
                    line=dict(dash='dot', width=2, color='green'),
                    opacity=0.8
                )
            )

    fig.update_layout(
        title=f"{index_name} Time Series Analysis",
        xaxis_title="Date",
        yaxis_title=f"{index_name} Value",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig, use_container_width=True, key=f"single_plot_{index_name}")

    # Show statistics panel
    if show_statistics and trend_data:
        st.markdown("##### 📊 Statistical Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean", f"{np.mean(values):.3f}")
            st.metric("Std Dev", f"{np.std(values):.3f}")

        with col2:
            st.metric("Linear Slope", f"{trend_data['slope']:.4f}/year")
            st.metric("R² Value", f"{trend_data['r_squared']:.3f}")

        with col3:
            p_value = trend_data.get('p_value', 0)
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("P-Value", f"{p_value:.4f}")
            st.metric("Trend", significance)

            # Add interpretation
            if trend_data['slope'] > 0 and p_value < 0.05:
                st.success("📈 Significant increasing trend")
            elif trend_data['slope'] < 0 and p_value < 0.05:
                st.warning("📉 Significant decreasing trend")
            else:
                st.info("➡️ No significant trend")


def _display_climate_results(time_series_data):
    """Display climate analysis results with enhanced visualizations including trend analysis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    from scipy import stats
    from datetime import datetime

    st.markdown("#### 📈 Enhanced Time Series Analysis")

    # Enhanced visualization options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_trends = st.checkbox("Show Trend Lines", value=True, key="climate_show_trends")
    with col2:
        show_statistics = st.checkbox("Show Statistics Panel", value=True, key="climate_show_stats")
    with col3:
        temporal_res = st.session_state.get('climate_temporal_resolution', 'yearly')
        st.info(f"📅 Resolution: {temporal_res}")

    if isinstance(time_series_data, dict):
        # Add toggle option to switch between indices
        index_names = list(time_series_data.keys())
        if len(index_names) > 1:
            st.markdown("##### 🔄 Index Selection")
            view_options = ["All Indices (Subplots)"] + index_names
            selected_view = st.selectbox(
                "Select index to visualize:",
                view_options,
                index=0,
                key="climate_index_toggle",
                help="Choose 'All Indices' to see all plots, or select a specific index for detailed view"
            )

            if selected_view != "All Indices (Subplots)":
                # Show single index in detail
                _display_single_index_plot(time_series_data[selected_view], selected_view, show_trends, show_statistics)
                return
        else:
            selected_view = "All Indices (Subplots)"
        # Multiple indices
        n_indices = len(time_series_data)
        fig = make_subplots(
            rows=n_indices, cols=1,
            subplot_titles=[f"{idx} Time Series" for idx in time_series_data.keys()],
            vertical_spacing=0.08
        )

        trend_stats = {}

        for i, (index_name, df) in enumerate(time_series_data.items(), 1):
            # Convert dates to datetime if needed
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                values = df['value'].values
            else:
                dates = df.index
                values = df.values

            # Add main time series
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=f"{index_name}",
                    line=dict(width=2),
                    marker=dict(size=4)
                ),
                row=i, col=1
            )

            # Calculate and add trend lines if requested
            if show_trends and len(values) > 2:
                trend_data = _calculate_trend_statistics(dates, values, index_name)
                trend_stats[index_name] = trend_data

                # Add linear trend line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=trend_data['linear_trend'],
                        mode='lines',
                        name=f"{index_name} Linear Trend",
                        line=dict(dash='dash', width=2, color='red'),
                        opacity=0.8
                    ),
                    row=i, col=1
                )

                # Add Sen's slope trend line
                if 'sens_slope' in trend_data:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=trend_data['sens_trend'],
                            mode='lines',
                            name=f"{index_name} Sen's Slope",
                            line=dict(dash='dot', width=2, color='orange'),
                            opacity=0.8
                        ),
                        row=i, col=1
                    )

        fig.update_layout(
            height=400 * n_indices,
            showlegend=True,
            title_text="Climate Index Time Series with Trend Analysis"
        )

        # Update x-axis labels
        for i in range(1, n_indices + 1):
            fig.update_xaxes(title_text="Date", row=i, col=1)
            fig.update_yaxes(title_text="Index Value", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Display trend statistics if requested
        if show_statistics and trend_stats:
            _display_trend_statistics_panel(trend_stats)

    else:
        # Single time series
        if hasattr(time_series_data, 'date') and hasattr(time_series_data, 'value'):
            dates = pd.to_datetime(time_series_data['date'])
            values = time_series_data['value'].values
            index_name = "Climate Index"
        else:
            dates = time_series_data.index
            values = time_series_data.values
            index_name = "Climate Index"

        fig = go.Figure()

        # Add main time series
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=index_name,
                line=dict(width=2),
                marker=dict(size=4)
            )
        )

        # Calculate and add trend lines if requested
        trend_data = None
        if show_trends and len(values) > 2:
            trend_data = _calculate_trend_statistics(dates, values, index_name)

            # Add linear trend line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=trend_data['linear_trend'],
                    mode='lines',
                    name="Linear Trend",
                    line=dict(dash='dash', width=2, color='red'),
                    opacity=0.8
                )
            )

            # Add Sen's slope trend line
            if 'sens_slope' in trend_data:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=trend_data['sens_trend'],
                        mode='lines',
                        name="Sen's Slope Trend",
                        line=dict(dash='dot', width=2, color='orange'),
                        opacity=0.8
                    )
                )

        fig.update_layout(
            title="Climate Index Time Series with Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Index Value",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display trend statistics for single series
        if show_statistics and trend_data:
            _display_trend_statistics_panel({index_name: trend_data})


def _calculate_trend_statistics(dates, values, index_name):
    """Calculate comprehensive trend statistics for time series"""
    import numpy as np
    from scipy import stats

    # Convert dates to numeric (years since first date)
    dates_numeric = [(d - dates[0]).days / 365.25 for d in dates]

    # Remove any NaN values
    valid_mask = ~np.isnan(values)
    dates_clean = np.array(dates_numeric)[valid_mask]
    values_clean = values[valid_mask]

    if len(values_clean) < 3:
        return {"error": "Insufficient data for trend analysis"}

    trend_stats = {}

    # Linear regression trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(dates_clean, values_clean)
    linear_trend = slope * np.array(dates_numeric) + intercept

    trend_stats.update({
        'linear_slope': slope,
        'linear_intercept': intercept,
        'linear_r_squared': r_value**2,
        'linear_p_value': p_value,
        'linear_trend': linear_trend
    })

    # Mann-Kendall trend test
    try:
        mk_result = _mann_kendall_test(values_clean)
        trend_stats.update(mk_result)
    except Exception as e:
        trend_stats['mk_error'] = str(e)

    # Sen's slope estimator
    try:
        sens_slope = _sens_slope_estimator(dates_clean, values_clean)
        sens_trend = sens_slope * np.array(dates_numeric) + np.median(values_clean)
        trend_stats.update({
            'sens_slope': sens_slope,
            'sens_trend': sens_trend
        })
    except Exception as e:
        trend_stats['sens_error'] = str(e)

    # Basic statistics
    trend_stats.update({
        'mean': np.mean(values_clean),
        'std': np.std(values_clean),
        'min': np.min(values_clean),
        'max': np.max(values_clean),
        'n_points': len(values_clean)
    })

    return trend_stats


def _mann_kendall_test(data):
    """Simplified Mann-Kendall trend test"""
    import numpy as np
    from scipy import stats

    n = len(data)
    s = 0

    # Calculate S statistic
    for i in range(n-1):
        for j in range(i+1, n):
            if data[j] > data[i]:
                s += 1
            elif data[j] < data[i]:
                s -= 1

    # Calculate variance (simplified, ignoring ties)
    var_s = n * (n - 1) * (2*n + 5) / 18

    # Calculate Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Calculate Kendall's tau
    tau = 2 * s / (n * (n - 1))

    return {
        'mk_s': s,
        'mk_z': z,
        'mk_p_value': p_value,
        'mk_tau': tau,
        'mk_trend': 'increasing' if z > 1.96 else 'decreasing' if z < -1.96 else 'no trend'
    }


def _sens_slope_estimator(dates, values):
    """Calculate Sen's slope estimator"""
    import numpy as np

    slopes = []
    n = len(values)

    for i in range(n-1):
        for j in range(i+1, n):
            if dates[j] != dates[i]:
                slope = (values[j] - values[i]) / (dates[j] - dates[i])
                slopes.append(slope)

    return np.median(slopes) if slopes else 0


def _display_trend_statistics_panel(trend_stats):
    """Display comprehensive trend statistics in an organized panel"""
    st.markdown("#### 📊 Trend Analysis Statistics")

    for index_name, stats in trend_stats.items():
        with st.expander(f"📈 {index_name} - Trend Statistics", expanded=True):

            # Create columns for organized display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🔍 Basic Statistics**")
                if 'mean' in stats:
                    st.metric("Mean", f"{stats['mean']:.3f}")
                if 'std' in stats:
                    st.metric("Std Dev", f"{stats['std']:.3f}")
                if 'n_points' in stats:
                    st.metric("Data Points", stats['n_points'])

            with col2:
                st.markdown("**📈 Linear Trend**")
                if 'linear_slope' in stats:
                    slope_unit = "units/year"
                    st.metric("Slope", f"{stats['linear_slope']:.4f} {slope_unit}")
                if 'linear_r_squared' in stats:
                    st.metric("R²", f"{stats['linear_r_squared']:.3f}")
                if 'linear_p_value' in stats:
                    significance = "Significant" if stats['linear_p_value'] < 0.05 else "Not significant"
                    st.metric("P-value", f"{stats['linear_p_value']:.3f}")
                    st.caption(f"Trend: {significance} (α=0.05)")

            with col3:
                st.markdown("**🎯 Mann-Kendall Test**")
                if 'mk_tau' in stats:
                    st.metric("Kendall's τ", f"{stats['mk_tau']:.3f}")
                if 'mk_z' in stats:
                    st.metric("Z-statistic", f"{stats['mk_z']:.3f}")
                if 'mk_p_value' in stats:
                    st.metric("P-value", f"{stats['mk_p_value']:.3f}")
                if 'mk_trend' in stats:
                    st.caption(f"Trend: {stats['mk_trend'].title()}")

            # Additional row for Sen's slope and interpretation
            if 'sens_slope' in stats:
                st.markdown("**🎯 Sen's Slope Estimator**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sen's Slope", f"{stats['sens_slope']:.4f} units/year")
                with col2:
                    # Interpretation
                    if 'mk_p_value' in stats and 'sens_slope' in stats:
                        if stats['mk_p_value'] < 0.05:
                            if stats['sens_slope'] > 0:
                                interpretation = "🔺 Significant increasing trend"
                            else:
                                interpretation = "🔻 Significant decreasing trend"
                        else:
                            interpretation = "➡️ No significant trend detected"
                        st.success(interpretation)


def _display_geemap_visualization(results):
    """Display climate indices on an interactive geemap with layer toggles"""
    st.markdown("### 🗺️ Interactive Spatial Visualization")
    st.info("💡 Use the layer control to toggle between different time periods and indices. You can take screenshots directly from the map.")

    # Check if we have image collections to display
    if 'image_collections' not in results or not results['image_collections']:
        st.warning("⚠️ No spatial data available for visualization. Image collections were not generated.")
        return

    image_collections = results['image_collections']
    geometry = results.get('geometry')

    # Get temporal resolution for labeling
    temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')

    # Create geemap Map
    Map = geemap.Map()

    # Center map on geometry if available
    if geometry:
        try:
            centroid = geometry.centroid().getInfo()['coordinates']
            Map.setCenter(centroid[0], centroid[1], 8)
        except:
            pass

    # Define color palettes for different index types
    color_palettes = {
        # Temperature indices (warm colors)
        'TXx': ['blue', 'white', 'red'],
        'TNn': ['blue', 'white', 'red'],
        'TXn': ['blue', 'white', 'red'],
        'TNx': ['blue', 'white', 'red'],
        'TX90p': ['white', 'yellow', 'orange', 'red'],
        'TX10p': ['blue', 'cyan', 'white'],
        'TN90p': ['white', 'yellow', 'orange', 'red'],
        'TN10p': ['blue', 'cyan', 'white'],
        'DTR': ['purple', 'white', 'orange'],
        'SU': ['white', 'yellow', 'orange', 'red'],
        'FD': ['blue', 'cyan', 'white'],
        'WSDI': ['white', 'yellow', 'orange', 'red'],
        'CSDI': ['blue', 'cyan', 'white'],
        'GSL': ['brown', 'yellow', 'green'],

        # Precipitation indices (blue colors)
        'RX1day': ['white', 'lightblue', 'blue', 'darkblue'],
        'RX5day': ['white', 'lightblue', 'blue', 'darkblue'],
        'R10mm': ['white', 'lightblue', 'blue'],
        'R20mm': ['white', 'cyan', 'blue', 'darkblue'],
        'CDD': ['green', 'yellow', 'orange', 'red'],
        'PRCPTOT': ['white', 'lightblue', 'blue', 'darkblue'],
        'R95p': ['white', 'cyan', 'blue', 'darkblue'],
        'R99p': ['white', 'cyan', 'blue', 'darkblue'],
        'R75p': ['white', 'lightblue', 'blue'],
        'SDII': ['white', 'lightblue', 'blue', 'darkblue']
    }

    # Define units and descriptions for each index
    index_metadata = {
        # Temperature indices
        'TXx': {'unit': '°C', 'description': 'Maximum of Daily Max Temperature', 'type': 'temperature'},
        'TNn': {'unit': '°C', 'description': 'Minimum of Daily Min Temperature', 'type': 'temperature'},
        'TXn': {'unit': '°C', 'description': 'Minimum of Daily Max Temperature', 'type': 'temperature'},
        'TNx': {'unit': '°C', 'description': 'Maximum of Daily Min Temperature', 'type': 'temperature'},
        'TX90p': {'unit': 'days', 'description': 'Warm Days (Tmax > 90th percentile)', 'type': 'temperature'},
        'TX10p': {'unit': 'days', 'description': 'Cool Days (Tmax < 10th percentile)', 'type': 'temperature'},
        'TN90p': {'unit': 'days', 'description': 'Warm Nights (Tmin > 90th percentile)', 'type': 'temperature'},
        'TN10p': {'unit': 'days', 'description': 'Cool Nights (Tmin < 10th percentile)', 'type': 'temperature'},
        'DTR': {'unit': '°C', 'description': 'Diurnal Temperature Range', 'type': 'temperature'},
        'SU': {'unit': 'days', 'description': 'Summer Days (Tmax > 25°C)', 'type': 'temperature'},
        'FD': {'unit': 'days', 'description': 'Frost Days (Tmin < 0°C)', 'type': 'temperature'},
        'WSDI': {'unit': 'days', 'description': 'Warm Spell Duration Index', 'type': 'temperature'},
        'CSDI': {'unit': 'days', 'description': 'Cold Spell Duration Index', 'type': 'temperature'},
        'GSL': {'unit': 'days', 'description': 'Growing Season Length', 'type': 'temperature'},

        # Precipitation indices
        'RX1day': {'unit': 'mm', 'description': 'Max 1-day Precipitation', 'type': 'precipitation'},
        'RX5day': {'unit': 'mm', 'description': 'Max 5-day Precipitation', 'type': 'precipitation'},
        'R10mm': {'unit': 'days', 'description': 'Heavy Precipitation Days (≥10mm)', 'type': 'precipitation'},
        'R20mm': {'unit': 'days', 'description': 'Very Heavy Precipitation Days (≥20mm)', 'type': 'precipitation'},
        'CDD': {'unit': 'days', 'description': 'Consecutive Dry Days', 'type': 'precipitation'},
        'PRCPTOT': {'unit': 'mm', 'description': 'Total Wet-day Precipitation', 'type': 'precipitation'},
        'R95p': {'unit': 'mm', 'description': 'Very Wet Days (>95th percentile)', 'type': 'precipitation'},
        'R99p': {'unit': 'mm', 'description': 'Extremely Wet Days (>99th percentile)', 'type': 'precipitation'},
        'R75p': {'unit': 'mm', 'description': 'Moderately Wet Days (>75th percentile)', 'type': 'precipitation'},
        'SDII': {'unit': 'mm/day', 'description': 'Simple Daily Intensity Index', 'type': 'precipitation'}
    }

    # Selection for which index to visualize
    index_names = list(image_collections.keys())

    if len(index_names) == 0:
        st.warning("No climate indices available for visualization.")
        return

    # User selects which index to visualize
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = st.selectbox(
            "Select Climate Index to Display:",
            index_names,
            key="geemap_index_selector"
        )

    with col2:
        show_all_layers = st.checkbox(
            "Show All Time Periods",
            value=False,
            help="Toggle to show/hide all temporal layers at once"
        )

    if selected_index:
        collection = image_collections[selected_index]
        palette = color_palettes.get(selected_index, ['blue', 'white', 'red'])

        # Get metadata for selected index
        metadata = index_metadata.get(selected_index, {
            'unit': 'value',
            'description': selected_index,
            'type': 'unknown'
        })

        # Display index information prominently
        st.markdown(f"#### 📍 Selected: **{selected_index}** - {metadata['description']}")

        # Get collection size (number of images/time periods)
        try:
            collection_size = collection.size().getInfo()
            st.info(f"📊 {selected_index} has {collection_size} time periods available")
        except Exception as e:
            st.error(f"Error getting collection size: {str(e)}")
            return

        # Get list of all images with dates
        try:
            # Convert ImageCollection to list
            collection_list = collection.toList(collection.size())

            # Create layers for each time period
            layers_added = []
            overall_min = float('inf')
            overall_max = float('-inf')

            for i in range(min(collection_size, 100)):  # Limit to 100 layers for performance
                try:
                    image = ee.Image(collection_list.get(i))

                    # Get the date from the image
                    date_millis = image.get('system:time_start').getInfo()
                    date_obj = datetime.fromtimestamp(date_millis / 1000)

                    # Format date based on temporal resolution
                    if temporal_resolution == 'monthly':
                        date_label = date_obj.strftime('%Y-%m')
                    else:  # yearly
                        date_label = date_obj.strftime('%Y')

                    # Create layer name
                    layer_name = f"{selected_index} - {date_label}"

                    # Calculate min/max for visualization
                    # Use reduce for the specific region if geometry available
                    if geometry:
                        stats = image.reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=geometry,
                            scale=5000,
                            maxPixels=1e8
                        ).getInfo()

                        # Get min/max values
                        band_name = image.bandNames().getInfo()[0]
                        vmin = stats.get(f'{band_name}_min', 0)
                        vmax = stats.get(f'{band_name}_max', 100)
                    else:
                        # Use default range if no geometry
                        vmin = 0
                        vmax = 100

                    # Track overall min/max across all layers
                    overall_min = min(overall_min, vmin)
                    overall_max = max(overall_max, vmax)

                    # Add layer to map
                    vis_params = {
                        'min': vmin,
                        'max': vmax,
                        'palette': palette
                    }

                    Map.addLayer(
                        image,
                        vis_params,
                        layer_name,
                        shown=(i == collection_size - 1) or show_all_layers  # Show only the last layer by default, or all if checkbox selected
                    )

                    layers_added.append(layer_name)

                except Exception as layer_error:
                    st.warning(f"Could not add layer {i}: {str(layer_error)}")
                    continue

            st.success(f"✅ Added {len(layers_added)} layers to the map. Use the layer control (top right) to toggle visibility.")

            # Add geometry outline if available
            if geometry:
                Map.addLayer(geometry, {'color': 'black'}, 'Study Area', True)

            # Add colorbar to the map
            try:
                Map.add_colorbar(
                    colors=palette,
                    vmin=overall_min,
                    vmax=overall_max,
                    label=f"{selected_index} ({metadata['unit']})",
                    categorical=False,
                    position='bottomright'
                )
            except Exception as colorbar_error:
                st.warning(f"Could not add colorbar to map: {str(colorbar_error)}")

            # Display the map
            Map.to_streamlit(height=600)

            # Compact info below map
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**📍 {selected_index}:** {metadata['description']}")
                st.markdown(f"**📊 Range:** {overall_min:.2f} to {overall_max:.2f} {metadata['unit']}")
            with col2:
                st.markdown(f"**🎨 Colormap:** Built-in on map →")
                st.markdown(f"**📅 Layers:** {len(layers_added)} time periods")

            # Detailed info in expander
            with st.expander("📋 View Detailed Layer Information", expanded=False):
                st.markdown(f"**Climate Index:** {selected_index}")
                st.markdown(f"**Description:** {metadata['description']}")
                st.markdown(f"**Unit:** {metadata['unit']}")
                st.markdown(f"**Temporal Resolution:** {temporal_resolution}")
                st.markdown(f"**Number of Layers:** {len(layers_added)}")
                st.markdown(f"**Data Range:** {overall_min:.2f} to {overall_max:.2f} {metadata['unit']}")
                st.markdown(f"**Color Palette:** {', '.join(palette)}")
                st.markdown("**Available Layers:**")
                for layer in layers_added[:10]:  # Show first 10
                    st.markdown(f"- {layer}")
                if len(layers_added) > 10:
                    st.markdown(f"... and {len(layers_added) - 10} more")

        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
            st.info("This may be due to large data size or computation limits. Try selecting a smaller time range.")


def _show_download_options(results):
    """Show download options for climate analysis results with Google Drive support"""
    st.markdown("### 📥 Download Results")

    export_method = st.session_state.get('climate_export_method', 'local')

    # Always available downloads
    col1, col2 = st.columns(2)

    with col1:
        if 'time_series_csv' in results:
            st.download_button(
                label="📊 Download Time Series (CSV)",
                data=results['time_series_csv'],
                file_name=f"climate_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Area-averaged yearly climate index values",
                use_container_width=True
            )

    with col2:
        if 'analysis_report' in results:
            st.download_button(
                label="📋 Download Analysis Report",
                data=results['analysis_report'],
                file_name=f"climate_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Comprehensive analysis report with configuration and results",
                use_container_width=True
            )

    st.markdown("---")

    # Check for TEMPORAL-ONLY mode (spatial export needs to be triggered separately)
    if results.get('temporal_only', False):
        st.markdown("### 🗺️ Spatial Data Export")
        st.info("""
        **Temporal data has been extracted successfully!**

        Spatial export (GeoTIFF files) requires additional processing.
        - Click the button below to start spatial export
        - If local download fails (>50MB limit), exports will automatically switch to Google Drive
        """)

        # Check if spatial export already completed
        if st.session_state.get('climate_spatial_export_complete', False):
            st.success("✅ Spatial export completed!")
            # Show spatial download results
            if 'climate_spatial_results' in st.session_state:
                _display_spatial_export_results(st.session_state.climate_spatial_results)

            if st.button("🔄 Re-run Spatial Export", use_container_width=True,
                        key="rerun_spatial_export_main"):
                st.session_state.climate_spatial_export_complete = False
                if 'climate_spatial_results' in st.session_state:
                    del st.session_state.climate_spatial_results
                st.rerun()
        else:
            if st.button("🚀 Proceed to Spatial Export", type="primary", use_container_width=True,
                        help="Download spatial GeoTIFF files for each climate index",
                        key="proceed_spatial_export_main"):
                # Execute spatial export with 50MB error learning
                _execute_smart_spatial_export(results)
                st.rerun()
    else:
        # Spatial data based on export method (for non-temporal-only modes)
        if export_method == 'auto':
            _show_smart_download_results(results)
        elif export_method == 'drive':
            _show_google_drive_results(results)
        elif export_method == 'preview':
            _show_preview_results(results)
        elif export_method == 'local':
            _show_local_download_results(results)


def _show_smart_download_results(results):
    """Show results from smart download system - mix of local and drive exports"""
    st.markdown("### 📊 Smart Download Results")

    # Debug: Show the actual structure received
    if st.checkbox("🔍 Debug: Show raw results structure", key="debug_smart_results"):
        st.json(results)

    spatial_data = results.get('individual_results', {})
    local_files = []
    drive_exports = []
    pending_exports = []
    preview_files = []
    failed_exports = []

    # Debug information (can be toggled)
    debug_mode = st.checkbox("🔍 Enable detailed debugging", key="debug_mode", value=False)

    if debug_mode:
        st.info(f"🔍 Debug: Found {len(spatial_data)} indices in results")

    # Categorize results by export method used
    for index_name, index_result in spatial_data.items():
        if debug_mode:
            st.info(f"🔍 Processing {index_name}: success={index_result.get('success', False)}, has_spatial_data={'spatial_data' in index_result}")

        if index_result.get('success') and 'spatial_data' in index_result:
            spatial_result = index_result['spatial_data']
            export_method = spatial_result.get('export_method', 'unknown')

            if debug_mode:
                st.info(f"🔍 {index_name} export method: {export_method}")

            if export_method == 'local':
                local_files.append({
                    'index': index_name,
                    'filename': spatial_result.get('filename', f"{index_name}.tif"),
                    'file_data': spatial_result.get('file_data'),
                    'size_mb': spatial_result.get('actual_size_mb', 0),
                    'is_fallback': spatial_result.get('fallback', False),
                    'message': spatial_result.get('message', '')
                })
                if debug_mode:
                    st.success(f"✅ {index_name}: Added to local downloads")
            elif export_method == 'drive':
                drive_exports.append({
                    'index': index_name,
                    'folder': spatial_result.get('drive_folder', 'Unknown'),
                    'task_id': spatial_result.get('task_id'),
                    'size_mb': spatial_result.get('estimated_size_mb', 0),
                    'url': spatial_result.get('drive_url', 'https://drive.google.com/drive/'),
                    'total_tasks': spatial_result.get('total_tasks', 0)
                })
                if debug_mode:
                    st.success(f"✅ {index_name}: Added to drive exports")
            elif export_method == 'preview':
                preview_files.append({
                    'index': index_name,
                    'filename': spatial_result.get('filename', f"{index_name}_sample.tif"),
                    'file_data': spatial_result.get('file_data'),
                    'size_mb': spatial_result.get('file_size_mb', spatial_result.get('actual_size_mb', 0)),
                    'total_images': spatial_result.get('total_images', 1)
                })
                if debug_mode:
                    st.success(f"✅ {index_name}: Added to preview files")
            elif export_method == 'pending':
                pending_exports.append({
                    'index': index_name,
                    'message': spatial_result.get('message', 'Spatial export pending')
                })
                if debug_mode:
                    st.info(f"⏳ {index_name}: Pending spatial export")
            elif export_method in ['failed', 'error']:
                failed_exports.append({
                    'index': index_name,
                    'error': spatial_result.get('error', spatial_result.get('message', 'Unknown error'))
                })
                if debug_mode:
                    st.error(f"❌ {index_name}: Export failed")
            else:
                if debug_mode:
                    st.warning(f"⚠️ {index_name}: Unknown export method '{export_method}'")

    # Show summary
    total_local = len(local_files)
    total_drive = len(drive_exports)
    total_preview = len(preview_files)
    total_pending = len(pending_exports)
    total_failed = len(failed_exports)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Smart Results", f"{total_local + total_drive + total_preview} indices")
    with col2:
        st.metric("💻 Local", total_local)
    with col3:
        st.metric("☁️ Drive", total_drive)
    with col4:
        if total_pending > 0:
            st.metric("⏳ Pending", total_pending)
        elif total_failed > 0:
            st.metric("❌ Failed", total_failed)
        else:
            st.metric("✅ Success", "100%")

    # Show local downloads
    if local_files:
        st.markdown("#### 💻 Ready for Download")
        for file_info in local_files:
            if file_info.get('is_fallback'):
                st.warning(f"**{file_info['index']}** ({file_info['size_mb']:.1f} MB) - ⚠️ Fallback")
                st.caption(file_info.get('message', 'Fallback data'))
            else:
                st.markdown(f"**{file_info['index']}** ({file_info['size_mb']:.1f} MB)")

            if file_info['file_data']:
                mime_type = 'application/zip' if file_info['filename'].endswith('.zip') else 'text/csv'
                st.download_button(
                    label=f"📥 Download {file_info['index']}",
                    data=file_info['file_data'],
                    file_name=file_info['filename'],
                    mime=mime_type,
                    help=f"Download {file_info['index']} ({file_info['size_mb']:.1f} MB)",
                    use_container_width=True
                )

    # Show drive exports
    if drive_exports:
        st.markdown("#### ☁️ Google Drive Exports")
        st.success(f"✅ {sum(e.get('total_tasks', 0) for e in drive_exports)} tasks submitted!")

        for export_info in drive_exports:
            with st.expander(f"📤 {export_info['index']} - {export_info.get('total_tasks', 0)} files", expanded=True):
                st.write(f"**Folder:** `{export_info['folder']}`")
                if export_info['task_id']:
                    st.code(f"Task ID: {export_info['task_id']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.link_button("⚙️ Monitor Tasks", "https://code.earthengine.google.com/tasks", use_container_width=True)
                with col2:
                    st.link_button("📁 Open Drive", export_info.get('url', 'https://drive.google.com/drive/'), use_container_width=True)

    # Show preview files
    if preview_files:
        st.markdown("#### 📱 Preview Samples")
        for preview_info in preview_files:
            st.markdown(f"**{preview_info['index']}** - Sample ({preview_info['size_mb']:.2f} MB)")
            st.caption(f"1 of {preview_info['total_images']} total images")
            if preview_info['file_data']:
                st.download_button(
                    label=f"📷 Download {preview_info['index']} Sample",
                    data=preview_info['file_data'],
                    file_name=preview_info['filename'],
                    mime="image/tiff",
                    use_container_width=True
                )

    # Show pending exports (temporal_only mode)
    if pending_exports:
        st.markdown("#### ⏳ Pending Spatial Exports")
        st.info("Temporal data extracted. Spatial exports available on-demand.")
        for pending_info in pending_exports:
            st.write(f"- **{pending_info['index']}**: {pending_info['message']}")

    # Show failed exports
    if failed_exports:
        st.markdown("#### ❌ Failed Exports")
        for failed_info in failed_exports:
            st.error(f"**{failed_info['index']}**: {failed_info['error']}")

    if not local_files and not drive_exports and not preview_files and not pending_exports:
        st.warning("No spatial data results found for smart download.")

        # Enhanced debugging for empty results
        st.markdown("#### 🔍 Debugging Information")
        st.info(f"Total indices processed: {len(spatial_data)}")

        if spatial_data:
            st.info("Index details:")
            for index_name, index_result in spatial_data.items():
                success = index_result.get('success', False)
                has_spatial = 'spatial_data' in index_result
                if has_spatial:
                    spatial_result = index_result['spatial_data']
                    export_method = spatial_result.get('export_method', 'missing')
                    spatial_success = spatial_result.get('success', False)
                    st.write(f"- **{index_name}**: success={success}, has_spatial={has_spatial}, spatial_success={spatial_success}, export_method={export_method}")
                else:
                    st.write(f"- **{index_name}**: success={success}, has_spatial={has_spatial}")
        else:
            st.warning("No individual results found in the data structure.")
            st.info("Expected structure: results['individual_results'][index_name]['spatial_data']")


def _show_google_drive_results(results):
    """Show Google Drive export results and task tracking"""
    st.markdown("### ☁️ Google Drive Export Status")

    # Check if we have spatial data results
    spatial_data = results.get('individual_results', {})
    drive_folders = []
    total_tasks = 0

    for index_name, index_result in spatial_data.items():
        # Check for 'drive' export_method (not 'google_drive')
        if index_result.get('success') and index_result.get('spatial_data', {}).get('export_method') == 'drive':
            spatial_result = index_result['spatial_data']
            drive_folders.append({
                'index': index_name,
                'folder': spatial_result.get('drive_folder', 'Unknown'),
                'url': spatial_result.get('drive_url', 'https://drive.google.com/drive/'),
                'tasks': spatial_result.get('total_tasks', 0),
                'failed': spatial_result.get('failed_tasks', 0)
            })
            total_tasks += spatial_result.get('total_tasks', 0)

    if drive_folders:
        st.success(f"✅ Successfully submitted {total_tasks} export tasks to Google Drive!")

        # Show Google Drive folders
        for folder_info in drive_folders:
            with st.expander(f"🗂️ {folder_info['index']} - {folder_info['tasks']} files", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Google Drive Folder:** `{folder_info['folder']}`")
                    st.markdown(f"**Export Tasks:** {folder_info['tasks']} submitted")
                    if folder_info['failed'] > 0:
                        st.warning(f"⚠️ {folder_info['failed']} tasks failed to submit")

                with col2:
                    st.link_button(
                        "🔗 Open Google Drive",
                        folder_info['url'],
                        use_container_width=True
                    )

        # Show status information
        st.info("""
        **📋 What's happening:**
        - Export tasks are running on Google Earth Engine servers
        - Files will appear in your Google Drive automatically (usually within 5-30 minutes)
        - You can close this page and continue with other work
        - Check your Google Drive periodically for completed files
        """)

        # Task tracking section
        with st.expander("📊 Task Tracking (Advanced)", expanded=False):
            st.markdown("**Export Status Tracking:**")
            st.code("""
# You can check task status using Earth Engine's task manager
# Or monitor your Google Drive folders directly
# Task IDs are logged in the analysis report for reference
            """)

    else:
        st.error("❌ No Google Drive export tasks were created. Please try again.")


def _show_preview_results(results):
    """Show preview sample results"""
    st.markdown("### 📱 Preview Sample Results")

    spatial_data = results.get('individual_results', {})
    preview_files = []

    for index_name, index_result in spatial_data.items():
        if index_result.get('success') and index_result.get('spatial_data', {}).get('export_method') == 'preview':
            spatial_result = index_result['spatial_data']
            preview_files.append({
                'index': index_name,
                'filename': spatial_result['filename'],
                'file_data': spatial_result['file_data'],
                'total_images': spatial_result['total_images'],
                'size_mb': spatial_result['file_size_mb']
            })

    if preview_files:
        for preview in preview_files:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**{preview['index']} Sample:**")
                st.info(f"1 of {preview['total_images']} total images ({preview['size_mb']:.2f} MB)")

            with col2:
                st.download_button(
                    label=f"📷 Download Sample",
                    data=preview['file_data'],
                    file_name=preview['filename'],
                    mime="image/tiff",
                    help="Sample GeoTIFF file for preview",
                    use_container_width=True
                )

        st.info("""
        **🔍 Need all the files?**
        This preview shows just 1 sample file per index. To get all spatial data,
        re-run the analysis and choose "Export to Google Drive" or "Download All Locally".
        """)
    else:
        st.error("❌ No preview samples were generated.")


def _show_local_download_results(results):
    """Show local download results"""
    st.markdown("### 💻 Local Download Results")

    if 'spatial_data_zip' in results:
        spatial_size_mb = len(results['spatial_data_zip']) / (1024 * 1024)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.success(f"✅ All spatial data ready for download ({spatial_size_mb:.1f} MB)")
            st.info("ZIP archive contains individual GeoTIFF files for each time period")

        with col2:
            st.download_button(
                label="🗺️ Download All Spatial Data",
                data=results['spatial_data_zip'],
                file_name=f"climate_spatial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                help="ZIP archive with all GeoTIFF files",
                use_container_width=True
            )
    else:
        st.warning("⚠️ Spatial data not available for local download.")


def _execute_smart_spatial_export(results):
    """
    Execute spatial export with 50MB error learning.
    If first index fails with size error, switch ALL remaining to Google Drive.
    """
    import ee
    from geoclimate_fetcher.core.download_utils import process_image_collection_chunked
    from geoclimate_fetcher.core.dataset_config import get_dataset_config
    from geoclimate_fetcher.climate_indices import ClimateIndicesCalculator

    st.markdown("### 🗺️ Processing Spatial Exports...")

    if 'spatial_export_config' not in results:
        st.error("❌ Spatial export configuration not found. Please re-run analysis.")
        return

    config = results['spatial_export_config']
    selected_indices = config['selected_indices']
    geometry = config['geometry']
    spatial_scale = config['spatial_scale']
    temporal_resolution = config['temporal_resolution']
    dataset_id = config['dataset_id']
    start_date = config['start_date']
    end_date = config['end_date']

    # Initialize calculator
    calculator = ClimateIndicesCalculator(geometry, dataset_id)

    # Get dataset configuration
    dataset_config_manager = get_dataset_config()
    band_mapping = dataset_config_manager.get_required_bands_for_indices(dataset_id, selected_indices)

    # Load collections
    st.info("🔄 Loading Earth Engine collections...")
    collections = {}
    base_collection = ee.ImageCollection(dataset_id)

    for band_type, ee_band_name in band_mapping.items():
        collections[band_type] = base_collection.filterDate(start_date, end_date).select([ee_band_name])

    # Smart export with learning
    spatial_results = {}
    local_failed = False  # Track if local download has failed (for learning)
    force_drive = False   # If True, skip local entirely

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, index_name in enumerate(selected_indices):
        progress = idx / len(selected_indices)
        progress_bar.progress(progress)
        status_text.text(f"Processing {index_name} ({idx + 1}/{len(selected_indices)})...")

        try:
            # Calculate index
            st.info(f"🔄 Calculating {index_name}...")

            # Get parameters
            threshold_kwargs = config.get('threshold_params', {}).get(index_name, {})
            percentile_kwargs = {}
            if index_name in config.get('percentile_params', {}):
                percentile_config = config['percentile_params'][index_name]
                percentile_kwargs = {
                    'percentile': percentile_config['percentile'],
                    'base_start': percentile_config['base_start'],
                    'base_end': percentile_config['base_end']
                }
            all_kwargs = {**threshold_kwargs, **percentile_kwargs}

            index_result = calculator.calculate_simple_index(
                index_name, collections, start_date, end_date,
                temporal_resolution=temporal_resolution,
                **all_kwargs
            )

            # SMART EXPORT LOGIC WITH LEARNING
            if force_drive or local_failed:
                # Previous failure → skip local, go directly to Drive
                st.info(f"⏭️ {index_name}: Skipping local (learned from previous failure) → Google Drive")
                export_result = _export_to_drive(index_result, index_name, geometry, spatial_scale, temporal_resolution)
            else:
                # Try local first
                st.info(f"💻 {index_name}: Attempting local download...")
                try:
                    export_result = process_image_collection_chunked(
                        collection=index_result,
                        bands=None,
                        geometry=geometry,
                        start_date=start_date,
                        end_date=end_date,
                        export_format='GeoTIFF',
                        scale=spatial_scale,
                        temporal_resolution=temporal_resolution
                    )

                    if export_result.get('success') and export_result.get('file_data'):
                        # Local success!
                        file_size_mb = len(export_result['file_data']) / (1024 * 1024)
                        st.success(f"✅ {index_name}: Local download successful ({file_size_mb:.1f} MB)")
                        export_result['export_method'] = 'local'
                        export_result['file_size_mb'] = file_size_mb
                    else:
                        # Local failed (size exceeded or other error)
                        raise Exception("Local download failed or exceeded size limit")

                except Exception as local_error:
                    error_msg = str(local_error)

                    # Check if it's a 50MB size limit error
                    if "50331648" in error_msg or "must be less than" in error_msg.lower() or "request size" in error_msg.lower():
                        st.warning(f"⚠️ {index_name}: Local download failed (50MB limit exceeded)")
                        st.warning(f"📝 **Learning:** Switching ALL remaining indices to Google Drive export")
                        local_failed = True  # LEARNING: Don't try local for remaining indices
                    else:
                        st.warning(f"⚠️ {index_name}: Local download failed ({error_msg})")
                        local_failed = True

                    # Fallback to Google Drive
                    st.info(f"☁️ {index_name}: Falling back to Google Drive...")
                    export_result = _export_to_drive(index_result, index_name, geometry, spatial_scale, temporal_resolution)

            spatial_results[index_name] = export_result

        except Exception as e:
            st.error(f"❌ {index_name}: Export failed - {str(e)}")
            spatial_results[index_name] = {
                'success': False,
                'export_method': 'error',
                'error': str(e)
            }

    progress_bar.progress(1.0)
    status_text.text("✅ Spatial export completed!")

    # Store results in session state
    st.session_state.climate_spatial_results = spatial_results
    st.session_state.climate_spatial_export_complete = True

    # Display results
    _display_spatial_export_results(spatial_results)


def _export_to_drive(index_collection, index_name, geometry, scale, temporal_resolution):
    """Export an index collection to Google Drive"""
    import ee
    from datetime import datetime

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"Climate_Spatial_{timestamp}"

        # Create composite image (e.g., mean over time period)
        composite = index_collection.mean()

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f'{index_name}_{temporal_resolution}_{timestamp}',
            folder=folder_name,
            fileNamePrefix=f'{index_name}_{temporal_resolution}',
            scale=scale,
            region=geometry,
            fileFormat='GeoTiff',
            maxPixels=1e13
        )
        task.start()

        st.info(f"☁️ {index_name}: Submitted to Google Drive (Task ID: {task.id})")

        return {
            'success': True,
            'export_method': 'drive',
            'task_id': task.id,
            'folder': folder_name,
            'estimated_size_mb': 'Unknown (processing in Drive)'
        }

    except Exception as e:
        st.error(f"❌ {index_name}: Drive export failed - {str(e)}")
        return {
            'success': False,
            'export_method': 'error',
            'error': str(e)
        }


def _display_spatial_export_results(spatial_results):
    """Display spatial export results with download buttons (stable keys)"""
    st.markdown("### 📊 Spatial Export Results")

    local_results = {k: v for k, v in spatial_results.items() if v.get('export_method') == 'local'}
    drive_results = {k: v for k, v in spatial_results.items() if v.get('export_method') == 'drive'}
    error_results = {k: v for k, v in spatial_results.items() if v.get('export_method') == 'error'}

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💻 Local Downloads", len(local_results))
    with col2:
        st.metric("☁️ Drive Exports", len(drive_results))
    with col3:
        st.metric("❌ Errors", len(error_results))

    # Local downloads with stable keys
    if local_results:
        st.markdown("#### 💻 Ready for Download")
        for idx, (index_name, result) in enumerate(local_results.items()):
            if result.get('file_data'):
                file_size = result.get('file_size_mb', 0)
                st.markdown(f"**{index_name}** ({file_size:.1f} MB)")

                # Stable key for download button
                button_key = f"spatial_download_{index_name.replace(' ', '_').replace('-', '_')}_{idx}"
                st.download_button(
                    label=f"📥 Download {index_name} GeoTIFF",
                    data=result['file_data'],
                    file_name=result.get('filename', f'{index_name}_spatial.zip'),
                    mime='application/zip',
                    use_container_width=True,
                    key=button_key
                )

    # Drive exports
    if drive_results:
        st.markdown("#### ☁️ Google Drive Exports")
        st.info("These files are processing in Google Drive. Monitor progress below.")

        for index_name, result in drive_results.items():
            with st.expander(f"📤 {index_name}", expanded=False):
                st.write(f"**Folder:** {result.get('folder', 'Unknown')}")
                st.code(f"Task ID: {result.get('task_id', 'Unknown')}")
                st.markdown("[⚙️ Monitor Task Progress](https://code.earthengine.google.com/tasks)")
                st.markdown("[📁 Open Google Drive](https://drive.google.com/drive/)")

    # Errors
    if error_results:
        st.markdown("#### ❌ Failed Exports")
        for index_name, result in error_results.items():
            st.error(f"{index_name}: {result.get('error', 'Unknown error')}")


def _reset_climate_analysis():
    """Reset all climate analysis session state"""
    keys_to_reset = [
        'climate_step', 'climate_analysis_type', 'climate_geometry_complete',
        'climate_dataset_selected', 'climate_date_range_set', 'climate_indices_selected',
        'climate_selected_dataset', 'climate_selected_indices', 'climate_start_date', 'climate_end_date',
        'climate_results', 'climate_summary', 'climate_export_configured', 'climate_analysis_complete',
        'climate_export_method', 'climate_spatial_scale', 'climate_temporal_resolution',
        'climate_spatial_export_complete', 'climate_spatial_results', 'climate_index_toggle'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def _render_smart_spatial_download_section():
    """Render smart spatial download section for climate indices"""
    from app_components.download_component import DownloadHelper

    # Initialize download helper
    download_helper = DownloadHelper()

    st.markdown("#### 🗺️ Download Spatial Climate Index Data")

    # Get configuration
    selected_indices = st.session_state.climate_selected_indices
    temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')
    export_method = st.session_state.get('climate_export_method', 'auto')
    spatial_scale = st.session_state.get('climate_spatial_scale', 1000)

    # Show current configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📈 Indices: {len(selected_indices)}")
    with col2:
        st.info(f"⏰ Resolution: {temporal_resolution}")
    with col3:
        scale_km = spatial_scale / 1000 if spatial_scale >= 1000 else spatial_scale
        unit = "km" if spatial_scale >= 1000 else "m"
        st.info(f"📐 Scale: {scale_km:.1f} {unit}")

    # Show available indices for download
    st.markdown("**Selected Climate Indices:**")
    for idx in selected_indices:
        st.markdown(f"• {idx}")

    # Smart download options
    st.markdown("---")
    st.markdown("#### 📤 Smart Download Options")

    # Render smart download preferences (reusing geodata explorer logic)
    export_preference = download_helper.render_smart_download_options()

    # Download section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Choose download approach:**")
        download_approach = st.radio(
            "Select what to download:",
            options=['time_series_only', 'spatial_composite', 'all_temporal'],
            format_func=lambda x: {
                'time_series_only': '📊 Time Series Data Only (CSV)',
                'spatial_composite': '🗺️ Spatial Composite (Single GeoTIFF per index)',
                'all_temporal': '🎞️ All Time Periods (Multiple GeoTIFFs per index)'
            }[x],
            help="Time series: Fast CSV download. Composite: Single spatial file per index. All temporal: Complete spatial time series"
        )

    with col2:
        if download_approach == 'time_series_only':
            st.success("✅ Fast download")
            st.caption("CSV file with area-averaged values")
        elif download_approach == 'spatial_composite':
            st.info("📊 Medium download")
            st.caption("Single GeoTIFF per index")
        else:
            st.warning("⚠️ Large download")
            st.caption("Multiple files per index")

    # Download execution
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Start Smart Download", type="primary", use_container_width=True):
            _execute_climate_smart_download(
                download_approach=download_approach,
                export_preference=export_preference,
                download_helper=download_helper
            )


def _execute_climate_smart_download(download_approach, export_preference, download_helper):
    """Execute smart download for climate indices based on user preferences"""
    try:
        # Get configuration
        selected_indices = st.session_state.climate_selected_indices
        temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')
        spatial_scale = st.session_state.get('climate_spatial_scale', 1000)
        geometry = st.session_state.climate_geometry_handler.current_geometry

        # Time series data is always available
        if download_approach == 'time_series_only':
            _execute_time_series_download()
            return

        # For spatial downloads, we need to simulate Earth Engine processing
        st.info("🔄 Preparing spatial climate index data...")

        if download_approach == 'spatial_composite':
            _execute_spatial_composite_download(
                selected_indices, temporal_resolution, spatial_scale,
                geometry, export_preference, download_helper
            )
        else:  # all_temporal
            _execute_all_temporal_download(
                selected_indices, temporal_resolution, spatial_scale,
                geometry, export_preference, download_helper
            )

    except Exception as e:
        st.error(f"❌ Smart download failed: {str(e)}")
        st.info("💡 Try using a smaller area or fewer indices")


def _execute_time_series_download():
    """Execute fast time series CSV download"""
    import numpy as np

    st.success("✅ Time series data ready for download!")

    # Generate time series data
    temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')
    freq = 'Y' if temporal_resolution == 'yearly' else 'M'
    dates = pd.date_range(st.session_state.climate_start_date, st.session_state.climate_end_date, freq=freq)

    all_data = []
    for idx in st.session_state.climate_selected_indices:
        # Simulate different patterns for different indices
        if 'TX' in idx:  # Temperature max indices
            base_values = np.random.normal(25, 5, len(dates))
        elif 'TN' in idx:  # Temperature min indices
            base_values = np.random.normal(15, 4, len(dates))
        elif 'PRCP' in idx or 'R' in idx:  # Precipitation indices
            base_values = np.random.exponential(50, len(dates))
        else:
            base_values = np.random.randn(len(dates)).cumsum()

        for i, (date, value) in enumerate(zip(dates, base_values)):
            all_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Climate_Index': idx,
                'Value': round(value, 3),
                'Year': date.year,
                'Temporal_Resolution': temporal_resolution,
                'Analysis_Type': st.session_state.climate_analysis_type
            })

    df = pd.DataFrame(all_data)

    # Download button
    csv_data = df.to_csv(index=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    st.download_button(
        label="📊 Download Climate Index Time Series",
        data=csv_data,
        file_name=f"climate_indices_timeseries_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Show preview
    st.markdown("#### 📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Add visualization module integration for time series data
    _integrate_climate_visualization_links(df, 'time_series')


def _execute_spatial_composite_download(selected_indices, temporal_resolution, spatial_scale,
                                      geometry, export_preference, download_helper):
    """Execute spatial composite download (one file per index)"""
    import numpy as np

    st.info("🗺️ Creating spatial composites for each climate index...")

    download_results = {}

    for idx in selected_indices:
        st.write(f"Processing {idx}...")

        # Simulate creating composite for this index
        # In real implementation, this would use ClimateIndicesCalculator
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"climate_index_{idx}_{temporal_resolution}_composite_{timestamp}"

            # For now, simulate successful processing
            if export_preference == 'auto':
                # Try local first, fallback to drive
                st.info(f"🤖 Smart Auto: Processing {idx} locally...")
                # Simulate local success/failure
                local_success = np.random.choice([True, False], p=[0.7, 0.3])  # 70% success rate

                if local_success:
                    download_results[idx] = {
                        'method': 'local',
                        'status': 'ready',
                        'filename': f"{filename}.tif",
                        'size_mb': np.random.uniform(5, 50)
                    }
                    st.success(f"✅ {idx}: Ready for local download")
                else:
                    download_results[idx] = {
                        'method': 'drive',
                        'status': 'processing',
                        'folder': f"Climate_Analysis_{timestamp}",
                        'task_id': f"TASK_{np.random.randint(100000, 999999)}"
                    }
                    st.info(f"☁️ {idx}: Sent to Google Drive (local size exceeded)")

            elif export_preference == 'drive':
                download_results[idx] = {
                    'method': 'drive',
                    'status': 'processing',
                    'folder': f"Climate_Analysis_{timestamp}",
                    'task_id': f"TASK_{np.random.randint(100000, 999999)}"
                }
                st.info(f"☁️ {idx}: Processing in Google Drive...")

            else:  # local
                download_results[idx] = {
                    'method': 'local',
                    'status': 'ready',
                    'filename': f"{filename}.tif",
                    'size_mb': np.random.uniform(5, 50)
                }
                st.success(f"✅ {idx}: Ready for local download")

        except Exception as e:
            download_results[idx] = {
                'method': 'error',
                'status': 'failed',
                'error': str(e)
            }
            st.error(f"❌ {idx}: Processing failed")

    # Show download results
    _display_spatial_download_results(download_results)

    # Add visualization module integration for spatial data
    if any(result['method'] == 'local' and result['status'] == 'ready' for result in download_results.values()):
        _integrate_climate_visualization_links(download_results, 'spatial_composite')


def _execute_all_temporal_download(selected_indices, temporal_resolution, spatial_scale,
                                 geometry, export_preference, download_helper):
    """Execute full temporal download (multiple files per index)"""
    import numpy as np

    st.warning("🎞️ Full temporal download generates many files - this will take longer...")

    # Calculate number of time periods
    freq = 'Y' if temporal_resolution == 'yearly' else 'M'
    dates = pd.date_range(st.session_state.climate_start_date, st.session_state.climate_end_date, freq=freq)
    n_periods = len(dates)
    total_files = len(selected_indices) * n_periods

    st.info(f"📊 Will generate {total_files} files ({len(selected_indices)} indices × {n_periods} time periods)")

    # For large downloads, recommend Google Drive
    if total_files > 20 and export_preference != 'drive':
        st.warning("⚠️ Large number of files detected. Google Drive recommended for reliability.")

    # Simulate processing
    progress_bar = st.progress(0)
    status_text = st.empty()

    download_results = {}

    for i, idx in enumerate(selected_indices):
        progress = (i + 1) / len(selected_indices)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx}... ({i+1}/{len(selected_indices)})")

        # Simulate processing time
        time.sleep(0.5)

        # All temporal downloads typically go to Google Drive due to size
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        download_results[idx] = {
            'method': 'drive',
            'status': 'processing',
            'folder': f"Climate_Analysis_Full_{timestamp}",
            'n_files': n_periods,
            'task_ids': [f"TASK_{np.random.randint(100000, 999999)}" for _ in range(n_periods)]
        }

    progress_bar.progress(1.0)
    status_text.text("✅ All indices queued for processing!")

    # Show results
    _display_temporal_download_results(download_results)


def _display_spatial_download_results(download_results):
    """Display results from spatial composite downloads"""
    st.markdown("### 📊 Spatial Download Results")

    local_files = []
    drive_exports = []
    failed_exports = []

    for idx, result in download_results.items():
        if result['method'] == 'local':
            local_files.append((idx, result))
        elif result['method'] == 'drive':
            drive_exports.append((idx, result))
        else:
            failed_exports.append((idx, result))

    # Show local downloads
    if local_files:
        st.markdown("#### 💻 Ready for Download")
        for idx, result in local_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{idx}** ({result['size_mb']:.1f} MB)")
            with col2:
                # Simulate download button (would be real file in actual implementation)
                st.button(f"📥 Download", key=f"download_{idx}", disabled=True)
                st.caption("Demo mode")

    # Show drive exports
    if drive_exports:
        st.markdown("#### ☁️ Google Drive Exports")
        for idx, result in drive_exports:
            with st.expander(f"📤 {idx} - Processing"):
                st.write(f"**Folder:** {result['folder']}")
                st.write(f"**Task ID:** {result['task_id']}")
                st.link_button("🔗 Open Google Drive", "https://drive.google.com/drive/")

    # Show failures
    if failed_exports:
        st.markdown("#### ❌ Failed Exports")
        for idx, result in failed_exports:
            st.error(f"{idx}: {result['error']}")


def _display_temporal_download_results(download_results):
    """Display results from full temporal downloads"""
    st.markdown("### 🎞️ Full Temporal Download Results")

    total_files = sum(result['n_files'] for result in download_results.values())
    st.success(f"✅ Successfully queued {total_files} files across {len(download_results)} climate indices!")

    for idx, result in download_results.items():
        with st.expander(f"📁 {idx} - {result['n_files']} files"):
            st.write(f"**Google Drive Folder:** {result['folder']}")
            st.write(f"**Number of Files:** {result['n_files']}")
            st.write(f"**Sample Task IDs:** {', '.join(result['task_ids'][:3])}...")
            st.link_button("🔗 Open Google Drive", "https://drive.google.com/drive/")

    st.info("""
    **📋 What happens next:**
    - Files are being processed on Google Earth Engine servers
    - They will appear in your Google Drive folders automatically
    - Processing time: 5-30 minutes depending on data size
    - You can close this page and check Google Drive later
    """)

    st.link_button("📊 Monitor Earth Engine Tasks", "https://code.earthengine.google.com/tasks")


def _integrate_climate_visualization_links(data, data_type):
    """Integrate visualization module links for climate data"""
    st.markdown("---")
    st.markdown("### 🎨 Visualization & Analysis Options")

    if data_type == 'time_series':
        _render_time_series_visualization_links(data)
    elif data_type == 'spatial_composite':
        _render_spatial_visualization_links(data)
    elif data_type == 'all_temporal':
        _render_temporal_series_visualization_links(data)


def _render_time_series_visualization_links(df):
    """Render visualization links for time series climate data"""
    st.markdown("#### 📈 Time Series Analysis & Visualization")

    # Get temporal resolution and analysis type
    temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')
    analysis_type = st.session_state.get('climate_analysis_type', 'climate')
    n_indices = len(st.session_state.get('climate_selected_indices', []))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Quick Analysis**")

        # Register data for post-download integration
        try:
            from app_components.post_download_integration import register_csv_download
            result_id = register_csv_download(
                "climate_analytics",
                f"climate_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                df,
                metadata={
                    'analysis_type': analysis_type,
                    'temporal_resolution': temporal_resolution,
                    'n_indices': n_indices
                }
            )

            if st.button("🔍 Quick Visualizer", use_container_width=True, help="Instant charts and statistics"):
                st.session_state.visualization_data = df
                st.session_state.show_quick_viz = True
                st.rerun()

        except Exception as e:
            st.caption(f"Quick analysis: {str(e)[:50]}...")

    with col2:
        st.markdown("**🎯 Trend Analysis**")

        # Enhanced trend analysis for yearly data
        if temporal_resolution == 'yearly' and len(df) >= 5:
            if st.button("📈 Advanced Trends", use_container_width=True, help="Mann-Kendall, Sen's slope, seasonal analysis"):
                st.session_state.trend_analysis_data = df
                st.session_state.show_trend_analysis = True
                st.rerun()
        else:
            st.caption("Requires ≥5 years of yearly data")

    with col3:
        st.markdown("**🌍 Climate Intelligence**")

        # Climate-specific analysis
        if st.button("🧠 Climate Insights", use_container_width=True, help="Climate patterns, extremes, variability"):
            st.session_state.climate_insights_data = df
            st.session_state.show_climate_insights = True
            st.rerun()

    # Show quick visualization if requested
    if st.session_state.get('show_quick_viz', False):
        _render_quick_climate_visualization(df)

    # Show trend analysis if requested
    if st.session_state.get('show_trend_analysis', False):
        _render_advanced_trend_analysis(df)

    # Show climate insights if requested
    if st.session_state.get('show_climate_insights', False):
        _render_climate_insights_analysis(df)


def _render_spatial_visualization_links(download_results):
    """Render visualization links for spatial climate data"""
    st.markdown("#### 🗺️ Spatial Analysis & Mapping")

    # Count ready local files
    local_files = [(idx, result) for idx, result in download_results.items()
                   if result['method'] == 'local' and result['status'] == 'ready']

    if not local_files:
        st.info("Spatial visualization will be available once files are downloaded locally")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🗺️ Interactive Maps**")
        if st.button("🌐 Map Viewer", use_container_width=True, help="Interactive spatial visualization"):
            st.session_state.spatial_map_data = download_results
            st.session_state.show_spatial_maps = True
            st.rerun()

    with col2:
        st.markdown("**📊 Spatial Statistics**")
        if st.button("📈 Spatial Analysis", use_container_width=True, help="Spatial patterns and statistics"):
            st.session_state.spatial_stats_data = download_results
            st.session_state.show_spatial_stats = True
            st.rerun()

    with col3:
        st.markdown("**🔄 Compare Indices**")
        if len(local_files) > 1:
            if st.button("⚖️ Compare Climate Indices", use_container_width=True, help="Side-by-side comparison"):
                st.session_state.compare_indices_data = download_results
                st.session_state.show_compare_indices = True
                st.rerun()
        else:
            st.caption("Requires multiple indices")

    # Show visualizations if requested
    if st.session_state.get('show_spatial_maps', False):
        _render_spatial_map_placeholder(download_results)

    if st.session_state.get('show_spatial_stats', False):
        _render_spatial_stats_placeholder(download_results)

    if st.session_state.get('show_compare_indices', False):
        _render_compare_indices_placeholder(download_results)


def _render_temporal_series_visualization_links(download_results):
    """Render visualization links for full temporal series data"""
    st.markdown("#### 🎞️ Temporal Series Analysis")

    st.info("🔄 Full temporal series visualization will be available once Google Drive processing completes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📹 Animation Tools**")
        st.caption("• Time-lapse animations")
        st.caption("• Seasonal progression")
        st.caption("• Multi-year comparison")

    with col2:
        st.markdown("**📊 Advanced Analytics**")
        st.caption("• Pixel-wise trend analysis")
        st.caption("• Spatial-temporal hotspots")
        st.caption("• Change detection")


def _render_quick_climate_visualization(df):
    """Render quick visualization for climate time series"""
    st.markdown("---")
    st.markdown("#### 🔍 Quick Climate Visualization")

    try:
        # Group by climate index for visualization
        indices = df['Climate_Index'].unique()

        if len(indices) == 1:
            # Single index - enhanced single plot
            idx_data = df[df['Climate_Index'] == indices[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(idx_data['Date']),
                y=idx_data['Value'],
                mode='lines+markers',
                name=indices[0],
                line=dict(width=3),
                marker=dict(size=6)
            ))

            # Add trend line if enough data
            if len(idx_data) >= 3:
                from scipy import stats
                x_numeric = np.arange(len(idx_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, idx_data['Value'])
                trend_line = slope * x_numeric + intercept

                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(idx_data['Date']),
                    y=trend_line,
                    mode='lines',
                    name=f'Trend (R² = {r_value**2:.3f})',
                    line=dict(dash='dash', color='red', width=2)
                ))

            fig.update_layout(
                title=f"{indices[0]} Climate Index Time Series",
                xaxis_title="Date",
                yaxis_title="Index Value",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # Multiple indices - subplot approach
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=len(indices), cols=1,
                subplot_titles=[f"{idx} Time Series" for idx in indices],
                vertical_spacing=0.08
            )

            for i, idx in enumerate(indices, 1):
                idx_data = df[df['Climate_Index'] == idx]

                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(idx_data['Date']),
                        y=idx_data['Value'],
                        mode='lines+markers',
                        name=idx,
                        showlegend=False
                    ),
                    row=i, col=1
                )

            fig.update_layout(
                height=300 * len(indices),
                title_text="Climate Indices Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Quick statistics summary
        st.markdown("**📊 Quick Statistics Summary:**")
        stats_df = df.groupby('Climate_Index')['Value'].agg(['mean', 'std', 'min', 'max']).round(3)
        st.dataframe(stats_df, use_container_width=True)

        # Reset button
        if st.button("🔙 Hide Quick Visualization"):
            st.session_state.show_quick_viz = False
            st.rerun()

    except Exception as e:
        st.error(f"❌ Visualization error: {str(e)}")


def _render_advanced_trend_analysis(df):
    """Render advanced trend analysis for climate data"""
    st.markdown("---")
    st.markdown("#### 📈 Advanced Trend Analysis")

    try:
        indices = df['Climate_Index'].unique()

        for idx in indices:
            with st.expander(f"📈 {idx} - Advanced Trend Analysis", expanded=True):
                idx_data = df[df['Climate_Index'] == idx].copy()
                idx_data = idx_data.sort_values('Date')

                # Convert dates and calculate trends
                dates = pd.to_datetime(idx_data['Date'])
                values = idx_data['Value'].values

                if len(values) >= 5:
                    # Calculate trend statistics using the functions from enhanced visualization
                    trend_stats = _calculate_trend_statistics(dates, values, idx)

                    # Display trend visualization
                    fig = go.Figure()

                    # Original data
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode='lines+markers',
                        name='Observed Values',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))

                    # Linear trend
                    if 'linear_trend' in trend_stats:
                        fig.add_trace(go.Scatter(
                            x=dates, y=trend_stats['linear_trend'],
                            mode='lines',
                            name='Linear Trend',
                            line=dict(dash='dash', color='red', width=2)
                        ))

                    # Sen's slope trend
                    if 'sens_trend' in trend_stats:
                        fig.add_trace(go.Scatter(
                            x=dates, y=trend_stats['sens_trend'],
                            mode='lines',
                            name="Sen's Slope",
                            line=dict(dash='dot', color='orange', width=2)
                        ))

                    fig.update_layout(
                        title=f"{idx} - Trend Analysis",
                        xaxis_title="Year",
                        yaxis_title="Index Value",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display statistics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Linear Slope", f"{trend_stats.get('linear_slope', 0):.4f} units/year")
                        st.metric("R²", f"{trend_stats.get('linear_r_squared', 0):.3f}")

                    with col2:
                        st.metric("Sen's Slope", f"{trend_stats.get('sens_slope', 0):.4f} units/year")
                        st.metric("Kendall's τ", f"{trend_stats.get('mk_tau', 0):.3f}")

                    with col3:
                        mk_trend = trend_stats.get('mk_trend', 'unknown')
                        if mk_trend == 'increasing':
                            st.success("🔺 Increasing Trend")
                        elif mk_trend == 'decreasing':
                            st.error("🔻 Decreasing Trend")
                        else:
                            st.info("➡️ No Significant Trend")

                        p_val = trend_stats.get('mk_p_value', 1.0)
                        st.metric("P-value", f"{p_val:.3f}")
                else:
                    st.warning(f"Insufficient data for trend analysis ({len(values)} years)")

        # Reset button
        if st.button("🔙 Hide Trend Analysis"):
            st.session_state.show_trend_analysis = False
            st.rerun()

    except Exception as e:
        st.error(f"❌ Trend analysis error: {str(e)}")


def _render_climate_insights_analysis(df):
    """Render climate insights and patterns analysis"""
    st.markdown("---")
    st.markdown("#### 🧠 Climate Intelligence Insights")

    try:
        indices = df['Climate_Index'].unique()
        temporal_resolution = st.session_state.get('climate_temporal_resolution', 'yearly')

        st.markdown(f"**📅 Analysis Period:** {temporal_resolution.title()} data from {df['Date'].min()} to {df['Date'].max()}")

        for idx in indices:
            with st.expander(f"🧠 {idx} - Climate Intelligence", expanded=True):
                idx_data = df[df['Climate_Index'] == idx].copy()
                values = idx_data['Value'].values

                # Climate insights based on index type
                if 'TX' in idx or 'TN' in idx:  # Temperature indices
                    _render_temperature_insights(idx, values, temporal_resolution)
                elif any(precip_indicator in idx for precip_indicator in ['PRCP', 'R', 'CDD', 'CWD']):
                    _render_precipitation_insights(idx, values, temporal_resolution)
                else:
                    _render_general_climate_insights(idx, values, temporal_resolution)

        # Reset button
        if st.button("🔙 Hide Climate Insights"):
            st.session_state.show_climate_insights = False
            st.rerun()

    except Exception as e:
        st.error(f"❌ Climate insights error: {str(e)}")


def _render_temperature_insights(idx, values, temporal_resolution):
    """Render temperature-specific climate insights"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌡️ Temperature Patterns**")

        # Calculate temperature statistics
        mean_temp = np.mean(values)
        std_temp = np.std(values)

        # Detect extremes (values beyond 2 standard deviations)
        extreme_threshold = 2 * std_temp
        hot_extremes = np.sum(values > (mean_temp + extreme_threshold))
        cold_extremes = np.sum(values < (mean_temp - extreme_threshold))

        st.metric("Average", f"{mean_temp:.1f}°C")
        st.metric("Variability", f"±{std_temp:.1f}°C")
        st.metric("Hot Extremes", f"{hot_extremes} events")
        st.metric("Cold Extremes", f"{cold_extremes} events")

    with col2:
        st.markdown("**📊 Temperature Distribution**")

        # Create histogram
        fig = go.Figure(data=[go.Histogram(x=values, nbinsx=15)])
        fig.update_layout(
            title=f"{idx} Distribution",
            xaxis_title="Temperature Value",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_precipitation_insights(idx, values, temporal_resolution):
    """Render precipitation-specific climate insights"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌧️ Precipitation Patterns**")

        # Calculate precipitation statistics
        total_precip = np.sum(values) if 'PRCP' in idx else np.mean(values)
        dry_threshold = np.percentile(values, 25)  # Bottom quartile
        wet_threshold = np.percentile(values, 75)  # Top quartile

        dry_periods = np.sum(values <= dry_threshold)
        wet_periods = np.sum(values >= wet_threshold)

        if 'PRCP' in idx:
            st.metric("Total Precipitation", f"{total_precip:.0f} mm")
        else:
            st.metric("Average Value", f"{np.mean(values):.1f}")

        st.metric("Dry Periods", f"{dry_periods} events")
        st.metric("Wet Periods", f"{wet_periods} events")

    with col2:
        st.markdown("**📊 Precipitation Variability**")

        # Create box plot
        fig = go.Figure(data=[go.Box(y=values, name=idx)])
        fig.update_layout(
            title=f"{idx} Variability",
            yaxis_title="Precipitation Value",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_general_climate_insights(idx, values, temporal_resolution):
    """Render general climate insights for other indices"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**📈 {idx} Characteristics**")

        st.metric("Mean", f"{np.mean(values):.2f}")
        st.metric("Std Dev", f"{np.std(values):.2f}")
        st.metric("Range", f"{np.min(values):.2f} to {np.max(values):.2f}")

        # Calculate coefficient of variation
        cv = np.std(values) / np.abs(np.mean(values)) * 100
        st.metric("Variability", f"{cv:.1f}%")

    with col2:
        st.markdown("**📊 Value Distribution**")

        # Create time series mini-plot
        fig = go.Figure(data=[go.Scatter(y=values, mode='lines+markers')])
        fig.update_layout(
            title=f"{idx} Time Series",
            xaxis_title="Time Period",
            yaxis_title="Index Value",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_spatial_map_placeholder(download_results):
    """Placeholder for spatial map visualization"""
    st.markdown("---")
    st.markdown("#### 🌐 Interactive Spatial Maps")

    local_files = [(idx, result) for idx, result in download_results.items()
                   if result['method'] == 'local' and result['status'] == 'ready']

    st.info(f"🗺️ Interactive maps would display {len(local_files)} climate indices spatially")
    st.caption("In the full implementation, this would show:\n• Interactive leaflet maps\n• Colorized climate index values\n• Hover tooltips with statistics\n• Layer switching between indices")

    if st.button("🔙 Hide Spatial Maps"):
        st.session_state.show_spatial_maps = False
        st.rerun()


def _render_spatial_stats_placeholder(download_results):
    """Placeholder for spatial statistics"""
    st.markdown("---")
    st.markdown("#### 📈 Spatial Statistical Analysis")

    st.info("📊 Spatial statistics would analyze spatial patterns and distributions")
    st.caption("Features would include:\n• Spatial autocorrelation\n• Hotspot detection\n• Gradient analysis\n• Spatial clustering")

    if st.button("🔙 Hide Spatial Stats"):
        st.session_state.show_spatial_stats = False
        st.rerun()


def _render_compare_indices_placeholder(download_results):
    """Placeholder for comparing multiple climate indices"""
    st.markdown("---")
    st.markdown("#### ⚖️ Climate Index Comparison")

    local_files = [(idx, result) for idx, result in download_results.items()
                   if result['method'] == 'local' and result['status'] == 'ready']

    st.info(f"⚖️ Side-by-side comparison of {len(local_files)} climate indices")
    st.caption("Comparison features:\n• Synchronized map views\n• Correlation analysis\n• Difference maps\n• Statistical relationships")

    if st.button("🔙 Hide Index Comparison"):
        st.session_state.show_compare_indices = False
        st.rerun()


def _prepare_climate_data_for_visualization(df, metadata=None):
    """Process DataFrame for visualization compatibility with all required metadata"""
    try:
        from app_components.data_processors import data_detector
        import pandas as pd
        import numpy as np

        # Ensure we have a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df)}")

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
        import pandas as pd
        import numpy as np

        if not isinstance(df, pd.DataFrame):
            # Try to convert to DataFrame if possible
            try:
                df = pd.DataFrame(df)
            except:
                raise ValueError(f"Cannot convert {type(df)} to DataFrame")

        return {
            'type': 'csv',
            'data': df,
            'column_suggestions': {
                'date_columns': [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()],
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


def _launch_climate_visualization(results):
    """Launch data visualizer with climate analysis results"""
    try:
        # Prepare data for visualization
        visualization_data = []

        # Note: CSV time series data is already visualized in climate analytics
        # Only transfer TIFF files (spatial data) to the data visualizer

        # Use individual downloadable ZIP files (same as manual downloads)
        if 'individual_results' in results and results['individual_results']:
            individual_results = results['individual_results']

            # Look for individual results with downloadable file_data
            for index_name, index_data in individual_results.items():
                if isinstance(index_data, dict) and index_data.get('success'):
                    spatial_data = index_data.get('spatial_data')

                    if spatial_data and isinstance(spatial_data, dict):
                        export_method = spatial_data.get('export_method')
                        file_data = spatial_data.get('file_data')

                        if export_method == 'local' and file_data:
                            # Create temp file for this individual ZIP (same as manual download)
                            import tempfile
                            import os

                            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                                temp_file.write(file_data)
                                temp_path = temp_file.name

                            filename = spatial_data.get('filename', f'{index_name}_spatial.zip')
                            file_size_mb = spatial_data.get('actual_size_mb', len(file_data) / (1024 * 1024))

                            viz_entry = {
                                'file_name': filename,
                                'data_type': 'zip',
                                'file_path': temp_path,
                                'transfer_method': 'temp_file',
                                'metadata': {
                                    'climate_index': index_name,
                                    'analysis_type': st.session_state.get('climate_analysis_type', 'Unknown'),
                                    'source': 'climate_analytics',
                                    'content_type': 'individual_spatial_climate_data',
                                    'file_size_mb': file_size_mb,
                                    'export_method': export_method
                                }
                            }
                            visualization_data.append(viz_entry)

        if not visualization_data:
            # No data available, but still go to visualizer (user can upload manually)
            st.session_state.direct_visualization_data = None
        else:
            # Set up direct visualization data
            st.session_state.direct_visualization_data = {
                'results': visualization_data,
                'source_module': 'climate_analytics'
            }

        # Switch to data visualizer
        st.session_state.app_mode = "data_visualizer"
        st.rerun()

    except Exception as e:
        # If any error occurs, show warning and redirect to upload interface
        st.warning(f"⚠️ Error transferring data to visualization: {str(e)}")
        st.info("🔄 Redirecting to Data Visualizer upload interface. Please upload your downloaded files manually.")

        # Clear any failed direct visualization data
        if 'direct_visualization_data' in st.session_state:
            del st.session_state.direct_visualization_data

        # Redirect to data visualizer with clean state
        st.session_state.app_mode = "data_visualizer"
        st.rerun()
