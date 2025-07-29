import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium.plugins import Draw
import json
from pathlib import Path
import sys
from datetime import datetime, date, timedelta
import zipfile
import io
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import ee
import logging

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import GeometryHandler, MetadataCatalog, GEEExporter
from geoclimate_fetcher.core.product_selector import MeteostatHandler, GriddedDataHandler, StatisticalAnalyzer
from streamlit_folium import st_folium
from app_components.geometry_component import GeometryComponent
from app_components.product_selector_visualizer import ProductSelectorVisualizer
from app_components.product_selector_data_manager import DataManager

class ProductSelectorComponent:
    """Component for optimal product selection analysis"""
    
    def __init__(self):
        """Initialize the component"""
        # Initialize session state variables
        self._init_session_state()
        
        # Initialize components
        self.geometry_component = GeometryComponent()
        
        # Initialize analysis components
        self.meteostat_handler = MeteostatHandler()
        self.gridded_handler = GriddedDataHandler()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ProductSelectorVisualizer()
        self.data_manager = DataManager()
        
        # Load station metadata and dataset catalogs
        self._load_data_catalogs()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'ps_geometry_complete' not in st.session_state:
            st.session_state.ps_geometry_complete = False
        if 'ps_stations_loaded' not in st.session_state:
            st.session_state.ps_stations_loaded = False
        if 'ps_variable_selected' not in st.session_state:
            st.session_state.ps_variable_selected = False
        if 'ps_datasets_selected' not in st.session_state:
            st.session_state.ps_datasets_selected = False
        if 'ps_timerange_selected' not in st.session_state:
            st.session_state.ps_timerange_selected = False
        if 'ps_analysis_complete' not in st.session_state:
            st.session_state.ps_analysis_complete = False
        
        # Data storage
        if 'ps_selected_geometry' not in st.session_state:
            st.session_state.ps_selected_geometry = None
        if 'ps_available_stations' not in st.session_state:
            st.session_state.ps_available_stations = None
        if 'ps_selected_variable' not in st.session_state:
            st.session_state.ps_selected_variable = None
        if 'ps_selected_datasets' not in st.session_state:
            st.session_state.ps_selected_datasets = []
        if 'ps_analysis_results' not in st.session_state:
            st.session_state.ps_analysis_results = None
        if 'ps_custom_data_mode' not in st.session_state:
            st.session_state.ps_custom_data_mode = False
    
    def _load_data_catalogs(self):
        """Load station metadata and dataset catalogs"""
        try:
            # Load meteostat stations
            stations_path = project_root / "geoclimate_fetcher" / "data" / "meteostat_stations.csv"
            if stations_path.exists():
                self.meteostat_stations = pd.read_csv(stations_path)
                st.session_state.meteostat_available = True
            else:
                self.meteostat_stations = None
                st.session_state.meteostat_available = False
            
            # Load climate datasets
            climate_index_path = project_root / "geoclimate_fetcher" / "data" / "climate_index"
            
            precipitation_path = climate_index_path / "precipitation.csv"
            temperature_path = climate_index_path / "temperature.csv"
            
            if precipitation_path.exists():
                self.precipitation_datasets = pd.read_csv(precipitation_path)
            else:
                self.precipitation_datasets = pd.DataFrame()
            
            if temperature_path.exists():
                self.temperature_datasets = pd.read_csv(temperature_path)
            else:
                self.temperature_datasets = pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error loading data catalogs: {str(e)}")
            self.meteostat_stations = None
            self.precipitation_datasets = pd.DataFrame()
            self.temperature_datasets = pd.DataFrame()
    
    def render(self):
        """Render the main component interface"""
        # Add home button
        if st.button("🏠 Back to Home", key="ps_home_button"):
            st.session_state.app_mode = None
            st.rerun()
        
        # Show progress indicator
        self._show_progress_indicator()
        
        # Step 1: Area Selection
        if not st.session_state.ps_geometry_complete:
            self._render_geometry_selection()
        
        # Step 2: Station Discovery/Upload
        elif not st.session_state.ps_stations_loaded:
            self._render_station_selection()
        
        # Step 3: Variable Selection
        elif not st.session_state.ps_variable_selected:
            self._render_variable_selection()
        
        # Step 4: Dataset Selection
        elif not st.session_state.ps_datasets_selected:
            self._render_dataset_selection()
        
        # Step 5: Time Range Selection
        elif not st.session_state.ps_timerange_selected:
            self._render_timerange_selection()
        
        # Step 6: Analysis and Results
        else:
            self._render_analysis_results()
    
    def _show_progress_indicator(self):
        """Display progress indicator"""
        steps = [
            ("🗺️", "Area", st.session_state.ps_geometry_complete),
            ("📡", "Stations", st.session_state.ps_stations_loaded),
            ("📊", "Variable", st.session_state.ps_variable_selected),
            ("🗂️", "Datasets", st.session_state.ps_datasets_selected),
            ("📅", "Timeframe", st.session_state.ps_timerange_selected),
            ("📈", "Analysis", st.session_state.ps_analysis_complete)
        ]
        
        cols = st.columns(len(steps))
        for i, (icon, label, completed) in enumerate(steps):
            with cols[i]:
                if completed:
                    st.markdown(f"✅ **{icon} {label}**")
                else:
                    st.markdown(f"⏳ {icon} {label}")
        
        st.markdown("---")
    
    def _render_geometry_selection(self):
        """Render area selection interface"""
        st.markdown("### 🗺️ Step 1: Select Area of Interest")
        st.markdown("Choose your study area using one of the methods below:")
        
        # Geometry selection options
        geometry_method = st.radio(
            "Select method:",
            ["🗺️ Draw on Map", "📁 Upload GeoJSON", "📍 Enter Coordinates"],
            horizontal=True
        )
        
        if geometry_method == "🗺️ Draw on Map":
            self._render_map_selection()
        elif geometry_method == "📁 Upload GeoJSON":
            self._render_geojson_upload()
        else:
            self._render_coordinate_input()
    
    def _render_map_selection(self):
        """Render interactive map for area selection"""
        st.markdown("**Draw a polygon on the map to define your area of interest:**")
        
        # Create map
        m = self.geometry_component.create_map()
        
        # Get map data
        map_data = st_folium(
            m,
            key="product_selector_map",
            width=700,
            height=500,
            returned_objects=["all_drawings"]
        )
        
        # Process map selection
        if map_data["all_drawings"]:
            if len(map_data["all_drawings"]) > 0:
                geometry = map_data["all_drawings"][-1]["geometry"]
                st.session_state.ps_selected_geometry = geometry
                
                st.success("✅ Area selected successfully!")
                if st.button("Continue to Station Discovery", type="primary", key="map_continue"):
                    st.session_state.ps_geometry_complete = True
                    st.rerun()
            else:
                st.info("👆 Draw a polygon on the map to select your area of interest")
        else:
            st.info("👆 Draw a polygon on the map to select your area of interest")
    
    def _render_geojson_upload(self):
        """Render GeoJSON upload interface"""
        st.markdown("**Upload a GeoJSON file:**")
        
        uploaded_file = st.file_uploader(
            "Choose GeoJSON file", 
            type=['geojson', 'json'],
            key="geojson_upload",
            help="Upload a GeoJSON file containing your area of interest"
        )
        
        if uploaded_file is not None:
            try:
                geojson_data = json.loads(uploaded_file.read().decode())
                
                # Extract geometry
                if geojson_data.get("type") == "FeatureCollection":
                    if len(geojson_data["features"]) > 0:
                        geometry = geojson_data["features"][0]["geometry"]
                    else:
                        st.error("No features found in GeoJSON")
                        return
                elif geojson_data.get("type") == "Feature":
                    geometry = geojson_data["geometry"]
                else:
                    geometry = geojson_data
                
                st.session_state.ps_selected_geometry = geometry
                st.success("✅ GeoJSON uploaded successfully!")
                
                if st.button("Continue to Station Discovery", type="primary", key="geojson_continue"):
                    st.session_state.ps_geometry_complete = True
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error reading GeoJSON: {str(e)}")
    
    def _render_coordinate_input(self):
        """Render coordinate input interface"""
        st.markdown("**Enter bounding box coordinates:**")
        
        col1, col2 = st.columns(2)
        with col1:
            min_lat = st.number_input("Minimum Latitude", value=40.0, step=0.001, format="%.3f")
            min_lon = st.number_input("Minimum Longitude", value=-75.0, step=0.001, format="%.3f")
        
        with col2:
            max_lat = st.number_input("Maximum Latitude", value=41.0, step=0.001, format="%.3f")
            max_lon = st.number_input("Maximum Longitude", value=-74.0, step=0.001, format="%.3f")
        
        if st.button("Create Bounding Box", type="primary", key="create_bbox"):
            # Create polygon geometry from bounding box
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            }
            
            st.session_state.ps_selected_geometry = geometry
            st.success("✅ Bounding box created successfully!")
            
            if st.button("Continue to Station Discovery", type="primary", key="bbox_continue"):
                st.session_state.ps_geometry_complete = True
                st.rerun()
    
    def _render_station_selection(self):
        """Render station discovery/upload interface"""
        st.markdown("### 📡 Step 2: Station Data Source")
        
        # Add back button
        if st.button("← Back to Area Selection", key="back_to_area"):
            st.session_state.ps_geometry_complete = False
            st.rerun()
        
        # Check if meteostat is available
        if st.session_state.meteostat_available:
            data_source = st.radio(
                "Choose data source:",
                ["🌐 Use Meteostat Stations", "📁 Upload Custom Data"],
                horizontal=True
            )
            
            if data_source == "🌐 Use Meteostat Stations":
                self._render_meteostat_discovery()
            else:
                st.session_state.ps_custom_data_mode = True
                self._render_custom_data_upload()
        else:
            st.warning("⚠️ Meteostat stations data not found. Please upload custom data.")
            st.session_state.ps_custom_data_mode = True
            self._render_custom_data_upload()
    
    def _render_meteostat_discovery(self):
        """Render meteostat station discovery"""
        st.markdown("**Discovering stations within your selected area...**")
        
        # Extract bounds from geometry
        bounds = self._extract_bounds_from_geometry(st.session_state.ps_selected_geometry)
        
        if bounds:
            # Filter stations within bounds
            filtered_stations = self._filter_stations_by_bounds(self.meteostat_stations, bounds)
            
            if not filtered_stations.empty:
                st.success(f"✅ Found {len(filtered_stations)} stations in the selected area")
                
                # Show station information
                st.markdown("**Available Stations:**")
                
                # Create a summary table
                summary_cols = ['id', 'name', 'latitude', 'longitude', 'daily_start', 'daily_end']
                available_cols = [col for col in summary_cols if col in filtered_stations.columns]
                
                st.dataframe(
                    filtered_stations[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Store stations
                st.session_state.ps_available_stations = filtered_stations
                
                if st.button("Continue with These Stations", type="primary", key="continue_stations"):
                    st.session_state.ps_stations_loaded = True
                    st.rerun()
            else:
                st.warning("⚠️ No meteostat stations found in the selected area.")
                st.markdown("**Options:**")
                st.markdown("1. Try a larger area")
                st.markdown("2. Upload custom station data")
                
                if st.button("Upload Custom Data", key="upload_custom"):
                    st.session_state.ps_custom_data_mode = True
                    st.rerun()
        else:
            st.error("Could not extract bounds from selected geometry")
    
    def _render_custom_data_upload(self):
        """Render custom data upload interface"""
        st.markdown("**Upload Custom Station Data:**")
        
        st.markdown("""
        **Required Format:**
        - **Metadata File**: CSV with columns: id, latitude, longitude, daily_start, daily_end
        - **Data File**: CSV with columns: date, station_id, value
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Station Metadata:**")
            metadata_file = st.file_uploader(
                "Upload metadata CSV",
                type=['csv'],
                key="metadata_upload",
                help="CSV file with station information"
            )
        
        with col2:
            st.markdown("**Station Data:**")
            data_file = st.file_uploader(
                "Upload data CSV", 
                type=['csv'],
                key="data_upload",
                help="CSV file with time series data"
            )
        
        if metadata_file is not None and data_file is not None:
            try:
                # Load and validate files
                metadata_df = pd.read_csv(metadata_file)
                data_df = pd.read_csv(data_file)
                
                # Validate metadata columns
                required_meta_cols = ['id', 'latitude', 'longitude', 'daily_start', 'daily_end']
                missing_meta_cols = [col for col in required_meta_cols if col not in metadata_df.columns]
                
                # Validate data columns
                required_data_cols = ['date', 'station_id', 'value']
                missing_data_cols = [col for col in required_data_cols if col not in data_df.columns]
                
                if missing_meta_cols or missing_data_cols:
                    st.error("❌ Missing required columns:")
                    if missing_meta_cols:
                        st.error(f"Metadata: {missing_meta_cols}")
                    if missing_data_cols:
                        st.error(f"Data: {missing_data_cols}")
                else:
                    # Filter stations within selected area if geometry is available
                    if st.session_state.ps_selected_geometry:
                        bounds = self._extract_bounds_from_geometry(st.session_state.ps_selected_geometry)
                        if bounds:
                            filtered_metadata = self._filter_stations_by_bounds(metadata_df, bounds)
                        else:
                            filtered_metadata = metadata_df
                    else:
                        filtered_metadata = metadata_df
                    
                    if not filtered_metadata.empty:
                        st.success(f"✅ Loaded {len(filtered_metadata)} stations")
                        st.dataframe(filtered_metadata, use_container_width=True, hide_index=True)
                        
                        # Store data
                        st.session_state.ps_available_stations = filtered_metadata
                        st.session_state.ps_custom_station_data = data_df
                        
                        if st.button("Continue with Custom Data", type="primary", key="continue_custom"):
                            st.session_state.ps_stations_loaded = True
                            st.rerun()
                    else:
                        st.warning("⚠️ No stations found within the selected area")
                        
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
    
    def _render_variable_selection(self):
        """Render variable selection interface"""
        st.markdown("### 📊 Step 3: Select Variable for Analysis")
        
        # Add back button
        if st.button("← Back to Station Selection", key="back_to_stations"):
            st.session_state.ps_stations_loaded = False
            st.rerun()
        
        st.markdown("Choose one variable to analyze (only one variable can be analyzed at a time):")
        
        variable_options = {
            "🌧️ Precipitation": "prcp",
            "🌡️ Maximum Temperature": "tmax", 
            "❄️ Minimum Temperature": "tmin"
        }
        
        selected_var = st.radio(
            "Variable:",
            list(variable_options.keys()),
            help="Select the climate variable you want to analyze"
        )
        
        variable_code = variable_options[selected_var]
        
        # Show variable information
        if variable_code == "prcp":
            st.info("📊 **Precipitation Analysis** - Compare daily precipitation totals between station and gridded data")
        elif variable_code == "tmax":
            st.info("📊 **Maximum Temperature Analysis** - Compare daily maximum temperature between station and gridded data")
        else:
            st.info("📊 **Minimum Temperature Analysis** - Compare daily minimum temperature between station and gridded data")
        
        if st.button("Continue with Selected Variable", type="primary", key="continue_variable"):
            st.session_state.ps_selected_variable = variable_code
            st.session_state.ps_variable_selected = True
            st.rerun()
    
    def _render_dataset_selection(self):
        """Render dataset selection interface"""
        st.markdown("### 🗂️ Step 4: Select Gridded Datasets for Comparison")
        
        # Add back button
        if st.button("← Back to Variable Selection", key="back_to_variable"):
            st.session_state.ps_variable_selected = False
            st.rerun()
        
        variable = st.session_state.ps_selected_variable
        
        # Choose appropriate catalog based on variable
        if variable == "prcp":
            available_datasets = self.precipitation_datasets
            st.markdown("**Available Precipitation Datasets:**")
        else:  # tmax or tmin
            available_datasets = self.temperature_datasets
            st.markdown("**Available Temperature Datasets:**")
        
        if available_datasets.empty:
            st.error("❌ No datasets available for the selected variable")
            return
        
        # Display datasets for selection
        st.markdown("Select one or more datasets to compare:")
        
        selected_datasets = []
        for idx, row in available_datasets.iterrows():
            dataset_name = row.get('Dataset Name', f'Dataset {idx}')
            description = row.get('Description', 'No description available')
            temporal_res = row.get('Temporal Resolution', 'Unknown')
            pixel_size = row.get('Pixel Size (m)', 'Unknown')
            
            # Create checkbox for each dataset
            if st.checkbox(
                f"**{dataset_name}**",
                key=f"dataset_{idx}",
                help=f"Temporal Resolution: {temporal_res} | Pixel Size: {pixel_size}m"
            ):
                selected_datasets.append(row.to_dict())
                
                # Show dataset details
                with st.expander(f"Details for {dataset_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Provider:** {row.get('Provider', 'N/A')}")
                        st.write(f"**Start Date:** {row.get('Start Date', 'N/A')}")
                        st.write(f"**End Date:** {row.get('End Date', 'N/A')}")
                    with col2:
                        st.write(f"**Temporal Resolution:** {temporal_res}")
                        st.write(f"**Pixel Size:** {pixel_size}m")
                        st.write(f"**Band Names:** {row.get('Band Names', 'N/A')}")
                    
                    st.write(f"**Description:** {description}")
        
        if selected_datasets:
            st.success(f"✅ Selected {len(selected_datasets)} dataset(s)")
            
            if st.button("Continue with Selected Datasets", type="primary", key="continue_datasets"):
                st.session_state.ps_selected_datasets = selected_datasets
                st.session_state.ps_datasets_selected = True
                st.rerun()
        else:
            st.info("👆 Select at least one dataset to continue")
    
    def _render_timerange_selection(self):
        """Render time range selection interface"""
        st.markdown("### 📅 Step 5: Select Analysis Time Period")
        
        # Add back button
        if st.button("← Back to Dataset Selection", key="back_to_datasets"):
            st.session_state.ps_datasets_selected = False
            st.rerun()
        
        # Auto-detect optimal time range
        optimal_range = self._detect_optimal_timerange()
        
        if optimal_range:
            start_date, end_date = optimal_range
            st.success(f"✅ Optimal time period detected: {start_date} to {end_date}")
            
            # Allow user to modify the range
            st.markdown("**Adjust time period if needed:**")
            
            col1, col2 = st.columns(2)
            with col1:
                user_start = st.date_input("Start Date", value=start_date, min_value=start_date)
            with col2:
                user_end = st.date_input("End Date", value=end_date, max_value=end_date)
            
            # Validate date range
            if user_start <= user_end:
                st.session_state.ps_analysis_start_date = user_start
                st.session_state.ps_analysis_end_date = user_end
                
                period_days = (user_end - user_start).days
                st.info(f"📊 Analysis period: {period_days} days")
                
                if st.button("Start Analysis", type="primary", key="start_analysis"):
                    st.session_state.ps_timerange_selected = True
                    st.rerun()
            else:
                st.error("❌ Start date must be before end date")
        else:
            st.error("❌ Could not detect optimal time period. No overlapping data found.")
            st.markdown("**Possible issues:**")
            st.markdown("- Selected datasets have no temporal overlap with station data")
            st.markdown("- Station data periods don't overlap with gridded data periods")
    
    def _render_analysis_results(self):
        """Render analysis and results interface"""
        st.markdown("### 📈 Step 6: Analysis Results")
        
        # Add back button and restart button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Time Range", key="back_to_timerange"):
                st.session_state.ps_timerange_selected = False
                st.rerun()
        with col2:
            if st.button("🔄 Start New Analysis", key="restart_analysis"):
                # Reset all session state variables
                st.session_state.ps_geometry_complete = False
                st.session_state.ps_stations_loaded = False
                st.session_state.ps_variable_selected = False
                st.session_state.ps_datasets_selected = False
                st.session_state.ps_timerange_selected = False
                st.session_state.ps_analysis_complete = False
                st.rerun()
        
        # Run analysis if not already done
        if not st.session_state.ps_analysis_complete:
            self._run_analysis()
        
        # Display results
        if st.session_state.ps_analysis_complete and st.session_state.ps_analysis_results:
            self._display_analysis_results()
        else:
            st.error("❌ Analysis failed or no results available")
    
    def _extract_bounds_from_geometry(self, geometry: dict) -> Optional[Tuple[float, float, float, float]]:
        """Extract bounding box from geometry"""
        try:
            if geometry["type"] == "Polygon":
                coords = geometry["coordinates"][0]
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                return min(lons), min(lats), max(lons), max(lats)
            # Add support for other geometry types if needed
        except Exception:
            pass
        return None
    
    def _filter_stations_by_bounds(self, stations_df: pd.DataFrame, bounds: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Filter stations within bounding box"""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        return stations_df[
            (stations_df['latitude'] >= min_lat) &
            (stations_df['latitude'] <= max_lat) &
            (stations_df['longitude'] >= min_lon) &
            (stations_df['longitude'] <= max_lon)
        ]
    
    def _detect_optimal_timerange(self) -> Optional[Tuple[date, date]]:
        """Detect optimal overlapping time range"""
        try:
            # Get station data periods
            stations = st.session_state.ps_available_stations
            station_starts = []
            station_ends = []
            
            for _, station in stations.iterrows():
                if pd.notna(station.get('daily_start')):
                    station_starts.append(pd.to_datetime(station['daily_start']).date())
                if pd.notna(station.get('daily_end')):
                    station_ends.append(pd.to_datetime(station['daily_end']).date())
            
            if not station_starts or not station_ends:
                return None
            
            # Get dataset periods
            datasets = st.session_state.ps_selected_datasets
            dataset_starts = []
            dataset_ends = []
            
            for dataset in datasets:
                if dataset.get('Start Date'):
                    try:
                        start_str = str(dataset['Start Date'])
                        # Handle different date formats
                        if '/' in start_str:
                            start_date = datetime.strptime(start_str, '%m/%d/%Y').date()
                        else:
                            start_date = pd.to_datetime(start_str).date()
                        dataset_starts.append(start_date)
                    except:
                        continue
                
                if dataset.get('End Date'):
                    try:
                        end_str = str(dataset['End Date'])
                        if '/' in end_str:
                            end_date = datetime.strptime(end_str, '%m/%d/%Y').date()
                        else:
                            end_date = pd.to_datetime(end_str).date()
                        dataset_ends.append(end_date)
                    except:
                        continue
            
            if not dataset_starts or not dataset_ends:
                return None
            
            # Find optimal overlap
            latest_start = max(max(station_starts), max(dataset_starts))
            earliest_end = min(min(station_ends), min(dataset_ends))
            
            if latest_start <= earliest_end:
                return latest_start, earliest_end
            
        except Exception as e:
            st.error(f"Error detecting time range: {str(e)}")
        
        return None
    
    def _run_analysis(self):
        """Run the statistical analysis"""
        st.markdown("🔄 **Running Analysis...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Get analysis parameters
            stations = st.session_state.ps_available_stations
            datasets = st.session_state.ps_selected_datasets
            variable = st.session_state.ps_selected_variable
            start_date = st.session_state.ps_analysis_start_date
            end_date = st.session_state.ps_analysis_end_date
            
            all_results = {}
            total_steps = len(stations) * len(datasets)
            current_step = 0
            
            # Analyze each station
            for _, station in stations.iterrows():
                station_id = station['id']
                station_results = {}
                
                status_text.text(f"Analyzing station {station_id}...")
                
                # Get station data
                try:
                    station_data = self.meteostat_handler.get_station_data(
                        station_id, variable, start_date, end_date
                    )
                    
                    if station_data is None or len(station_data) == 0:
                        logging.warning(f"No station data for {station_id}")
                        continue
                        
                except Exception as e:
                    logging.error(f"Error getting station data for {station_id}: {str(e)}")
                    continue
                
                # Compare with each dataset
                for dataset in datasets:
                    # Create dataset ID mapping from dataset name to simple identifier
                    dataset_name = dataset.get('Dataset Name', 'unknown')
                    
                    # Map dataset names to simple IDs for the analysis engine
                    dataset_mapping = {
                        'Daymet V4 Daily Meteorology (NA)': 'daymet',
                        'CHIRPS Daily Precipitation': 'chirps',
                        'ERA5 Daily Aggregates': 'era5',
                        'ERA5-Land Hourly Climate Reanalysis': 'era5',
                        'GridMET Daily Meteorology (CONUS)': 'gridmet',
                        'GLDAS-2.1 Noah (0.25° 3-hourly)': 'gldas',
                        'GLDAS-2.2 CLSM (0.25° daily)': 'gldas',
                        'TerraClimate Monthly Climate': 'terraclimate',
                        'GPM IMERG Monthly Precipitation (V07)': 'imerg',
                        'CPC Unified Gauge Precipitation (Daily 0.5°)': 'cpc',
                        'FLDAS NOAH Land Data Assimilation (Monthly 0.1°)': 'gldas'
                    }
                    
                    # First try exact match, then fallback to a cleaner ID
                    if dataset_name in dataset_mapping:
                        dataset_id = dataset_mapping[dataset_name]
                    else:
                        # Create a cleaner ID from the name
                        dataset_id = dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('.', '')
                        # Simplify further by taking key parts
                        if 'daymet' in dataset_id:
                            dataset_id = 'daymet'
                        elif 'era5' in dataset_id:
                            dataset_id = 'era5'  
                        elif 'chirps' in dataset_id:
                            dataset_id = 'chirps'
                        elif 'gridmet' in dataset_id:
                            dataset_id = 'gridmet'
                        elif 'gldas' in dataset_id:
                            dataset_id = 'gldas'
                        elif 'terraclimate' in dataset_id:
                            dataset_id = 'terraclimate'
                        elif 'imerg' in dataset_id:
                            dataset_id = 'imerg'
                        elif 'cpc' in dataset_id:
                            dataset_id = 'cpc'
                    
                    logging.info(f"Processing dataset: '{dataset_name}' -> '{dataset_id}'")
                    
                    try:
                        # Get gridded data
                        gridded_data = self.gridded_handler.get_gridded_data(
                            dataset_id, variable, station['latitude'], station['longitude'],
                            start_date, end_date
                        )
                        
                        if gridded_data is None or len(gridded_data) == 0:
                            logging.warning(f"No gridded data for {dataset_id} at {station_id}")
                            continue
                        
                        # Merge and analyze
                        merged_data = self.statistical_analyzer.merge_datasets(
                            station_data, gridded_data
                        )
                        
                        if merged_data is None or len(merged_data) == 0:
                            logging.warning(f"No overlapping data for {station_id} - {dataset_id}")
                            continue
                        
                        # Calculate statistics
                        stats = self.statistical_analyzer.calculate_statistics(merged_data)
                        seasonal_stats = self.statistical_analyzer.calculate_seasonal_statistics(merged_data)
                        
                        station_results[dataset_id] = {
                            'stats': stats,
                            'seasonal_stats': seasonal_stats,
                            'merged_data': merged_data
                        }
                        
                    except Exception as e:
                        logging.error(f"Error analyzing {station_id} - {dataset_id}: {str(e)}")
                        continue
                    
                    current_step += 1
                    progress_bar.progress(min(current_step / total_steps, 1.0))
                
                if station_results:
                    all_results[station_id] = station_results
            
            if not all_results:
                st.error("❌ No valid analysis results found. Please check your data selection.")
                return
            
            # Store results
            stations_dict = {}
            for _, station in stations.iterrows():
                stations_dict[station['id']] = station.to_dict()
            
            results = {
                'stations_data': stations_dict,
                'analysis_results': all_results,
                'analysis_summary': {
                    'total_stations': len(stations),
                    'analyzed_stations': len(all_results),
                    'total_datasets': len(datasets),
                    'variable': variable,
                    'start_date': str(start_date),
                    'end_date': str(end_date)
                }
            }
            
            st.session_state.ps_analysis_results = results
            st.session_state.ps_analysis_complete = True
            
            status_text.text("✅ Analysis completed successfully!")
            progress_bar.empty()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logging.error(f"Analysis error: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _display_analysis_results(self):
        """Display analysis results"""
        results = st.session_state.ps_analysis_results
        
        st.success("✅ **Analysis Completed Successfully!**")
        
        # Summary information
        summary = results['analysis_summary']
        analysis_results = results['analysis_results']
        stations_data = results['stations_data']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stations Analyzed", summary['analyzed_stations'])
        with col2:
            st.metric("Datasets Compared", summary['total_datasets'])
        with col3:
            st.metric("Variable", summary['variable'].upper())
        with col4:
            start_date = summary['start_date']
            end_date = summary['end_date']
            period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            st.metric("Analysis Period", f"{period_days} days")
        
        # Overall comparison overview
        st.markdown("---")
        st.markdown("### � Overall Performance Comparison")
        
        if analysis_results:
            overview_fig = self.visualizer.create_comparison_overview(
                analysis_results, summary['variable']
            )
            st.plotly_chart(overview_fig, use_container_width=True)
            
            # Station locations map
            stations_df = pd.DataFrame([station for station in stations_data.values()])
            map_fig = self.visualizer.create_station_map(stations_df, analysis_results)
            st.plotly_chart(map_fig, use_container_width=True)
        
        # Station-by-station results
        st.markdown("---")
        st.markdown("### 📊 Station-by-Station Results")
        
        for i, (station_id, station_results) in enumerate(analysis_results.items()):
            with st.expander(f"📍 {station_id} Results", expanded=i==0):
                self._display_station_results(station_id, stations_data[station_id], station_results)
        
        # Download section
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        with col1:
            include_timeseries = st.checkbox("Include detailed time series data", value=False,
                                           help="Warning: This will significantly increase file size")
        with col2:
            if st.button("📦 Prepare Download Package", type="primary", key="prepare_download"):
                self._prepare_download_package(include_timeseries)
    
    def _display_station_results(self, station_id: str, station_info: dict, station_results: dict):
        """Display results for individual station"""
        st.markdown(f"**Station Information:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ID:** {station_id}")
            st.write(f"**Latitude:** {station_info.get('latitude', 'N/A')}")
            st.write(f"**Longitude:** {station_info.get('longitude', 'N/A')}")
        with col2:
            st.write(f"**Name:** {station_info.get('name', 'N/A')}")
            st.write(f"**Elevation:** {station_info.get('elevation', 'N/A')} m")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["📊 Statistics", "📈 Scatter Plots", "📉 Time Series", "📦 Seasonal Analysis"])
        
        with tabs[0]:  # Statistics
            st.markdown("**Statistical Summary:**")
            
            # Create summary table
            stats_data = []
            for dataset_name, results in station_results.items():
                if 'stats' in results:
                    stats = results['stats']
                    stats_data.append({
                        'Dataset': dataset_name,
                        'N_Obs': stats.get('n_observations', 0),
                        'RMSE': f"{stats.get('rmse', 0):.3f}",
                        'MAE': f"{stats.get('mae', 0):.3f}",
                        'R²': f"{stats.get('r2', 0):.3f}",
                        'Correlation': f"{stats.get('correlation', 0):.3f}",
                        'Bias': f"{stats.get('bias', 0):.3f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Find best performing dataset
                best_dataset = max(station_results.keys(), 
                                 key=lambda x: station_results[x]['stats'].get('r2', -1))
                best_r2 = station_results[best_dataset]['stats'].get('r2', 0)
                st.success(f"🏆 **Best performing dataset:** {best_dataset} (R² = {best_r2:.3f})")
        
        with tabs[1]:  # Scatter plots
            for dataset_name, results in station_results.items():
                if 'merged_data' in results:
                    st.markdown(f"**{dataset_name}**")
                    scatter_fig = self.visualizer.create_scatter_plot(
                        results['merged_data'], station_id, dataset_name, 
                        st.session_state.ps_selected_variable
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)
        
        with tabs[2]:  # Time series
            for dataset_name, results in station_results.items():
                if 'merged_data' in results:
                    st.markdown(f"**{dataset_name}**")
                    ts_fig = self.visualizer.create_time_series_plot(
                        results['merged_data'], station_id, dataset_name,
                        st.session_state.ps_selected_variable
                    )
                    st.plotly_chart(ts_fig, use_container_width=True)
        
        with tabs[3]:  # Seasonal analysis
            for dataset_name, results in station_results.items():
                if 'merged_data' in results:
                    st.markdown(f"**{dataset_name}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        seasonal_fig = self.visualizer.create_seasonal_boxplot(
                            results['merged_data'], station_id, dataset_name,
                            st.session_state.ps_selected_variable
                        )
                        st.plotly_chart(seasonal_fig, use_container_width=True)
                    
                    with col2:
                        if 'seasonal_stats' in results:
                            seasonal_table = self.visualizer.create_seasonal_stats_table(
                                results['seasonal_stats'], station_id, dataset_name
                            )
                            st.plotly_chart(seasonal_table, use_container_width=True)
    
    def _prepare_download_package(self, include_timeseries: bool = False):
        """Prepare and offer download package"""
        try:
            with st.spinner("Preparing download package..."):
                results = st.session_state.ps_analysis_results
                
                # Validate results
                is_valid, error_msg = self.data_manager.validate_analysis_results(
                    results['analysis_results']
                )
                
                if not is_valid:
                    st.error(f"❌ Cannot create download package: {error_msg}")
                    return
                
                # Prepare download data
                download_data = self.data_manager.prepare_download_data(
                    results['analysis_results'],
                    results['stations_data'],
                    results['analysis_summary']['variable'],
                    include_timeseries
                )
                
                if not download_data:
                    st.error("❌ No data available for download")
                    return
                
                # Create ZIP file
                zip_data = self.data_manager.create_download_zip(download_data)
                
                if not zip_data:
                    st.error("❌ Failed to create download package")
                    return
                
                # Generate filename
                filename = self.data_manager.get_download_filename(
                    results['analysis_summary']['variable'],
                    results['analysis_summary']['analyzed_stations'],
                    results['analysis_summary']['total_datasets']
                )
                
                # Offer download
                st.download_button(
                    label="📥 Download Analysis Results",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    type="primary"
                )
                
                st.success("✅ Download package ready!")
                
                # Show what's included
                with st.expander("📋 Package Contents"):
                    for file_key in download_data.keys():
                        st.write(f"• {file_key.replace('_', ' ').title()}")
        
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            logging.error(f"Download preparation error: {str(e)}")
