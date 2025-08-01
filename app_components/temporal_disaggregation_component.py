"""
Temporal Disaggregation Component for high-resolution precipitation analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

from geoclimate_fetcher.core import GeometryHandler
from geoclimate_fetcher.core.temporal_disaggregation import TemporalDisaggregationHandler, BiasCorrection, OptimalSelection

class TemporalDisaggregationComponent:
    """Component for temporal disaggregation analysis"""
    
    def __init__(self):
        """Initialize the component"""
        # Initialize session state variables
        self._init_session_state()
        
        # Initialize analysis components
        self.td_handler = TemporalDisaggregationHandler()
        self.bias_correction = BiasCorrection()
        self.optimal_selection = OptimalSelection()
        
        # Pre-configured datasets
        self.satellite_datasets = {
            'era5_land': {
                'name': 'ERA5-Land Hourly',
                'ee_id': 'ECMWF/ERA5_LAND/HOURLY',
                'band': 'total_precipitation',
                'temporal_resolution': 'hourly',
                'units': 'meters',
                'units_type': 'total',
                'description': 'Hourly land surface reanalysis data',
                'start_date': '1950-01-01',
                'end_date': '2025-07-22'
            },
            'gpm_imerg': {
                'name': 'GPM IMERG V07',
                'ee_id': 'NASA/GPM_L3/IMERG_V07',
                'band': 'precipitation',
                'temporal_resolution': '30min',
                'units': 'mm/hr',
                'units_type': 'rate',
                'description': '30-minute global precipitation measurement',
                'start_date': '2000-06-01',
                'end_date': '2025-07-28'
            },
            'gsmap': {
                'name': 'GSMaP Operational V8',
                'ee_id': 'JAXA/GPM_L3/GSMaP/v8/operational',
                'band': 'hourlyPrecipRate',
                'temporal_resolution': 'hourly',
                'units': 'mm/hr',
                'units_type': 'rate',
                'description': 'Hourly satellite precipitation mapping',
                'start_date': '1998-01-01',
                'end_date': '2025-07-28'
            }
        }
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'td_data_uploaded' not in st.session_state:
            st.session_state.td_data_uploaded = False
        if 'td_metadata_uploaded' not in st.session_state:
            st.session_state.td_metadata_uploaded = False
        if 'td_timerange_selected' not in st.session_state:
            st.session_state.td_timerange_selected = False
        if 'td_satellite_data_downloaded' not in st.session_state:
            st.session_state.td_satellite_data_downloaded = False
        if 'td_analysis_complete' not in st.session_state:
            st.session_state.td_analysis_complete = False
        
        # Data storage
        if 'td_station_data' not in st.session_state:
            st.session_state.td_station_data = None
        if 'td_station_metadata' not in st.session_state:
            st.session_state.td_station_metadata = None
        if 'td_station_columns' not in st.session_state:
            st.session_state.td_station_columns = []
        if 'td_satellite_data' not in st.session_state:
            st.session_state.td_satellite_data = {}
        if 'td_analysis_results' not in st.session_state:
            st.session_state.td_analysis_results = None
        if 'td_corrected_data' not in st.session_state:
            st.session_state.td_corrected_data = {}
    
    def render(self):
        """Render the main component interface"""
        # Show progress indicator
        self._show_progress_indicator()
        
        # Step 1: Data Upload
        if not st.session_state.td_data_uploaded or not st.session_state.td_metadata_uploaded:
            self._render_data_upload()
        
        # Step 2: Time Range Selection
        elif not st.session_state.td_timerange_selected:
            self._render_timerange_selection()
        
        # Step 3: Satellite Data Download
        elif not st.session_state.td_satellite_data_downloaded:
            self._render_satellite_download()
        
        # Step 4: Analysis and Results
        else:
            self._render_analysis_results()
    
    def _show_progress_indicator(self):
        """Display progress indicator"""
        steps = [
            ("📁", "Data Upload", st.session_state.td_data_uploaded and st.session_state.td_metadata_uploaded),
            ("📅", "Time Range", st.session_state.td_timerange_selected),
            ("🛰️", "Satellite Data", st.session_state.td_satellite_data_downloaded),
            ("📊", "Analysis", st.session_state.td_analysis_complete),
            ("📥", "Export", False)  # Always false until download
        ]
        
        cols = st.columns(len(steps))
        for i, (icon, label, completed) in enumerate(steps):
            with cols[i]:
                if completed:
                    st.markdown(f"✅ **{icon} {label}**")
                else:
                    st.markdown(f"⏳ {icon} {label}")
        
        st.markdown("---")
    
    def _render_data_upload(self):
        """Render data upload interface"""
        st.markdown("### 📁 Step 1: Upload Station Data")
        st.markdown("Upload your daily precipitation data and station metadata files:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daily Precipitation Data:**")
            st.markdown("Format: Date, STATION_001, STATION_002, ...")
            
            precip_file = st.file_uploader(
                "Upload precipitation CSV",
                type=['csv'],
                key="precip_upload",
                help="CSV file with Date (YYYY-MM-DD) and station columns (mm)"
            )
            
            if precip_file is not None:
                try:
                    precip_df = pd.read_csv(precip_file)
                    
                    # Validate Date column exists
                    if 'Date' not in precip_df.columns:
                        st.error("❌ Missing required 'Date' column")
                    else:
                        # Convert date column
                        precip_df['Date'] = pd.to_datetime(precip_df['Date'])
                        
                        # Get station columns (all columns except Date)
                        station_columns = [col for col in precip_df.columns if col != 'Date']
                        
                        if len(station_columns) == 0:
                            st.error("❌ No station data columns found")
                        else:
                            # Show preview
                            st.success(f"✅ Loaded {len(precip_df)} records for {len(station_columns)} station(s)")
                            st.write(f"**Date range:** {precip_df['Date'].min().date()} to {precip_df['Date'].max().date()}")
                            st.write(f"**Stations found:** {', '.join(station_columns)}")
                            st.dataframe(precip_df.head(), use_container_width=True)
                            
                            # Store data
                            st.session_state.td_station_data = precip_df
                            st.session_state.td_station_columns = station_columns
                            st.session_state.td_data_uploaded = True
                            
                            # If metadata is already loaded, enhance it with start/end dates
                            if (st.session_state.td_metadata_uploaded and 
                                st.session_state.td_station_metadata is not None):
                                metadata_df = st.session_state.td_station_metadata.copy()
                                
                                # Add start_date and end_date for each station
                                for idx, row in metadata_df.iterrows():
                                    station_id = str(row['id'])
                                    if station_id in station_columns:
                                        # Find first and last non-null precipitation values
                                        station_precip = precip_df[['Date', station_id]].dropna()
                                        if not station_precip.empty:
                                            metadata_df.loc[idx, 'start_date'] = station_precip['Date'].min()
                                            metadata_df.loc[idx, 'end_date'] = station_precip['Date'].max()
                                        else:
                                            # If no valid data, use full date range
                                            metadata_df.loc[idx, 'start_date'] = precip_df['Date'].min()
                                            metadata_df.loc[idx, 'end_date'] = precip_df['Date'].max()
                                
                                # Update stored metadata
                                st.session_state.td_station_metadata = metadata_df
                                st.info("✨ Metadata enhanced with start/end dates from precipitation data")
                        
                except Exception as e:
                    st.error(f"Error reading precipitation file: {str(e)}")
        
        with col2:
            st.markdown("**Station Metadata:**")
            st.markdown("Format: id, lat, long")
            
            metadata_file = st.file_uploader(
                "Upload metadata CSV",
                type=['csv'],
                key="metadata_upload",
                help="CSV file with station coordinates (id, lat, long)"
            )
            
            if metadata_file is not None:
                try:
                    metadata_df = pd.read_csv(metadata_file)
                    
                    # Validate required columns
                    required_cols = ['id', 'lat', 'long']
                    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                    else:
                        # Convert id to string to match precipitation data column names
                        metadata_df['id'] = metadata_df['id'].astype(str)
                        
                        # If precipitation data is already loaded, add start/end dates from precipitation data
                        if st.session_state.td_data_uploaded and st.session_state.td_station_data is not None:
                            precip_data = st.session_state.td_station_data
                            station_columns = st.session_state.td_station_columns
                            
                            # Add start_date and end_date for each station based on precipitation data
                            metadata_enhanced = []
                            for _, row in metadata_df.iterrows():
                                station_id = str(row['id'])
                                if station_id in station_columns:
                                    # Find first and last non-null precipitation values for this station
                                    station_precip = precip_data[['Date', station_id]].dropna()
                                    if not station_precip.empty:
                                        start_date = station_precip['Date'].min()
                                        end_date = station_precip['Date'].max()
                                    else:
                                        # If no valid data, use full precipitation date range
                                        start_date = precip_data['Date'].min()
                                        end_date = precip_data['Date'].max()
                                    
                                    metadata_enhanced.append({
                                        'id': station_id,
                                        'lat': row['lat'],
                                        'long': row['long'],
                                        'start_date': start_date,
                                        'end_date': end_date
                                    })
                            
                            if metadata_enhanced:
                                metadata_df = pd.DataFrame(metadata_enhanced)
                                st.info("✨ Start and end dates automatically determined from precipitation data")
                        
                        # Show preview
                        st.success(f"✅ Loaded {len(metadata_df)} station(s)")
                        st.dataframe(metadata_df, use_container_width=True)
                        
                        # Store data
                        st.session_state.td_station_metadata = metadata_df
                        st.session_state.td_metadata_uploaded = True
                        
                except Exception as e:
                    st.error(f"Error reading metadata file: {str(e)}")
        
        # Continue button
        if st.session_state.td_data_uploaded and st.session_state.td_metadata_uploaded:
            # Ensure metadata has start_date and end_date columns
            metadata_df = st.session_state.td_station_metadata
            if 'start_date' not in metadata_df.columns or 'end_date' not in metadata_df.columns:
                st.warning("⚠️ Enhancing metadata with date ranges from precipitation data...")
                
                # Enhance metadata with dates from precipitation data
                precip_data = st.session_state.td_station_data
                station_columns = st.session_state.td_station_columns
                
                metadata_enhanced = []
                for _, row in metadata_df.iterrows():
                    station_id = str(row['id'])
                    if station_id in station_columns:
                        # Find first and last non-null precipitation values
                        station_precip = precip_data[['Date', station_id]].dropna()
                        if not station_precip.empty:
                            start_date = station_precip['Date'].min()
                            end_date = station_precip['Date'].max()
                        else:
                            # If no valid data, use full date range
                            start_date = precip_data['Date'].min()
                            end_date = precip_data['Date'].max()
                        
                        metadata_enhanced.append({
                            'id': station_id,
                            'lat': row['lat'],
                            'long': row['long'],
                            'start_date': start_date,
                            'end_date': end_date
                        })
                
                if metadata_enhanced:
                    st.session_state.td_station_metadata = pd.DataFrame(metadata_enhanced)
                    metadata_df = st.session_state.td_station_metadata
                    st.success("✅ Metadata enhanced with automatic date ranges")
            
            # Validate that station IDs match between metadata and precipitation data
            metadata_stations = set(metadata_df['id'].astype(str).tolist())
            precip_stations = set(st.session_state.td_station_columns)
            
            missing_in_precip = metadata_stations - precip_stations
            missing_in_metadata = precip_stations - metadata_stations
            
            if missing_in_precip:
                st.error(f"❌ Station IDs in metadata but not in precipitation data: {missing_in_precip}")
            elif missing_in_metadata:
                st.warning(f"⚠️ Station IDs in precipitation data but not in metadata: {missing_in_metadata}")
                st.info("💡 Only stations with metadata will be analyzed")
            
            # Show matching stations
            matching_stations = metadata_stations & precip_stations
            if matching_stations:
                st.success(f"✅ {len(matching_stations)} station(s) ready for analysis: {', '.join(sorted(matching_stations))}")
                
                if st.button("Continue to Time Range Selection", type="primary", key="continue_timerange"):
                    st.rerun()
            else:
                st.error("❌ No matching stations found between metadata and precipitation data")
    
    def _render_timerange_selection(self):
        """Render time range selection interface"""
        st.markdown("### 📅 Step 2: Select Analysis Time Period")
        
        # Add back button
        if st.button("← Back to Data Upload", key="back_to_upload"):
            st.session_state.td_data_uploaded = False
            st.session_state.td_metadata_uploaded = False
            st.rerun()
        
        # Get date constraints from uploaded data
        precip_data = st.session_state.td_station_data
        metadata = st.session_state.td_station_metadata
        
        # Calculate available date range from precipitation data and metadata
        precip_start = precip_data['Date'].min().date()
        precip_end = precip_data['Date'].max().date()
        
        # Get the date range from metadata (automatically calculated from precipitation data)
        if 'start_date' in metadata.columns and 'end_date' in metadata.columns:
            meta_start = pd.to_datetime(metadata['start_date']).min().date()
            meta_end = pd.to_datetime(metadata['end_date']).max().date()
            
            available_start = max(precip_start, meta_start)
            available_end = min(precip_end, meta_end)
        else:
            # Fallback to precipitation data range only
            available_start = precip_start
            available_end = precip_end
        
        st.info(f"📊 Available data period: **{available_start}** to **{available_end}**")
        st.info(f"📈 Total precipitation records: **{len(precip_data):,}** days across **{len(metadata)}** stations")
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Analysis Start Date:",
                value=available_start,
                min_value=available_start,
                max_value=available_end
            )
        
        with col2:
            end_date = st.date_input(
                "Analysis End Date:",
                value=available_end,
                min_value=start_date,
                max_value=available_end
            )
        
        # Validate and show summary
        if start_date <= end_date:
            num_days = (end_date - start_date).days + 1
            st.success(f"✅ Selected range: **{start_date}** to **{end_date}** ({num_days} days)")
            
            # Store selection
            st.session_state.td_analysis_start_date = start_date
            st.session_state.td_analysis_end_date = end_date
            
            if st.button("Continue to Satellite Data Download", type="primary", key="continue_satellite"):
                st.session_state.td_timerange_selected = True
                st.rerun()
        else:
            st.error("❌ Start date must be before end date")
    
    def _render_satellite_download(self):
        """Render satellite data download interface"""
        st.markdown("### 🛰️ Step 3: Download Satellite Data")
        
        # Add back button
        if st.button("← Back to Time Range", key="back_to_timerange"):
            st.session_state.td_timerange_selected = False
            st.rerun()
        
        st.markdown("**Pre-configured satellite datasets will be downloaded automatically:**")
        
        # Show dataset information
        for dataset_id, config in self.satellite_datasets.items():
            with st.expander(f"📡 {config['name']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Earth Engine ID:** {config['ee_id']}")
                    st.write(f"**Band:** {config['band']}")
                    st.write(f"**Resolution:** {config['temporal_resolution']}")
                with col2:
                    st.write(f"**Units:** {config['units']} ({config['units_type']})")
                    st.write(f"**Coverage:** {config['start_date']} to {config['end_date']}")
                
                st.write(f"**Description:** {config['description']}")
        
        # Download button
        if st.button("🚀 Download Satellite Data", type="primary", key="download_satellite"):
            self._download_satellite_data()
    
    def _download_satellite_data(self):
        """Download satellite data for all stations"""
        metadata = st.session_state.td_station_metadata
        start_date = st.session_state.td_analysis_start_date
        end_date = st.session_state.td_analysis_end_date
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_operations = len(metadata) * len(self.satellite_datasets)
        current_op = 0
        
        satellite_data = {}
        
        try:
            for _, station in metadata.iterrows():
                station_id = station['id']
                lat = station['lat']
                lon = station['long']
                
                status_text.text(f"Processing station {station_id}...")
                
                station_satellite_data = {}
                
                for dataset_id, config in self.satellite_datasets.items():
                    try:
                        # Download data using the temporal disaggregation handler
                        data = self.td_handler.get_satellite_data(
                            config['ee_id'],
                            config['band'],
                            lat, lon,
                            start_date, end_date,
                            config['temporal_resolution']
                        )
                        
                        if not data.empty:
                            station_satellite_data[dataset_id] = data
                            
                    except Exception as e:
                        st.warning(f"⚠️ Error downloading {config['name']} for station {station_id}: {str(e)}")
                    
                    current_op += 1
                    progress_bar.progress(current_op / total_operations)
                
                if station_satellite_data:
                    satellite_data[station_id] = station_satellite_data
            
            # Store results
            st.session_state.td_satellite_data = satellite_data
            st.session_state.td_satellite_data_downloaded = True
            
            status_text.text("✅ Satellite data download completed!")
            progress_bar.empty()
            
            st.success("🎉 Successfully downloaded satellite data for all stations!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error downloading satellite data: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _render_analysis_results(self):
        """Render analysis and results interface"""
        st.markdown("### 📊 Step 4: Analysis Results")
        
        # Add back buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Satellite Download", key="back_to_satellite"):
                st.session_state.td_satellite_data_downloaded = False
                st.rerun()
        with col2:
            if st.button("🔄 Start New Analysis", key="restart_td_analysis"):
                # Reset all session state variables
                keys_to_clear = [key for key in st.session_state.keys() if key.startswith('td_')]
                for key in keys_to_clear:
                    del st.session_state[key]
                st.rerun()
        
        # Run analysis if not already done
        if not st.session_state.td_analysis_complete:
            self._run_analysis()
        
        # Display results
        if st.session_state.td_analysis_complete and st.session_state.td_analysis_results:
            self._display_analysis_results()
        else:
            st.error("❌ Analysis failed or no results available")
    
    def _run_analysis(self):
        """Run the temporal disaggregation analysis"""
        st.markdown("🔄 **Running Analysis...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            station_data = st.session_state.td_station_data
            satellite_data = st.session_state.td_satellite_data
            metadata = st.session_state.td_station_metadata
            station_columns = st.session_state.td_station_columns
            
            analysis_results = {}
            corrected_data = {}
            
            total_stations = len(metadata)
            
            for i, (_, station) in enumerate(metadata.iterrows()):
                station_id = station['id']
                
                status_text.text(f"Analyzing station {station_id}...")
                
                if station_id not in satellite_data:
                    continue
                
                # Check if station_id exists in precipitation data columns
                if station_id not in station_columns:
                    st.warning(f"⚠️ Station {station_id} not found in precipitation data columns")
                    continue
                
                # Get station precipitation data for this specific station
                station_precip = station_data[['Date', station_id]].copy()
                station_precip = station_precip.rename(columns={station_id: 'Prcp'})
                
                # Analyze each satellite dataset
                station_results = {}
                station_corrected = {}
                
                for dataset_id, sat_data in satellite_data[station_id].items():
                    try:
                        # Get dataset configuration for units handling
                        dataset_config = self.satellite_datasets[dataset_id]
                        
                        # Aggregate satellite data to daily with proper units handling
                        daily_sat = self.td_handler.aggregate_to_daily(
                            sat_data, 
                            dataset_config['units_type'],
                            dataset_config['temporal_resolution']
                        )
                        
                        # Merge with station data
                        merged = self._merge_station_satellite(station_precip, daily_sat)
                        
                        if len(merged) > 5:  # Need minimum data for analysis
                            # Calculate statistics
                            stats = self._calculate_statistics(merged)
                            
                            # Calculate optimal score
                            score = self.optimal_selection.calculate_composite_score(stats)
                            
                            # Apply comprehensive bias correction
                            correction_result = self.bias_correction.apply_correction(merged, sat_data)
                            corrected_data_df = correction_result['corrected_data']
                            correction_metadata = correction_result['metadata']
                            
                            station_results[dataset_id] = {
                                'stats': stats,
                                'score': score,
                                'merged_data': merged,
                                'correction_metadata': correction_metadata
                            }
                            
                            station_corrected[dataset_id] = corrected_data_df
                            
                    except Exception as e:
                        st.warning(f"Error analyzing {dataset_id} for station {station_id}: {str(e)}")
                
                if station_results:
                    analysis_results[station_id] = station_results
                    corrected_data[station_id] = station_corrected
                
                progress_bar.progress((i + 1) / total_stations)
            
            # Store results
            st.session_state.td_analysis_results = analysis_results
            st.session_state.td_corrected_data = corrected_data
            st.session_state.td_analysis_complete = True
            
            status_text.text("✅ Analysis completed!")
            progress_bar.empty()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _merge_station_satellite(self, station_data: pd.DataFrame, satellite_data: pd.DataFrame) -> pd.DataFrame:
        """Merge station and satellite data"""
        # Ensure both have date columns
        station_data = station_data.copy()
        satellite_data = satellite_data.copy()
        
        # Rename columns for merging
        station_data = station_data.rename(columns={'Date': 'date', 'Prcp': 'station_prcp'})
        satellite_data = satellite_data.rename(columns={'value': 'satellite_prcp'})
        
        # Merge on date
        merged = pd.merge(station_data, satellite_data, on='date', how='inner')
        
        # Remove NaN values
        merged = merged.dropna(subset=['station_prcp', 'satellite_prcp'])
        
        return merged
    
    def _calculate_statistics(self, merged_data: pd.DataFrame) -> Dict:
        """Calculate statistical metrics"""
        try:
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error
            except ImportError:
                # Fallback implementations if sklearn not available
                def mean_squared_error(y_true, y_pred):
                    return np.mean((y_true - y_pred) ** 2)
                
                def mean_absolute_error(y_true, y_pred):
                    return np.mean(np.abs(y_true - y_pred))
            
            station_vals = merged_data['station_prcp']
            satellite_vals = merged_data['satellite_prcp']
            
            # Basic metrics
            rmse = np.sqrt(mean_squared_error(station_vals, satellite_vals))
            mae = mean_absolute_error(station_vals, satellite_vals)
            correlation = np.corrcoef(station_vals, satellite_vals)[0, 1]
            bias = np.mean(satellite_vals - station_vals)
            
            # KGE calculation
            r = correlation
            alpha = np.std(satellite_vals) / np.std(station_vals) if np.std(station_vals) > 0 else 1.0
            beta = np.mean(satellite_vals) / np.mean(station_vals) if np.mean(station_vals) > 0 else 1.0
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation,
                'bias': bias,
                'kge': kge,
                'n_observations': len(merged_data)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _display_analysis_results(self):
        """Display analysis results"""
        results = st.session_state.td_analysis_results
        metadata = st.session_state.td_station_metadata
        
        st.success("✅ **Analysis Completed Successfully!**")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stations Analyzed", len(results))
        with col2:
            st.metric("Satellite Datasets", len(self.satellite_datasets))
        with col3:
            total_comparisons = sum(len(station_results) for station_results in results.values())
            st.metric("Total Comparisons", total_comparisons)
        
        # Station-by-station results
        st.markdown("---")
        st.markdown("### �️ Spatial Analysis Results")
        
        # Spatial visualization controls
        col1, col2 = st.columns([2, 1])
        with col1:
            metric_options = ['Score', 'RMSE', 'Correlation', 'Bias', 'KGE']
            selected_metric = st.selectbox(
                "Select metric to display spatially:",
                metric_options,
                index=0,
                key="spatial_metric_select"
            )
        with col2:
            show_optimal_only = st.checkbox("Show optimal datasets only", value=True, key="show_optimal_only")
        
        # Create spatial visualization
        spatial_fig = self._create_spatial_plot(results, metadata, selected_metric, show_optimal_only)
        st.plotly_chart(spatial_fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### �📊 Station Analysis Results")
        
        for station_id, station_results in results.items():
            with st.expander(f"📍 Station {station_id}", expanded=False):
                self._display_station_results(station_id, station_results)
        
        # Download section
        st.markdown("---")
        st.markdown("### 📥 Download Corrected Data")
        
        if st.button("📦 Prepare Download Package", type="primary", key="prepare_td_download"):
            self._prepare_download_package()
    
    def _create_spatial_plot(self, results: Dict, metadata: pd.DataFrame, 
                           metric: str, show_optimal_only: bool = True):
        """Create spatial plot of analysis results"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Prepare data for plotting
            plot_data = []
            
            for station_id, station_results in results.items():
                # Get station metadata
                station_meta = metadata[metadata['id'] == station_id]
                if station_meta.empty:
                    continue
                    
                lat = station_meta.iloc[0]['lat']
                lon = station_meta.iloc[0]['long']
                
                if show_optimal_only:
                    # Show only optimal dataset for each station
                    best_dataset = max(station_results.keys(), 
                                     key=lambda x: station_results[x]['score'])
                    dataset_results = {best_dataset: station_results[best_dataset]}
                else:
                    # Show all datasets for each station
                    dataset_results = station_results
                
                for dataset_id, data in dataset_results.items():
                    stats = data['stats']
                    dataset_name = self.satellite_datasets[dataset_id]['name']
                    
                    # Get metric value
                    if metric == 'Score':
                        value = data['score']
                    elif metric == 'RMSE':
                        value = stats.get('rmse', 0)
                    elif metric == 'Correlation':
                        value = stats.get('correlation', 0)
                    elif metric == 'Bias':
                        value = stats.get('bias', 0)
                    elif metric == 'KGE':
                        value = stats.get('kge', 0)
                    else:
                        value = 0
                    
                    plot_data.append({
                        'station_id': station_id,
                        'lat': lat,
                        'lon': lon,
                        'dataset': dataset_name,
                        'metric': metric,
                        'value': value,
                        'is_optimal': dataset_id == max(station_results.keys(), 
                                                      key=lambda x: station_results[x]['score'])
                    })
            
            if not plot_data:
                # Return empty figure if no data
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for spatial plotting",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            df = pd.DataFrame(plot_data)
            
            # Create color mapping based on metric
            if metric in ['Score', 'Correlation', 'KGE']:
                # Higher is better
                color_scale = 'Viridis'
                color_title = f"{metric} (Higher is Better)"
            else:
                # Lower is better (RMSE, Bias)
                color_scale = 'Viridis_r'
                color_title = f"{metric} (Lower is Better)"
            
            # Create scatter mapbox
            if show_optimal_only:
                # Single color for optimal datasets
                fig = px.scatter_mapbox(
                    df, lat="lat", lon="lon", 
                    color="value",
                    size="value" if metric in ['Score', 'Correlation', 'KGE'] else None,
                    hover_data=["station_id", "dataset"],
                    color_continuous_scale=color_scale,
                    title=f"Spatial Distribution of {metric} - Optimal Datasets Only",
                    labels={"value": color_title},
                    mapbox_style="open-street-map",
                    zoom=6
                )
            else:
                # Different shapes/colors for different datasets
                fig = px.scatter_mapbox(
                    df, lat="lat", lon="lon", 
                    color="dataset",
                    size="value" if metric in ['Score', 'Correlation', 'KGE'] else None,
                    hover_data=["station_id", "value"],
                    title=f"Spatial Distribution of {metric} - All Datasets",
                    mapbox_style="open-street-map",
                    zoom=6
                )
            
            # Update layout
            fig.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Set default center if we have data
            if not df.empty:
                center_lat = df['lat'].mean()
                center_lon = df['lon'].mean()
                fig.update_layout(
                    mapbox=dict(
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=6
                    )
                )
            
            return fig
            
        except Exception as e:
            # Fallback figure on error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating spatial plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _display_station_results(self, station_id: str, station_results: Dict):
        """Display results for individual station"""
        # Find optimal dataset
        best_dataset = max(station_results.keys(), 
                          key=lambda x: station_results[x]['score'])
        best_score = station_results[best_dataset]['score']
        
        st.success(f"🏆 **Optimal Dataset:** {self.satellite_datasets[best_dataset]['name']} (Score: {best_score:.3f})")
        
        # Display bias correction details for optimal dataset
        if 'correction_metadata' in station_results[best_dataset]:
            correction_meta = station_results[best_dataset]['correction_metadata']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Scaling Factor (α)", f"{correction_meta.get('scaling_factor', 1.0):.3f}")
            with col2:
                st.metric("Improvement", f"{correction_meta.get('improvement_percent', 0):.1f}%")
            with col3:
                st.metric("Method", correction_meta.get('method_used', 'unknown').replace('_', ' ').title())
        
        # Temporal disaggregation section
        st.markdown("---")
        st.markdown("### 📊 **Time Series Analysis**")
        
        # Time series comparison plot
        st.markdown("#### � **Daily Precipitation Comparison**")
        timeseries_fig = self._create_timeseries_comparison_plot(station_id, station_results, best_dataset, 'Full Period')
        st.plotly_chart(timeseries_fig, use_container_width=True)
        
        # Temporal disaggregation section
        st.markdown("#### �🕒 **Temporal Disaggregation to 30-minute Resolution**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📈 Generate 30-min Data for {station_id}", key=f"disagg_{station_id}"):
                self._perform_temporal_disaggregation(station_id, best_dataset)
        
        with col2:
            disagg_method = st.selectbox(
                "Disaggregation Method:",
                ['auto', 'imerg_guided', 'statistical'],
                index=0,
                key=f"method_{station_id}",
                help="Auto: Uses IMERG when available, statistical otherwise"
            )
        
        # Show 30-minute data if available
        if (hasattr(st.session_state, 'td_disaggregated_data') and 
            station_id in st.session_state.td_disaggregated_data and 
            best_dataset in st.session_state.td_disaggregated_data[station_id]):
            
            st.markdown("#### ⏱️ **30-minute Precipitation Time Series**")
            disagg_fig = self._create_30min_precipitation_plot(station_id, best_dataset)
            st.plotly_chart(disagg_fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 **Statistical Summary**")
        
        # Create summary table
        summary_data = []
        for dataset_id, results in station_results.items():
            stats = results['stats']
            summary_data.append({
                'Dataset': self.satellite_datasets[dataset_id]['name'],
                'Score': f"{results['score']:.3f}",
                'RMSE': f"{stats.get('rmse', 0):.3f}",
                'Correlation': f"{stats.get('correlation', 0):.3f}",
                'Bias': f"{stats.get('bias', 0):.3f}",
                'KGE': f"{stats.get('kge', 0):.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Create visualization
        fig = self._create_comparison_plot(station_results, station_id)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_timeseries_comparison_plot(self, station_id: str, station_results: Dict, best_dataset: str, time_range: str = "Full Period"):
        """Create time series comparison plot showing all datasets and final corrected data"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = go.Figure()
            
            # Get station data
            station_data = st.session_state.td_station_data
            station_precip = station_data[['Date', station_id]].copy()
            station_precip['Date'] = pd.to_datetime(station_precip['Date'])
            
            # Apply time range filtering
            if time_range == "Last 30 Days":
                cutoff_date = station_precip['Date'].max() - pd.Timedelta(days=30)
                station_precip = station_precip[station_precip['Date'] >= cutoff_date]
            elif time_range == "Last 7 Days":
                cutoff_date = station_precip['Date'].max() - pd.Timedelta(days=7)
                station_precip = station_precip[station_precip['Date'] >= cutoff_date]
            elif time_range == "Precipitation Events Only":
                # Show only days with precipitation > 0.1 mm at any dataset
                precip_mask = station_precip[station_id] > 0.1
                station_precip = station_precip[precip_mask]
            
            # Add station data (ground truth)
            fig.add_trace(go.Scatter(
                x=station_precip['Date'],
                y=station_precip[station_id],
                mode='lines+markers',
                name='🌧️ Station (Ground Truth)',
                line=dict(color='black', width=3),
                marker=dict(size=6)
            ))
            
            # Add satellite datasets
            satellite_data = st.session_state.td_satellite_data[station_id]
            colors = {'era5_land': '#1f77b4', 'gpm_imerg': '#ff7f0e', 'gsmap': '#2ca02c'}
            
            for dataset_id, sat_data in satellite_data.items():
                if dataset_id in station_results:
                    # Get daily aggregated data
                    dataset_config = self.satellite_datasets[dataset_id]
                    daily_sat = self.td_handler.aggregate_to_daily(
                        sat_data, 
                        dataset_config['units_type'],
                        dataset_config['temporal_resolution']
                    )
                    
                    if not daily_sat.empty:
                        dataset_name = self.satellite_datasets[dataset_id]['name']
                        color = colors.get(dataset_id, '#d62728')
                        
                        # Determine if this is the optimal dataset
                        line_style = 'solid' if dataset_id == best_dataset else 'dash'
                        line_width = 3 if dataset_id == best_dataset else 2
                        
                        fig.add_trace(go.Scatter(
                            x=daily_sat['date'],
                            y=daily_sat['value'],
                            mode='lines',
                            name=f"🛰️ {dataset_name}" + (" (Optimal)" if dataset_id == best_dataset else ""),
                            line=dict(color=color, width=line_width, dash=line_style)
                        ))
            
            # Add final corrected data for the best dataset
            if hasattr(st.session_state, 'td_corrected_data') and station_id in st.session_state.td_corrected_data:
                if best_dataset in st.session_state.td_corrected_data[station_id]:
                    corrected_data = st.session_state.td_corrected_data[station_id][best_dataset]
                    
                    if not corrected_data.empty and 'value_corrected' in corrected_data.columns:
                        # Aggregate corrected data to daily
                        corrected_daily = corrected_data.copy()
                        corrected_daily['date'] = pd.to_datetime(corrected_daily['datetime']).dt.date
                        corrected_daily = corrected_daily.groupby('date')['value_corrected'].sum().reset_index()
                        corrected_daily['date'] = pd.to_datetime(corrected_daily['date'])
                        
                        fig.add_trace(go.Scatter(
                            x=corrected_daily['date'],
                            y=corrected_daily['value_corrected'],
                            mode='lines',
                            name='✨ Final Corrected (Best)',
                            line=dict(color='red', width=4, dash='dot'),
                            opacity=0.8
                        ))
            
            # Update layout
            fig.update_layout(
                title=f"Daily Precipitation Time Series Comparison - Station {station_id}",
                xaxis_title="Date",
                yaxis_title="Precipitation (mm/day)",
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add rangeslider for better navigation
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating time series plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _create_30min_precipitation_plot(self, station_id: str, dataset_id: str):
        """Create 30-minute precipitation time series plot"""
        try:
            import plotly.graph_objects as go
            
            # Get disaggregated data
            disagg_data = st.session_state.td_disaggregated_data[station_id][dataset_id]
            
            fig = go.Figure()
            
            # Add 30-minute precipitation data
            fig.add_trace(go.Scatter(
                x=disagg_data['datetime'],
                y=disagg_data['value'],
                mode='lines',
                name='30-minute Precipitation',
                line=dict(color='blue', width=2),
                fill='tonexty' if len(disagg_data) > 1 else None,
                fillcolor='rgba(0,100,255,0.2)'
            ))
            
            # Add method information as annotations
            if hasattr(disagg_data, 'attrs') and 'method_stats' in disagg_data.attrs:
                stats = disagg_data.attrs['method_stats']
                imerg_hours = stats.get('imerg_used', 0)
                stat_hours = stats.get('statistical_used', 0)
                total_hours = imerg_hours + stat_hours
                
                if total_hours > 0:
                    imerg_percent = (imerg_hours / total_hours) * 100
                    
                    fig.add_annotation(
                        text=f"Methods: {imerg_percent:.1f}% IMERG-guided, {100-imerg_percent:.1f}% Statistical",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
            
            # Highlight different time periods with different colors based on method
            if 'method' in disagg_data.columns:
                # Color code by disaggregation method
                imerg_data = disagg_data[disagg_data['method'] == 'imerg_guided']
                stat_data = disagg_data[disagg_data['method'] == 'statistical']
                
                if not imerg_data.empty:
                    fig.add_trace(go.Scatter(
                        x=imerg_data['datetime'],
                        y=imerg_data['value'],
                        mode='markers',
                        name='IMERG-guided periods',
                        marker=dict(color='green', size=4, symbol='circle'),
                        showlegend=True
                    ))
                
                if not stat_data.empty:
                    fig.add_trace(go.Scatter(
                        x=stat_data['datetime'],
                        y=stat_data['value'],
                        mode='markers',
                        name='Statistical periods',
                        marker=dict(color='orange', size=4, symbol='square'),
                        showlegend=True
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"30-minute Precipitation Time Series - Station {station_id}<br><sub>Dataset: {self.satellite_datasets[dataset_id]['name']}</sub>",
                xaxis_title="Date and Time",
                yaxis_title="Precipitation (mm/30min)",
                height=450,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add rangeslider for detailed examination
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating 30-minute plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _create_comparison_plot(self, station_results: Dict, station_id: str) -> go.Figure:
        """Create comparison plot for station results"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE', 'Correlation', 'Bias', 'Composite Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        datasets = list(station_results.keys())
        dataset_names = [self.satellite_datasets[d]['name'] for d in datasets]
        
        # RMSE
        rmse_vals = [station_results[d]['stats'].get('rmse', 0) for d in datasets]
        fig.add_trace(go.Bar(x=dataset_names, y=rmse_vals, name='RMSE'), row=1, col=1)
        
        # Correlation
        corr_vals = [station_results[d]['stats'].get('correlation', 0) for d in datasets]
        fig.add_trace(go.Bar(x=dataset_names, y=corr_vals, name='Correlation'), row=1, col=2)
        
        # Bias
        bias_vals = [station_results[d]['stats'].get('bias', 0) for d in datasets]
        fig.add_trace(go.Bar(x=dataset_names, y=bias_vals, name='Bias'), row=2, col=1)
        
        # Composite Score
        score_vals = [station_results[d]['score'] for d in datasets]
        fig.add_trace(go.Bar(x=dataset_names, y=score_vals, name='Composite Score'), row=2, col=2)
        
        fig.update_layout(
            height=600,
            title_text=f"Performance Metrics for Station {station_id}",
            showlegend=False
        )
        
        return fig
    
    def _perform_temporal_disaggregation(self, station_id: str, optimal_dataset_id: str):
        """Perform temporal disaggregation for a specific station"""
        try:
            # Get corrected data
            corrected_data = st.session_state.td_corrected_data[station_id][optimal_dataset_id]
            
            # Get IMERG data if available for pattern guidance
            imerg_data = None
            if 'gpm_imerg' in st.session_state.td_satellite_data[station_id]:
                imerg_data = st.session_state.td_satellite_data[station_id]['gpm_imerg']
            
            # Get disaggregation method selection
            method = st.session_state.get(f"method_{station_id}", 'auto')
            
            with st.spinner("🔄 Performing temporal disaggregation..."):
                # Perform disaggregation
                disaggregated_data = self.td_handler.disaggregate_to_30min(
                    corrected_data, 
                    imerg_data, 
                    method=method
                )
                
                if not disaggregated_data.empty:
                    # Store results
                    if 'td_disaggregated_data' not in st.session_state:
                        st.session_state.td_disaggregated_data = {}
                    if station_id not in st.session_state.td_disaggregated_data:
                        st.session_state.td_disaggregated_data[station_id] = {}
                    
                    st.session_state.td_disaggregated_data[station_id][optimal_dataset_id] = disaggregated_data
                    
                    # Display results
                    st.success("✅ Temporal disaggregation completed! The 30-minute plot will appear above.")
                    
                    # Show statistics in a nice layout
                    if hasattr(disaggregated_data, 'attrs') and 'method_stats' in disaggregated_data.attrs:
                        stats = disaggregated_data.attrs['method_stats']
                        
                        st.markdown("**📈 Disaggregation Statistics:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Hours", stats.get('imerg_used', 0) + stats.get('statistical_used', 0))
                        with col2:
                            st.metric("IMERG Guided", stats.get('imerg_used', 0))
                        with col3:
                            st.metric("Statistical", stats.get('statistical_used', 0))
                        with col4:
                            coverage = disaggregated_data.attrs.get('imerg_coverage', 0) * 100
                            st.metric("IMERG Coverage", f"{coverage:.1f}%")
                    
                    # Show summary statistics
                    st.markdown("**📊 30-minute Data Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(disaggregated_data))
                    with col2:
                        st.metric("Max Intensity", f"{disaggregated_data['value'].max():.2f} mm/30min")
                    with col3:
                        st.metric("Total Volume", f"{disaggregated_data['value'].sum():.2f} mm")
                    with col4:
                        non_zero = (disaggregated_data['value'] > 0).sum()
                        st.metric("Precipitation Periods", non_zero)
                    
                    # Show preview with better formatting
                    st.markdown("**🔍 Data Preview (First 20 records):**")
                    preview_data = disaggregated_data.head(20).copy()
                    if 'datetime' in preview_data.columns:
                        preview_data['datetime'] = preview_data['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(preview_data, use_container_width=True)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = disaggregated_data.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download CSV",
                            data=csv_data,
                            file_name=f"temporal_disaggregated_{station_id}_{optimal_dataset_id}.csv",
                            mime="text/csv",
                            key=f"download_disagg_{station_id}"
                        )
                    
                    with col2:
                        # Create JSON format for modeling software
                        json_data = disaggregated_data.to_json(orient='records', date_format='iso')
                        st.download_button(
                            label=f"📥 Download JSON",
                            data=json_data,
                            file_name=f"temporal_disaggregated_{station_id}_{optimal_dataset_id}.json",
                            mime="application/json",
                            key=f"download_disagg_json_{station_id}"
                        )
                    
                    # Trigger refresh to show the plot
                    st.rerun()
                else:
                    st.error("❌ Temporal disaggregation failed - no data generated")
                    
        except Exception as e:
            st.error(f"❌ Error performing temporal disaggregation: {str(e)}")
    
    def _prepare_download_package(self):
        """Prepare download package with corrected data"""
        try:
            with st.spinner("Preparing download package..."):
                # Create download files
                download_data = {}
                
                # Analysis summary
                summary_data = []
                for station_id, station_results in st.session_state.td_analysis_results.items():
                    for dataset_id, results in station_results.items():
                        summary_data.append({
                            'station_id': station_id,
                            'dataset': self.satellite_datasets[dataset_id]['name'],
                            'score': results['score'],
                            **results['stats']
                        })
                
                download_data['analysis_summary.csv'] = pd.DataFrame(summary_data).to_csv(index=False)
                
                # Corrected data files
                for station_id, corrected_datasets in st.session_state.td_corrected_data.items():
                    for dataset_id, corrected_data in corrected_datasets.items():
                        filename = f"corrected_{station_id}_{dataset_id}.csv"
                        download_data[filename] = corrected_data.to_csv(index=False)
                
                # Create ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, content in download_data.items():
                        zip_file.writestr(filename, content)
                
                zip_buffer.seek(0)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"temporal_disaggregation_results_{timestamp}.zip"
                
                # Offer download
                st.download_button(
                    label="📥 Download Results Package",
                    data=zip_buffer.getvalue(),
                    file_name=filename,
                    mime="application/zip",
                    type="primary"
                )
                
                st.success("✅ Download package ready!")
        
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
