import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, date
import pandas as pd

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import MetadataCatalog

class DatasetComponent:
    """Component for dataset, band, and time range selection"""
    
    def __init__(self):
        if 'metadata_catalog' not in st.session_state:
            st.session_state.metadata_catalog = MetadataCatalog()
        self.catalog = st.session_state.metadata_catalog
    
    def get_bands_for_dataset(self, dataset_name):
        """Get bands for a dataset with fallback methods"""
        bands = []
        
        # Method 1: From dataset metadata
        datasets = self.catalog.all_datasets.to_dict('records')
        dataset = next((d for d in datasets if d.get("Dataset Name") == dataset_name), None)
        
        if dataset:
            bands_str = dataset.get('Band Names', '')
            if isinstance(bands_str, str) and bands_str:
                bands = [band.strip() for band in bands_str.split(',')]
        
        # Method 2: Common defaults based on dataset type
        if not bands:
            if "Daymet" in dataset_name:
                bands = ["tmax", "tmin", "prcp", "srad", "dayl", "swe", "vp"]
            elif "MODIS" in dataset_name and "Temperature" in dataset_name:
                bands = ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"]
            elif "CHIRPS" in dataset_name or "Precipitation" in dataset_name:
                bands = ["precipitation", "error"]
            elif "NDVI" in dataset_name or "Vegetation" in dataset_name:
                bands = ["NDVI", "EVI", "EVI2"]
            elif "Landsat" in dataset_name:
                bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "pixel_qa"]
            elif "Sentinel" in dataset_name:
                bands = ["B2", "B3", "B4", "B8", "B11", "B12", "QA60"]
            else:
                bands = ["B1", "B2", "B3", "B4"]  # Generic fallback
        
        return bands
    
    def render(self):
        """Render the dataset selection component"""
        st.markdown("## 📊 Dataset Selection")
        
        # Check if already completed
        if st.session_state.get('dataset_complete', False):
            st.success("✅ Dataset configuration complete!")
            
            # Show current selection summary
            dataset_name = st.session_state.get('selected_dataset_name', 'Unknown')
            selected_bands = st.session_state.get('selected_bands', [])
            start_date = st.session_state.get('start_date')
            end_date = st.session_state.get('end_date')
            
            with st.expander("📋 Current Selection", expanded=True):
                st.write(f"**Dataset:** {dataset_name}")
                st.write(f"**Bands:** {', '.join(selected_bands) if selected_bands else 'None'}")
                if start_date and end_date:
                    st.write(f"**Time Range:** {start_date} to {end_date}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Continue to Download", type="primary"):
                    return True
            with col2:
                if st.button("Change Selection"):
                    st.session_state.dataset_complete = False
                    st.rerun()
            return False
        
        # Step 1: Dataset Selection
        if not st.session_state.get('dataset_selected', False):
            return self.render_dataset_selection()
        
        # Step 2: Band Selection
        elif not st.session_state.get('bands_selected', False):
            return self.render_band_selection()
        
        # Step 3: Time Range Selection (if needed)
        elif not st.session_state.get('dates_selected', False):
            return self.render_time_selection()
        
        # All steps complete
        else:
            st.session_state.dataset_complete = True
            st.rerun()
    
    def render_dataset_selection(self):
        """Render dataset selection interface"""
        st.markdown("### 📊 Choose Dataset")
        
        # Get all datasets
        datasets = self.catalog.all_datasets.to_dict('records')
        
        # Search functionality
        search_term = st.text_input(
            "🔍 Search datasets:",
            placeholder="e.g., CHIRPS, MODIS, precipitation, temperature...",
            help="Search by dataset name, description, or keywords"
        )
        
        # Filter datasets
        filtered_datasets = datasets
        if search_term:
            search_lower = search_term.lower()
            filtered_datasets = [
                d for d in datasets 
                if search_lower in d.get('Dataset Name', '').lower() or
                   search_lower in d.get('Description', '').lower()
            ]
        
        # Category filter
        categories = sorted(list(set([d.get('Category', 'Other') for d in datasets if d.get('Category')])))
        selected_category = st.selectbox(
            "Filter by category:",
            ['All Categories'] + categories,
            help="Filter datasets by category"
        )
        
        if selected_category != 'All Categories':
            filtered_datasets = [d for d in filtered_datasets if d.get('Category') == selected_category]
        
        # Display results
        if not filtered_datasets:
            st.warning(f"No datasets found matching '{search_term}'. Try different keywords.")
            return False
        
        st.write(f"Found {len(filtered_datasets)} dataset(s):")
        
        # Dataset selection with improved display
        for i, dataset in enumerate(filtered_datasets):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    dataset_name = dataset.get('Dataset Name', 'Unknown')
                    description = dataset.get('Description', 'No description available')
                    snippet_type = dataset.get('Snippet Type', 'Unknown')
                    
                    st.markdown(f"**{dataset_name}**")
                    st.caption(f"Type: {snippet_type}")
                    st.write(description[:200] + "..." if len(description) > 200 else description)
                
                with col2:
                    if st.button(f"Select", key=f"select_{i}", type="secondary"):
                        st.session_state.current_dataset = dataset
                        st.session_state.selected_dataset_name = dataset_name
                        st.session_state.dataset_selected = True
                        st.success(f"✅ Selected: {dataset_name}")
                        st.rerun()
                
                st.divider()
        
        return False
    
    def render_band_selection(self):
        """Render band selection interface"""
        st.markdown("### 🎚️ Select Bands")
        
        dataset = st.session_state.current_dataset
        dataset_name = dataset.get('Dataset Name')
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back"):
                st.session_state.dataset_selected = False
                st.rerun()
        
        with col2:
            st.info(f"Selected dataset: **{dataset_name}**")
        
        # Get available bands
        bands = self.get_bands_for_dataset(dataset_name)
        
        if not bands:
            st.warning("No band information available for this dataset.")
            bands = ["Band_1", "Band_2", "Band_3"]  # Fallback
        
        st.write(f"Available bands ({len(bands)}):")
        
        # Band selection interface
        if len(bands) <= 10:
            # Use checkboxes for small number of bands
            selected_bands = []
            
            # Organize in columns
            num_cols = min(3, len(bands))
            cols = st.columns(num_cols)
            
            for i, band in enumerate(bands):
                with cols[i % num_cols]:
                    if st.checkbox(band, key=f"band_{band}"):
                        selected_bands.append(band)
        else:
            # Use multiselect for large number of bands
            selected_bands = st.multiselect(
                "Choose bands:",
                bands,
                help=f"Select from {len(bands)} available bands"
            )
        
        # Quick selection buttons
        if len(bands) > 5:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Select All"):
                    st.session_state.temp_selected_bands = bands
                    st.rerun()
            with col2:
                if st.button("Clear All"):
                    st.session_state.temp_selected_bands = []
                    st.rerun()
            with col3:
                # Common combinations
                if "NDVI" in bands and "EVI" in bands:
                    if st.button("Vegetation Indices"):
                        st.session_state.temp_selected_bands = [b for b in bands if b in ["NDVI", "EVI", "EVI2"]]
                        st.rerun()
        
        # Apply temporary selection
        if hasattr(st.session_state, 'temp_selected_bands'):
            selected_bands = st.session_state.temp_selected_bands
        
        # Continue button
        if selected_bands:
            st.success(f"Selected {len(selected_bands)} band(s): {', '.join(selected_bands)}")
            if st.button("Continue", type="primary"):
                st.session_state.selected_bands = selected_bands
                st.session_state.bands_selected = True
                st.rerun()
        else:
            st.warning("Please select at least one band.")
        
        return False
    
    def render_time_selection(self):
        """Render time range selection interface"""
        dataset = st.session_state.current_dataset
        snippet_type = dataset.get('Snippet Type')
        
        # Skip time selection for static images
        if snippet_type != 'ImageCollection':
            st.session_state.dates_selected = True
            st.rerun()
            return False
        
        st.markdown("### 📅 Select Time Range")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back"):
                st.session_state.bands_selected = False
                st.rerun()
        
        with col2:
            st.info(f"Dataset: **{st.session_state.selected_dataset_name}** (ImageCollection)")
        
        # Get date range from metadata
        try:
            ee_id = dataset.get('Earth Engine ID')
            date_range = self.catalog.get_date_range(ee_id)
            
            if date_range and date_range[0] and date_range[1]:
                min_date_str, max_date_str = date_range
                # Parse dates with multiple formats
                try:
                    min_date = datetime.strptime(min_date_str, "%Y-%m-%d").date()
                    max_date = datetime.strptime(max_date_str, "%Y-%m-%d").date()
                except:
                    min_date = date(2000, 1, 1)
                    max_date = date.today()
            else:
                min_date = date(2000, 1, 1)
                max_date = date.today()
        except:
            min_date = date(2000, 1, 1)
            max_date = date.today()
        
        # Display available date range
        st.info(f"Available data range: {min_date} to {max_date}")
        
        # Date selection
        with st.form("date_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=max(min_date, date(2020, 1, 1)),
                    min_value=min_date,
                    max_value=max_date,
                    help="Start date for data collection"
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=min(max_date, date(2021, 12, 31)),
                    min_value=min_date,
                    max_value=max_date,
                    help="End date for data collection"
                )
            
            # Quick date range buttons
            st.markdown("**Quick selections:**")
            quick_cols = st.columns(4)
            
            with quick_cols[0]:
                if st.form_submit_button("Last Year"):
                    end_date = date.today()
                    start_date = date(end_date.year - 1, end_date.month, end_date.day)
            
            with quick_cols[1]:
                if st.form_submit_button("Last 6 Months"):
                    end_date = date.today()
                    start_date = date(end_date.year, max(1, end_date.month - 6), end_date.day)
            
            with quick_cols[2]:
                if st.form_submit_button("2023 Full Year"):
                    start_date = date(2023, 1, 1)
                    end_date = date(2023, 12, 31)
            
            with quick_cols[3]:
                if st.form_submit_button("2022 Full Year"):
                    start_date = date(2022, 1, 1)
                    end_date = date(2022, 12, 31)
            
            # Validation and submission
            if start_date >= end_date:
                st.error("❌ Start date must be before end date")
                valid = False
            elif start_date < min_date or end_date > max_date:
                st.error(f"❌ Date range must be within {min_date} to {max_date}")
                valid = False
            else:
                valid = True
                days_diff = (end_date - start_date).days
                st.success(f"✅ Selected range: {days_diff} days")
                
                if days_diff > 365:
                    st.warning("⚠️ Large date range may result in long processing time")
            
            submitted = st.form_submit_button("Confirm Date Range", type="primary", disabled=not valid)
            
            if submitted and valid:
                st.session_state.start_date = start_date.strftime("%Y-%m-%d")
                st.session_state.end_date = end_date.strftime("%Y-%m-%d")
                st.session_state.dates_selected = True
                st.success(f"✅ Time range selected: {start_date} to {end_date}")
                st.rerun()
        
        return False 