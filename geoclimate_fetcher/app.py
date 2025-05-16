import streamlit as st
import ee
import geemap.foliumap as geemap
import folium
from folium.plugins import Draw
from datetime import datetime
import os
import json
from pathlib import Path
from streamlit_folium import folium_static
import pandas as pd
import time
import re

# Import GeoClimate-Fetcher modules
from geoclimate_fetcher.core import (
    authenticate,
    MetadataCatalog,
    GeometryHandler,
    GEEExporter, 
    ImageCollectionFetcher,
    StaticRasterFetcher
)

# Initialize session state to store variables across reruns
if 'auth_complete' not in st.session_state:
    st.session_state.auth_complete = False
if 'geometry_complete' not in st.session_state:
    st.session_state.geometry_complete = False
if 'dataset_selected' not in st.session_state:
    st.session_state.dataset_selected = False
if 'bands_selected' not in st.session_state:
    st.session_state.bands_selected = False
if 'dates_selected' not in st.session_state:
    st.session_state.dates_selected = False
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'selected_bands' not in st.session_state:
    st.session_state.selected_bands = []
if 'start_date' not in st.session_state:
    st.session_state.start_date = None
if 'end_date' not in st.session_state:
    st.session_state.end_date = None
if 'geometry_handler' not in st.session_state:
    st.session_state.geometry_handler = GeometryHandler()
if 'download_path' not in st.session_state:
    st.session_state.download_path = None
if 'drawn_features' not in st.session_state:
    st.session_state.drawn_features = None

# Function to go back to a previous step
def go_back_to_step(step):
    """Reset the app state to go back to a specific step"""
    if step == "geometry":
        st.session_state.geometry_complete = False
        st.session_state.dataset_selected = False
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False
    elif step == "dataset":
        st.session_state.dataset_selected = False
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False
    elif step == "bands":
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False
    elif step == "dates":
        st.session_state.dates_selected = False
    st.rerun()

# Initialize core objects
metadata_catalog = MetadataCatalog()
exporter = GEEExporter()

# App title
st.title("GeoClimate Fetcher Web App")

# Authentication step
def authenticate_gee(project_id, service_account=None, key_file=None):
    try:
        auth = authenticate(project_id, service_account, key_file)
        if auth.is_initialized():
            st.session_state.auth_complete = True
            return True, "Authentication successful!"
        else:
            return False, "Authentication failed. Please check your credentials."
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"

# Step 1: Authentication
if not st.session_state.auth_complete:
    st.header("Step 1: Google Earth Engine Authentication")
    
    # Load credentials from file if available
    credentials_file = os.path.expanduser("~/.geoclimate-fetcher/credentials.json")
    saved_project_id = ""
    saved_service_account = ""
    saved_key_file = ""
    
    if os.path.exists(credentials_file):
        try:
            with open(credentials_file, 'r') as f:
                saved_credentials = json.load(f)
                saved_project_id = saved_credentials.get("project_id", "")
                saved_service_account = saved_credentials.get("service_account", "")
                saved_key_file = saved_credentials.get("key_file", "")
        except Exception:
            pass
    
    # Project ID input
    project_id = st.text_input("Google Earth Engine Project ID", 
                               value=saved_project_id,
                               help="Enter your Google Cloud project ID that has Earth Engine enabled")
    
    # Advanced options expander
    with st.expander("Advanced Authentication Options"):
        service_account = st.text_input("Service Account Email (optional)", 
                                       value=saved_service_account,
                                       help="For service account authentication")
        key_file = st.text_input("Key File Path (optional)", 
                                value=saved_key_file,
                                help="Path to service account JSON key file")
        remember = st.checkbox("Remember credentials", value=True)
    
    if st.button("Authenticate"):
        if not project_id:
            st.error("Project ID is required")
        else:
            success, message = authenticate_gee(project_id, service_account, key_file)
            
            if success:
                st.success(message)
                
                # Save credentials if remember is checked
                if remember:
                    os.makedirs(os.path.dirname(credentials_file), exist_ok=True)
                    credentials = {"project_id": project_id}
                    if service_account:
                        credentials["service_account"] = service_account
                    if key_file:
                        credentials["key_file"] = key_file
                    
                    try:
                        with open(credentials_file, 'w') as f:
                            json.dump(credentials, f)
                    except Exception as e:
                        st.warning(f"Could not save credentials: {str(e)}")
                
                st.rerun()
            else:
                st.error(message)
else:
    # Step 2: Area of Interest Selection
    if not st.session_state.geometry_complete:
        st.header("Step 2: Select Area of Interest")
        
        # Options for AOI selection
        selection_method = st.radio(
            "Select AOI method:",
            ["Draw on map", "Upload GeoJSON", "Enter coordinates"]
        )
        
        if selection_method == "Draw on map":
            st.info("Use the drawing tools on the map to select your area of interest. Click the rectangle or polygon tool in the top right of the map, draw your area, then click 'Confirm Drawn Area'.")
            
            # Create a folium map
            m = folium.Map(location=[37.0, -95.0], zoom_start=4)
            
            # Add drawing controls without edit options
            draw = Draw(
                export=False,
                position='topright',
                draw_options={
                    'polyline': False,
                    'rectangle': True,
                    'polygon': True,
                    'circle': False,
                    'marker': False,
                    'circlemarker': False
                }
            )
            draw.add_to(m)
            
            # Display the map
            folium_static(m, width=800, height=500)
            
            # Store the drawn features in session state
            if st.button("Confirm Drawn Area"):
                # For now, we'll use a dummy geometry since we can't directly capture the drawn features
                # Create a proper GeoJSON geometry object (not a FeatureCollection)
                default_geojson = {
                    "type": "Polygon",
                    "coordinates": [[
                        [-95, 30], 
                        [-94, 30], 
                        [-94, 31], 
                        [-95, 31], 
                        [-95, 30]
                    ]]
                }
                
                try:
                    # Create an ee.Geometry object directly
                    geometry = ee.Geometry.Polygon([
                        [-95, 30], 
                        [-94, 30], 
                        [-94, 31], 
                        [-95, 31], 
                        [-95, 30]
                    ])
                    st.session_state.geometry_handler._current_geometry = geometry
                    st.session_state.geometry_handler._current_geometry_name = "drawn_aoi"
                    st.session_state.geometry_complete = True
                    st.success("Area of interest selected! (Note: Currently using a default area as drawing capture is limited)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating geometry: {str(e)}")
        
        elif selection_method == "Upload GeoJSON":
            uploaded_file = st.file_uploader("Upload GeoJSON file", type=["geojson", "json"])
            if uploaded_file is not None:
                try:
                    geojson_dict = json.loads(uploaded_file.getvalue())
                    
                    # If it's a FeatureCollection, extract the first feature's geometry
                    if geojson_dict.get("type") == "FeatureCollection" and "features" in geojson_dict and len(geojson_dict["features"]) > 0:
                        geometry_dict = geojson_dict["features"][0]["geometry"]
                    # If it's a Feature, extract its geometry
                    elif geojson_dict.get("type") == "Feature" and "geometry" in geojson_dict:
                        geometry_dict = geojson_dict["geometry"]
                    # Otherwise assume it's already a geometry
                    else:
                        geometry_dict = geojson_dict
                    
                    # Create an ee.Geometry object directly
                    geometry = ee.Geometry(geometry_dict)
                    st.session_state.geometry_handler._current_geometry = geometry
                    st.session_state.geometry_handler._current_geometry_name = "uploaded_aoi"
                    st.session_state.geometry_complete = True
                    st.success("GeoJSON file uploaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing GeoJSON: {str(e)}")
        
        elif selection_method == "Enter coordinates":
            st.write("Enter the coordinates of your bounding box:")
            col1, col2 = st.columns(2)
            with col1:
                min_lon = st.number_input("Min Longitude", value=-95.0)
                min_lat = st.number_input("Min Latitude", value=30.0)
            with col2:
                max_lon = st.number_input("Max Longitude", value=-94.0)
                max_lat = st.number_input("Max Latitude", value=31.0)
            
            # Show a preview map with the bounding box
            preview_map = folium.Map()
            
            # Add the bounding box to the preview map
            bbox = [[min_lat, min_lon], [min_lat, max_lon], [max_lat, max_lon], [max_lat, min_lon]]
            folium.Polygon(locations=bbox, color="red", fill_color="red", fill_opacity=0.1).add_to(preview_map)
            
            # Fit the map to the bounding box
            preview_map.fit_bounds(bbox)
            
            # Display the preview map
            st.write("Preview of selected area:")
            folium_static(preview_map, width=800, height=500)
            
            if st.button("Confirm Coordinates"):
                try:
                    # Create an ee.Geometry object directly using a rectangle
                    geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
                    st.session_state.geometry_handler._current_geometry = geometry
                    st.session_state.geometry_handler._current_geometry_name = "coordinates_aoi"
                    st.session_state.geometry_complete = True
                    st.success("Area of interest created from coordinates!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating geometry: {str(e)}")
    
    # Step 3: Dataset Selection
    elif not st.session_state.dataset_selected:
        st.header("Step 3: Select Dataset")
        
        # Add back button
        if st.button("← Back to Area of Interest"):
            go_back_to_step("geometry")
        
        # Get all datasets from the metadata catalog
        datasets = metadata_catalog.all_datasets.to_dict('records')
        
        # Create a list of dataset names for the selectbox
        dataset_names = [dataset.get("Dataset Name") for dataset in datasets]
        
        # Add a search box for filtering datasets
        search_term = st.text_input("Search datasets:", "")
        
        # Filter datasets based on search term
        if search_term:
            filtered_dataset_names = [name for name in dataset_names if search_term.lower() in name.lower()]
            if not filtered_dataset_names:
                st.warning(f"No datasets found matching '{search_term}'. Showing all datasets.")
                filtered_dataset_names = dataset_names
        else:
            filtered_dataset_names = dataset_names
        
        # Display the selectbox with filtered datasets
        selected_name = st.selectbox("Select a dataset:", filtered_dataset_names)
        
        # Find the selected dataset
        selected_dataset = next((d for d in datasets if d.get("Dataset Name") == selected_name), None)
        
        if selected_dataset:
            # Display dataset information
            st.write(f"**Description:** {selected_dataset.get('Description', 'No description available')}")
            st.write(f"**Earth Engine ID:** {selected_dataset.get('Earth Engine ID', 'N/A')}")
            st.write(f"**Snippet Type:** {selected_dataset.get('Snippet Type', 'N/A')}")
            
            if st.button("Confirm Dataset Selection"):
                st.session_state.current_dataset = selected_dataset
                st.session_state.dataset_selected = True
                st.success(f"Dataset '{selected_name}' selected!")
                st.rerun()
    
    # Step 4: Band Selection
    elif not st.session_state.bands_selected:
        st.header("Step 4: Select Bands")
        
        # Add back button
        if st.button("← Back to Dataset Selection"):
            go_back_to_step("dataset")
        
        # Get available bands for the selected dataset
        dataset = st.session_state.current_dataset
        dataset_name = dataset.get('Dataset Name')
        
        # Try multiple methods to get bands
        bands = []
        
        # Method 1: Try to get bands from the dataset directly if available
        bands_str = dataset.get('Band Names', '')
        if isinstance(bands_str, str) and bands_str:
            bands = [band.strip() for band in bands_str.split(',')]
            st.info(f"Using bands from dataset metadata.")
        
        # Method 2: Try our custom function to get bands from CSV files
        if not bands:
            bands = get_bands_for_dataset(dataset_name)
            if bands:
                st.info(f"Using bands from CSV catalog files.")
        
        # Method 3: Try the catalog method
        if not bands:
            try:
                bands = metadata_catalog.get_bands_for_dataset(dataset.get("Earth Engine ID"))
                if bands:
                    st.info(f"Using bands from metadata catalog.")
            except Exception as e:
                st.warning(f"Error getting bands from metadata catalog: {str(e)}")
        
        # Method 4: If still no bands, provide some default common bands based on the dataset name
        if not bands:
            # Check dataset name to provide appropriate default bands
            if "Daymet" in dataset_name:
                bands = ["tmax", "tmin", "prcp", "srad", "dayl", "swe", "vp"]
            elif "MODIS" in dataset_name and "Temperature" in dataset_name:
                bands = ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"]
            elif "Precipitation" in dataset_name or "Rain" in dataset_name:
                bands = ["precipitation", "error", "gauge_relative_weighting"]
            elif "NDVI" in dataset_name or "Vegetation" in dataset_name:
                bands = ["NDVI", "EVI", "EVI2"]
            else:
                # Generic bands for various Earth Engine datasets
                bands = ["B1", "B2", "B3", "B4", "B5", "B7", "ndvi", "evi", "precipitation", "temperature"]
            
            st.warning(f"No band information found for dataset '{dataset_name}'. Using default bands based on the dataset type.")
        
        # Display band selection
        st.write(f"Available bands for {dataset_name}:")
        
        # Add search for bands if there are many bands
        if len(bands) > 10:
            band_search = st.text_input("Search bands:", "")
            if band_search:
                filtered_bands = [band for band in bands if band_search.lower() in band.lower()]
                if not filtered_bands:
                    st.warning(f"No bands found matching '{band_search}'. Showing all bands.")
                    filtered_bands = bands
            else:
                filtered_bands = bands
        else:
            filtered_bands = bands
        
        # Create columns for band checkboxes to save space
        num_cols = 3  # Number of columns for band selection
        cols = st.columns(num_cols)
        
        selected_bands = []
        for i, band in enumerate(filtered_bands):
            with cols[i % num_cols]:
                if st.checkbox(f"{band}", key=band):
                    selected_bands.append(band)
        
        # Add a "Select All" button if there are many bands
        if len(filtered_bands) > 5:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All"):
                    st.session_state.selected_bands = filtered_bands
                    st.rerun()
            with col2:
                if st.button("Clear All"):
                    st.session_state.selected_bands = []
                    st.rerun()
        
        if st.button("Confirm Band Selection") and selected_bands:
            st.session_state.selected_bands = selected_bands
            st.session_state.bands_selected = True
            st.success(f"Selected bands: {', '.join(selected_bands)}")
            st.rerun()
        elif not selected_bands:
            st.warning("Please select at least one band.")
    
    # Step 5: Time Range Selection (for ImageCollections)
    elif not st.session_state.dates_selected:
        dataset = st.session_state.current_dataset
        snippet_type = dataset.get('Snippet Type')
        
        # Add back button
        if st.button("← Back to Band Selection"):
            go_back_to_step("bands")
        
        if snippet_type == 'ImageCollection':
            st.header("Step 5: Select Time Range")
            
            # Get date range from metadata
            date_range = metadata_catalog.get_date_range(dataset.get("Earth Engine ID"))
            
            # Set default dates if not available in metadata
            if date_range and date_range[0] and date_range[1]:
                min_date_str, max_date_str = date_range
                try:
                    # Try different date formats
                    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y"]
                    min_date = None
                    max_date = None
                    
                    for date_format in date_formats:
                        try:
                            min_date = datetime.strptime(min_date_str, date_format).date()
                            max_date = datetime.strptime(max_date_str, date_format).date()
                            break
                        except (ValueError, TypeError):
                            continue
                    
                    # If all formats failed, use defaults
                    if min_date is None or max_date is None:
                        min_date = datetime(2000, 1, 1).date()
                        max_date = datetime.now().date()
                except Exception:
                    # Use default dates if parsing fails
                    min_date = datetime(2000, 1, 1).date()
                    max_date = datetime.now().date()
            else:
                # Default date range if not available
                st.info("No date range information available for this dataset. Using default date range.")
                min_date = datetime(2000, 1, 1).date()
                max_date = datetime.now().date()
            
            # Date selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
            
            if start_date > end_date:
                st.error("Error: End date must be after start date.")
            else:
                if st.button("Confirm Date Range"):
                    st.session_state.start_date = start_date.strftime("%Y-%m-%d")
                    st.session_state.end_date = end_date.strftime("%Y-%m-%d")
                    st.session_state.dates_selected = True
                    st.success(f"Selected time range: {start_date} to {end_date}")
                    st.rerun()
        else:
            # For Image type, skip date selection
            st.session_state.dates_selected = True
            st.rerun()
    
    # Step 6: Download Configuration
    else:
        st.header("Step 6: Download Configuration")
        
        # Add back buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Time Range"):
                go_back_to_step("dates")
        with col2:
            if st.button("← Back to Band Selection"):
                go_back_to_step("bands")
        
        dataset = st.session_state.current_dataset
        snippet_type = dataset.get('Snippet Type')
        dataset_name = dataset.get('Dataset Name')
        ee_id = dataset.get('Earth Engine ID')
        
        # Display summary of selections
        st.write("### Summary of Selections:")
        st.write(f"**Dataset:** {dataset_name}")
        st.write(f"**Selected Bands:** {', '.join(st.session_state.selected_bands)}")
        
        if snippet_type == 'ImageCollection':
            st.write(f"**Time Range:** {st.session_state.start_date} to {st.session_state.end_date}")
        
        # Download options
        st.write("### Download Options:")
        
        # File format selection
        file_format = st.selectbox("Select file format:", ["GeoTIFF", "NetCDF", "CSV"])
        
        # Scale selection
        scale = st.number_input("Resolution (meters):", min_value=10, max_value=1000, value=30)
        
        # Chunking options for large collections
        with st.expander("Advanced Options"):
            use_chunking = st.checkbox("Enable chunking for large collections", value=True)
            if use_chunking:
                chunk_size = st.slider("Maximum items per chunk", min_value=100, max_value=5000, value=1000)
                st.info("Chunking splits large collections into smaller batches to avoid Earth Engine limitations.")
            else:
                chunk_size = 5000
            
            # Add options to handle large files
            st.write("### Large File Handling:")
            use_drive_for_large = st.checkbox("Use Google Drive for large files (>50MB)", value=True)
            drive_folder = "GeoClimate_Downloads"  # Default value
            if use_drive_for_large:
                drive_folder = st.text_input("Google Drive folder name:", "GeoClimate_Downloads")
                
            clip_to_region = st.checkbox("Clip data to exact region boundary", value=True)
            st.info("Clipping to the exact boundary can reduce file size but may take longer.")
            
            if file_format.lower() == 'geotiff':
                compression = st.checkbox("Use compression for GeoTIFF files", value=True)
                st.info("Compression reduces file size but may slightly increase processing time.")
            else:
                compression = False
        
        # Output directory selection
        st.write("### Output Location:")
        
        # Default directory
        default_dir = "data/downloads"
        
        # Option to use file browser dialog
        use_browser = st.checkbox("Select output directory using file browser", value=False)
        
        if use_browser:
            # Create a temporary button to trigger file dialog
            if 'output_dir' not in st.session_state:
                st.session_state.output_dir = default_dir
            
            if st.button("Browse for folder..."):
                # Use Python's tkinter to create a folder selection dialog
                import tkinter as tk
                from tkinter import filedialog
                
                # Create and hide the Tkinter root window
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                # Show the folder dialog
                folder_path = filedialog.askdirectory(title="Select Output Directory")
                
                # Update the session state if a folder was selected
                if folder_path:
                    st.session_state.output_dir = folder_path
                    st.rerun()
            
            # Display the selected directory
            output_dir = st.session_state.output_dir
            st.success(f"Selected directory: {output_dir}")
        else:
            # Manual input
            output_dir = st.text_input("Output directory:", default_dir)
        
        # Generate filename
        default_filename = f"{dataset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add option for custom filename
        use_custom_filename = st.checkbox("Use custom filename", value=False)
        if use_custom_filename:
            filename = st.text_input("Enter custom filename (without extension):", value=default_filename)
        else:
            filename = default_filename
        
        def download_data(drive_folder=drive_folder, use_drive_for_large=use_drive_for_large):
            """Function to handle the download process"""
            try:                
                with st.spinner("Downloading data... This may take a while."):
                    # Show info about Google Drive usage for large files
                    if use_drive_for_large:
                        st.info(f"Files larger than 50MB will be automatically exported to Google Drive. Drive folder: '{drive_folder}'")
                    else:
                        st.info("Large file handling is disabled. Files exceeding 50MB limit may fail to download.")
                    
                    # Get the geometry from the geometry handler
                    geometry = st.session_state.geometry_handler.current_geometry
                    
                    # Get required parameters
                    ee_id = dataset.get('Earth Engine ID')
                    selected_bands = st.session_state.selected_bands
                    
                    if not geometry:
                        st.error("No geometry selected. Please go back to Step 2 and select an area of interest.")
                        return
                        
                    if not ee_id:
                        st.error("No Earth Engine ID found for the selected dataset.")
                        return
                        
                    if not selected_bands:
                        st.error("No bands selected. Please go back to Step 4 and select at least one band.")
                        return
                    
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Set the appropriate file extension based on format
                    if file_format.lower() == 'geotiff':
                        file_ext = '.tif'
                    elif file_format.lower() == 'netcdf':
                        file_ext = '.nc'
                    elif file_format.lower() == 'csv':
                        file_ext = '.csv'
                    else:
                        file_ext = f'.{file_format.lower()}'
                    
                    # Ensure the filename doesn't already have the extension
                    if filename.lower().endswith(file_ext.lower()):
                        base_filename = filename[:-len(file_ext)]
                    else:
                        base_filename = filename
                    
                    # Create the full output path with correct extension
                    output_path = os.path.join(output_dir, f"{base_filename}{file_ext}")
                    
                    # Log the output path for debugging
                    print(f"Output path: {output_path}")
                    
                    # Initialize the appropriate fetcher based on snippet type
                    if snippet_type == 'ImageCollection':
                        # Get date range
                        start_date = st.session_state.start_date
                        end_date = st.session_state.end_date
                        
                        if not start_date or not end_date:
                            st.error("No date range selected. Please go back to Step 5 and select a date range.")
                            return
                        
                        try:
                            # Create and configure the fetcher
                            fetcher = ImageCollectionFetcher(
                                ee_id=ee_id,
                                bands=selected_bands,
                                geometry=geometry
                            )
                            
                            # Filter by date range
                            fetcher = fetcher.filter_dates(start_date=start_date, end_date=end_date)
                            
                            # Define output paths based on format
                            if file_format.lower() == 'geotiff':
                                # Create a directory for the GeoTIFF files
                                geotiff_dir = os.path.join(output_dir, f"{filename}_geotiffs")
                                os.makedirs(geotiff_dir, exist_ok=True)
                                final_output_path = geotiff_dir
                            else:
                                # For CSV and NetCDF, use the original output path
                                final_output_path = output_path
                            
                            if file_format.lower() == 'csv':
                                # For CSV, get time series average data
                                if use_chunking:
                                    st.info("Using chunked processing for time series data...")
                                    df = fetcher.get_time_series_average_chunked(chunk_months=3)
                                else:
                                    df = fetcher.get_time_series_average()
                                    
                                exporter.export_time_series_to_csv(df, output_path)
                                st.success(f"Exported time series data to {output_path}")
                            elif file_format.lower() == 'netcdf':
                                # For NetCDF, get gridded data
                                ds = fetcher.get_gridded_data(scale=scale)
                                exporter.export_gridded_data_to_netcdf(ds, output_path)
                                st.success(f"Exported NetCDF data to {output_path}")
                            else:  # GeoTIFF - export each image separately
                                # Get the collection
                                collection = fetcher.collection
                                
                                # Initialize collection_size to avoid reference error
                                collection_size = 0
                                
                                # Get collection size
                                collection_size = collection.size().getInfo()
                                st.info(f"Found {collection_size} images in the collection.")
                                
                                # Check if the estimated file size exceeds the direct download limit (50MB)
                                # Rough estimate: 5MB per image for high-resolution data
                                estimated_size_mb = collection_size * 5  # Rough estimate
                                
                                # Process images one by one, trying local download first
                                # Get the list of images
                                image_list = collection.toList(collection.size())
                                
                                # Show a progress bar
                                progress_bar = st.progress(0)
                                
                                # Track successful downloads and drive exports
                                successful_downloads = 0
                                drive_exports = 0
                                
                                # Create a directory for the GeoTIFF files if not already done
                                if not os.path.exists(geotiff_dir):
                                    os.makedirs(geotiff_dir)
                                
                                for i in range(collection_size):
                                    # Update progress
                                    progress_bar.progress((i + 1) / collection_size)
                                    
                                    try:
                                        # Get the image
                                        image = ee.Image(image_list.get(i))
                                        
                                        # Get the date
                                        date_millis = image.get('system:time_start').getInfo()
                                        date_str = datetime.fromtimestamp(date_millis / 1000).strftime('%Y%m%d')
                                        
                                        # If bands are specified, select them
                                        if selected_bands:
                                            image = image.select(selected_bands)
                                        
                                        # Create the output path for this image
                                        image_output_path = os.path.join(geotiff_dir, f"{date_str}.tif")
                                        
                                        # Create a unique filename for potential Google Drive export
                                        image_filename = f"{filename}_{date_str}"
                                        
                                        # Always try local download first
                                        try:
                                            # Try to export the image locally
                                            exporter.export_image_to_local(
                                                image=image,
                                                output_path=image_output_path,
                                                region=geometry,
                                                scale=scale,
                                                use_drive_for_large=use_drive_for_large,
                                                drive_folder=drive_folder
                                            )
                                            
                                            # Check if it was actually exported to Drive
                                            drive_export_marker = Path(str(image_output_path) + '.drive_export.txt')
                                            if drive_export_marker.exists():
                                                # It was too large and exported to Drive instead
                                                st.info(f"Image {i+1}/{collection_size} (date {date_str}): Exported to Google Drive (>50MB)")
                                                # Read the task ID from the marker file
                                                try:
                                                    with open(drive_export_marker, 'r') as f:
                                                        lines = f.readlines()
                                                        for line in lines:
                                                            if line.startswith('Task ID:'):
                                                                task_id = line.strip().split('Task ID:')[1].strip()
                                                                st.info(f"Google Drive export Task ID: {task_id}")
                                                                break
                                                except:
                                                    pass  # If we can't read the file, just continue
                                                drive_exports += 1
                                            else:
                                                # Verify the file actually exists and has content
                                                if image_output_path.exists() and image_output_path.stat().st_size > 0:
                                                    # It was successfully downloaded locally
                                                    st.info(f"Image {i+1}/{collection_size} (date {date_str}): Downloaded successfully")
                                                    successful_downloads += 1
                                                else:
                                                    # File doesn't exist or is empty - automatically export to Drive
                                                    st.warning(f"Failed to download image for date {date_str}. Exporting to Google Drive instead.")
                                                    
                                                    # Export this specific image to Google Drive
                                                    safe_description = image_filename.replace(' ', '_')
                                                    safe_description = re.sub(r'[^a-zA-Z0-9.,;:_\-]', '', safe_description)
                                                    if len(safe_description) > 100:
                                                        safe_description = safe_description[:100]
                                                    
                                                    # Start the export task for this individual image
                                                    task = ee.batch.Export.image.toDrive(
                                                        image=image,
                                                        description=safe_description,
                                                        folder=drive_folder,
                                                        fileNamePrefix=image_filename,
                                                        region=geometry.bounds().getInfo()['coordinates'],
                                                        scale=scale,
                                                        crs='EPSG:4326',
                                                        maxPixels=1e10,
                                                        fileFormat='GeoTIFF'
                                                    )
                                                    
                                                    task.start()
                                                    st.info(f"Started Google Drive export for image {date_str}, Task ID: {task.id}")
                                                    drive_exports += 1
                                        except Exception as e:
                                            # Check if the error is due to size limit
                                            if "exceeds maximum" in str(e) or "size" in str(e).lower() or "too large" in str(e).lower():
                                                st.warning(f"Image for date {date_str} is too large for direct download. Exporting to Google Drive instead.")
                                                
                                                # Export this specific image to Google Drive
                                                safe_description = image_filename.replace(' ', '_')
                                                safe_description = re.sub(r'[^a-zA-Z0-9.,;:_\-]', '', safe_description)
                                                if len(safe_description) > 100:
                                                    safe_description = safe_description[:100]
                                                
                                                # Start the export task for this individual image
                                                task = ee.batch.Export.image.toDrive(
                                                    image=image,
                                                    description=safe_description,
                                                    folder=drive_folder,
                                                    fileNamePrefix=image_filename,
                                                    region=geometry.bounds().getInfo()['coordinates'],
                                                    scale=scale,
                                                    crs='EPSG:4326',
                                                    maxPixels=1e10,
                                                    fileFormat='GeoTIFF'
                                                )
                                                
                                                task.start()
                                                st.info(f"Started Google Drive export for image {date_str}, Task ID: {task.id}")
                                                drive_exports += 1
                                            else:
                                                st.warning(f"Error processing image for date {date_str}: {str(e)}")
                                    except Exception as e:
                                        st.warning(f"Error processing image {i+1}/{collection_size}: {str(e)}")
                                        st.info("Continuing with next image...")
                                        continue
                                
                                # Show summary of exports
                                if drive_exports > 0:
                                    st.info(f"{drive_exports} images were exported to Google Drive folder '{drive_folder}'")
                                    st.info("You can check the status of these exports in the Earth Engine Code Editor.")
                                    st.info("Export status URL: https://code.earthengine.google.com/tasks")
                                
                                if successful_downloads > 0:
                                    # Check if files actually exist in the directory
                                    actual_files = list(Path(geotiff_dir).glob('*.tif'))
                                    if len(actual_files) == successful_downloads:
                                        st.success(f"Successfully downloaded {successful_downloads} out of {collection_size} images to {geotiff_dir}")
                                    else:
                                        st.warning(f"Only {len(actual_files)} files were found in {geotiff_dir}, though {successful_downloads} were reported as downloaded.")
                                        # If no files were found but some were reported as downloaded
                                        if len(actual_files) == 0 and successful_downloads > 0:
                                            # All exports might have failed silently - try Google Drive for all
                                            st.error("No files were found in the output directory. All exports may have failed.")
                                            st.info("Please check the Google Drive exports, or try again with smaller time range.")
                                
                                # Update to show combined message when files were sent to both locations
                                if drive_exports > 0 and successful_downloads > 0:
                                    actual_files = list(Path(geotiff_dir).glob('*.tif'))
                                    if len(actual_files) > 0:
                                        st.warning(f"Note: {drive_exports} files were too large (>50MB) and were sent to Google Drive instead of local download.")
                                        st.success(f"Export complete. {len(actual_files)} files saved locally, {drive_exports} files sent to Google Drive.")
                                    else:
                                        st.warning("All files may have been sent to Google Drive as no local files were found.")
                                        st.info(f"Please check your Google Drive folder '{drive_folder}' for all {collection_size} files.")
                                elif drive_exports == collection_size:
                                    st.warning(f"All {drive_exports} files were too large (>50MB) and were sent to Google Drive instead of local download.")
                                    st.success(f"Export complete. All files were sent to Google Drive folder '{drive_folder}'.")
                                elif successful_downloads == collection_size:
                                    actual_files = list(Path(geotiff_dir).glob('*.tif'))
                                    if len(actual_files) == successful_downloads:
                                        st.success(f"Export complete. All {successful_downloads} files saved locally to {geotiff_dir}")
                                    else:
                                        st.warning(f"Export incomplete. Only {len(actual_files)} out of {successful_downloads} reported downloads were found in {geotiff_dir}")
                                elif drive_exports + successful_downloads < collection_size:
                                    st.warning(f"Export may be incomplete. Only processed {drive_exports + successful_downloads} out of {collection_size} images.")
                                    st.info("Some exports may have failed. Try with a smaller date range or area.")
                        
                        except Exception as e:
                            st.error(f"Error with ImageCollection fetcher: {str(e)}")
                            raise
                    
                    else:  # Image type
                        try:
                            # Create and configure the fetcher
                            fetcher = StaticRasterFetcher(
                                ee_id=ee_id,
                                bands=selected_bands,
                                geometry=geometry
                            )
                            
                            # Get the image
                            image = fetcher.image
                            
                            if file_format.lower() == 'csv':
                                # For CSV, get zonal statistics
                                stats = fetcher.get_zonal_statistics()
                                # Convert to DataFrame
                                rows = []
                                for band, band_stats in stats.items():
                                    row = {'band': band}
                                    row.update(band_stats)
                                    rows.append(row)
                                df = pd.DataFrame(rows)
                                exporter.export_time_series_to_csv(df, output_path)
                                final_output_path = output_path
                                st.success(f"Exported statistics to {output_path}")
                            else:  # GeoTIFF or NetCDF
                                # Try local export first
                                try:
                                    # Export the image
                                    exporter.export_image_to_local(
                                        image=image,
                                        output_path=output_path,
                                        region=geometry,
                                        scale=scale,
                                        use_drive_for_large=use_drive_for_large,
                                        drive_folder=drive_folder
                                    )
                                    
                                    # Check if it was actually exported to Drive
                                    drive_export_marker = Path(str(output_path) + '.drive_export.txt')
                                    if drive_export_marker.exists():
                                        # It was too large and exported to Drive instead
                                        st.info(f"Image was automatically exported to Google Drive folder '{drive_folder}' (>50MB).")
                                        st.warning("Note: The file was too large for direct download and was sent to Google Drive.")
                                        st.success(f"Export complete! Check Google Drive folder '{drive_folder}' for your file.")
                                        final_output_path = f"Google Drive: {drive_folder}/{filename}.tif"
                                        
                                        # Read and display the task ID for tracking
                                        try:
                                            with open(drive_export_marker, 'r') as f:
                                                lines = f.readlines()
                                                for line in lines:
                                                    if line.startswith('Task ID:'):
                                                        task_id = line.strip().split('Task ID:')[1].strip()
                                                        st.info(f"Google Drive export Task ID: {task_id}")
                                                        break
                                        except:
                                            pass  # If we can't read the file, just continue
                                    else:
                                        # It was successfully downloaded locally
                                        final_output_path = output_path
                                        st.success(f"Export complete! Your file has been saved to {output_path}")
                                except Exception as e:
                                    # Check if the error is due to size limit
                                    if "exceeds maximum" in str(e) or "size" in str(e).lower() or "too large" in str(e).lower():
                                        st.warning(f"Image is too large for direct download. Exporting to Google Drive instead.")
                                        
                                        # Export to Google Drive
                                        safe_description = filename.replace(' ', '_')
                                        safe_description = re.sub(r'[^a-zA-Z0-9.,;:_\-]', '', safe_description)
                                        if len(safe_description) > 100:
                                            safe_description = safe_description[:100]
                                        
                                        # Start the export task
                                        task_id = exporter.export_image_to_drive(
                                            image=image,
                                            filename=filename,
                                            folder=drive_folder,
                                            region=geometry,
                                            scale=scale,
                                            wait=False
                                        )
                                        
                                        # Show export task information
                                        st.success(f"Export started! Your file will be available in Google Drive folder '{drive_folder}'")
                                        st.info(f"Task ID: {task_id}")
                                        st.info("You can check the status of your export in the Earth Engine Code Editor Tasks tab.")
                                        st.info("Export URL: https://code.earthengine.google.com/tasks")
                                        
                                        # Store the final output path
                                        final_output_path = f"Google Drive: {drive_folder}/{filename}"
                                    else:
                                        # Re-raise the error if it's not size-related
                                        st.error(f"Error exporting image: {str(e)}")
                                        raise
                        
                        except Exception as e:
                            st.error(f"Error with StaticRaster fetcher: {str(e)}")
                            raise
                    
                    # Store the final output path in session state
                    st.session_state.download_path = final_output_path
                    
                    # Final confirmation of where the file actually went
                    if "Google Drive:" in str(final_output_path):
                        st.warning("⚠️ NOTE: Your file exceeded the 50MB direct download limit and was exported to Google Drive instead.")
                        st.info("Google Drive exports run in the background. Please check your Google Drive folder when the task completes.")
                        st.info("You can check export status at: https://code.earthengine.google.com/tasks")
                    else:
                        # Check if the file actually exists locally before confirming
                        local_file = Path(final_output_path)
                        if local_file.exists() and local_file.stat().st_size > 0:
                            st.info(f"✓ Confirmed: Your file was successfully saved to {final_output_path}")
                        else:
                            # If we got here but the file doesn't exist, check for a drive_export marker
                            marker_file = Path(str(local_file) + '.drive_export.txt')
                            if marker_file.exists():
                                drive_folder = "unknown"
                                try:
                                    with open(marker_file, 'r') as f:
                                        for line in f.readlines():
                                            if line.startswith("Folder:"):
                                                drive_folder = line.strip().split("Folder:")[1].strip()
                                                break
                                except:
                                    pass
                                
                                st.warning("⚠️ NOTE: Your file exceeded the 50MB direct download limit and was exported to Google Drive.")
                                st.info(f"Check Google Drive folder: {drive_folder}")
                                st.info("You can check export status at: https://code.earthengine.google.com/tasks")
                            else:
                                st.warning("⚠️ NOTE: The expected output file wasn't found.")
                                if isinstance(final_output_path, str) and os.path.isdir(final_output_path):
                                    # It's a directory - check if it has any files
                                    files = list(Path(final_output_path).glob('*.tif'))
                                    if files:
                                        st.info(f"Found {len(files)} files in the output directory: {final_output_path}")
                                    else:
                                        st.error("No files were found in the output directory. The download may have failed.")
                                        st.info("Please try exporting to Google Drive instead or select a smaller region.")
                                else:
                                    st.error("The download may have failed. Please check the logs for errors.")
                                    st.info("Try exporting to Google Drive instead or select a smaller region.")
                    
            except Exception as e:
                st.error(f"Error during download: {str(e)}")
                # Print more detailed error information
                import traceback
                st.code(traceback.format_exc(), language="python")
        
        if st.button("Download Data"):
            download_data(drive_folder=drive_folder, use_drive_for_large=use_drive_for_large)

# Add a reset button at the bottom
if st.button("Reset Application"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Function to get bands directly from the dataset name
def get_bands_for_dataset(dataset_name):
    """Get bands for a dataset directly from the CSV files"""
    import os
    import pandas as pd
    from pathlib import Path
    
    # Look in the data directory for CSV files
    data_dir = Path('data')
    if not data_dir.exists():
        return []
    
    # Try to find the dataset in any CSV file
    for csv_file in data_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'Dataset Name' not in df.columns or 'Band Names' not in df.columns:
                continue
                
            # Find the dataset
            dataset_row = df[df['Dataset Name'] == dataset_name]
            if not dataset_row.empty:
                bands_str = dataset_row.iloc[0].get('Band Names', '')
                if isinstance(bands_str, str) and bands_str:
                    return [band.strip() for band in bands_str.split(',')]
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # If not found, try the Datasets.csv file specifically
    datasets_file = data_dir / 'Datasets.csv'
    if datasets_file.exists():
        try:
            df = pd.read_csv(datasets_file)
            dataset_row = df[df['Dataset Name'] == dataset_name]
            if not dataset_row.empty:
                bands_str = dataset_row.iloc[0].get('Band Names', '')
                if isinstance(bands_str, str) and bands_str:
                    return [band.strip() for band in bands_str.split(',')]
        except Exception as e:
            print(f"Error reading Datasets.csv: {e}")
    
    return [] 