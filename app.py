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

# Configure Streamlit page
st.set_page_config(
    page_title="GeoClimate Fetcher",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo/geoclimate-fetcher',
        'Report a bug': "https://github.com/your-repo/geoclimate-fetcher/issues",
        'About': "# GeoClimate Fetcher\nA powerful web app for downloading Earth Engine climate and geospatial data!"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Step headers */
    .step-header {
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-complete {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-current {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-pending {
        color: #6c757d;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress indicator */
    .progress-steps {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .step-item {
        text-align: center;
        flex: 1;
        padding: 0.5rem;
    }
    
    .step-number {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .step-complete .step-number {
        background: #28a745;
        color: white;
    }
    
    .step-current .step-number {
        background: #ffc107;
        color: black;
    }
    
    .step-pending .step-number {
        background: #e9ecef;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

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

# App title and header
st.markdown('<h1 class="main-title">🌍 GeoClimate Fetcher</h1>', unsafe_allow_html=True)
st.markdown("### Your gateway to Earth Engine climate and geospatial data")

# Progress indicator
def show_progress_indicator():
    """Display a visual progress indicator showing current step"""
    steps = [
        ("🔐", "Auth", st.session_state.auth_complete),
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

# Show progress if any step is started
if any([st.session_state.auth_complete, st.session_state.geometry_complete, 
        st.session_state.dataset_selected, st.session_state.bands_selected, 
        st.session_state.dates_selected]):
    st.markdown('<div class="progress-steps">', unsafe_allow_html=True)
    show_progress_indicator()
    st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="step-header"><h2>🔐 Step 1: Google Earth Engine Authentication</h2></div>', unsafe_allow_html=True)
    
    # Authentication status and info
    col_info, col_status = st.columns([3, 1])
    
    with col_info:
        st.info("🔑 **Required:** You need a Google Earth Engine account to use this app. "
                "[Sign up here](https://earthengine.google.com/signup/) if you don't have one.")
    
    with col_status:
        if 'auth_attempts' not in st.session_state:
            st.session_state.auth_attempts = 0
        
        if st.session_state.auth_attempts > 0:
            st.warning(f"⚠️ {st.session_state.auth_attempts} failed attempts")
    
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
            st.success("💾 Found saved credentials!")
        except Exception:
            pass
    
    # Create authentication form
    with st.form("auth_form"):
        st.write("#### Authentication Details")
        
        # Project ID input
        project_id = st.text_input(
            "🏗️ Google Earth Engine Project ID *", 
            value=saved_project_id,
            help="Enter your Google Cloud project ID that has Earth Engine enabled",
            placeholder="my-gee-project-123"
        )
        
        # Advanced options expander
        with st.expander("🔧 Advanced Authentication Options"):
            service_account = st.text_input(
                "📧 Service Account Email (optional)", 
                value=saved_service_account,
                help="For service account authentication",
                placeholder="service-account@my-project.iam.gserviceaccount.com"
            )
            key_file = st.text_input(
                "🔑 Key File Path (optional)", 
                value=saved_key_file,
                help="Path to service account JSON key file",
                placeholder="/path/to/service-account-key.json"
            )
            remember = st.checkbox("💾 Remember credentials", value=True)
            st.caption("Credentials will be saved locally for future use")
        
        # Submit button
        col_auth, col_help = st.columns([2, 1])
        
        with col_auth:
            auth_submitted = st.form_submit_button("🚀 Authenticate", type="primary", use_container_width=True)
        
        with col_help:
            if st.form_submit_button("❓ Need Help?", use_container_width=True):
                st.info("""
                **Common Issues:**
                - Make sure your project has Earth Engine enabled
                - Check that you have the correct project ID
                - Ensure you're signed into the right Google account
                
                **Getting Started:**
                1. Go to [Google Earth Engine](https://earthengine.google.com/)
                2. Sign up with your Google account
                3. Create a new project or use existing one
                4. Copy the project ID and paste it above
                """)
    
    # Handle authentication
    if auth_submitted:
        if not project_id:
            st.error("❌ Project ID is required")
            st.session_state.auth_attempts += 1
        else:
            with st.spinner("🔄 Authenticating with Google Earth Engine..."):
                success, message = authenticate_gee(project_id, service_account, key_file)
                
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.auth_attempts = 0  # Reset attempts on success
                    
                    # Save credentials if remember is checked
                    if remember:
                        try:
                            os.makedirs(os.path.dirname(credentials_file), exist_ok=True)
                            credentials = {"project_id": project_id}
                            if service_account:
                                credentials["service_account"] = service_account
                            if key_file:
                                credentials["key_file"] = key_file
                            
                            with open(credentials_file, 'w') as f:
                                json.dump(credentials, f, indent=2)
                            st.info("💾 Credentials saved successfully!")
                        except Exception as e:
                            st.warning(f"⚠️ Could not save credentials: {str(e)}")
                    
                    time.sleep(1)  # Brief pause to show success message
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    st.session_state.auth_attempts += 1
                    
                    # Provide helpful suggestions
                    if st.session_state.auth_attempts >= 3:
                        st.warning("🔍 **Multiple failed attempts detected**")
                        st.info("""
                        **Try these solutions:**
                        - Verify your project ID is correct
                        - Check if Earth Engine is enabled for your project
                        - Make sure you're using the right Google account
                        - Try using a service account for automated access
                        """)

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
        st.markdown('<div class="step-header"><h2>📊 Step 3: Select Dataset</h2></div>', unsafe_allow_html=True)
        
        # Add back button
        if st.button("← Back to Area of Interest"):
            go_back_to_step("geometry")
        
        # Get all datasets from the metadata catalog
        datasets = metadata_catalog.all_datasets.to_dict('records')
        
        # Create a list of dataset names for the selectbox
        dataset_names = [dataset.get("Dataset Name") for dataset in datasets]
        
        # Search and filter interface
        col_search, col_filter = st.columns([2, 1])
        
        with col_search:
            search_term = st.text_input("🔍 Search datasets:", "", placeholder="Enter keywords (e.g., temperature, MODIS, precipitation)")
        
        with col_filter:
            # Get unique providers for filtering
            providers = sorted(list(set([d.get('Provider', 'Unknown') for d in datasets if d.get('Provider')])))
            selected_provider = st.selectbox("Filter by Provider:", ["All Providers"] + providers)
        
        # Filter datasets based on search term and provider
        filtered_datasets = datasets
        
        # Apply provider filter
        if selected_provider != "All Providers":
            filtered_datasets = [d for d in filtered_datasets if d.get('Provider') == selected_provider]
        
        # Apply search filter
        if search_term:
            search_lower = search_term.lower()
            filtered_datasets = [
                d for d in filtered_datasets 
                if (search_lower in d.get("Dataset Name", "").lower() or 
                    search_lower in d.get("Description", "").lower() or
                    search_lower in d.get("Provider", "").lower())
            ]
            
            if not filtered_datasets:
                st.warning(f"No datasets found matching '{search_term}'. Showing all datasets.")
                filtered_datasets = datasets
        
        # Get filtered dataset names
        filtered_dataset_names = [dataset.get("Dataset Name") for dataset in filtered_datasets]
        
        # Show results count
        st.info(f"📊 Showing {len(filtered_dataset_names)} of {len(dataset_names)} datasets")
        
        # Dataset selection
        if filtered_dataset_names:
            selected_name = st.selectbox("Select a dataset:", filtered_dataset_names, key="dataset_selector")
            
            # Find the selected dataset
            selected_dataset = next((d for d in filtered_datasets if d.get("Dataset Name") == selected_name), None)
            
            if selected_dataset:
                # Display comprehensive dataset information
                st.write("### 📋 Dataset Information")
                
                # Create tabs for organized information display
                tab_overview, tab_technical, tab_temporal = st.tabs(["📖 Overview", "🔧 Technical", "📅 Temporal"])
                
                with tab_overview:
                    # Basic information
                    col_info1, col_info2 = st.columns([2, 1])
                    
                    with col_info1:
                        st.write(f"**📊 Dataset:** {selected_dataset.get('Dataset Name', 'N/A')}")
                        st.write(f"**🏢 Provider:** {selected_dataset.get('Provider', 'N/A')}")
                        st.write(f"**🔗 Earth Engine ID:** `{selected_dataset.get('Earth Engine ID', 'N/A')}`")
                        st.write(f"**📦 Type:** {selected_dataset.get('Snippet Type', 'N/A')}")
                    
                    with col_info2:
                        # Show some key metrics
                        pixel_size = selected_dataset.get('Pixel Size (m)', 'N/A')
                        if pixel_size and str(pixel_size).replace('.', '').isdigit():
                            try:
                                pixel_size_float = float(pixel_size)
                                if pixel_size_float < 1000:
                                    st.metric("📏 Resolution", f"{pixel_size_float} m")
                                else:
                                    st.metric("📏 Resolution", f"{pixel_size_float/1000:.1f} km")
                            except:
                                st.metric("📏 Resolution", str(pixel_size))
                        else:
                            st.metric("📏 Resolution", "N/A")
                        
                        temporal_res = selected_dataset.get('Temporal Resolution', 'N/A')
                        st.metric("⏱️ Temporal Res.", temporal_res)
                    
                    # Description
                    description = selected_dataset.get('Description', 'No description available')
                    st.write("**📝 Description:**")
                    st.info(description)
                
                with tab_technical:
                    # Technical details
                    st.write("**🎛️ Available Bands:**")
                    bands_str = selected_dataset.get('Band Names', 'Not specified')
                    if bands_str and bands_str != 'Not specified':
                        # Parse and display bands nicely
                        bands = [band.strip() for band in bands_str.split(',')]
                        if len(bands) <= 10:
                            # Show all bands if not too many
                            for i, band in enumerate(bands, 1):
                                st.write(f"  {i}. `{band}`")
                        else:
                            # Show first few and count
                            for i, band in enumerate(bands[:5], 1):
                                st.write(f"  {i}. `{band}`")
                            st.write(f"  ... and {len(bands) - 5} more bands")
                    else:
                        st.write("  Band information not available in metadata")
                    
                    # Band units if available
                    band_units = selected_dataset.get('Band Units', '')
                    if band_units:
                        st.write("**📐 Band Units:**")
                        st.code(band_units)
                    
                    # Technical specifications
                    st.write("**🔧 Technical Specifications:**")
                    tech_col1, tech_col2 = st.columns(2)
                    
                    with tech_col1:
                        st.write(f"• **Pixel Size:** {selected_dataset.get('Pixel Size (m)', 'N/A')} meters")
                        st.write(f"• **Data Type:** {selected_dataset.get('Snippet Type', 'N/A')}")
                    
                    with tech_col2:
                        st.write(f"• **Provider:** {selected_dataset.get('Provider', 'N/A')}")
                        st.write(f"• **Temporal Resolution:** {selected_dataset.get('Temporal Resolution', 'N/A')}")
                
                with tab_temporal:
                    # Temporal information
                    start_date = selected_dataset.get('Start Date', 'N/A')
                    end_date = selected_dataset.get('End Date', 'N/A')
                    temporal_res = selected_dataset.get('Temporal Resolution', 'N/A')
                    
                    st.write("**📅 Temporal Coverage:**")
                    
                    temp_col1, temp_col2, temp_col3 = st.columns(3)
                    
                    with temp_col1:
                        st.metric("📅 Start Date", start_date)
                    
                    with temp_col2:
                        st.metric("📅 End Date", end_date)
                    
                    with temp_col3:
                        st.metric("⏱️ Resolution", temporal_res)
                    
                    # Calculate data span if possible
                    if start_date != 'N/A' and end_date != 'N/A':
                        try:
                            # Try to parse dates and calculate span
                            from datetime import datetime
                            
                            # Try common date formats
                            date_formats = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%Y"]
                            
                            start_parsed = None
                            end_parsed = None
                            
                            for fmt in date_formats:
                                try:
                                    start_parsed = datetime.strptime(start_date.strip(), fmt)
                                    break
                                except:
                                    continue
                            
                            for fmt in date_formats:
                                try:
                                    end_parsed = datetime.strptime(end_date.strip(), fmt)
                                    break
                                except:
                                    continue
                            
                            if start_parsed and end_parsed:
                                span_days = (end_parsed - start_parsed).days
                                span_years = span_days / 365.25
                                
                                st.success(f"📊 **Data Span:** {span_years:.1f} years ({span_days:,} days)")
                                
                                # Estimate data volume based on temporal resolution
                                if 'daily' in temporal_res.lower():
                                    est_images = span_days
                                elif 'monthly' in temporal_res.lower():
                                    est_images = int(span_years * 12)
                                elif 'yearly' in temporal_res.lower() or 'annual' in temporal_res.lower():
                                    est_images = int(span_years)
                                else:
                                    est_images = "Unknown"
                                
                                if isinstance(est_images, int):
                                    st.info(f"📈 **Estimated Images:** ~{est_images:,}")
                        except:
                            st.info("Could not calculate data span from available dates")
                    
                    # Show temporal resolution details
                    if temporal_res != 'N/A':
                        st.write("**⏱️ Temporal Resolution Details:**")
                        if 'daily' in temporal_res.lower():
                            st.info("🗓️ **Daily data** - New image every day")
                        elif 'monthly' in temporal_res.lower():
                            st.info("📅 **Monthly data** - New image every month")
                        elif 'yearly' in temporal_res.lower() or 'annual' in temporal_res.lower():
                            st.info("📆 **Yearly data** - New image every year")
                        elif 'hourly' in temporal_res.lower():
                            st.info("🕐 **Hourly data** - New image every hour")
                        else:
                            st.info(f"⏱️ **Custom resolution:** {temporal_res}")
                
                # Confirmation button
                st.write("---")
                if st.button("✅ Confirm Dataset Selection", type="primary", use_container_width=True):
                    st.session_state.current_dataset = selected_dataset
                    st.session_state.dataset_selected = True
                    st.success(f"🎉 Dataset '{selected_name}' selected!")
                    time.sleep(1)  # Brief pause to show success
                    st.rerun()
        else:
            st.error("❌ No datasets match your search criteria. Please try different keywords.")
    
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
        dataset_name = dataset.get('Dataset Name')
        
        # Add back button
        if st.button("← Back to Band Selection"):
            go_back_to_step("bands")
        
        if snippet_type == 'ImageCollection':
            st.markdown('<div class="step-header"><h2>📅 Step 5: Select Time Range</h2></div>', unsafe_allow_html=True)
            
            # Get date range from CSV metadata using dataset name
            date_range = metadata_catalog.get_date_range(dataset_name)
            
            # Display dataset temporal information
            col_info, col_temporal = st.columns([2, 1])
            
            with col_info:
                st.info(f"📊 **Dataset:** {dataset_name}")
                temporal_res = dataset.get('Temporal Resolution', 'Unknown')
                st.info(f"⏱️ **Temporal Resolution:** {temporal_res}")
            
            with col_temporal:
                # Show data availability period
                if date_range and date_range[0] and date_range[1]:
                    start_str, end_str = date_range
                    st.success(f"📅 **Data Available:**\n{start_str} to {end_str}")
                else:
                    st.warning("⚠️ Date range not specified in metadata")
            
            # Parse dates from CSV metadata
            if date_range and date_range[0] and date_range[1]:
                min_date_str, max_date_str = date_range
                try:
                    # Try different date formats commonly used in CSV
                    date_formats = [
                        "%m/%d/%Y",    # 1/1/2015 (most common in CSV)
                        "%Y-%m-%d",    # 2015-01-01
                        "%Y/%m/%d",    # 2015/1/1
                        "%d/%m/%Y",    # 1/1/2015 (European)
                        "%Y",          # Just year: 2015
                        "%m/%d/%y",    # 1/1/15
                        "%Y-%m",       # 2015-01
                        "%m/%Y"        # 1/2015
                    ]
                    
                    min_date = None
                    max_date = None
                    
                    # Try to parse start date
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.strptime(min_date_str.strip(), date_format)
                            min_date = parsed_date.date()
                            break
                        except (ValueError, TypeError):
                            continue
                    
                    # Try to parse end date
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.strptime(max_date_str.strip(), date_format)
                            max_date = parsed_date.date()
                            break
                        except (ValueError, TypeError):
                            continue
                    
                    # If parsing failed, use reasonable defaults
                    if min_date is None:
                        st.warning(f"⚠️ Could not parse start date '{min_date_str}', using default")
                        min_date = datetime(2000, 1, 1).date()
                    
                    if max_date is None:
                        st.warning(f"⚠️ Could not parse end date '{max_date_str}', using current date")
                        max_date = datetime.now().date()
                        
                    # Show parsed dates
                    st.success(f"✅ Parsed date range: {min_date} to {max_date}")
                    
                except Exception as e:
                    st.error(f"❌ Error parsing dates: {str(e)}")
                    # Use default dates if parsing fails
                    min_date = datetime(2000, 1, 1).date()
                    max_date = datetime.now().date()
                    st.info(f"Using default date range: {min_date} to {max_date}")
            else:
                # No date range in metadata - use defaults
                st.warning("⚠️ No date range information available in dataset metadata")
                min_date = datetime(2000, 1, 1).date()
                max_date = datetime.now().date()
                st.info(f"Using default date range: {min_date} to {max_date}")
            
            # Date selection interface
            st.write("### 📅 Select Your Time Range")
            
            # Show some helpful presets
            col_preset, col_custom = st.columns([1, 2])
            
            with col_preset:
                st.write("**Quick Presets:**")
                
                # Calculate some useful presets
                today = datetime.now().date()
                
                preset_options = {
                    "Last Year": (datetime(today.year - 1, 1, 1).date(), datetime(today.year - 1, 12, 31).date()),
                    "Last 2 Years": (datetime(today.year - 2, 1, 1).date(), today),
                    "Last 5 Years": (datetime(today.year - 5, 1, 1).date(), today),
                    "Full Dataset": (min_date, max_date),
                    "Recent Data": (max(min_date, datetime(today.year - 1, today.month, 1).date()), max_date)
                }
                
                # Filter presets to only show valid ones
                valid_presets = {}
                for name, (start, end) in preset_options.items():
                    if start >= min_date and end <= max_date and start <= end:
                        valid_presets[name] = (start, end)
                
                selected_preset = st.selectbox("Choose preset:", ["Custom"] + list(valid_presets.keys()))
                
                if selected_preset != "Custom":
                    preset_start, preset_end = valid_presets[selected_preset]
                    st.info(f"📅 {selected_preset}: {preset_start} to {preset_end}")
            
            with col_custom:
                st.write("**Custom Date Selection:**")
                
                # Set initial values based on preset or defaults
                if selected_preset != "Custom":
                    initial_start, initial_end = valid_presets[selected_preset]
                else:
                    # Default to last year of available data
                    initial_end = max_date
                    initial_start = max(min_date, datetime(max_date.year - 1, max_date.month, max_date.day).date())
                
                # Date input widgets
                col_start, col_end = st.columns(2)
                
                with col_start:
                    start_date = st.date_input(
                        "📅 Start date", 
                        value=initial_start,
                        min_value=min_date, 
                        max_value=max_date,
                        help=f"Available from {min_date}"
                    )
                
                with col_end:
                    end_date = st.date_input(
                        "📅 End date", 
                        value=initial_end,
                        min_value=min_date, 
                        max_value=max_date,
                        help=f"Available until {max_date}"
                    )
            
            # Validation and summary
            if start_date > end_date:
                st.error("❌ Error: End date must be after start date.")
            elif start_date < min_date or end_date > max_date:
                st.error(f"❌ Error: Selected dates must be within available range ({min_date} to {max_date})")
            else:
                # Calculate some statistics
                date_diff = (end_date - start_date).days
                
                # Show selection summary
                st.write("### 📊 Selection Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("📅 Start Date", start_date.strftime("%Y-%m-%d"))
                
                with summary_col2:
                    st.metric("📅 End Date", end_date.strftime("%Y-%m-%d"))
                
                with summary_col3:
                    st.metric("📊 Duration", f"{date_diff} days")
                
                # Estimate data volume
                temporal_res = dataset.get('Temporal Resolution', '').lower()
                if 'daily' in temporal_res:
                    estimated_images = date_diff
                elif 'monthly' in temporal_res:
                    estimated_images = max(1, date_diff // 30)
                elif 'yearly' in temporal_res or 'annual' in temporal_res:
                    estimated_images = max(1, date_diff // 365)
                elif '16' in temporal_res and 'day' in temporal_res:
                    estimated_images = max(1, date_diff // 16)
                elif '8' in temporal_res and 'day' in temporal_res:
                    estimated_images = max(1, date_diff // 8)
                else:
                    estimated_images = "Unknown"
                
                if isinstance(estimated_images, int):
                    st.info(f"📈 Estimated number of images: ~{estimated_images}")
                    
                    if estimated_images > 1000:
                        st.warning("⚠️ Large number of images detected. Consider using chunking or a shorter time range for better performance.")
                    elif estimated_images > 100:
                        st.info("💡 Tip: Enable chunking in advanced options for large collections.")
                
                # Confirm button
                if st.button("✅ Confirm Date Range", type="primary", use_container_width=True):
                    st.session_state.start_date = start_date.strftime("%Y-%m-%d")
                    st.session_state.end_date = end_date.strftime("%Y-%m-%d")
                    st.session_state.dates_selected = True
                    st.success(f"🎉 Selected time range: {start_date} to {end_date}")
                    time.sleep(1)  # Brief pause to show success
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
        with st.expander("View Selected Parameters", expanded=False):
            st.write(f"**Dataset:** {dataset_name}")
            st.write(f"**Earth Engine ID:** {ee_id}")
            st.write(f"**Snippet Type:** {snippet_type}")
            st.write(f"**Selected Bands:** {', '.join(st.session_state.selected_bands)}")
            
            if snippet_type == 'ImageCollection':
                st.write(f"**Time Range:** {st.session_state.start_date} to {st.session_state.end_date}")
        
        # Download options in a more organized layout
        st.write("### Download Configuration:")
        
        # Create two columns for better layout
        config_col1, config_col2 = st.columns([1, 1])
        
        with config_col1:
            st.write("#### File Format & Quality")
            # File format selection
            file_format = st.selectbox("Select file format:", ["GeoTIFF", "NetCDF", "CSV"])
            
            # Scale selection with CSV metadata integration
            st.write("**Resolution (Scale in meters):**")
            
            # Get default pixel size from CSV metadata
            default_pixel_size = dataset.get('Pixel Size (m)', None)
            
            # Display dataset's native resolution
            if default_pixel_size and str(default_pixel_size).replace('.', '').isdigit():
                try:
                    native_resolution = float(default_pixel_size)
                    st.info(f"📏 **Dataset's Native Resolution:** {native_resolution} meters")
                    
                    # Determine if it's a reasonable default
                    if native_resolution <= 100000:  # Less than 100km
                        recommended_scale = native_resolution
                        st.success(f"💡 **Recommended:** Use native resolution ({native_resolution}m) for best quality")
                    else:
                        recommended_scale = 1000  # Default to 1km for very coarse datasets
                        st.warning(f"⚠️ Native resolution is very coarse ({native_resolution}m). Consider using {recommended_scale}m for faster processing.")
                except (ValueError, TypeError):
                    native_resolution = None
                    recommended_scale = 30  # Default fallback
                    st.warning("⚠️ Could not parse native resolution from metadata")
            else:
                native_resolution = None
                recommended_scale = 30  # Default fallback
                st.info("ℹ️ No native resolution specified in metadata")
            
            scale_option = st.radio(
                "Choose resolution option:",
                ["Use recommended", "Common resolutions", "Custom resolution"],
                help="Recommended uses the dataset's native resolution for optimal quality"
            )
            
            if scale_option == "Use recommended":
                scale = recommended_scale
                st.success(f"✅ Using recommended resolution: {scale} meters")
                
                # Show some context about the resolution
                if scale <= 1:
                    st.info("🔍 **Very High Resolution** - Excellent detail, large file sizes")
                elif scale <= 10:
                    st.info("🔍 **High Resolution** - Great detail, moderate file sizes")
                elif scale <= 100:
                    st.info("🔍 **Medium Resolution** - Good balance of detail and file size")
                elif scale <= 1000:
                    st.info("🔍 **Moderate Resolution** - Regional analysis, smaller files")
                else:
                    st.info("🔍 **Coarse Resolution** - Global/continental analysis, small files")
                    
            elif scale_option == "Common resolutions":
                # Create a more intelligent common scales list
                common_scales = {
                    "1m (Very High Resolution)": 1,
                    "10m (Sentinel-2)": 10,
                    "30m (Landsat)": 30,
                    "100m (High Resolution)": 100,
                    "250m (MODIS Terra/Aqua)": 250,
                    "500m (MODIS)": 500,
                    "1km (1000m)": 1000,
                    "5km (5000m)": 5000,
                    "10km (10000m)": 10000,
                    "25km (25000m)": 25000,
                    "55km (55000m)": 55000,
                    "100km (100000m)": 100000
                }
                
                # Add the native resolution if it's not already in the list
                if native_resolution and native_resolution not in common_scales.values():
                    # Find appropriate description
                    if native_resolution < 1000:
                        desc = f"{native_resolution}m (Native Resolution)"
                    else:
                        km_res = native_resolution / 1000
                        desc = f"{km_res}km ({int(native_resolution)}m) (Native Resolution)"
                    common_scales[desc] = native_resolution
                
                # Sort by resolution value
                sorted_scales = dict(sorted(common_scales.items(), key=lambda x: x[1]))
                
                # Find the default selection (native resolution or closest)
                if native_resolution and native_resolution in sorted_scales.values():
                    # Find the key for native resolution
                    default_key = next(k for k, v in sorted_scales.items() if v == native_resolution)
                else:
                    # Default to 30m Landsat
                    default_key = "30m (Landsat)"
                
                scale_choice = st.selectbox(
                    "Select resolution:", 
                    list(sorted_scales.keys()),
                    index=list(sorted_scales.keys()).index(default_key) if default_key in sorted_scales else 0
                )
                scale = sorted_scales[scale_choice]
                
                # Show file size warning for high resolution
                if scale <= 10:
                    st.warning("⚠️ **High resolution selected** - Files may be very large. Consider using Google Drive backup.")
                elif scale <= 100:
                    st.info("💡 **Medium resolution** - Good balance of quality and file size.")
                
            else:  # Custom resolution
                scale = st.number_input(
                    "Enter custom resolution (meters):", 
                    min_value=0.1, 
                    max_value=1000000.0, 
                    value=float(recommended_scale),
                    step=0.1,
                    help="Resolution in meters. Smaller values = higher resolution but larger file sizes."
                )
                
                # Provide guidance based on custom input
                if scale < 1:
                    st.warning("⚠️ **Sub-meter resolution** - Extremely large files expected!")
                elif scale < 10:
                    st.warning("⚠️ **Very high resolution** - Large files expected. Enable Google Drive backup.")
                elif scale > 50000:
                    st.info("ℹ️ **Very coarse resolution** - Suitable for global/continental analysis.")
            
            # Show final resolution info
            st.info(f"📐 **Selected resolution:** {scale} meters")
            
            # Estimate pixel coverage for the AOI (if geometry is available)
            if hasattr(st.session_state, 'geometry_handler') and st.session_state.geometry_handler.current_geometry:
                try:
                    # This is a rough estimate - actual calculation would need the geometry bounds
                    st.caption(f"💾 Higher resolution = larger file sizes | Lower resolution = faster processing")
                except:
                    pass
            
            # Quality and compression options
            if file_format.lower() == 'geotiff':
                compression = st.checkbox("Use compression for GeoTIFF files", value=True)
                st.caption("Compression reduces file size but may slightly increase processing time.")
            else:
                compression = False
        
        with config_col2:
            st.write("#### Output Location & Naming")
            
            # Output directory selection - FIXED for all formats including NetCDF
            use_browser = st.checkbox("Use folder browser", value=True, help="Select output folder using a dialog")
            
            if use_browser:
                # Initialize session state for output directory
                if 'output_dir' not in st.session_state:
                    st.session_state.output_dir = os.path.abspath("data/downloads")
                
                col_browse, col_reset = st.columns([3, 1])
                with col_browse:
                    if st.button("📁 Browse for Output Folder", use_container_width=True):
                        try:
                            import tkinter as tk
                            from tkinter import filedialog
                            
                            # Create and hide the Tkinter root window
                            root = tk.Tk()
                            root.withdraw()
                            root.attributes('-topmost', True)
                            
                            # Show the folder dialog
                            folder_path = filedialog.askdirectory(
                                title="Select Output Directory",
                                initialdir=st.session_state.output_dir
                            )
                            
                            # Update the session state if a folder was selected
                            if folder_path:
                                st.session_state.output_dir = os.path.abspath(folder_path)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error opening folder dialog: {str(e)}")
                            st.info("Please use manual input instead.")
                
                with col_reset:
                    if st.button("🔄", help="Reset to default folder"):
                        st.session_state.output_dir = os.path.abspath("data/downloads")
                        st.rerun()
                
                # Display current directory
                output_dir = st.session_state.output_dir
                st.success(f"📂 **Selected:** `{output_dir}`")
                
                # Verify directory exists or can be created
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    st.info("✅ Directory is accessible")
                except Exception as e:
                    st.error(f"❌ Cannot access directory: {str(e)}")
                    st.info("Please select a different directory or use manual input.")
            else:
                # Manual input
                default_dir = os.path.abspath("data/downloads")
                output_dir = st.text_input("Output directory:", value=default_dir)
                output_dir = os.path.abspath(output_dir)  # Convert to absolute path
                
                # Verify directory
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    st.success("✅ Directory is accessible")
                except Exception as e:
                    st.error(f"❌ Cannot access directory: {str(e)}")
            
            # Filename configuration
            st.write("**Filename Options:**")
            default_filename = f"{dataset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            filename_option = st.radio(
                "Filename type:",
                ["Auto-generated", "Custom filename"],
                help="Auto-generated includes dataset name and timestamp"
            )
            
            if filename_option == "Custom filename":
                filename = st.text_input(
                    "Enter filename (without extension):", 
                    value=default_filename,
                    help="Extension will be added automatically based on format"
                )
            else:
                filename = default_filename
            
            # Show preview of final filename
            file_ext = {'geotiff': '.tif', 'netcdf': '.nc', 'csv': '.csv'}[file_format.lower()]
            final_filename = f"{filename}{file_ext}"
            st.info(f"📄 Final filename: `{final_filename}`")
        
        # Advanced Options in expandable section
        with st.expander("🔧 Advanced Options", expanded=False):
            # Processing options
            st.write("#### Processing Options")
            
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                clip_to_region = st.checkbox("Clip to exact region boundary", value=True)
                st.caption("Clips data to the exact AOI shape (slower but more precise)")
                
                if snippet_type == 'ImageCollection':
                    use_chunking = st.checkbox("Enable chunking for large collections", value=True)
                    if use_chunking:
                        chunk_size = st.slider("Max items per chunk", 50, 5000, 1000, 50)
                        st.caption("Splits large collections into smaller batches")
                    else:
                        chunk_size = 5000
                else:
                    use_chunking = False
                    chunk_size = 5000
            
            with col_adv2:
                # Google Drive options
                st.write("#### Large File Handling")
                use_drive_for_large = st.checkbox("Auto-export large files to Google Drive", value=True)
                st.caption("Files >50MB will be exported to Google Drive instead")
                
                if use_drive_for_large:
                    drive_folder = st.text_input("Google Drive folder name:", "GeoClimate_Downloads")
                else:
                    drive_folder = "GeoClimate_Downloads"
            
            # Export options specific to format
            if file_format.lower() == 'netcdf':
                st.write("#### NetCDF Specific Options")
                netcdf_compression = st.checkbox("Compress NetCDF file", value=True)
                include_metadata = st.checkbox("Include detailed metadata", value=True)
                st.caption("Compression reduces file size, metadata includes processing info")
            
            # Coordinate system options
            st.write("#### Coordinate System")
            crs_option = st.selectbox(
                "Output coordinate system:",
                ["EPSG:4326 (WGS84 Lat/Lon)", "EPSG:3857 (Web Mercator)", "Custom CRS"],
                help="Most datasets use WGS84 (EPSG:4326)"
            )
            
            if crs_option == "Custom CRS":
                custom_crs = st.text_input("Enter EPSG code or proj4 string:", "EPSG:4326")
                crs = custom_crs
            else:
                crs = crs_option.split(" ")[0]
        
        # Preview section
        st.write("### 📋 Export Preview")
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            st.write("**File Information:**")
            st.write(f"• Format: {file_format}")
            st.write(f"• Resolution: {scale} meters")
            st.write(f"• Output: `{output_dir}`")
            st.write(f"• Filename: `{final_filename}`")
        
        with preview_col2:
            st.write("**Processing Options:**")
            st.write(f"• Clip to boundary: {'Yes' if clip_to_region else 'No'}")
            st.write(f"• Google Drive backup: {'Yes' if use_drive_for_large else 'No'}")
            if snippet_type == 'ImageCollection':
                st.write(f"• Chunking: {'Yes' if use_chunking else 'No'}")
            st.write(f"• Coordinate system: {crs}")

        # Download function with all the improvements
        def download_data():
            """Enhanced download function with better error handling and flexibility"""
            try:
                with st.spinner("🚀 Downloading data... This may take a while."):
                    # Show processing info
                    if use_drive_for_large:
                        st.info(f"📤 Large files (>50MB) will be exported to Google Drive folder: '{drive_folder}'")
                    else:
                        st.warning("⚠️ Large file handling disabled. Files >50MB may fail.")
                    
                    # Get the geometry from the geometry handler
                    geometry = st.session_state.geometry_handler.current_geometry
                    
                    # Get required parameters
                    ee_id = dataset.get('Earth Engine ID')
                    selected_bands = st.session_state.selected_bands
                    
                    # Validation
                    if not geometry:
                        st.error("❌ No geometry selected. Please go back to Step 2.")
                        return
                    if not ee_id:
                        st.error("❌ No Earth Engine ID found for the dataset.")
                        return
                    if not selected_bands:
                        st.error("❌ No bands selected. Please go back to Step 4.")
                        return
                    
                    # Ensure output directory exists
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                    except Exception as e:
                        st.error(f"❌ Cannot create output directory: {str(e)}")
                        return
                    
                    # Create output path
                    output_path = os.path.join(output_dir, final_filename)
                    st.info(f"📂 Output path: `{output_path}`")
                    
                    # Apply clipping if requested
                    if clip_to_region:
                        processing_geometry = geometry
                    else:
                        # Use bounding box for faster processing
                        bounds = geometry.bounds().getInfo()['coordinates'][0]
                        xs = [p[0] for p in bounds]
                        ys = [p[1] for p in bounds]
                        processing_geometry = ee.Geometry.Rectangle([min(xs), min(ys), max(xs), max(ys)])
                    
                    # Process based on dataset type
                    if snippet_type == 'ImageCollection':
                        # Get date range
                        start_date = st.session_state.start_date
                        end_date = st.session_state.end_date
                        
                        if not start_date or not end_date:
                            st.error("❌ No date range selected. Please go back to Step 5.")
                            return
                        
                        # Create fetcher
                        fetcher = ImageCollectionFetcher(ee_id=ee_id, bands=selected_bands, geometry=processing_geometry)
                        fetcher = fetcher.filter_dates(start_date=start_date, end_date=end_date)
                        
                        # Process based on format
                        if file_format.lower() == 'csv':
                            st.info("📊 Extracting time series data...")
                            if use_chunking:
                                df = fetcher.get_time_series_average_chunked(chunk_months=3)
                            else:
                                df = fetcher.get_time_series_average()
                            
                            if not df.empty:
                                exporter.export_time_series_to_csv(df, output_path)
                                st.success(f"✅ CSV exported successfully to `{output_path}`")
                            else:
                                st.error("❌ No data retrieved for the time series.")
                                
                        elif file_format.lower() == 'netcdf':
                            st.info("🌐 Creating NetCDF file...")
                            try:
                                ds = fetcher.get_gridded_data(scale=scale, crs=crs)
                                if ds and len(ds.data_vars) > 0:
                                    # Add extra metadata if requested
                                    if 'include_metadata' in locals() and include_metadata:
                                        ds.attrs.update({
                                            'processing_date': datetime.now().isoformat(),
                                            'source_dataset': ee_id,
                                            'bands': ', '.join(selected_bands),
                                            'time_range': f"{start_date} to {end_date}",
                                            'scale_meters': scale,
                                            'coordinate_system': crs,
                                            'created_by': 'GeoClimate Fetcher'
                                        })
                                    
                                    exporter.export_gridded_data_to_netcdf(ds, output_path)
                                    st.success(f"✅ NetCDF exported successfully to `{output_path}`")
                                else:
                                    st.error("❌ No gridded data retrieved.")
                                    st.info("💡 Try: smaller time range, larger scale, or CSV format")
                            except Exception as e:
                                st.error(f"❌ NetCDF export failed: {str(e)}")
                                st.info("💡 Try: CSV format or different time range")
                                
                        else:  # GeoTIFF
                            st.info("🖼️ Exporting individual GeoTIFF files...")
                            collection = fetcher.collection
                            collection_size = collection.size().getInfo()
                            
                            if collection_size == 0:
                                st.error("❌ No images found in collection.")
                                return
                            
                            # Create subdirectory for GeoTIFFs
                            geotiff_dir = os.path.join(output_dir, f"{filename}_geotiffs")
                            os.makedirs(geotiff_dir, exist_ok=True)
                            
                            st.info(f"📸 Processing {collection_size} images...")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process images
                            image_list = collection.toList(collection.size())
                            successful_downloads = 0
                            drive_exports = 0
                            
                            for i in range(collection_size):
                                progress_bar.progress((i + 1) / collection_size)
                                status_text.text(f"Processing image {i+1}/{collection_size}")
                                
                                try:
                                    image = ee.Image(image_list.get(i))
                                    date_millis = image.get('system:time_start').getInfo()
                                    date_str = datetime.fromtimestamp(date_millis / 1000).strftime('%Y%m%d')
                                    
                                    if selected_bands:
                                        image = image.select(selected_bands)
                                    
                                    image_output_path = os.path.join(geotiff_dir, f"{date_str}.tif")
                                    
                                    try:
                                        result_path = exporter.export_image_to_local(
                                            image=image, output_path=image_output_path,
                                            region=processing_geometry, scale=scale
                                        )
                                        
                                        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                                            successful_downloads += 1
                                        else:
                                            raise ValueError("Local export failed")
                                            
                                    except Exception:
                                        if use_drive_for_large:
                                            task_id = exporter.export_image_to_drive(
                                                image=image, filename=f"{filename}_{date_str}",
                                                folder=drive_folder, region=processing_geometry,
                                                scale=scale, wait=False
                                            )
                                            drive_exports += 1
                                        
                                except Exception as e:
                                    st.warning(f"⚠️ Failed to process image {i+1}: {str(e)}")
                            
                            # Summary
                            if successful_downloads > 0:
                                st.success(f"✅ {successful_downloads} images saved to `{geotiff_dir}`")
                            if drive_exports > 0:
                                st.info(f"📤 {drive_exports} images sent to Google Drive folder '{drive_folder}'")
                    
                    else:  # Static Image
                        fetcher = StaticRasterFetcher(ee_id=ee_id, bands=selected_bands, geometry=processing_geometry)
                        image = fetcher.image
                        
                        if file_format.lower() == 'csv':
                            st.info("📊 Extracting zonal statistics...")
                            stats = fetcher.get_zonal_statistics()
                            rows = [{'band': band, **band_stats} for band, band_stats in stats.items()]
                            df = pd.DataFrame(rows)
                            exporter.export_time_series_to_csv(df, output_path)
                            st.success(f"✅ Statistics exported to `{output_path}`")
                            
                        elif file_format.lower() == 'netcdf':
                            st.info("🌐 Creating NetCDF from static image...")
                            pixel_data = fetcher.get_pixel_values(scale=scale)
                            
                            if pixel_data:
                                import xarray as xr
                                import numpy as np
                                
                                # Get bounds and create coordinates
                                bounds = processing_geometry.bounds().getInfo()['coordinates'][0]
                                xs, ys = [p[0] for p in bounds], [p[1] for p in bounds]
                                xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
                                
                                first_band_data = next(iter(pixel_data.values()))
                                height, width = first_band_data.shape
                                
                                coords = {
                                    'lat': np.linspace(ymax, ymin, height),
                                    'lon': np.linspace(xmin, xmax, width)
                                }
                                
                                data_vars = {band: (['lat', 'lon'], array) for band, array in pixel_data.items()}
                                ds = xr.Dataset(data_vars=data_vars, coords=coords)
                                
                                # Add metadata
                                ds.attrs.update({
                                    'description': f"Data from {ee_id}",
                                    'created': datetime.now().isoformat(),
                                    'scale_meters': scale,
                                    'coordinate_system': crs,
                                    'bounds': f"[{xmin}, {ymin}, {xmax}, {ymax}]",
                                    'created_by': 'GeoClimate Fetcher'
                                })
                                
                                exporter.export_gridded_data_to_netcdf(ds, output_path)
                                st.success(f"✅ NetCDF exported to `{output_path}`")
                            else:
                                st.error("❌ No pixel data retrieved")
                                
                        else:  # GeoTIFF
                            st.info("🖼️ Exporting GeoTIFF...")
                            try:
                                result_path = exporter.export_image_to_local(
                                    image=image, output_path=output_path,
                                    region=processing_geometry, scale=scale
                                )
                                
                                if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                                    st.success(f"✅ GeoTIFF exported to `{output_path}`")
                                else:
                                    raise ValueError("Local export failed")
                                    
                            except Exception:
                                if use_drive_for_large:
                                    st.warning("📤 Local export failed, trying Google Drive...")
                                    task_id = exporter.export_image_to_drive(
                                        image=image, filename=filename, folder=drive_folder,
                                        region=processing_geometry, scale=scale, wait=False
                                    )
                                    st.success(f"✅ Export started to Google Drive (Task ID: {task_id})")
                                    st.info("🔗 Check status: https://code.earthengine.google.com/tasks")
                                else:
                                    st.error("❌ Export failed. Try enabling Google Drive backup.")
                    
                    # Final success message
                    st.balloons()
                    st.success("🎉 Download completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                with st.expander("🐛 Error Details", expanded=False):
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        
        # Download button with better styling
        st.write("---")
        col_download, col_reset = st.columns([3, 1])
        
        with col_download:
            if st.button("🚀 Start Download", type="primary", use_container_width=True):
                download_data()
        
        with col_reset:
            if st.button("🔄 Reset App", help="Reset all selections"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

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