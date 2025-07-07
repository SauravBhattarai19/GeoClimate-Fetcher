import streamlit as st
import ee
import geemap.foliumap as geemap
import folium
from folium.plugins import Draw
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from streamlit_folium import folium_static, st_folium
import pandas as pd
import time
import re
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import xarray as xr
from geoclimate_fetcher.climate_indices import ClimateIndicesCalculator
import hashlib
import extra_streamlit_components as stx

# Create cookie manager
cookie_manager = stx.CookieManager()

# Function to create a secure token
def create_auth_token(project_id, timestamp):
    """Create a secure authentication token"""
    secret = "geoclimate-fetcher-secret-key"  # Change this in production
    token_string = f"{project_id}:{timestamp}:{secret}"
    return hashlib.sha256(token_string.encode()).hexdigest()

# Function to validate auth token
def validate_auth_token(token, project_id):
    """Validate if the auth token is valid and not expired"""
    # Token expires after 30 days
    expiry_days = 30
    current_time = time.time()
    
    # Try to validate the token for the last 30 days
    for days_ago in range(expiry_days):
        timestamp = current_time - (days_ago * 86400)
        expected_token = create_auth_token(project_id, int(timestamp // 86400))
        if token == expected_token:
            return True
    return False

# Function to check and load stored credentials
def check_stored_auth():
    """Check if user has valid stored authentication"""
    auth_cookie = cookie_manager.get(cookie="gee_auth_token")
    project_cookie = cookie_manager.get(cookie="gee_project_id")
    
    if auth_cookie and project_cookie:
        if validate_auth_token(auth_cookie, project_cookie):
            return project_cookie
    return None

# Configure Streamlit page
st.set_page_config(
    page_title="GeoClimate Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/sauravbhattarai19/geoclimate-platform',
        'Report a bug': "https://github.com/sauravbhattarai19/geoclimate-platform/issues",
        'About': "# GeoClimate Intelligence Platform\nA comprehensive platform for Earth Engine climate data analysis and intelligence!"
    }
)

# Import app components
from app_components.theme_utils import apply_dark_mode_css

# Apply universal dark mode support
apply_dark_mode_css()

# Enhanced CSS with dark mode support
st.markdown("""
<style>
    /* Dark mode support - automatically detects user's browser preference */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Landing page styles - dark mode */
        .landing-hero {
            background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .tool-card {
            background: #262730;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            border: 2px solid #464852;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .tool-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            border-color: #667eea;
        }
        
        .tool-title {
            color: #fafafa;
        }
        
        .tool-description {
            color: #d0d0d0;
        }
        
        .author-section {
            background: #262730;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 3rem;
            text-align: center;
            border-left: 5px solid #667eea;
            color: #fafafa;
        }
        
        .feature-item {
            background: #262730;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #464852;
            color: #fafafa;
        }
        
        .step-header {
            padding: 1rem;
            background: linear-gradient(90deg, #1e293b, #334155);
            border-left: 5px solid #1f77b4;
            border-radius: 5px;
            margin: 1rem 0;
            color: #fafafa;
        }
        
        .metric-card {
            background: #262730;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
            color: #fafafa;
        }
        
        .info-box {
            background: #262730;
            border: 1px solid #464852;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #fafafa;
        }
        
        .progress-steps {
            display: flex;
            justify-content: space-between;
            margin: 2rem 0;
            padding: 1rem;
            background: #262730;
            border-radius: 10px;
            border: 1px solid #464852;
        }
        
        .step-item {
            color: #fafafa;
        }
    }
    
    /* Light mode styles (default) */
    @media (prefers-color-scheme: light) {
        .landing-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .tool-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .tool-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        
        .tool-title {
            color: #333;
        }
        
        .tool-description {
            color: #666;
        }
        
        .author-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 3rem;
            text-align: center;
            border-left: 5px solid #667eea;
        }
        
        .feature-item {
            background: #f0f4ff;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #e0e7ff;
        }
        
        .step-header {
            padding: 1rem;
            background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
            border-left: 5px solid #1f77b4;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        
        .info-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .progress-steps {
            display: flex;
            justify-content: space-between;
            margin: 2rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 10px;
        }
    }
    
    /* Universal styles that work in both modes */
    .landing-title {
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .landing-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 2rem;
    }
    
    .tool-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .tool-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .tool-description {
        line-height: 1.6;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #667eea;
    }
    
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
    
    /* Back button styling */
    .back-button {
        background: #6c757d;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin-bottom: 1rem;
    }
    
    .back-button:hover {
        background: #5a6268;
    }
    
    /* Enhanced Streamlit component styling for both themes */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .landing-title {
            font-size: 2.5rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .progress-steps {
            flex-direction: column;
            gap: 1rem;
        }
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

# Initialize session state
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = None  # None, 'data_explorer', 'climate_analytics'
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
if 'project_id' not in st.session_state:
    st.session_state.project_id = None

# Function to go back to a previous step
def go_back_to_step(step):
    """Reset the app state to go back to a specific step"""
    if step == "home":
        st.session_state.app_mode = None
        st.session_state.auth_complete = False
        st.session_state.geometry_complete = False
        st.session_state.dataset_selected = False
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False
        # Clear stored auth when going back to home
        try:
            # Get all cookies first
            all_cookies = cookie_manager.get_all()
            # Only delete if cookies exist
            if 'gee_auth_token' in all_cookies:
                cookie_manager.delete('gee_auth_token')
            if 'gee_project_id' in all_cookies:
                cookie_manager.delete('gee_project_id')
        except Exception as e:
            # Log error but don't stop execution
            print(f"Error clearing cookies: {str(e)}")
    elif step == "geometry":
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

# Function to authenticate GEE
def authenticate_gee(project_id, service_account=None, key_file=None, auth_token=None):
    """Authenticate with Google Earth Engine and store credentials"""
    try:
        auth = authenticate(project_id, service_account, key_file, auth_token)
        if auth.is_initialized():
            st.session_state.auth_complete = True
            st.session_state.project_id = project_id
            # Store authentication in cookies
            auth_token = create_auth_token(project_id, int(time.time() // 86400))
            cookie_manager.set("gee_auth_token", auth_token, expires_at=datetime.now() + timedelta(days=30))
            cookie_manager.set("gee_project_id", project_id, expires_at=datetime.now() + timedelta(days=30))
            return True, "Authentication successful!"
        else:
            return False, "Authentication failed. Please check your credentials."
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"

# Landing Page
if st.session_state.app_mode is None:
    # Hero Section
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-title">🌍 GeoClimate Intelligence Platform</div>
        <div class="landing-subtitle">
            Unlock the power of Earth Engine climate data with advanced analytics and visualization
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("### 🚀 Platform Capabilities")
    
    features = [
        ("🛰️", "Satellite Data Access", "Access petabytes of Earth observation data"),
        ("📊", "Climate Analytics", "Calculate standard climate indices"),
        ("🗺️", "Interactive Maps", "Visualize data spatially and temporally"),
        ("📈", "Time Series Analysis", "Extract and analyze temporal patterns"),
        ("☁️", "Cloud Processing", "Leverage Google Earth Engine's power"),
        ("💾", "Multiple Formats", "Export as GeoTIFF, NetCDF, or CSV")
    ]
    
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-item">
                <div class="feature-icon">{icon}</div>
                <strong>{title}</strong><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tool Selection
    st.markdown("### 🎯 Choose Your Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tool-card">
            <span class="tool-icon">🔍</span>
            <div class="tool-title">GeoData Explorer</div>
            <div class="tool-description">
                Download and visualize Earth Engine datasets. Perfect for researchers needing raw climate data
                with options for GeoTIFF, NetCDF, and CSV formats. Includes interactive previews.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Data Explorer", use_container_width=True, type="primary"):
            st.session_state.app_mode = "data_explorer"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="tool-card">
            <span class="tool-icon">🧠</span>
            <div class="tool-title">Climate Intelligence Hub</div>
            <div class="tool-description">
                Calculate climate indices and analyze extreme events. Compute SPI, temperature anomalies,
                drought indicators, and more directly in the cloud.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Climate Analytics", use_container_width=True, type="primary"):
            st.session_state.app_mode = "climate_analytics"
            st.rerun()
    
    # Author Section
    st.markdown("""
    <div class="author-section">
        <h3>👨‍💻 About the Developer</h3>
        <h3><strong>Saurav Bhattarai</strong></h3>
        <p>Climate Data Scientist & Geospatial Developer</p>
        <p>Under Supervision of Dr. Rocky Talchabhadeli</p>
        <p>📧 Email: <a href="mailto:saurav.bhattarai.1999@gmail.com">saurav.bhattarai.1999@gmail.com</a></p>
        <p>🔗 <a href="https://github.com/sauravbhattarai19">GitHub</a> | 
           <a href="https://www.linkedin.com/in/saurav-bhattarai-7133a3176/">LinkedIn</a></p>
        <br>
        <small>Built with ❤️ using Google Earth Engine, Streamlit, and Python</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start Guide
    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        ### Getting Started
        
        **🔍 GeoData Explorer**
        1. Authenticate with Google Earth Engine
        2. Select your area of interest
        3. Choose a dataset from our curated catalog
        4. Select bands and time range
        5. Preview and download your data
        
        **🧠 Climate Intelligence Hub**
        1. Authenticate with Google Earth Engine
        2. Define your study area
        3. Select climate indices to calculate
        4. Set parameters and time period
        5. Visualize results and export
        
        ### 🔑 Requirements
        - Google Earth Engine account ([Sign up free](https://earthengine.google.com/signup/))
        - Basic understanding of climate data
        - Internet connection for cloud processing
        """)

# Data Explorer Mode
elif st.session_state.app_mode == "data_explorer":
    # Add home button
    if st.button("🏠 Back to Home"):
        go_back_to_step("home")
    
    # Continue with existing data download code...
    # Initialize core objects
    metadata_catalog = MetadataCatalog()
    exporter = GEEExporter()

    # App title and header
    st.markdown('<h1 class="main-title">🔍 GeoData Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Download and visualize Earth Engine climate datasets")

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

    # Step 1: Authentication
    if not st.session_state.auth_complete:
        # First check for stored authentication
        stored_project_id = check_stored_auth()
        
        if stored_project_id:
            # Try to authenticate with stored credentials
            try:
                success, message = authenticate_gee(stored_project_id)
                if success:
                    st.success("✅ Authenticated using stored credentials!")
                    time.sleep(1)
                    st.rerun()
            except:
                # Stored auth failed, proceed with manual auth
                pass
        
        # If no stored auth or it failed, show authentication form
        st.markdown('<div class="step-header"><h2>🔐 Step 1: Google Earth Engine Authentication</h2></div>', unsafe_allow_html=True)
        
        # Use the updated AuthComponent
        from app_components.auth_component import AuthComponent
        
        auth_component = AuthComponent()
        if auth_component.render():
            # Authentication is complete, automatically proceed to next step
            st.rerun()

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
                
                # Display the map using st_folium instead of folium_static
                st_folium(m, width=800, height=500, returned_objects=[])
                
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
                
                # Display the preview map using st_folium instead of folium_static
                st.write("Preview of selected area:")
                st_folium(preview_map, width=800, height=500, returned_objects=[])
                
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
                        
                        # Estimate request size to prevent Google Earth Engine size limit errors
                        def estimate_request_size(geometry, scale, num_bands=1, num_images=1):
                            """Estimate the size of an Earth Engine request in bytes"""
                            try:
                                # Get geometry area in square meters
                                area_sqm = geometry.area().getInfo()
                                
                                # Calculate number of pixels
                                pixels_per_band = area_sqm / (scale * scale)
                                
                                # Estimate bytes per pixel (typically 4 bytes for float32)
                                bytes_per_pixel = 4
                                
                                # Total size estimate
                                total_size = pixels_per_band * num_bands * num_images * bytes_per_pixel
                                
                                return total_size
                            except:
                                # Return conservative estimate if calculation fails
                                return 100 * 1024 * 1024  # 100MB
                        
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
                                    
                                    # Show success and automatic download for CSV
                                    from app_components.download_component import DownloadHelper
                                    download_helper = DownloadHelper()
                                    download_helper.create_automatic_download(
                                        output_path, 
                                        download_name=os.path.basename(output_path),
                                        show_success=True
                                    )
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
                                        
                                        # Show success and automatic download for NetCDF
                                        from app_components.download_component import DownloadHelper
                                        download_helper = DownloadHelper()
                                        download_helper.create_automatic_download(
                                            output_path, 
                                            download_name=os.path.basename(output_path),
                                            show_success=True
                                        )
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
                                
                                # Estimate total request size for collection
                                estimated_size_per_image = estimate_request_size(
                                    processing_geometry, scale, len(selected_bands), 1
                                )
                                total_estimated_size = estimated_size_per_image * collection_size
                                max_ee_size = 50 * 1024 * 1024  # 50MB limit
                                
                                if estimated_size_per_image > max_ee_size:
                                    st.error(f"❌ Estimated size per image ({estimated_size_per_image/1024/1024:.1f} MB) exceeds Earth Engine limit (50 MB)")
                                    st.info("💡 **Solutions:**")
                                    st.info("• Increase the scale parameter (lower resolution)")
                                    st.info("• Reduce the area of interest")
                                    st.info("• Select fewer bands")
                                    st.info("• Use CSV format for time series data")
                                    return
                                elif total_estimated_size > max_ee_size * 5:  # Warning if total is very large
                                    st.warning(f"⚠️ Large download estimated ({total_estimated_size/1024/1024:.1f} MB total)")
                                    st.info("💡 Consider using Google Drive backup for large downloads")
                                
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
                                            # Note: Data type harmonization is handled automatically in the exporter
                                            result_path = exporter.export_image_to_local(
                                                image=image, output_path=image_output_path,
                                                region=processing_geometry, scale=scale
                                            )
                                            
                                            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                                                successful_downloads += 1
                                            else:
                                                raise ValueError("Local export failed")
                                                
                                        except Exception as export_error:
                                            error_str = str(export_error)
                                            if "Total request size" in error_str and "bytes" in error_str:
                                                st.warning(f"⚠️ Image {i+1} too large for direct download")
                                                if use_drive_for_large:
                                                    try:
                                                        # Note: Data type harmonization handled in exporter
                                                        task_id = exporter.export_image_to_drive(
                                                            image=image, filename=f"{filename}_{date_str}",
                                                            folder=drive_folder, region=processing_geometry,
                                                            scale=scale, wait=False
                                                        )
                                                        drive_exports += 1
                                                    except Exception as drive_error:
                                                        st.warning(f"⚠️ Drive export also failed for image {i+1}: {str(drive_error)}")
                                                else:
                                                    st.warning(f"⚠️ Skipping image {i+1}: {error_str}")
                                            else:
                                                if use_drive_for_large:
                                                    # Note: Data type harmonization handled in exporter
                                                    task_id = exporter.export_image_to_drive(
                                                        image=image, filename=f"{filename}_{date_str}",
                                                        folder=drive_folder, region=processing_geometry,
                                                        scale=scale, wait=False
                                                    )
                                                    drive_exports += 1
                                            
                                    except Exception as e:
                                        st.warning(f"⚠️ Failed to process image {i+1}: {str(e)}")
                                
                                # Summary with download options
                                if successful_downloads > 0 or drive_exports > 0:
                                    # Import download component
                                    from app_components.download_component import DownloadComponent
                                    download_component = DownloadComponent()
                                    download_component.show_download_summary(
                                        output_dir=geotiff_dir,
                                        successful_downloads=successful_downloads,
                                        drive_exports=drive_exports
                                    )
                                else:
                                    st.warning("⚠️ No files were successfully processed")
                        
                        else:  # Static Image
                            fetcher = StaticRasterFetcher(ee_id=ee_id, bands=selected_bands, geometry=processing_geometry)
                            image = fetcher.image
                            
                            if file_format.lower() == 'csv':
                                st.info("📊 Extracting zonal statistics...")
                                stats = fetcher.get_zonal_statistics()
                                rows = [{'band': band, **band_stats} for band, band_stats in stats.items()]
                                df = pd.DataFrame(rows)
                                exporter.export_time_series_to_csv(df, output_path)
                                
                                # Show success and automatic download for CSV
                                from app_components.download_component import DownloadHelper
                                download_helper = DownloadHelper()
                                
                                download_helper.create_automatic_download(
                                    output_path, 
                                    download_name=os.path.basename(output_path),
                                    show_success=False
                                )
                                
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
                                    
                                    # Show success and automatic download for NetCDF
                                    from app_components.download_component import DownloadHelper
                                    download_helper = DownloadHelper()
                                    
                                    download_helper.create_automatic_download(
                                        output_path, 
                                        download_name=os.path.basename(output_path),
                                        show_success=False
                                    )
                                else:
                                    st.error("❌ No pixel data retrieved")
                                    
                            else:  # GeoTIFF
                                st.info("🖼️ Exporting GeoTIFF...")
                                
                                # Check estimated size before export
                                estimated_size = estimate_request_size(
                                    processing_geometry, scale, len(selected_bands), 1
                                )
                                max_ee_size = 50 * 1024 * 1024  # 50MB limit
                                
                                if estimated_size > max_ee_size:
                                    st.error(f"❌ Estimated size ({estimated_size/1024/1024:.1f} MB) exceeds Earth Engine limit (50 MB)")
                                    st.info("💡 **Solutions:**")
                                    st.info("• Increase the scale parameter (lower resolution)")
                                    st.info("• Reduce the area of interest")
                                    st.info("• Select fewer bands")
                                    st.info("• Use CSV format for statistics")
                                    if use_drive_for_large:
                                        st.info("🚀 Trying Google Drive export instead...")
                                        st.info("📊 **Data Type Harmonization**: Converting all bands to Float32 to ensure compatibility")
                                        try:
                                            task_id = exporter.export_image_to_drive(
                                                image=image, filename=filename, folder=drive_folder,
                                                region=processing_geometry, scale=scale, wait=False
                                            )
                                            st.success(f"✅ Export started to Google Drive (Task ID: {task_id})")
                                            st.info("🔗 Check status: https://code.earthengine.google.com/tasks")
                                        except Exception as drive_e:
                                            st.error(f"❌ Google Drive export also failed: {str(drive_e)}")
                                    return
                                elif estimated_size > max_ee_size * 0.8:  # Warning if approaching limit
                                    st.warning(f"⚠️ Large download ({estimated_size/1024/1024:.1f} MB). May be slow.")
                                
                                # Add informational message about data type handling
                                st.info("📊 **Data Type Harmonization**: Converting all bands to Float32 to ensure compatibility")
                                
                                try:
                                    result_path = exporter.export_image_to_local(
                                        image=image, output_path=output_path,
                                        region=processing_geometry, scale=scale
                                    )
                                    
                                    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                                        # Show success and instant download for single GeoTIFF
                                        from app_components.download_component import DownloadHelper
                                        download_helper = DownloadHelper()
                                        
                                        download_helper.create_instant_download(
                                            result_path, 
                                            download_name=os.path.basename(result_path),
                                            show_success=False
                                        )
                                    else:
                                        raise ValueError("Local export failed")
                                        
                                except Exception as export_error:
                                    error_str = str(export_error)
                                    if "Total request size" in error_str and "bytes" in error_str:
                                        st.error(f"❌ Export failed: Request too large ({error_str})")
                                        st.info("💡 **Try these solutions:**")
                                        st.info("• Increase scale parameter (reduce resolution)")
                                        st.info("• Reduce area of interest size")
                                        st.info("• Select fewer bands")
                                        st.info("• Enable Google Drive for large files")
                                    elif use_drive_for_large:
                                        st.warning("📤 Local export failed, trying Google Drive...")
                                        st.info("📊 **Data Type Harmonization**: Converting all bands to Float32 to ensure compatibility")
                                        try:
                                            task_id = exporter.export_image_to_drive(
                                                image=image, filename=filename, folder=drive_folder,
                                                region=processing_geometry, scale=scale, wait=False
                                            )
                                            st.success(f"✅ Export started to Google Drive (Task ID: {task_id})")
                                            st.info("🔗 Check status: https://code.earthengine.google.com/tasks")
                                        except Exception as drive_error:
                                            st.error(f"❌ Both local and Drive export failed: {str(drive_error)}")
                                    else:
                                        st.error(f"❌ Export failed: {error_str}")
                                        st.info("💡 Try enabling Google Drive backup or reducing data size.")
                        
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
            
            # Visualization Section
            st.write("---")
            st.write("### 📊 Data Visualization")
            
            # Check if output directory exists and has files
            if 'output_dir' in locals() and os.path.exists(output_dir):
                # List files in output directory
                files = []
                for ext in ['.nc', '.tif', '.csv']:
                    files.extend([f for f in os.listdir(output_dir) if f.endswith(ext)])
                
                if files:
                    st.write("#### 🗂️ Available Files for Visualization")
                    
                    # File selection
                    selected_file = st.selectbox(
                        "Select a file to visualize:",
                        ["-- Select a file --"] + files
                    )
                    
                    if selected_file != "-- Select a file --":
                        file_path = os.path.join(output_dir, selected_file)
                        
                        # Initialize visualizer
                        from geoclimate_fetcher.visualization import DataVisualizer
                        visualizer = DataVisualizer()
                        
                        # Visualize based on file type
                        st.write(f"#### 🎨 Visualizing: {selected_file}")
                        
                        try:
                            if selected_file.endswith('.nc'):
                                visualizer.visualize_netcdf(file_path)
                            elif selected_file.endswith('.tif'):
                                visualizer.visualize_geotiff(file_path)
                            elif selected_file.endswith('.csv'):
                                visualizer.visualize_csv(file_path)
                            else:
                                st.error("Unsupported file format for visualization")
                        except Exception as e:
                            st.error(f"Error visualizing file: {str(e)}")
                            with st.expander("Error details"):
                                import traceback
                                st.code(traceback.format_exc())
                    
                    # Option to upload external files
                    st.write("---")
                    st.write("#### 📤 Or Upload a File to Visualize")
                else:
                    st.info("No files found in the output directory. Download some data first!")
                    st.write("#### 📤 Upload a File to Visualize")
            else:
                st.write("#### 📤 Upload a File to Visualize")
            
            # File upload for visualization
            uploaded_viz_file = st.file_uploader(
                "Choose a file to visualize",
                type=['nc', 'tif', 'csv'],
                help="Upload NetCDF, GeoTIFF, or CSV files"
            )
            
            if uploaded_viz_file is not None:
                # Save uploaded file temporarily
                temp_path = os.path.join("temp", uploaded_viz_file.name)
                os.makedirs("temp", exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_viz_file.getbuffer())
                
                # Initialize visualizer
                from geoclimate_fetcher.visualization import DataVisualizer
                visualizer = DataVisualizer()
                
                st.write(f"#### 🎨 Visualizing: {uploaded_viz_file.name}")
                
                try:
                    if uploaded_viz_file.name.endswith('.nc'):
                        visualizer.visualize_netcdf(temp_path)
                    elif uploaded_viz_file.name.endswith('.tif'):
                        visualizer.visualize_geotiff(temp_path)
                    elif uploaded_viz_file.name.endswith('.csv'):
                        visualizer.visualize_csv(temp_path)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error visualizing file: {str(e)}")
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# Climate Intelligence Hub Mode
elif st.session_state.app_mode == "climate_analytics":
    # Add home button
    if st.button("🏠 Back to Home"):
        go_back_to_step("home")
    
    # App title and header
    st.markdown('<h1 class="main-title">🧠 Climate Intelligence Hub</h1>', unsafe_allow_html=True)
    st.markdown("### Calculate climate indices and analyze extreme events")
    
    # Check if authenticated, if not, redirect to authentication
    if not st.session_state.get('auth_complete', False):
        # First check for stored authentication
        stored_project_id = check_stored_auth()
        
        if stored_project_id:
            # Try to authenticate with stored credentials
            try:
                success, message = authenticate_gee(stored_project_id)
                if success:
                    st.success("✅ Authenticated using stored credentials!")
                    time.sleep(1)
                    st.rerun()
            except:
                # Stored auth failed, proceed with manual auth
                pass
        
        # If no stored auth or it failed, show authentication form
        st.markdown('<div class="step-header"><h2>🔐 Step 1: Google Earth Engine Authentication</h2></div>', unsafe_allow_html=True)
        
        # Use the updated AuthComponent
        from app_components.auth_component import AuthComponent
        
        auth_component = AuthComponent()
        if auth_component.render():
            # Authentication is complete, automatically proceed to next step
            st.rerun()
        else:
            st.stop()
      # Initialize session state for climate analytics
    if 'climate_step' not in st.session_state:
        st.session_state.climate_step = 1
    if 'climate_geometry_complete' not in st.session_state:
        st.session_state.climate_geometry_complete = False
    if 'climate_dataset_selected' not in st.session_state:
        st.session_state.climate_dataset_selected = False
    if 'climate_geometry_handler' not in st.session_state:
        st.session_state.climate_geometry_handler = GeometryHandler()
    if 'climate_selected_dataset' not in st.session_state:
        st.session_state.climate_selected_dataset = None
    if 'climate_analysis_type' not in st.session_state:
        st.session_state.climate_analysis_type = None
    if 'climate_date_range_set' not in st.session_state:
        st.session_state.climate_date_range_set = False
    if 'climate_indices_selected' not in st.session_state:
        st.session_state.climate_indices_selected = False

    # Progress indicator for climate analytics
    def show_climate_progress():
        """Display progress indicator for climate analytics workflow"""
        steps = [
            ("🔐", "Auth", st.session_state.auth_complete),
            ("🌡️", "Type", st.session_state.climate_analysis_type is not None),
            ("🗺️", "Area", st.session_state.climate_geometry_complete),
            ("📊", "Dataset", st.session_state.climate_dataset_selected),
            ("📅", "Dates", st.session_state.climate_date_range_set),
            ("🧮", "Indices", st.session_state.climate_indices_selected),
            ("�", "Results", False)
        ]
        
        st.markdown('<div class="progress-steps">', unsafe_allow_html=True)
        cols = st.columns(len(steps))
        
        for i, (icon, name, complete) in enumerate(steps):
            with cols[i]:
                if complete:
                    status_class = "step-complete"
                elif i == st.session_state.climate_step - 1:
                    status_class = "step-current"
                else:
                    status_class = "step-pending"
                
                st.markdown(f"""
                <div class="step-item {status_class}">
                    <div class="step-number">{icon}</div>
                    <div class="step-name">{name}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Show progress
    show_climate_progress()

    # Step 1: Analysis Type Selection
    if st.session_state.climate_analysis_type is None:
        st.markdown('<div class="step-header"><h2>🌡️ Step 1: Choose Analysis Type</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="tool-card">
                <span class="tool-icon">🌡️</span>
                <div class="tool-title">Temperature Indices</div>
                <div class="tool-description">
                    Calculate temperature-based climate indices such as heat waves, cold spells, 
                    growing degree days, and temperature percentiles using various satellite datasets.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🌡️ Analyze Temperature", use_container_width=True, type="primary"):
                st.session_state.climate_analysis_type = "temperature"
                st.session_state.climate_step = 2
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="tool-card">
                <span class="tool-icon">💧</span>
                <div class="tool-title">Precipitation Indices</div>
                <div class="tool-description">
                    Calculate precipitation-based climate indices such as dry spells, extreme precipitation,
                    standardized precipitation index (SPI), and rainfall percentiles.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("💧 Analyze Precipitation", use_container_width=True, type="primary"):
                st.session_state.climate_analysis_type = "precipitation"
                st.session_state.climate_step = 2
                st.rerun()
    
    else:
        # Show current analysis type and allow changing
        analysis_type = st.session_state.climate_analysis_type
        st.info(f"🎯 Current Analysis: **{analysis_type.title()} Indices**")
        
        if st.button("🔄 Change Analysis Type", key="change_analysis"):
            st.session_state.climate_analysis_type = None
            st.session_state.climate_step = 1
            st.session_state.climate_geometry_complete = False
            st.session_state.climate_dataset_selected = False
            st.session_state.climate_date_range_set = False
            st.session_state.climate_indices_selected = False
            st.rerun()
        # Step 2: Area of Interest Selection
        if not st.session_state.climate_geometry_complete:
            st.markdown('<div class="step-header"><h2>🗺️ Step 2: Select Study Area</h2></div>', unsafe_allow_html=True)
            
            # Add back button
            if st.button("← Back to Analysis Type"):
                st.session_state.climate_analysis_type = None
                st.session_state.climate_step = 1
                st.rerun()
            
            st.markdown("Choose your study area using one of the methods below:")
            
            # Method selection
            method = st.radio(
                "Select method:",
                ["🗺️ Draw on Map", "📁 Upload GeoJSON", "📐 Enter Coordinates"],
                horizontal=True,
                key="climate_method"
            )
            
            if method == "🗺️ Draw on Map":
                st.info("Use the drawing tools in the top-right corner to draw a rectangle or polygon on the map.")
                
                # Create the map
                m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
                folium.TileLayer( 
                    tiles='OpenStreetMap',
                    attr='© OpenStreetMap contributors',
                    name='OpenStreetMap'
                ).add_to(m)
                folium.TileLayer(
                    tiles='CartoDB positron',
                    attr='© OpenStreetMap contributors © CARTO',
                    name='Light'
                ).add_to(m)
                
                # Add drawing tools
                draw = Draw(
                    export=True,
                    position='topright',
                    draw_options={
                        'polyline': False,
                        'rectangle': {
                            'shapeOptions': {
                                'color': '#ff7f0e',
                                'fillColor': '#ff7f0e',
                                'fillOpacity': 0.3
                            }
                        },
                        'polygon': {
                            'shapeOptions': {
                                'color': '#1f77b4',
                                'fillColor': '#1f77b4',
                                'fillOpacity': 0.3
                            }
                        },
                        'circle': False,
                        'marker': False,
                        'circlemarker': False
                    },
                    edit_options={
                        'edit': True,
                        'remove': True
                    }
                )
                draw.add_to(m)
                folium.LayerControl().add_to(m)
                
                # Display the map and capture interactions
                map_data = st_folium(
                    m,
                    key="climate_aoi_map",
                    width=700,
                    height=500,
                    returned_objects=["all_drawings", "last_object_clicked"]
                )
                
                # Process drawn features
                if map_data['all_drawings'] and len(map_data['all_drawings']) > 0:
                    st.success(f"✅ Found {len(map_data['all_drawings'])} drawn feature(s)")
                    
                    # Use the most recent drawing
                    latest_drawing = map_data['all_drawings'][-1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Selected Area:**")
                        st.json(latest_drawing, expanded=False)
                    
                    with col2:
                        if st.button("Use This Area", type="primary", key="climate_use_area"):
                            try:
                                # Handle different GeoJSON formats
                                geometry_dict = latest_drawing['geometry']
                                
                                # Create Earth Engine geometry
                                geometry = ee.Geometry(geometry_dict)
                                st.session_state.climate_geometry_handler._current_geometry = geometry
                                st.session_state.climate_geometry_handler._current_geometry_name = "climate_drawn_aoi"
                                st.session_state.climate_geometry_complete = True
                                st.session_state.climate_step = 3
                                st.success("✅ Area of interest selected successfully!")
                                st.balloons()
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error processing drawn area: {str(e)}")
                else:
                    st.info("👆 Draw a rectangle or polygon on the map above to select your area of interest.")
            
            elif method == "📁 Upload GeoJSON":
                st.markdown("### 📁 Upload GeoJSON File")
                
                uploaded_file = st.file_uploader(
                    "Choose a GeoJSON file",
                    type=["geojson", "json"],
                    help="Upload a GeoJSON file containing your area of interest",
                    key="climate_geojson_upload"
                )
                
                if uploaded_file is not None:
                    try:
                        # Read the file
                        geojson_data = json.loads(uploaded_file.getvalue().decode())
                        
                        # Show preview
                        st.success("✅ File loaded successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**File Contents:**")
                            st.json(geojson_data, expanded=False)
                        
                        with col2:
                            st.markdown("**Actions:**")
                            if st.button("Use This File", type="primary", key="climate_use_geojson"):
                                try:
                                    # Handle different GeoJSON formats
                                    if isinstance(geojson_data, str):
                                        geojson_dict = json.loads(geojson_data)
                                    else:
                                        geojson_dict = geojson_data
                                    
                                    # Extract geometry from different GeoJSON types
                                    if geojson_dict.get("type") == "FeatureCollection" and "features" in geojson_dict:
                                        if len(geojson_dict["features"]) > 0:
                                            geometry_dict = geojson_dict["features"][0]["geometry"]
                                        else:
                                            st.error("❌ FeatureCollection is empty")
                                            st.stop()
                                    elif geojson_dict.get("type") == "Feature" and "geometry" in geojson_dict:
                                        geometry_dict = geojson_dict["geometry"]
                                    else:
                                        geometry_dict = geojson_dict
                                    
                                    # Create Earth Engine geometry
                                    geometry = ee.Geometry(geometry_dict)
                                    st.session_state.climate_geometry_handler._current_geometry = geometry
                                    st.session_state.climate_geometry_handler._current_geometry_name = "climate_uploaded_aoi"
                                    st.session_state.climate_geometry_complete = True
                                    st.session_state.climate_step = 3
                                    st.success("✅ Geometry created successfully from GeoJSON")
                                    st.balloons()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error processing GeoJSON: {str(e)}")
                                    
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON file: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
            
            elif method == "📐 Enter Coordinates":
                st.markdown("### 📐 Enter Bounding Box Coordinates")
                
                # Coordinate input form
                with st.form("climate_coordinates_form"):
                    st.markdown("Enter the bounding box coordinates (in decimal degrees):")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Southwest Corner:**")
                        min_lon = st.number_input(
                            "Minimum Longitude",
                            value=-95.0,
                            min_value=-180.0,
                            max_value=180.0,
                            format="%.6f",
                            help="Western boundary",
                            key="climate_min_lon"
                        )
                        min_lat = st.number_input(
                            "Minimum Latitude",
                            value=30.0,
                            min_value=-90.0,
                            max_value=90.0,
                            format="%.6f",
                            help="Southern boundary",
                            key="climate_min_lat"
                        )
                    
                    with col2:
                        st.markdown("**Northeast Corner:**")
                        max_lon = st.number_input(
                            "Maximum Longitude",
                            value=-94.0,
                            min_value=-180.0,
                            max_value=180.0,
                            format="%.6f",
                            help="Eastern boundary",
                            key="climate_max_lon"
                        )
                        max_lat = st.number_input(
                            "Maximum Latitude",
                            value=31.0,
                            min_value=-90.0,
                            max_value=90.0,
                            format="%.6f",
                            help="Northern boundary",
                            key="climate_max_lat"
                        )
                    
                    # Validation
                    valid = True
                    if min_lon >= max_lon:
                        st.error("❌ Minimum longitude must be less than maximum longitude")
                        valid = False
                    if min_lat >= max_lat:
                        st.error("❌ Minimum latitude must be less than maximum latitude")
                        valid = False
                    
                    submitted = st.form_submit_button("Create Bounding Box", type="primary", disabled=not valid)
                    
                    if submitted and valid:
                        try:
                            geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
                            st.session_state.climate_geometry_handler._current_geometry = geometry
                            st.session_state.climate_geometry_handler._current_geometry_name = "climate_coordinates_aoi"
                            st.session_state.climate_geometry_complete = True
                            st.session_state.climate_step = 3
                            st.success("✅ Geometry created successfully from coordinates")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error creating geometry: {str(e)}")
                            
        # Step 3: Dataset Selection
        elif not st.session_state.climate_dataset_selected:
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
                    
                    # Load appropriate datasets based on analysis type
                    analysis_type = st.session_state.climate_analysis_type
                    csv_file = f'geoclimate_fetcher/data/climate_index/{analysis_type}.csv'
                    
                    try:
                        # Check if the CSV file exists
                        from pathlib import Path
                        if not Path(csv_file).exists():
                            st.error(f"❌ Dataset file not found: {csv_file}")
                            st.info("Please make sure the climate index CSV files are in the correct location.")
                            st.stop()
                        
                        df = pd.read_csv(csv_file)
                        st.info(f"📊 Found {len(df)} {analysis_type} datasets")
                        
                        # Create dataset selection interface
                        dataset_options = {}
                        for _, row in df.iterrows():
                            name = row['Dataset Name']
                            provider = row['Provider']
                            temporal_res = row['Temporal Resolution']
                            start_date = row['Start Date']
                            end_date = row['End Date']
                            bands = row['Band Names']
                            
                            display_name = f"{name} ({provider})"
                            dataset_options[display_name] = {
                                'name': name,
                                'ee_id': row['Earth Engine ID'],
                                'provider': provider,
                                'temporal_resolution': temporal_res,
                                'start_date': start_date,
                                'end_date': end_date,
                                'bands': bands,
                                'band_units': row.get('Band Units', ''),
                                'description': row['Description'],
                                'pixel_size': row.get('Pixel Size (m)', 'N/A'),
                                'snippet_type': row.get('Snippet Type', 'ImageCollection')
                            }
                        
                        # Display dataset selection
                        st.markdown(f"### Available {analysis_type.title()} Datasets:")
                        
                        # Create tabs for different providers if there are many datasets
                        if len(dataset_options) > 6:
                            providers = list(set([info['provider'] for info in dataset_options.values()]))
                            provider_tabs = st.tabs(providers)
                            
                            for i, provider in enumerate(providers):
                                with provider_tabs[i]:
                                    provider_datasets = {k: v for k, v in dataset_options.items() if v['provider'] == provider}
                                    selected_dataset_key = None
                                    
                                    for display_name, dataset_info in provider_datasets.items():
                                        with st.expander(f"📊 {dataset_info['name']}", expanded=False):
                                            col1, col2 = st.columns([2, 1])
                                            
                                            with col1:
                                                st.markdown(f"**Provider:** {dataset_info['provider']}")
                                                st.markdown(f"**Temporal Resolution:** {dataset_info['temporal_resolution']}")
                                                st.markdown(f"**Data Period:** {dataset_info['start_date']} to {dataset_info['end_date']}")
                                                st.markdown(f"**Pixel Size:** {dataset_info['pixel_size']} meters")
                                                st.markdown(f"**Available Bands:** {dataset_info['bands']}")
                                                if dataset_info['band_units']:
                                                    st.markdown(f"**Band Units:** {dataset_info['band_units']}")
                                                st.markdown(f"**Description:** {dataset_info['description']}")
                                            
                                            with col2:
                                                if st.button(f"Select {dataset_info['name']}", key=f"select_{display_name.replace(' ', '_')}", type="primary"):
                                                    selected_dataset_key = display_name
                                    
                                    # Handle dataset selection
                                    if selected_dataset_key:
                                        st.session_state.climate_selected_dataset = dataset_options[selected_dataset_key]
                                        st.session_state.climate_dataset_selected = True
                                        st.session_state.climate_step = 4
                                        st.success(f"✅ Selected dataset: {dataset_options[selected_dataset_key]['name']}")
                                        st.rerun()
                        else:
                            # Display all datasets in a simpler format
                            selected_dataset_key = None
                            
                            for display_name, dataset_info in dataset_options.items():
                                with st.expander(f"📊 {dataset_info['name']}", expanded=False):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Provider:** {dataset_info['provider']}")
                                        st.markdown(f"**Temporal Resolution:** {dataset_info['temporal_resolution']}")
                                        st.markdown(f"**Data Period:** {dataset_info['start_date']} to {dataset_info['end_date']}")
                                        st.markdown(f"**Pixel Size:** {dataset_info['pixel_size']} meters")
                                        st.markdown(f"**Available Bands:** {dataset_info['bands']}")
                                        if dataset_info['band_units']:
                                            st.markdown(f"**Band Units:** {dataset_info['band_units']}")
                                        st.markdown(f"**Description:** {dataset_info['description']}")
                                    
                                    with col2:
                                        if st.button(f"Select Dataset", key=f"select_{display_name.replace(' ', '_')}", type="primary"):
                                            selected_dataset_key = display_name
                            
                            # Handle dataset selection
                            if selected_dataset_key:
                                st.session_state.climate_selected_dataset = dataset_options[selected_dataset_key]
                                st.session_state.climate_dataset_selected = True
                                st.session_state.climate_step = 4
                                st.success(f"✅ Selected dataset: {dataset_options[selected_dataset_key]['name']}")
                                st.rerun()
                                
                    except Exception as e:
                        st.error(f"❌ Error loading {analysis_type} datasets: {str(e)}")
                        st.info("Please check if the CSV files are available in the geoclimate_fetcher/data/climate_index/ directory.")
                        
            except Exception as e:
                st.error(f"❌ Error with geometry handler: {str(e)}")
                
        # Step 4: Date Range and Base Period Selection  
        elif not st.session_state.climate_date_range_set:
            st.markdown('<div class="step-header"><h2>📅 Step 4: Configure Time Parameters</h2></div>', unsafe_allow_html=True)
            
            # Add back button
            if st.button("← Back to Dataset Selection"):
                st.session_state.climate_dataset_selected = False
                st.session_state.climate_step = 3
                st.rerun()
            
            # Show current selections
            dataset = st.session_state.climate_selected_dataset
            st.success(f"✅ Dataset: {dataset['name']} ({dataset['provider']})")
            
            # Get dataset date constraints
            dataset_start = dataset['start_date']
            dataset_end = dataset['end_date']
            
            # Parse dataset dates to help with validation
            try:
                from datetime import datetime
                date_formats = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%Y"]
                
                dataset_start_parsed = None
                dataset_end_parsed = None
                
                for fmt in date_formats:
                    try:
                        dataset_start_parsed = datetime.strptime(dataset_start.strip(), fmt)
                        break
                    except:
                        continue
                
                for fmt in date_formats:
                    try:
                        dataset_end_parsed = datetime.strptime(dataset_end.strip(), fmt)
                        break
                    except:
                        continue
                        
                if dataset_start_parsed:
                    min_date = dataset_start_parsed.date()
                else:
                    min_date = datetime(1980, 1, 1).date()
                    
                if dataset_end_parsed:
                    max_date = dataset_end_parsed.date()
                else:
                    max_date = datetime(2023, 12, 31).date()
                    
            except:
                # Fallback dates
                min_date = datetime(1980, 1, 1).date()
                max_date = datetime(2023, 12, 31).date()
            
            st.info(f"📊 Dataset Coverage: {dataset_start} to {dataset_end}")
            
            with st.form("climate_time_config"):
                st.markdown("### 📅 Analysis Period")
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis_start = st.date_input(
                        "Analysis Start Date",
                        value=datetime(2000, 1, 1).date(),
                        min_value=min_date,
                        max_value=max_date,
                        help="Start date for climate indices calculation"
                    )
                
                with col2:
                    analysis_end = st.date_input(
                        "Analysis End Date",
                        value=datetime(2020, 12, 31).date(),
                        min_value=min_date,
                        max_value=max_date,
                        help="End date for climate indices calculation"
                    )
                
                st.markdown("### 📊 Base Period for Percentiles")
                st.info("The base period is used to calculate percentiles and anomalies. Typically 30 years (e.g., 1961-1990 or 1991-2020).")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    base_start = st.date_input(
                        "Base Period Start",
                        value=datetime(1981, 1, 1).date(),
                        min_value=min_date,
                        max_value=max_date,
                        help="Start of reference period for calculating percentiles"
                    )
                
                with col4:
                    base_end = st.date_input(
                        "Base Period End",
                        value=datetime(2010, 12, 31).date(),
                        min_value=min_date,
                        max_value=max_date,
                        help="End of reference period for calculating percentiles"
                    )
                
                # Validation
                valid = True
                if analysis_start >= analysis_end:
                    st.error("❌ Analysis start date must be before end date")
                    valid = False
                    
                if base_start >= base_end:
                    st.error("❌ Base period start date must be before end date")
                    valid = False
                    
                # Check if base period is reasonable (at least 10 years)
                if valid and (base_end - base_start).days < 3650:
                    st.warning("⚠️ Base period is less than 10 years. For reliable percentiles, consider using at least 20-30 years.")
                
                # Check overlap
                if valid and not (analysis_start <= base_end and analysis_end >= base_start):
                    st.warning("⚠️ Analysis period and base period don't overlap. This is unusual but may be intentional.")
                
                submitted = st.form_submit_button("Confirm Time Parameters", type="primary", disabled=not valid)
                
                if submitted and valid:
                    # Store the dates in session state
                    st.session_state.climate_analysis_start = analysis_start
                    st.session_state.climate_analysis_end = analysis_end
                    st.session_state.climate_base_start = base_start
                    st.session_state.climate_base_end = base_end
                    st.session_state.climate_date_range_set = True
                    st.session_state.climate_step = 5
                    
                    # Calculate analysis period info
                    analysis_years = (analysis_end - analysis_start).days / 365.25
                    base_years = (base_end - base_start).days / 365.25
                    
                    st.success(f"✅ Time parameters configured!")
                    st.info(f"📊 Analysis period: {analysis_years:.1f} years | Base period: {base_years:.1f} years")
                    st.rerun()
        
        # Step 5: Climate Indices Selection
        elif not st.session_state.climate_indices_selected:
            st.markdown('<div class="step-header"><h2>🧮 Step 5: Select Climate Indices</h2></div>', unsafe_allow_html=True)
            
            # Add back button
            if st.button("← Back to Time Parameters"):
                st.session_state.climate_date_range_set = False
                st.session_state.climate_step = 4
                st.rerun()
            
            # Show current selections
            dataset = st.session_state.climate_selected_dataset
            analysis_start = st.session_state.climate_analysis_start
            analysis_end = st.session_state.climate_analysis_end
            base_start = st.session_state.climate_base_start
            base_end = st.session_state.climate_base_end
            
            st.success(f"✅ Dataset: {dataset['name']}")
            st.success(f"✅ Analysis Period: {analysis_start} to {analysis_end}")
            st.success(f"✅ Base Period: {base_start} to {base_end}")
            
            # Define available climate indices based on analysis type
            analysis_type = st.session_state.climate_analysis_type
            
            if analysis_type == "temperature":
                available_indices = {
                    "TXx": {
                        "name": "Monthly Maximum of Daily Maximum Temperature",
                        "description": "Highest daily maximum temperature in each month",
                        "unit": "°C",
                        "category": "extreme_temperature"
                    },
                    "TNn": {
                        "name": "Monthly Minimum of Daily Minimum Temperature", 
                        "description": "Lowest daily minimum temperature in each month",
                        "unit": "°C",
                        "category": "extreme_temperature"
                    },
                    "TX90p": {
                        "name": "Hot Days (TX > 90th percentile)",
                        "description": "Number of days with maximum temperature above 90th percentile",
                        "unit": "days",
                        "category": "percentile_based"
                    },
                    "TN10p": {
                        "name": "Cold Nights (TN < 10th percentile)",
                        "description": "Number of days with minimum temperature below 10th percentile", 
                        "unit": "days",
                        "category": "percentile_based"
                    },
                    "WSDI": {
                        "name": "Warm Spell Duration Index",
                        "description": "Number of days in warm spells (6+ consecutive days with TX > 90th percentile)",
                        "unit": "days",
                        "category": "duration"
                    },
                    "CSDI": {
                        "name": "Cold Spell Duration Index", 
                        "description": "Number of days in cold spells (6+ consecutive days with TN < 10th percentile)",
                        "unit": "days",
                        "category": "duration"
                    },
                    "DTR": {
                        "name": "Diurnal Temperature Range",
                        "description": "Monthly mean difference between daily max and min temperature",
                        "unit": "°C",
                        "category": "variability"
                    },
                    "GSL": {
                        "name": "Growing Season Length",
                        "description": "Number of days between first span of 6+ days with mean temp > 5°C and first span of 6+ days with mean temp < 5°C",
                        "unit": "days",
                        "category": "agricultural"
                    }
                }
            else:  # precipitation
                available_indices = {
                    "PRCPTOT": {
                        "name": "Total Precipitation",
                        "description": "Annual total precipitation on wet days (≥1mm)",
                        "unit": "mm",
                        "category": "amount"
                    },
                    "RX1day": {
                        "name": "Maximum 1-day Precipitation",
                        "description": "Highest precipitation amount in a single day",
                        "unit": "mm",
                        "category": "extreme_precipitation"
                    },
                    "RX5day": {
                        "name": "Maximum 5-day Precipitation",
                        "description": "Highest precipitation amount in 5 consecutive days",
                        "unit": "mm", 
                        "category": "extreme_precipitation"
                    },
                    "R10mm": {
                        "name": "Heavy Precipitation Days (≥10mm)",
                        "description": "Number of days with precipitation ≥10mm",
                        "unit": "days",
                        "category": "threshold"
                    },
                    "R20mm": {
                        "name": "Very Heavy Precipitation Days (≥20mm)",
                        "description": "Number of days with precipitation ≥20mm", 
                        "unit": "days",
                        "category": "threshold"
                    },
                    "CDD": {
                        "name": "Consecutive Dry Days",
                        "description": "Maximum number of consecutive days with precipitation <1mm",
                        "unit": "days",
                        "category": "duration"
                    },
                    "CWD": {
                        "name": "Consecutive Wet Days",
                        "description": "Maximum number of consecutive days with precipitation ≥1mm",
                        "unit": "days",
                        "category": "duration"
                    },
                    "SDII": {
                        "name": "Simple Daily Intensity Index",
                        "description": "Average precipitation on wet days",
                        "unit": "mm/day",
                        "category": "intensity"
                    }
                }
            
            # Group indices by category
            categories = {}
            for idx_code, idx_info in available_indices.items():
                category = idx_info["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append((idx_code, idx_info))
            
            # Display indices selection by category
            st.markdown(f"### Available {analysis_type.title()} Indices")
            st.info("Select the climate indices you want to calculate. You can select multiple indices from different categories.")
            
            selected_indices = []
            
            for category, indices in categories.items():
                with st.expander(f"📊 {category.replace('_', ' ').title()} Indices", expanded=True):
                    for idx_code, idx_info in indices:
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if st.checkbox(f"**{idx_code}**", key=f"idx_{idx_code}"):
                                selected_indices.append(idx_code)
                        
                        with col2:
                            st.markdown(f"**{idx_info['name']}**")
                            st.markdown(f"*{idx_info['description']}*")
                            st.markdown(f"📏 Unit: {idx_info['unit']}")
            
            # Show selection summary
            if selected_indices:
                st.markdown("### 📋 Selected Indices")
                for idx in selected_indices:
                    st.markdown(f"• **{idx}**: {available_indices[idx]['name']}")
                
                # Confirm selection
                if st.button("✅ Confirm Index Selection", type="primary", use_container_width=True):
                    st.session_state.climate_selected_indices = selected_indices
                    st.session_state.climate_indices_metadata = {idx: available_indices[idx] for idx in selected_indices}
                    st.session_state.climate_indices_selected = True
                    st.session_state.climate_step = 6
                    st.success(f"✅ Selected {len(selected_indices)} climate indices for calculation!")
                    st.rerun()
            else:
                st.warning("Please select at least one climate index to continue.")
        
        # Step 6: Calculation and Results
        else:
            st.markdown('<div class="step-header"><h2>📈 Step 6: Calculate and View Results</h2></div>', unsafe_allow_html=True)
            
            # Add back button
            if st.button("← Back to Index Selection"):
                st.session_state.climate_indices_selected = False
                st.session_state.climate_step = 5
                st.rerun()
            
            # Show summary of selections
            dataset = st.session_state.climate_selected_dataset
            selected_indices = st.session_state.climate_selected_indices
            indices_metadata = st.session_state.climate_indices_metadata
            
            st.markdown("### 📋 Calculation Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Configuration:**")
                st.write(f"• **Analysis Type:** {st.session_state.climate_analysis_type.title()}")
                st.write(f"• **Dataset:** {dataset['name']}")
                st.write(f"• **Analysis Period:** {st.session_state.climate_analysis_start} to {st.session_state.climate_analysis_end}")
                st.write(f"• **Base Period:** {st.session_state.climate_base_start} to {st.session_state.climate_base_end}")
            
            with col2:
                st.markdown("**Selected Indices:**")
                for idx in selected_indices:
                    st.write(f"• **{idx}**: {indices_metadata[idx]['name']}")
            
            # Calculate button
            if st.button("🚀 Start Calculation", type="primary", use_container_width=True):
                try:
                    with st.spinner("Initializing calculation..."):
                        # Initialize calculator
                        geometry = st.session_state.climate_geometry_handler.current_geometry
                        
                        # For now, show a placeholder for the actual calculation
                        st.success("🎉 Calculation would start here!")
                        
                        # Create placeholder results
                        st.markdown("### 📊 Results")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        import time
                        
                        for i, idx in enumerate(selected_indices):
                            status_text.text(f"Calculating {indices_metadata[idx]['name']}...")
                            progress_bar.progress((i + 1) / len(selected_indices))
                            time.sleep(1)  # Simulate calculation time
                            
                            # Create a placeholder result visualization
                            with st.expander(f"📈 {indices_metadata[idx]['name']} ({idx})", expanded=True):
                                st.info(f"Calculation for {idx} completed successfully!")
                                st.markdown(f"**Description:** {indices_metadata[idx]['description']}")
                                st.markdown(f"**Unit:** {indices_metadata[idx]['unit']}")
                                
                                # Placeholder chart
                                import numpy as np
                                import pandas as pd
                                import plotly.graph_objects as go
                                
                                # Generate sample data
                                dates = pd.date_range(
                                    st.session_state.climate_analysis_start,
                                    st.session_state.climate_analysis_end,
                                    freq='M'
                                )
                                values = np.random.normal(50, 10, len(dates))
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates,
                                    y=values,
                                    mode='lines+markers',
                                    name=indices_metadata[idx]['name']
                                ))
                                
                                fig.update_layout(
                                    title=f"{indices_metadata[idx]['name']} ({idx})",
                                    xaxis_title="Date",
                                    yaxis_title=f"{indices_metadata[idx]['unit']}",
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download button
                                csv_data = pd.DataFrame({'Date': dates, f'{idx}': values}).to_csv(index=False)
                                st.download_button(
                                    f"📥 Download {idx} Data",
                                    csv_data,
                                    f"{idx}_data.csv",
                                    "text/csv",
                                    key=f"download_{idx}"
                                )
                        
                        status_text.text("✅ All calculations completed!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error during calculation: {str(e)}")
                    st.info("This is a demonstration. Full calculation functionality would be implemented here.")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("� Start Over", help="Reset all selections and start from the beginning"):
        # Clear all climate-related session state
        keys_to_clear = [key for key in st.session_state.keys() if key.startswith('climate_')]
        for key in keys_to_clear:
            del st.session_state[key]
        st.session_state.climate_step = 1
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