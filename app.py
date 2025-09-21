import streamlit as st

# Configure Streamlit page - MUST be first Streamlit command
st.set_page_config(
    page_title="GeoClimate Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/SauravBhattarai19/GeoClimate-Fetcher',
        'Report a bug': "https://github.com/SauravBhattarai19/GeoClimate-Fetcher/issues",
        'About': "# GeoClimate Intelligence Platform\nA comprehensive platform for Earth Engine climate data analysis and intelligence!"
    }
)

# Now import the rest of the modules
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
from plotly.subplots import make_subplots
import numpy as np
import xarray as xr
from geoclimate_fetcher.climate_indices import ClimateIndicesCalculator
import hashlib
from pathlib import Path

# Try to import cookie manager, fallback if not available
try:
    import extra_streamlit_components as stx
    cookie_manager = stx.CookieManager()
    COOKIES_AVAILABLE = True
except ImportError:
    # Fallback cookie manager that uses session state
    class FallbackCookieManager:
        def get(self, cookie):
            return st.session_state.get(f"cookie_{cookie}", None)
        
        def set(self, key, value, expires_at=None):
            st.session_state[f"cookie_{key}"] = value
        
        def delete(self, key):
            if f"cookie_{key}" in st.session_state:
                del st.session_state[f"cookie_{key}"]
    
    cookie_manager = FallbackCookieManager()
    COOKIES_AVAILABLE = False

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
    try:
        auth_cookie = cookie_manager.get(cookie="gee_auth_token")
        project_cookie = cookie_manager.get(cookie="gee_project_id")
        
        if auth_cookie and project_cookie:
            if validate_auth_token(auth_cookie, project_cookie):
                return project_cookie
    except Exception as e:
        # Handle any cookie-related errors gracefully
        print(f"Error checking stored auth: {str(e)}")
    return None

# Function to authenticate GEE and store credentials
def authenticate_gee(project_id, service_account=None, key_file=None, auth_token=None):
    """Authenticate with Google Earth Engine and store credentials"""
    try:
        from geoclimate_fetcher.core import authenticate
        auth = authenticate(project_id, service_account, key_file, auth_token)
        if auth.is_initialized():
            st.session_state.auth_complete = True
            st.session_state.project_id = project_id
            # Store authentication in cookies
            auth_token = create_auth_token(project_id, int(time.time() // 86400))
            try:
                cookie_manager.set("gee_auth_token", auth_token, expires_at=datetime.now() + timedelta(days=30))
                cookie_manager.set("gee_project_id", project_id, expires_at=datetime.now() + timedelta(days=30))
            except Exception as e:
                print(f"Warning: Could not set cookies: {str(e)}")
            return True, "Authentication successful!"
        else:
            return False, "Authentication failed. Please check your credentials."
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"


# Clear stored authentication (for logout functionality)
def clear_authentication():
    """Clear all authentication data"""
    # Clear session state
    st.session_state.auth_complete = False
    st.session_state.project_id = None
    
    # Clear cookies
    try:
        cookie_manager.delete('gee_auth_token')
        cookie_manager.delete('gee_project_id')
    except Exception as e:
        print(f"Error clearing cookies: {str(e)}")

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
    StaticRasterFetcher,
    GeometrySelectionWidget
)

# Utility functions moved to app_utils.py to avoid circular imports

# Initialize session state
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = None  # None, 'data_explorer', 'climate_analytics', 'hydrology'
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

# Climate Analytics session state
if 'climate_step' not in st.session_state:
    st.session_state.climate_step = 1
if 'climate_analysis_type' not in st.session_state:
    st.session_state.climate_analysis_type = None
if 'climate_geometry_complete' not in st.session_state:
    st.session_state.climate_geometry_complete = False
if 'climate_dataset_selected' not in st.session_state:
    st.session_state.climate_dataset_selected = False
if 'climate_date_range_set' not in st.session_state:
    st.session_state.climate_date_range_set = False
if 'climate_indices_selected' not in st.session_state:
    st.session_state.climate_indices_selected = False
if 'climate_geometry_handler' not in st.session_state:
    st.session_state.climate_geometry_handler = GeometryHandler()

# Hydrology-specific session state
if 'hydro_geometry_complete' not in st.session_state:
    st.session_state.hydro_geometry_complete = False
if 'hydro_dataset_selected' not in st.session_state:
    st.session_state.hydro_dataset_selected = False
if 'hydro_dates_selected' not in st.session_state:
    st.session_state.hydro_dates_selected = False
if 'hydro_current_dataset' not in st.session_state:
    st.session_state.hydro_current_dataset = None
if 'hydro_precipitation_data' not in st.session_state:
    st.session_state.hydro_precipitation_data = None
if 'hydro_analysis_results' not in st.session_state:
    st.session_state.hydro_analysis_results = {}


# =====================
# MAIN APP FLOW - LOGIN FIRST APPROACH
# =====================

# Check if user is authenticated - if not, show login page
if not st.session_state.get('auth_complete', False):
    # Check for stored authentication first
    stored_project_id = check_stored_auth()
    
    if stored_project_id:
        # Try to authenticate with stored credentials silently
        try:
            success, message = authenticate_gee(stored_project_id)
            if success:
                st.success("✅ Welcome back! Authenticated successfully.")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            # Stored auth failed, show login page
            print(f"Stored authentication failed: {str(e)}")
    
    # Show dedicated login page
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
         border-radius: 20px; color: white; margin: 0.5rem 0 0.75rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        <div style="font-size: 3rem; font-weight: bold; margin-bottom: 1rem;">🌍</div>
        <h1 style="margin: 0 0 1rem 0;">GeoClimate Intelligence Platform</h1>
        <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">Please authenticate to access the platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Use the AuthComponent for authentication
    from app_components.auth_component import AuthComponent

    auth_component = AuthComponent()
    if auth_component.render():
        # Authentication completed successfully
        st.success("🎉 Authentication successful! Welcome to the GeoClimate Intelligence Platform!")
        st.info("🔄 Loading platform...")
        time.sleep(2)
        st.rerun()

    # Platform info before author section
    st.markdown("---")
    st.info("🌍 **Authenticate once to access all platform tools:** GEE Data Explorer, Climate Intelligence Hub, Hydrology Analyzer, Product Selector, and Data Visualizer. Your session will be remembered for seamless analysis workflows.")

    # Author Section on Login Page
    st.markdown("---")
    st.markdown('<h3>👨‍💻 About the Developer</h3>', unsafe_allow_html=True)

    # Create two columns for photo and info
    col_photo, col_info = st.columns([1, 3])

    with col_photo:
        # Display developer photo
        try:
            st.image("pictures/Saurav.png", width=120, caption="Saurav Bhattarai")
        except:
            # Fallback if image not found
            st.markdown("**👨‍💻 Saurav Bhattarai**")

    with col_info:
        st.markdown("""
        <div class="developer-info" style="padding-left: 20px;">
            <h4><strong>Saurav Bhattarai</strong></h4>
            <p><strong>Civil Engineer & Geospatial Developer</strong></p>
            <p>Under Supervision of Dr. Rocky Talchabhadel and Dr. Nawaraj Pradhan</p>
            <p>📧 Email: <a href="mailto:saurav.bhattarai.1999@gmail.com">saurav.bhattarai.1999@gmail.com</a></p>
            <p>🌐 Website: <a href="https://sauravbhattarai19.github.io/" target="_blank">sauravbhattarai19.github.io</a></p>
            <p>🔗 <a href="https://github.com/sauravbhattarai19" target="_blank">GitHub</a> |
               <a href="https://www.linkedin.com/in/saurav-bhattarai-7133a3176/" target="_blank">LinkedIn</a></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .login-acknowledgments {
        margin-top: 15px;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Light mode styles */
    @media (prefers-color-scheme: light) {
        .login-acknowledgments {
            background-color: rgba(255,255,255,0.8);
            color: #333;
            border-color: rgba(0,0,0,0.1);
        }

        .login-acknowledgments a {
            color: #1f77b4;
        }

        .login-acknowledgments a:hover {
            color: #0066cc;
        }
    }

    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .login-acknowledgments {
            background-color: rgba(255,255,255,0.1);
            color: #f0f0f0;
            border-color: rgba(255,255,255,0.2);
        }

        .login-acknowledgments a {
            color: #87ceeb;
        }

        .login-acknowledgments a:hover {
            color: #add8e6;
        }
    }

    /* Default fallback for systems without preference */
    .login-acknowledgments {
        background-color: rgba(248, 249, 250, 0.9);
        color: #495057;
    }

    .login-acknowledgments a {
        color: #007bff;
        text-decoration: none;
    }

    .login-acknowledgments a:hover {
        color: #0056b3;
        text-decoration: underline;
    }
    </style>

    <div class="login-acknowledgments">
        <h5>🙏 Acknowledgments</h5>
        <p><small>Built with ❤️ using Google Earth Engine, Streamlit, and Python</small></p>
        <p><small>Development assistance provided by Claude AI (Anthropic)</small></p>
        <p><small>📖 <a href="https://github.com/SauravBhattarai19/GeoClimate-Fetcher" target="_blank">Documentation & Source Code</a></small></p>
    </div>
    """, unsafe_allow_html=True)

    # Stop here until authentication is complete
    st.stop()

# =====================
# AUTHENTICATED USER FLOW
# =====================
# If we reach here, user is authenticated

# Global Navigation Bar - Available across all tools
from app_components.global_navigation import render_global_navigation
render_global_navigation(clear_authentication)

# Landing Page (Now only shown to authenticated users)
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
    
    # First row of tools
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
    
    # Second row of tools
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="tool-card">
            <span class="tool-icon">💧</span>
            <div class="tool-title">Hydrology Analyzer</div>
            <div class="tool-description">
                Comprehensive precipitation analysis with return periods, frequency analysis, IDF curves, and drought indices. 
                Perfect for hydrology students and professionals.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Hydrology Analyzer", use_container_width=True, type="primary"):
            st.session_state.app_mode = "hydrology"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="tool-card">
            <span class="tool-icon">🎯</span>
            <div class="tool-title">Optimal Product Selector</div>
            <div class="tool-description">
                Compare meteostat station data with multiple gridded datasets to find the best data source. 
                Comprehensive statistical analysis with daily, monthly, yearly, seasonal, and extreme value comparisons.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Product Selector", use_container_width=True, type="primary"):
            st.session_state.app_mode = "product_selector"
            st.rerun()

    # Third row - centered tool
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing

    col_center = st.columns([1, 2, 1])  # Create centered layout

    with col_center[1]:  # Use middle column for centering
        st.markdown("""
        <div class="tool-card">
            <span class="tool-icon">📊</span>
            <div class="tool-title">Universal Data Visualizer</div>
            <div class="tool-description">
                Upload and visualize your downloaded data from any module. Supports CSV time series analysis
                and TIFF spatial visualization with interactive charts and statistical summaries.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Launch Data Visualizer", use_container_width=True, type="primary"):
            st.session_state.app_mode = "data_visualizer"
            st.rerun()

    # Author Section
    st.markdown('<h3>👨‍💻 About the Developer</h3>', unsafe_allow_html=True)

    # Create two columns for photo and info
    col_photo, col_info = st.columns([1, 3])

    with col_photo:
        # Display developer photo
        st.image("pictures/Saurav.png", width=150, caption="Saurav Bhattarai")

    with col_info:
        st.markdown("""
        <div class="developer-info" style="padding-left: 20px;">
            <h3><strong>Saurav Bhattarai</strong></h3>
            <p><strong>Civil Engineer & Geospatial Developer</strong></p>
            <p>Under Supervision of Dr. Rocky Talchabhadel and Dr. Nawaraj Pradhan</p>
            <p>📧 Email: <a href="mailto:saurav.bhattarai.1999@gmail.com">saurav.bhattarai.1999@gmail.com</a></p>
            <p>🌐 Website: <a href="https://sauravbhattarai19.github.io/" target="_blank">sauravbhattarai19.github.io</a></p>
            <p>🔗 <a href="https://github.com/sauravbhattarai19" target="_blank">GitHub</a> |
               <a href="https://www.linkedin.com/in/saurav-bhattarai-7133a3176/" target="_blank">LinkedIn</a></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="acknowledgments" style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 10px;">
        <h4>🙏 Acknowledgments</h4>
        <p><small>Built with ❤️ using Google Earth Engine, Streamlit, and Python</small></p>
        <p><small>Development assistance provided by Claude AI (Anthropic)</small></p>
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
        
        **💧 Hydrology Analyzer**
        1. Authenticate with Google Earth Engine
        2. Define study area and select precipitation dataset
        3. Set analysis period
        4. Run comprehensive precipitation analysis
        5. View return periods, IDF curves, and drought indices
        
        ### 🔑 Requirements
        - Google Earth Engine account ([Sign up free](https://earthengine.google.com/signup/))
        - Basic understanding of climate data
        - Internet connection for cloud processing
        """)

# Route to appropriate interface module
elif st.session_state.app_mode is not None:
    # Import interface router
    from interface.router import route_to_interface
    # Route to the appropriate interface module
    
    route_to_interface()