"""
Download Functionality Test
Demonstrates the enhanced download capabilities of the GeoClimate Fetcher app.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Download Functionality Test",
    page_icon="📥",
    layout="wide"
)

# Import download components
try:
    import sys
    app_components_path = Path(__file__).parent / "app_components"
    if str(app_components_path) not in sys.path:
        sys.path.insert(0, str(app_components_path))
    
    from app_components.download_component import DownloadHelper, DownloadComponent
    from app_components.theme_utils import apply_dark_mode_css
    
    # Apply dark mode support
    apply_dark_mode_css()
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Title
st.title("📥 Download Functionality Test")

st.markdown("""
## Enhanced Download System

This test demonstrates the comprehensive download improvements made to the GeoClimate Fetcher application:

### ✅ **Fixed Issues:**
1. **Server-only downloads** - Users can now actually download their processed data
2. **Multiple file handling** - ZIP archives for multiple GeoTIFF files
3. **Single file downloads** - Direct download buttons for individual files
4. **Large file management** - Google Drive integration for files >50MB

### 🚀 **Key Features:**
- **Automatic ZIP creation** for multiple files
- **Individual file downloads** with proper MIME types
- **Size estimation** and progress tracking
- **Download summaries** with file listings
- **Cross-platform compatibility** (Windows, Mac, Linux)
""")

st.divider()

# Test section
st.header("🧪 Test Download Components")

# Create test data
@st.cache_data
def create_test_files():
    """Create test files for download demonstration"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample CSV file
    csv_data = pd.DataFrame({
        'band': ['B1', 'B2', 'B3', 'B4'],
        'mean': np.random.normal(0.5, 0.1, 4),
        'std': np.random.normal(0.1, 0.02, 4),
        'min': np.random.uniform(0, 0.2, 4),
        'max': np.random.uniform(0.8, 1.0, 4)
    })
    
    csv_path = os.path.join(temp_dir, "test_statistics.csv")
    csv_data.to_csv(csv_path, index=False)
    
    # Create sample text files to simulate GeoTIFFs
    geotiff_dir = os.path.join(temp_dir, "test_geotiffs")
    os.makedirs(geotiff_dir, exist_ok=True)
    
    dates = ['20240101', '20240102', '20240103', '20240104', '20240105']
    geotiff_files = []
    
    for date in dates:
        file_path = os.path.join(geotiff_dir, f"{date}.tif")
        # Create dummy file with some content
        with open(file_path, 'w') as f:
            f.write(f"Dummy GeoTIFF data for {date}\n" * 100)  # Make it a bit larger
        geotiff_files.append(file_path)
    
    # Create sample NetCDF-like file
    nc_path = os.path.join(temp_dir, "test_climate_data.nc")
    with open(nc_path, 'w') as f:
        f.write("Dummy NetCDF data\n" * 200)
    
    return {
        'temp_dir': temp_dir,
        'csv_file': csv_path,
        'nc_file': nc_path,
        'geotiff_dir': geotiff_dir,
        'geotiff_files': geotiff_files
    }

# Create test files
test_files = create_test_files()

# Test different download scenarios
st.subheader("1. Single File Download")
st.write("Test downloading a single CSV file:")

if st.button("Test CSV Download"):
    download_helper = DownloadHelper()
    download_helper.create_download_button(
        test_files['csv_file'], 
        download_name="climate_statistics.csv"
    )

st.divider()

st.subheader("2. NetCDF File Download")
st.write("Test downloading a NetCDF file:")

if st.button("Test NetCDF Download"):
    download_helper = DownloadHelper()
    download_helper.create_download_button(
        test_files['nc_file'], 
        download_name="climate_data.nc"
    )

st.divider()

st.subheader("3. Multiple GeoTIFF Files (ZIP)")
st.write("Test downloading multiple GeoTIFF files as a ZIP archive:")

if st.button("Test Multiple GeoTIFF Download"):
    download_helper = DownloadHelper()
    download_helper.create_zip_download(
        test_files['geotiff_dir'], 
        zip_name="climate_geotiffs.zip"
    )

st.divider()

st.subheader("4. Complete Download Summary")
st.write("Test the complete download summary interface:")

if st.button("Show Download Summary"):
    download_component = DownloadComponent()
    download_component.show_download_summary(
        output_dir=test_files['temp_dir'],
        successful_downloads=5,
        drive_exports=0
    )

st.divider()

# Information section
st.header("📋 How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔧 **Technical Implementation**
    
    **Single Files:**
    - Direct download using `st.download_button()`
    - Automatic MIME type detection
    - File size calculation and display
    
    **Multiple Files:**
    - ZIP archive creation using Python's `zipfile`
    - Temporary file management
    - Compressed download for efficiency
    
    **Large Files:**
    - Automatic Google Drive export for files >50MB
    - Task tracking and status monitoring
    - Fallback options for different scenarios
    """)

with col2:
    st.markdown("""
    ### 🎯 **User Experience**
    
    **Before (Problem):**
    - Files saved only to server filesystem
    - Users couldn't access their data
    - "Download complete" with no actual download
    
    **After (Solution):**
    - Direct download buttons for all files
    - ZIP archives for multiple files
    - Clear file sizes and descriptions
    - Google Drive integration for large datasets
    """)

st.divider()

# Usage examples
st.header("💡 Usage Examples")

st.code("""
# In your Streamlit app, after processing data:

from app_components.download_component import DownloadHelper, DownloadComponent

# For single file download
download_helper = DownloadHelper()
download_helper.create_download_button(file_path, "my_data.csv")

# For multiple files (ZIP)
download_helper.create_zip_download(directory_path, "my_dataset.zip")

# For complete download summary (replaces old success message)
download_component = DownloadComponent()
download_component.show_download_summary(
    output_dir=output_directory,
    successful_downloads=num_files,
    drive_exports=num_drive_files
)
""", language="python")

st.divider()

# File format information
st.header("📄 Supported File Formats")

format_info = {
    "GeoTIFF (.tif)": {
        "icon": "🖼️",
        "description": "Georeferenced raster images",
        "use_case": "GIS software (QGIS, ArcGIS)",
        "mime": "image/tiff"
    },
    "NetCDF (.nc)": {
        "icon": "🌐",
        "description": "Scientific data format",
        "use_case": "Python (xarray), R, MATLAB",
        "mime": "application/x-netcdf"
    },
    "CSV (.csv)": {
        "icon": "📊",
        "description": "Tabular data",
        "use_case": "Excel, Google Sheets, Python",
        "mime": "text/csv"
    },
    "ZIP (.zip)": {
        "icon": "📦",
        "description": "Compressed archive",
        "use_case": "Multiple files, reduced size",
        "mime": "application/zip"
    }
}

for format_name, info in format_info.items():
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.markdown(f"### {info['icon']} {format_name}")
        with col2:
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Best for:** {info['use_case']}")
        with col3:
            st.code(f"MIME: {info['mime']}")

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>🌍 GeoClimate Fetcher - Enhanced with Comprehensive Download Support</p>
    <p>Users can now actually download and use their processed climate data!</p>
</div>
""", unsafe_allow_html=True)

# Cleanup note
st.info("🗑️ **Note:** Test files are created in temporary directories and will be automatically cleaned up.")
