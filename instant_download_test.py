"""
Instant Download Test
Tests the enhanced download functionality that automatically triggers browser save dialog.
"""

import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="Instant Download Test",
    page_icon="📥",
    layout="wide"
)

# Import components
try:
    import sys
    app_components_path = Path(__file__).parent / "app_components"
    if str(app_components_path) not in sys.path:
        sys.path.insert(0, str(app_components_path))
    
    from app_components.download_component import DownloadHelper
    from app_components.theme_utils import apply_dark_mode_css
    
    # Apply dark mode support
    apply_dark_mode_css()
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Title
st.title("📥 Instant Download Test")

st.markdown("""
## Enhanced Single File Downloads

This test demonstrates the new **instant download** functionality for single files (CSV, NetCDF, GeoTIFF).

### ✨ **Key Features:**
- **Automatic browser save dialog** - Prompts user to choose download location
- **Enhanced UI** - Professional download interface with progress feedback
- **Immediate trigger** - Downloads start immediately when button is clicked
- **Success feedback** - Clear indication when download has started
- **Cross-browser compatibility** - Works with all modern browsers

### 🎯 **Use Cases:**
- **CSV files** - Statistical data and time series
- **NetCDF files** - Scientific climate data
- **Single GeoTIFF files** - Individual raster images
""")

st.divider()

# Create test data
@st.cache_data
def create_test_files():
    """Create sample files for testing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample CSV file
    csv_data = pd.DataFrame({
        'band': ['B1_Temperature', 'B2_Humidity', 'B3_Pressure', 'B4_Wind_Speed'],
        'mean': [25.4, 68.2, 1013.5, 5.8],
        'std': [3.2, 12.1, 15.2, 2.1],
        'min': [18.1, 35.0, 980.3, 0.5],
        'max': [32.8, 95.0, 1045.2, 12.3],
        'count': [1000, 1000, 1000, 1000]
    })
    
    csv_path = os.path.join(temp_dir, "climate_statistics.csv")
    csv_data.to_csv(csv_path, index=False)
    
    # Create sample NetCDF-like file (simulation)
    nc_content = """netcdf climate_data {
dimensions:
    lat = 50 ;
    lon = 100 ;
    time = UNLIMITED ; // (365 currently)
variables:
    float temperature(time, lat, lon) ;
        temperature:units = "degrees_Celsius" ;
        temperature:long_name = "Air Temperature" ;
    float precipitation(time, lat, lon) ;
        precipitation:units = "mm/day" ;
        precipitation:long_name = "Daily Precipitation" ;
    double lat(lat) ;
        lat:units = "degrees_north" ;
    double lon(lon) ;
        lon:units = "degrees_east" ;
    int time(time) ;
        time:units = "days since 2024-01-01 00:00:00" ;

// global attributes:
        :title = "Climate Data from GeoClimate Fetcher" ;
        :source = "Google Earth Engine" ;
        :created_by = "GeoClimate Fetcher Application" ;
        :creation_date = "2025-07-05" ;
data:
    lat = 35.0, 35.1, 35.2, ... ;
    lon = -120.0, -119.9, -119.8, ... ;
    time = 0, 1, 2, 3, 4, 5, ... ;
}"""
    
    nc_path = os.path.join(temp_dir, "climate_data.nc")
    with open(nc_path, 'w') as f:
        f.write(nc_content)
    
    # Create sample GeoTIFF metadata file
    tiff_content = """TIFF Image Metadata:
Width: 1000 pixels
Height: 800 pixels
Bands: 4 (Red, Green, Blue, NIR)
Data Type: Float32
Coordinate System: EPSG:4326 (WGS84)
Pixel Size: 30 meters
Extent: [-120.5, 35.0, -119.5, 36.0]
Created: 2025-07-05
Source: Google Earth Engine
Dataset: Sentinel-2 Surface Reflectance

Band Information:
- Band 1 (Red): 620-670 nm
- Band 2 (Green): 540-580 nm  
- Band 3 (Blue): 450-520 nm
- Band 4 (NIR): 770-900 nm

Processing:
- Cloud masked
- Atmospheric correction applied
- Temporal composite (median)
"""
    
    tiff_path = os.path.join(temp_dir, "satellite_image.tif")
    with open(tiff_path, 'w') as f:
        f.write(tiff_content)
    
    return {
        'temp_dir': temp_dir,
        'csv_file': csv_path,
        'nc_file': nc_path,
        'tiff_file': tiff_path
    }

# Create test files
test_files = create_test_files()
download_helper = DownloadHelper()

# Test sections
st.header("🧪 Test Instant Downloads")

# CSV Test
st.subheader("1. 📊 CSV File Download")
st.write("**Climate Statistics CSV** - Contains statistical analysis of climate bands")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("**File contains:**")
    st.write("- Temperature, Humidity, Pressure, Wind Speed statistics")
    st.write("- Mean, Standard Deviation, Min/Max values")
    st.write("- Sample count for each measurement")
    
with col2:
    file_size = os.path.getsize(test_files['csv_file']) / 1024  # KB
    st.metric("File Size", f"{file_size:.1f} KB")
    st.metric("Format", "CSV")

if st.button("Test CSV Instant Download", key="csv_test"):
    download_helper.create_instant_download(test_files['csv_file'])

st.divider()

# NetCDF Test
st.subheader("2. 🌐 NetCDF File Download")
st.write("**Climate Data NetCDF** - Multi-dimensional scientific data format")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("**File contains:**")
    st.write("- Temperature and precipitation grids")
    st.write("- Time series data (365 days)")
    st.write("- Geospatial coordinates and metadata")
    
with col2:
    file_size = os.path.getsize(test_files['nc_file']) / 1024  # KB
    st.metric("File Size", f"{file_size:.1f} KB")
    st.metric("Format", "NetCDF")

if st.button("Test NetCDF Instant Download", key="nc_test"):
    download_helper.create_instant_download(test_files['nc_file'])

st.divider()

# GeoTIFF Test
st.subheader("3. 🖼️ GeoTIFF File Download")
st.write("**Satellite Image GeoTIFF** - Georeferenced raster image")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("**File contains:**")
    st.write("- Multi-band satellite imagery (RGB + NIR)")
    st.write("- Geospatial reference information")
    st.write("- Processed and cloud-masked data")
    
with col2:
    file_size = os.path.getsize(test_files['tiff_file']) / 1024  # KB
    st.metric("File Size", f"{file_size:.1f} KB")
    st.metric("Format", "GeoTIFF")

if st.button("Test GeoTIFF Instant Download", key="tiff_test"):
    download_helper.create_instant_download(test_files['tiff_file'])

st.divider()

# Comparison section
st.header("🔄 Before vs After Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ❌ **Before (Old Method)**
    
    **Process:**
    1. User processes data ✅
    2. System shows "Download complete" ❌
    3. Files saved to server only ❌
    4. User can't access their data ❌
    
    **Issues:**
    - No actual download
    - Files stuck on server
    - Poor user experience
    - Manual file retrieval needed
    """)

with col2:
    st.markdown("""
    ### ✅ **After (Instant Download)**
    
    **Process:**
    1. User processes data ✅
    2. Instant download interface appears ✅
    3. User clicks download button ✅
    4. Browser save dialog opens ✅
    5. User chooses location and saves ✅
    
    **Benefits:**
    - Immediate file access
    - Native browser experience
    - User controls save location
    - Professional interface
    """)

st.divider()

# Technical details
st.header("⚙️ Technical Implementation")

st.markdown("""
### 🛠️ **How It Works**

**JavaScript + Data URIs:**
```javascript
function startDownload() {
    const link = document.createElement('a');
    link.href = 'data:text/csv;base64,' + base64Data;
    link.download = 'filename.csv';
    link.click(); // Triggers browser save dialog
}
```

**Python Integration:**
```python
def create_instant_download(self, file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    b64 = base64.b64encode(file_data).decode()
    # Generate HTML with embedded JavaScript
```

**Browser Compatibility:**
- ✅ Chrome/Chromium: Full support with save dialog
- ✅ Firefox: Native download with location choice
- ✅ Safari: Automatic downloads folder or user choice
- ✅ Edge: Full save dialog functionality
- ✅ Mobile: Downloads to device storage
""")

# Usage tips
with st.expander("💡 Usage Tips for Downloaded Files"):
    st.markdown("""
    **CSV Files:**
    - Open in Excel, Google Sheets, or any spreadsheet software
    - Import into Python with `pandas.read_csv()`
    - Use for statistical analysis and data visualization
    
    **NetCDF Files:**
    - Open with Python: `import xarray as xr; ds = xr.open_dataset('file.nc')`
    - Use in R with ncdf4 package
    - Compatible with MATLAB, IDL, and other scientific software
    - View metadata with `ncdump -h filename.nc`
    
    **GeoTIFF Files:**
    - Open in QGIS (free GIS software)
    - Use in ArcGIS or other GIS applications
    - Python: `import rasterio; raster = rasterio.open('file.tif')`
    - View with standard image viewers (may not show georeferencing)
    """)

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>🌍 GeoClimate Fetcher - Enhanced with Instant Download Technology</p>
    <p>Professional file delivery with native browser integration</p>
</div>
""", unsafe_allow_html=True)
