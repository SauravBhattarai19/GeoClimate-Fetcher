"""
Test script for automatic download functionality in GeoClimate Fetcher
This demonstrates how CSV and NetCDF files now automatically trigger browser save dialogs.
"""

import streamlit as st
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Automatic Download Test",
    page_icon="📥",
    layout="wide"
)

st.title("🧪 Automatic Download Test")
st.markdown("This test demonstrates the new automatic download functionality for single files (CSV/NetCDF).")

# Add the app_components to path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app_components.download_component import DownloadHelper

def create_test_csv():
    """Create a test CSV file"""
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.uniform(30, 80, 100),
        'precipitation': np.random.exponential(2, 100)
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def create_test_netcdf():
    """Create a test NetCDF file"""
    try:
        import xarray as xr
        
        # Generate sample gridded data
        lats = np.linspace(-90, 90, 180)
        lons = np.linspace(-180, 180, 360)
        time = pd.date_range('2023-01-01', periods=12, freq='M')
        
        # Create random temperature data
        temp_data = np.random.normal(15, 10, (12, 180, 360))
        
        # Create xarray dataset
        ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'lat': lats,
            'lon': lons
        })
        
        # Add metadata
        ds.attrs.update({
            'title': 'Test Climate Data',
            'source': 'GeoClimate Fetcher Test',
            'created': pd.Timestamp.now().isoformat()
        })
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
        ds.to_netcdf(temp_file.name)
        return temp_file.name
        
    except ImportError:
        st.warning("⚠️ xarray not available - NetCDF test skipped")
        return None

# Main test interface
st.markdown("### 📥 Test Automatic Downloads")
st.markdown("Click the buttons below to test automatic downloads. The browser should immediately prompt you to save the file.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 CSV Test")
    st.markdown("This will create a sample CSV file with climate data and trigger an automatic download.")
    
    if st.button("🧪 Test CSV Download", type="primary"):
        with st.spinner("Creating test CSV file..."):
            csv_file = create_test_csv()
            
        st.success("✅ CSV file created!")
        
        # Use automatic download
        download_helper = DownloadHelper()
        download_helper.create_automatic_download(
            csv_file,
            download_name="test_climate_data.csv",
            show_success=False
        )
        
        st.info("👆 The download button above should immediately trigger your browser's save dialog when clicked.")
        
        # Cleanup
        try:
            os.unlink(csv_file)
        except:
            pass

with col2:
    st.subheader("🌐 NetCDF Test")
    st.markdown("This will create a sample NetCDF file with gridded climate data and trigger an automatic download.")
    
    if st.button("🧪 Test NetCDF Download", type="primary"):
        with st.spinner("Creating test NetCDF file..."):
            nc_file = create_test_netcdf()
            
        if nc_file:
            st.success("✅ NetCDF file created!")
            
            # Use automatic download
            download_helper = DownloadHelper()
            download_helper.create_automatic_download(
                nc_file,
                download_name="test_climate_grid.nc",
                show_success=False
            )
            
            st.info("👆 The download button above should immediately trigger your browser's save dialog when clicked.")
            
            # Cleanup
            try:
                os.unlink(nc_file)
            except:
                pass
        else:
            st.error("❌ Could not create NetCDF test file")

# Comparison section
st.markdown("---")
st.markdown("### 🔄 Comparison: Instant vs Automatic Downloads")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📋 Previous: Instant Download")
    st.markdown("""
    - Shows a styled download interface
    - Requires user to click a "Download Now" button
    - Updates UI after download starts
    - Good for GeoTIFF files where users need to see file info
    """)

with col2:
    st.markdown("#### ⚡ New: Automatic Download")
    st.markdown("""
    - Simple, clean download button
    - Immediately triggers browser save dialog
    - No additional UI or clicks required
    - Perfect for single CSV/NetCDF files
    """)

st.markdown("---")
st.markdown("### 🎯 Implementation Summary")
st.markdown("""
**For CSV and NetCDF files:**
- ✅ Use `create_automatic_download()` method
- ✅ Browser immediately shows save dialog
- ✅ Clean, simple user experience

**For GeoTIFF files (multiple files):**
- ✅ Use `show_download_summary()` method
- ✅ ZIP packaging for multiple files
- ✅ File listing and size information

This provides the best user experience for each file type!
""")
