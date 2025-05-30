import streamlit as st
import os
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import ee

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import GEEExporter, ImageCollectionFetcher, StaticRasterFetcher

class DownloadComponent:
    """Component for download configuration and execution"""
    
    def __init__(self):
        if 'exporter' not in st.session_state:
            st.session_state.exporter = GEEExporter()
        self.exporter = st.session_state.exporter
    
    def render(self):
        """Render the download component"""
        st.markdown("## 💾 Download Configuration")
        
        # Summary of previous selections
        self.render_summary()
        
        # Download configuration
        config = self.render_download_config()
        
        # Download execution
        if config:
            self.render_download_execution(config)
    
    def render_summary(self):
        """Render summary of all previous selections"""
        with st.expander("📋 Selection Summary", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔐 Authentication:**")
                if st.session_state.get('auth_complete'):
                    st.success("✅ Authenticated")
                    project_id = st.session_state.get('auth_project_id', 'Unknown')
                    st.caption(f"Project: {project_id}")
                
                st.markdown("**📍 Area of Interest:**")
                if st.session_state.get('geometry_complete'):
                    try:
                        handler = st.session_state.geometry_handler
                        area = handler.get_geometry_area()
                        name = handler.current_geometry_name
                        st.success(f"✅ {name}")
                        st.caption(f"Area: {area:.2f} km²")
                    except:
                        st.success("✅ Area selected")
            
            with col2:
                st.markdown("**📊 Dataset:**")
                dataset_name = st.session_state.get('selected_dataset_name', 'None')
                if dataset_name != 'None':
                    st.success(f"✅ {dataset_name}")
                    
                    # Dataset type
                    dataset = st.session_state.get('current_dataset', {})
                    snippet_type = dataset.get('Snippet Type', 'Unknown')
                    st.caption(f"Type: {snippet_type}")
                
                st.markdown("**🎚️ Bands:**")
                bands = st.session_state.get('selected_bands', [])
                if bands:
                    st.success(f"✅ {len(bands)} selected")
                    st.caption(f"{', '.join(bands[:3])}{'...' if len(bands) > 3 else ''}")
                
                # Time range (if applicable)
                start_date = st.session_state.get('start_date')
                end_date = st.session_state.get('end_date')
                if start_date and end_date:
                    st.markdown("**📅 Time Range:**")
                    st.success(f"✅ {start_date} to {end_date}")
                    
                    # Calculate duration
                    try:
                        start = datetime.strptime(start_date, "%Y-%m-%d")
                        end = datetime.strptime(end_date, "%Y-%m-%d")
                        days = (end - start).days
                        st.caption(f"Duration: {days} days")
                    except:
                        pass
    
    def render_download_config(self):
        """Render download configuration options"""
        st.markdown("### ⚙️ Download Configuration")
        
        with st.form("download_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📁 File Settings**")
                
                # File format
                file_format = st.selectbox(
                    "Output Format:",
                    ["GeoTIFF", "NetCDF", "CSV"],
                    help="Choose the output file format"
                )
                
                # Resolution/Scale
                scale = st.number_input(
                    "Resolution (meters):",
                    min_value=10,
                    max_value=10000,
                    value=30,
                    step=10,
                    help="Spatial resolution in meters"
                )
                
                # Custom filename
                default_name = f"{st.session_state.get('selected_dataset_name', 'data').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
                filename = st.text_input(
                    "Filename (without extension):",
                    value=default_name,
                    help="Custom filename for the output"
                )
            
            with col2:
                st.markdown("**🚀 Processing Options**")
                
                # Large file handling
                use_drive = st.checkbox(
                    "Use Google Drive for large files (>50MB)",
                    value=True,
                    help="Automatically export large files to Google Drive"
                )
                
                if use_drive:
                    drive_folder = st.text_input(
                        "Google Drive folder:",
                        value="GeoClimate_Downloads",
                        help="Folder name in Google Drive"
                    )
                else:
                    drive_folder = "GeoClimate_Downloads"
                
                # Processing options
                clip_to_region = st.checkbox(
                    "Clip to exact region boundary",
                    value=True,
                    help="Clip output to exact geometry (slower but more precise)"
                )
                
                # Chunking for large collections
                dataset = st.session_state.get('current_dataset', {})
                if dataset.get('Snippet Type') == 'ImageCollection':
                    use_chunking = st.checkbox(
                        "Enable chunking for large collections",
                        value=True,
                        help="Process large collections in smaller chunks"
                    )
                    
                    if use_chunking:
                        chunk_size = st.slider(
                            "Chunk size (images):",
                            min_value=10,
                            max_value=1000,
                            value=100,
                            help="Number of images to process at once"
                        )
                    else:
                        chunk_size = 1000
                else:
                    use_chunking = False
                    chunk_size = 1000
            
            # Output directory
            st.markdown("**📂 Output Location**")
            output_dir = st.text_input(
                "Output directory:",
                value="./downloads",
                help="Local directory to save files"
            )
            
            # Size estimation
            self.render_size_estimation(file_format, scale)
            
            # Submit configuration
            submitted = st.form_submit_button("📥 Start Download", type="primary")
            
            if submitted:
                # Validate inputs
                if not filename.strip():
                    st.error("❌ Please provide a filename")
                    return None
                
                if not output_dir.strip():
                    st.error("❌ Please provide an output directory")
                    return None
                
                return {
                    'file_format': file_format,
                    'scale': scale,
                    'filename': filename.strip(),
                    'output_dir': output_dir.strip(),
                    'use_drive': use_drive,
                    'drive_folder': drive_folder,
                    'clip_to_region': clip_to_region,
                    'use_chunking': use_chunking,
                    'chunk_size': chunk_size
                }
        
        return None
    
    def render_size_estimation(self, file_format, scale):
        """Provide size estimation for the download"""
        try:
            # Get area
            handler = st.session_state.geometry_handler
            area_km2 = handler.get_geometry_area()
            
            # Estimate size based on format and scale
            pixels_per_km2 = (1000 / scale) ** 2
            total_pixels = area_km2 * pixels_per_km2
            
            # Bytes per pixel estimates
            if file_format == "GeoTIFF":
                bytes_per_pixel = 4  # 32-bit float
            elif file_format == "NetCDF":
                bytes_per_pixel = 8  # Double precision + metadata
            else:  # CSV
                bytes_per_pixel = 20  # Text representation
            
            # Number of bands
            num_bands = len(st.session_state.get('selected_bands', [1]))
            
            # Time dimension for collections
            start_date = st.session_state.get('start_date')
            end_date = st.session_state.get('end_date')
            if start_date and end_date and file_format != "CSV":
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                    days = (end - start).days
                    # Estimate images per month (varies by dataset)
                    images_per_month = 30  # Conservative estimate
                    num_images = min(days, images_per_month * 12)  # Cap at 1 year worth
                except ValueError:
                    num_images = 1
            else:
                num_images = 1
            
            estimated_size_mb = (total_pixels * bytes_per_pixel * num_bands * num_images) / (1024 * 1024)
            
            # Display estimation
            with st.expander("📊 Size Estimation"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Area", f"{area_km2:.1f} km²")
                    st.metric("Resolution", f"{scale} m")
                    st.metric("Bands", num_bands)
                
                with col2:
                    st.metric("Estimated Size", f"{estimated_size_mb:.1f} MB")
                    
                    if estimated_size_mb > 50:
                        st.warning("⚠️ Large file - will be sent to Google Drive")
                    elif estimated_size_mb > 10:
                        st.info("ℹ️ Medium size file")
                    else:
                        st.success("✅ Small file - quick download")
                
                # Additional info
                if num_images > 1:
                    st.info(f"Estimated {num_images} images in collection")
                    
        except Exception as e:
            st.warning(f"Could not estimate size: {str(e)}")
    
    def render_download_execution(self, config):
        """Execute the download with progress tracking"""
        st.markdown("### 🚀 Download Execution")
        
        # Get all required data
        geometry = st.session_state.geometry_handler.current_geometry
        dataset = st.session_state.current_dataset
        selected_bands = st.session_state.selected_bands
        
        # Validate everything is ready
        if not geometry:
            st.error("❌ No geometry selected")
            return
        
        if not dataset:
            st.error("❌ No dataset selected")
            return
        
        if not selected_bands:
            st.error("❌ No bands selected")
            return
        
        # Start download process
        with st.spinner("🔄 Initializing download..."):
            try:
                # Create output directory
                os.makedirs(config['output_dir'], exist_ok=True)
                
                # Get dataset info
                ee_id = dataset.get('Earth Engine ID')
                snippet_type = dataset.get('Snippet Type')
                
                # Execute download based on type
                if snippet_type == 'ImageCollection':
                    success = self.download_image_collection(
                        ee_id, selected_bands, geometry, config
                    )
                else:
                    success = self.download_static_image(
                        ee_id, selected_bands, geometry, config
                    )
                
                if success:
                    st.success("🎉 Download completed successfully!")
                    st.balloons()
                    
                    # Provide next steps
                    with st.expander("📁 Output Files", expanded=True):
                        st.write(f"**Location:** {config['output_dir']}")
                        
                        # List files if local
                        try:
                            output_path = Path(config['output_dir'])
                            if output_path.exists():
                                files = list(output_path.glob("*"))
                                if files:
                                    st.write("**Files created:**")
                                    for file in files:
                                        st.write(f"- {file.name}")
                                else:
                                    st.info("Files may have been exported to Google Drive")
                        except:
                            pass
                
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                
                # Show troubleshooting tips
                with st.expander("🔧 Troubleshooting"):
                    st.markdown("""
                    **Common issues:**
                    - Large files are automatically sent to Google Drive
                    - Check your Google Earth Engine quotas
                    - Reduce the area size or time range
                    - Try a different file format
                    """)
    
    def download_image_collection(self, ee_id, bands, geometry, config):
        """Download ImageCollection data"""
        try:
            # Create fetcher
            fetcher = ImageCollectionFetcher(
                ee_id=ee_id,
                bands=bands,
                geometry=geometry
            )
            
            # Apply date filter
            start_date = st.session_state.get('start_date')
            end_date = st.session_state.get('end_date')
            if start_date and end_date:
                fetcher = fetcher.filter_dates(start_date=start_date, end_date=end_date)
            
            # Set up output path
            file_ext = {
                'GeoTIFF': '.tif',
                'NetCDF': '.nc',
                'CSV': '.csv'
            }.get(config['file_format'], '.tif')
            
            output_path = os.path.join(config['output_dir'], f"{config['filename']}{file_ext}")
            
            # Download based on format
            if config['file_format'] == 'CSV':
                if config['use_chunking']:
                    df = fetcher.get_time_series_average_chunked(chunk_months=3)
                else:
                    df = fetcher.get_time_series_average()
                self.exporter.export_time_series_to_csv(df, output_path)
                
            elif config['file_format'] == 'NetCDF':
                ds = fetcher.get_gridded_data(scale=config['scale'])
                self.exporter.export_gridded_data_to_netcdf(ds, output_path)
                
            else:  # GeoTIFF
                # Create directory for multiple files
                geotiff_dir = os.path.join(config['output_dir'], f"{config['filename']}_geotiffs")
                os.makedirs(geotiff_dir, exist_ok=True)
                
                # Export collection
                collection = fetcher.collection
                collection_size = collection.size().getInfo()
                
                st.info(f"Found {collection_size} images in collection")
                
                # Progress tracking
                progress = st.progress(0)
                status = st.empty()
                
                image_list = collection.toList(collection.size())
                
                for i in range(collection_size):
                    progress.progress((i + 1) / collection_size)
                    status.text(f"Processing image {i + 1} of {collection_size}")
                    
                    image = ee.Image(image_list.get(i))
                    date_millis = image.get('system:time_start').getInfo()
                    date_str = datetime.fromtimestamp(date_millis / 1000).strftime('%Y%m%d')
                    
                    if bands:
                        image = image.select(bands)
                    
                    image_output_path = os.path.join(geotiff_dir, f"{date_str}.tif")
                    
                    self.exporter.export_image_to_local(
                        image=image,
                        output_path=image_output_path,
                        region=geometry,
                        scale=config['scale'],
                        use_drive_for_large=config['use_drive'],
                        drive_folder=config['drive_folder']
                    )
                
                progress.progress(1.0)
                status.text("✅ All images processed")
            
            return True
            
        except Exception as e:
            st.error(f"Error downloading ImageCollection: {str(e)}")
            return False
    
    def download_static_image(self, ee_id, bands, geometry, config):
        """Download static Image data"""
        try:
            # Create fetcher
            fetcher = StaticRasterFetcher(
                ee_id=ee_id,
                bands=bands,
                geometry=geometry
            )
            
            image = fetcher.image
            
            # Set up output path
            file_ext = {
                'GeoTIFF': '.tif',
                'NetCDF': '.nc',
                'CSV': '.csv'
            }.get(config['file_format'], '.tif')
            
            output_path = os.path.join(config['output_dir'], f"{config['filename']}{file_ext}")
            
            # Download based on format
            if config['file_format'] == 'CSV':
                stats = fetcher.get_zonal_statistics()
                rows = []
                for band, band_stats in stats.items():
                    row = {'band': band}
                    row.update(band_stats)
                    rows.append(row)
                df = pd.DataFrame(rows)
                self.exporter.export_time_series_to_csv(df, output_path)
                
            else:  # GeoTIFF or NetCDF
                self.exporter.export_image_to_local(
                    image=image,
                    output_path=output_path,
                    region=geometry,
                    scale=config['scale'],
                    use_drive_for_large=config['use_drive'],
                    drive_folder=config['drive_folder']
                )
            
            return True
            
        except Exception as e:
            st.error(f"Error downloading Image: {str(e)}")
            return False 