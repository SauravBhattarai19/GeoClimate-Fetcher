"""
Enhanced Download Component for GeoClimate Fetcher
Provides file download functionality for users to get their processed data.
"""

import streamlit as st
import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import ee

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import GEEExporter, ImageCollectionFetcher, StaticRasterFetcher

class DownloadHelper:
    """Helper class for file downloads"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def create_download_button(self, file_path, download_name=None, mime_type=None):
        """Create a download button for a single file"""
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return False
            
        if download_name is None:
            download_name = os.path.basename(file_path)
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Read file content
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Determine MIME type if not provided
            if mime_type is None:
                ext = os.path.splitext(file_path)[1].lower()
                mime_types = {
                    '.tif': 'image/tiff',
                    '.tiff': 'image/tiff',
                    '.nc': 'application/x-netcdf',
                    '.csv': 'text/csv',
                    '.zip': 'application/zip',
                    '.json': 'application/json'
                }
                mime_type = mime_types.get(ext, 'application/octet-stream')
            
            # Create download button
            st.download_button(
                label=f"📥 Download {download_name} ({file_size_mb:.1f} MB)",
                data=file_data,
                file_name=download_name,
                mime=mime_type,
                type="primary"
            )
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            return False
    
    def create_zip_download(self, source_dir, zip_name=None, include_subdirs=True):
        """Create a ZIP file from a directory and provide download"""
        if not os.path.exists(source_dir):
            st.error(f"❌ Directory not found: {source_dir}")
            return False
            
        if zip_name is None:
            dir_name = os.path.basename(source_dir.rstrip('/\\'))
            zip_name = f"{dir_name}.zip"
        
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'
            
        # Create ZIP file in temporary directory
        zip_path = os.path.join(self.temp_dir, zip_name)
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                total_files = 0
                
                if include_subdirs:
                    # Include all files and subdirectories
                    for root, dirs, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Create relative path for ZIP
                            arcname = os.path.relpath(file_path, source_dir)
                            zipf.write(file_path, arcname)
                            total_files += 1
                else:
                    # Include only files in the main directory
                    for file in os.listdir(source_dir):
                        file_path = os.path.join(source_dir, file)
                        if os.path.isfile(file_path):
                            zipf.write(file_path, file)
                            total_files += 1
            
            if total_files == 0:
                st.warning("⚠️ No files found to zip")
                return False
            
            # Get ZIP file size
            zip_size = os.path.getsize(zip_path)
            zip_size_mb = zip_size / (1024 * 1024)
            
            # Read ZIP file for download
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
            
            # Create download button
            st.download_button(
                label=f"📦 Download ZIP ({total_files} files, {zip_size_mb:.1f} MB)",
                data=zip_data,
                file_name=zip_name,
                mime='application/zip',
                type="primary"
            )
            
            # Clean up temporary ZIP file
            os.remove(zip_path)
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating ZIP: {str(e)}")
            return False
    
    def create_multi_file_download(self, file_paths, zip_name=None):
        """Create a ZIP download for multiple specific files"""
        if not file_paths:
            st.warning("⚠️ No files provided for download")
            return False
            
        # Filter existing files
        existing_files = [f for f in file_paths if os.path.exists(f)]
        
        if not existing_files:
            st.error("❌ No valid files found for download")
            return False
            
        if len(existing_files) == 1:
            # Single file - direct download
            return self.create_download_button(existing_files[0])
        
        # Multiple files - create ZIP
        if zip_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"geoclimate_data_{timestamp}.zip"
            
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'
            
        zip_path = os.path.join(self.temp_dir, zip_name)
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in existing_files:
                    filename = os.path.basename(file_path)
                    zipf.write(file_path, filename)
            
            # Get ZIP file size
            zip_size = os.path.getsize(zip_path)
            zip_size_mb = zip_size / (1024 * 1024)
            
            # Read ZIP file for download
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
            
            # Create download button
            st.download_button(
                label=f"📦 Download ZIP ({len(existing_files)} files, {zip_size_mb:.1f} MB)",
                data=zip_data,
                file_name=zip_name,
                mime='application/zip',
                type="primary"
            )
            
            # Clean up temporary ZIP file
            os.remove(zip_path)
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating multi-file ZIP: {str(e)}")
            return False

    def render_smart_download_options(self, estimated_size_mb: float = None,
                                     export_format: str = None) -> str:
        """
        Render smart download options UI for user preference selection.

        Args:
            estimated_size_mb: Not used anymore - size estimation removed
            export_format: Format being exported ('CSV', 'GeoTIFF', etc.)

        Returns:
            Selected export preference: 'auto', 'local', or 'drive'
        """
        st.markdown("### 🎯 Smart Download Options")

        # Always recommend auto mode (no size estimation)
        recommended = 'auto'
        st.info("🤖 **Smart Auto Mode**: Try local download first, automatically fallback to Google Drive if needed")

        # Export preference options
        export_options = {
            'auto': {
                'title': '🤖 Auto (Recommended)',
                'description': 'Try local first, automatic Drive fallback if needed',
                'details': 'Always attempts local download first, seamlessly switches to Google Drive if local fails'
            },
            'local': {
                'title': '💻 Force Local Only',
                'description': 'Local download only (no fallback)',
                'details': 'Direct browser download - will fail if file is too large or processing fails'
            },
            'drive': {
                'title': '☁️ Force Google Drive',
                'description': 'Skip local attempt, go straight to Drive',
                'details': 'Cloud processing - handles any size, always works but requires Google Drive access'
            }
        }

        # Update Drive option description for CSV files
        if export_format == 'CSV':
            export_options['drive']['description'] = 'Fast CSV upload directly to Drive'
            export_options['drive']['details'] = 'Optimized for CSV files - instant upload to Google Drive with folder organization'

        # Create radio button options
        option_labels = []
        option_keys = []
        for key, option in export_options.items():
            label = option['title']
            if key == recommended:
                label += ' ⭐'
            option_labels.append(label)
            option_keys.append(key)

        # Default to recommended option
        default_index = option_keys.index(recommended)

        selected_label = st.radio(
            "Choose export method:",
            option_labels,
            index=default_index,
            help="Select how you want to receive your data"
        )

        # Get the selected key
        selected_key = option_keys[option_labels.index(selected_label)]

        # Show details for selected option
        selected_option = export_options[selected_key]
        st.info(f"ℹ️ **{selected_option['title']}**: {selected_option['details']}")

        # Show additional info based on selection
        if selected_key == 'local':
            st.warning("⚠️ Local-only mode: No fallback if download fails. Consider Auto mode for reliability.")
        elif selected_key == 'drive':
            if export_format == 'CSV':
                st.success("🚀 Fast CSV upload - data will be organized in Google Drive folders!")
            else:
                st.info("📤 Files will be exported to your Google Drive. You'll receive task monitoring links.")

        return selected_key

    def execute_smart_download(self, image, filename: str, region, scale: float,
                             export_preference: str = 'auto', crs: str = 'EPSG:4326') -> dict:
        """
        Execute smart download using enhanced GEEExporter with EPSG:4326 and Float32 support.

        Args:
            image: Earth Engine Image to export
            filename: Output filename
            region: Export region
            scale: Export scale in meters
            export_preference: Export preference from render_smart_download_options()
            crs: Coordinate reference system (default: EPSG:4326)

        Returns:
            Result dictionary from smart export
        """
        st.markdown("### 🚀 Executing Enhanced Smart Download")

        # Initialize exporter if not exists
        if not hasattr(self, 'exporter') or self.exporter is None:
            self.exporter = GEEExporter()

        # Show progress indicator
        progress_placeholder = st.empty()
        progress_placeholder.info(f"🔄 Preparing TIFF export with {crs} projection and Float32 data types...")

        try:
            # Execute enhanced smart export with explicit CRS
            result = self.exporter.smart_export_with_fallback(
                image=image,
                filename=filename,
                region=region,
                scale=scale,
                export_preference=export_preference,
                crs=crs  # Pass the coordinate reference system
            )

            # Update progress based on result
            if result['success']:
                progress_placeholder.success("✅ Export completed successfully!")

                # Handle result based on export method
                if result['export_method'] == 'local':
                    self._handle_local_result(result)
                elif result['export_method'] == 'drive':
                    self._handle_drive_result(result)

            else:
                progress_placeholder.error(f"❌ Export failed: {result['message']}")
                if 'error' in result:
                    st.error(f"Error details: {result['error']}")

            return result

        except Exception as e:
            progress_placeholder.error(f"❌ Smart download failed: {str(e)}")
            return {
                'success': False,
                'message': f"Smart download error: {str(e)}",
                'error': str(e)
            }

    def _handle_local_result(self, result: dict):
        """Handle local download result display and file download with enhanced TIFF info"""
        st.success(f"✅ {result['message']}")

        # Show enhanced file info including validation results
        col1, col2 = st.columns(2)

        with col1:
            if result.get('actual_size_mb'):
                st.info(f"📊 **File size:** {result['actual_size_mb']:.1f} MB")

            # Show validation results if available
            validation = result.get('validation', {})
            if validation and validation.get('success'):
                st.success(f"✅ **TIFF Validation:** Passed")
                if 'crs' in validation:
                    st.info(f"🗺️ **Projection:** {validation['crs']}")
                if 'dtypes' in validation:
                    dtypes_str = ', '.join(str(dt) for dt in validation['dtypes'])
                    st.info(f"🔢 **Data Types:** {dtypes_str}")
            elif validation and not validation.get('success'):
                st.warning(f"⚠️ **TIFF Validation:** {validation.get('message', 'Failed')}")

        with col2:
            # Show additional validation details
            if validation and 'dimensions' in validation:
                dims = validation['dimensions']
                st.info(f"📐 **Dimensions:** {dims[0]} × {dims[1]} pixels")
            if validation and 'band_count' in validation:
                st.info(f"🎚️ **Bands:** {validation['band_count']}")

        # Create download button
        if result.get('file_data') and result.get('filename'):
            filename_with_ext = f"{result['filename']}.tif"
            st.download_button(
                label="💾 Download Enhanced TIFF File",
                data=result['file_data'],
                file_name=filename_with_ext,
                mime='image/tiff',
                type="primary",
                help="Download TIFF file with EPSG:4326 projection and Float32 data types"
            )

    def _handle_drive_result(self, result: dict):
        """Handle Google Drive export result display and links"""
        st.success(f"✅ {result['message']}")

        # Show Drive info
        col1, col2 = st.columns(2)

        with col1:
            st.info("📁 **Google Drive Export**")
            if result.get('drive_folder'):
                st.write(f"**Folder:** {result['drive_folder']}")
            if result.get('estimated_size_mb'):
                st.write(f"**Size:** {result['estimated_size_mb']:.1f} MB")

        with col2:
            st.info("🔗 **Monitoring Links**")
            if result.get('drive_url'):
                st.markdown(f"[🗂️ Open Google Drive]({result['drive_url']})")
            if result.get('task_url'):
                st.markdown(f"[⚙️ Monitor Tasks]({result['task_url']})")

        # Task information
        if result.get('task_id'):
            st.code(f"Task ID: {result['task_id']}", language=None)

        # Instructions
        st.markdown("""
        **Next Steps:**
        1. 📱 Monitor task progress using the links above
        2. 📁 Check your Google Drive folder once tasks complete
        3. ⏱️ Large exports may take several minutes to complete
        """)

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass  # Ignore cleanup errors
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

    def create_auto_download_button(self, file_path, download_name=None, mime_type=None):
        """Create an enhanced download button that triggers browser download dialog automatically"""
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return False
            
        if download_name is None:
            download_name = os.path.basename(file_path)
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Read file content
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Determine MIME type if not provided
            if mime_type is None:
                ext = os.path.splitext(file_path)[1].lower()
                mime_types = {
                    '.tif': 'image/tiff',
                    '.tiff': 'image/tiff',
                    '.nc': 'application/x-netcdf',
                    '.csv': 'text/csv',
                    '.zip': 'application/zip',
                    '.json': 'application/json'
                }
                mime_type = mime_types.get(ext, 'application/octet-stream')
            
            # Create enhanced download interface with auto-download
            import base64
            
            # Encode file data for download
            b64 = base64.b64encode(file_data).decode()
            
            # Create download link with JavaScript to trigger save dialog
            download_id = f"download_{abs(hash(file_path))}"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <p style="margin-bottom: 1rem;">
                    📄 <strong>{download_name}</strong> ({file_size_mb:.1f} MB) is ready for download
                </p>
                <a id="{download_id}" 
                   href="data:{mime_type};base64,{b64}" 
                   download="{download_name}"
                   style="display: inline-block; 
                          background: #1f77b4; 
                          color: white; 
                          padding: 0.5rem 1rem; 
                          text-decoration: none; 
                          border-radius: 5px; 
                          font-weight: bold;
                          margin: 0.5rem;">
                    📥 Download {download_name}
                </a>
            </div>
            
            <script>
                // Auto-trigger download when button is clicked
                document.getElementById('{download_id}').addEventListener('click', function() {{
                    setTimeout(function() {{
                        // Show success message after a short delay
                        const successMsg = document.createElement('div');
                        successMsg.innerHTML = '✅ Download started! Check your browser\'s download location.';
                        successMsg.style.cssText = 'background: #d4edda; color: #155724; padding: 1rem; margin: 1rem 0; border-radius: 5px; text-align: center; font-weight: bold;';
                        document.getElementById('{download_id}').parentNode.appendChild(successMsg);
                    }}, 500);
                }});
            </script>
            """, unsafe_allow_html=True)
            
            # Also provide fallback Streamlit download button
            st.markdown("---")
            st.markdown("**Alternative download method:**")
            st.download_button(
                label=f"📥 Fallback Download {download_name}",
                data=file_data,
                file_name=download_name,
                mime=mime_type,
                help="Use this if the main download button doesn't work"
            )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            return False
    
    def create_instant_download(self, file_path, download_name=None, show_success=True):
        """Create an instant download that immediately triggers browser save dialog"""
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return False
            
        if download_name is None:
            download_name = os.path.basename(file_path)
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Check if file is too large for browser download (>100MB)
        if file_size_mb > 100:
            st.warning(f"⚠️ File is too large ({file_size_mb:.1f} MB) for browser download. Using fallback method.")
            return self.create_download_button(file_path, download_name)
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Determine MIME type
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                '.tif': 'image/tiff',
                '.tiff': 'image/tiff',
                '.nc': 'application/x-netcdf',
                '.csv': 'text/csv',
                '.zip': 'application/zip',
                '.json': 'application/json'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            
            # Create immediate download
            import base64
            try:
                b64 = base64.b64encode(file_data).decode()
            except Exception as e:
                st.error(f"❌ Error encoding file: {str(e)}")
                return self.create_download_button(file_path, download_name)
            
            # Generate unique ID for this download
            download_id = f"instant_download_{abs(hash(str(file_path) + str(file_size)))}"
            
            # Create auto-downloading link with JavaScript
            st.markdown(f"""
            <div id="download_container_{download_id}" style="text-align: center; margin: 2rem 0; padding: 1rem; border: 2px solid #1f77b4; border-radius: 10px; background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);">
                <h3 style="color: #1f77b4; margin-bottom: 1rem;">📥 Your Download is Ready!</h3>
                <p style="margin-bottom: 1rem; font-size: 1.1em;">
                    <strong>{download_name}</strong> ({file_size_mb:.1f} MB)
                </p>
                <div id="download_status_{download_id}" style="margin: 1rem 0;">
                    <button id="download_btn_{download_id}" 
                            onclick="startDownload_{download_id}()" 
                            style="background: #1f77b4; 
                                   color: white; 
                                   border: none; 
                                   padding: 1rem 2rem; 
                                   font-size: 1.1em; 
                                   border-radius: 8px; 
                                   cursor: pointer; 
                                   font-weight: bold;
                                   box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        📥 Download Now
                    </button>
                </div>
                <p style="font-size: 0.9em; color: #666; margin-top: 1rem;">
                    Click the button above to save the file to your computer
                </p>
            </div>
            
            <script>
                // Ensure function is defined in global scope
                window.startDownload_{download_id} = function() {{
                    try {{
                        console.log('Starting download for {download_name}');
                        
                        // Create download link
                        const link = document.createElement('a');
                        link.href = 'data:{mime_type};base64,{b64}';
                        link.download = '{download_name}';
                        link.style.display = 'none';
                        
                        // Add to DOM and trigger click
                        document.body.appendChild(link);
                        link.click();
                        
                        // Clean up
                        setTimeout(function() {{
                            document.body.removeChild(link);
                        }}, 100);
                        
                        // Update UI to show download started
                        const statusDiv = document.getElementById('download_status_{download_id}');
                        if (statusDiv) {{
                            statusDiv.innerHTML = `
                                <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                                    ✅ <strong>Download Started!</strong><br>
                                    <small>Check your browser's Downloads folder or the location you selected.</small>
                                </div>
                                <button onclick="startDownload_{download_id}()" 
                                        style="background: #28a745; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer; margin-top: 0.5rem;">
                                    📥 Download Again
                                </button>
                            `;
                        }}
                        
                        console.log('Download initiated successfully');
                    }} catch (error) {{
                        console.error('Download error:', error);
                        alert('Download failed. Please try using the fallback download button below.');
                    }}
                }};
                
                // Auto-execute for testing
                console.log('Download function loaded for {download_name}');
            </script>
            """, unsafe_allow_html=True)
            
            if show_success:
                st.success("🎉 File processed successfully and ready for download!")
            
            # Always provide a fallback Streamlit download button
            st.markdown("---")
            st.markdown("**Alternative Download Method:**")
            st.markdown("If the download button above doesn't work, use this fallback:")
            
            # Use Streamlit's native download button as backup
            st.download_button(
                label=f"📥 Fallback Download {download_name} ({file_size_mb:.1f} MB)",
                data=file_data,
                file_name=download_name,
                mime=mime_type,
                type="secondary",
                help="Use this if the main download button doesn't work"
            )
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            return False

    def create_automatic_download(self, file_path, download_name=None, show_success=True):
        """Create an automatic download that immediately triggers browser save dialog without UI"""
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return False
            
        if download_name is None:
            download_name = os.path.basename(file_path)
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Determine MIME type
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                '.tif': 'image/tiff',
                '.tiff': 'image/tiff',
                '.nc': 'application/x-netcdf',
                '.csv': 'text/csv',
                '.zip': 'application/zip',
                '.json': 'application/json'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            
            # Use Streamlit's built-in download_button for immediate download
            st.download_button(
                label=f"📥 Download {download_name} ({file_size_mb:.1f} MB)",
                data=file_data,
                file_name=download_name,
                mime=mime_type,
                type="primary",
                help="Click to save the file to your computer"
            )
            
            if show_success:
                st.success(f"🎉 {download_name} is ready for download!")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            return False

    def create_instant_download_debug(self, file_path, download_name=None, show_success=True):
        """Create an instant download with debugging information"""
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return False
            
        if download_name is None:
            download_name = os.path.basename(file_path)
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Show debug info
        st.markdown("### 🔍 Debug Information")
        st.write(f"**File Path:** {file_path}")
        st.write(f"**File Size:** {file_size_mb:.2f} MB")
        st.write(f"**File Exists:** {os.path.exists(file_path)}")
        
        # Check if file is too large for browser download (>50MB)
        if file_size_mb > 50:
            st.warning(f"⚠️ File is large ({file_size_mb:.1f} MB). Browser download may be slow.")
            
        # Check if file is too large for data URL (>100MB)
        if file_size_mb > 100:
            st.error(f"❌ File is too large ({file_size_mb:.1f} MB) for browser data URL. Using fallback method.")
            return self.create_download_button(file_path, download_name)
        
        try:
            st.info("📖 Reading file content...")
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            st.success(f"✅ File read successfully ({len(file_data)} bytes)")
            
            # Determine MIME type
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                '.tif': 'image/tiff',
                '.tiff': 'image/tiff',
                '.nc': 'application/x-netcdf',
                '.csv': 'text/csv',
                '.zip': 'application/zip',
                '.json': 'application/json'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            st.write(f"**MIME Type:** {mime_type}")
            
            # Create immediate download
            import base64
            try:
                st.info("🔄 Encoding file to base64...")
                b64 = base64.b64encode(file_data).decode()
                st.success(f"✅ Base64 encoding successful ({len(b64)} characters)")
            except Exception as e:
                st.error(f"❌ Error encoding file: {str(e)}")
                return self.create_download_button(file_path, download_name)
            
            # Generate unique ID for this download
            download_id = f"debug_download_{abs(hash(str(file_path) + str(file_size)))}"
            st.write(f"**Download ID:** {download_id}")
            
            # Create simple test download button first
            st.markdown("### 🧪 Test Download")
            st.download_button(
                label=f"📥 Test Download {download_name}",
                data=file_data,
                file_name=download_name,
                mime=mime_type,
                type="primary",
                help="This should work - native Streamlit download"
            )
            
            # Create enhanced download with JavaScript
            st.markdown("### 🚀 Enhanced Download")
            st.markdown(f"""
            <div id="download_container_{download_id}" style="text-align: center; margin: 2rem 0; padding: 1rem; border: 2px solid #1f77b4; border-radius: 10px; background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);">
                <h3 style="color: #1f77b4; margin-bottom: 1rem;">📥 Enhanced Download Ready!</h3>
                <p style="margin-bottom: 1rem; font-size: 1.1em;">
                    <strong>{download_name}</strong> ({file_size_mb:.1f} MB)
                </p>
                <div id="download_status_{download_id}" style="margin: 1rem 0;">
                    <button id="download_btn_{download_id}" 
                            onclick="startDownload_{download_id}()" 
                            style="background: #1f77b4; 
                                   color: white; 
                                   border: none; 
                                   padding: 1rem 2rem; 
                                   font-size: 1.1em; 
                                   border-radius: 8px; 
                                   cursor: pointer; 
                                   font-weight: bold;
                                   box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        📥 Download Now (Enhanced)
                    </button>
                </div>
                <div id="debug_info_{download_id}" style="font-size: 0.8em; color: #666; margin-top: 1rem;">
                    Click the button to test enhanced download
                </div>
            </div>
            
            <script>
                // Debug logging
                console.log('Setting up download function for {download_name}');
                console.log('Download ID: {download_id}');
                console.log('File size: {file_size_mb:.1f} MB');
                console.log('MIME type: {mime_type}');
                
                // Ensure function is defined in global scope
                window.startDownload_{download_id} = function() {{
                    const debugDiv = document.getElementById('debug_info_{download_id}');
                    try {{

                        // Create download link
                        const link = document.createElement('a');
                        link.href = 'data:{mime_type};base64,{b64}';
                        link.download = '{download_name}';
                        link.style.display = 'none';
                        
                        console.log('Download link created');
                        if (debugDiv) debugDiv.innerHTML = '⬇️ Initiating download...';
                        
                        // Add to DOM and trigger click
                        document.body.appendChild(link);
                        link.click();
                        
                        console.log('Download click triggered');
                        if (debugDiv) debugDiv.innerHTML = '✅ Download initiated!';
                        
                        // Clean up
                        setTimeout(function() {{
                            document.body.removeChild(link);
                            console.log('Download link cleaned up');
                        }}, 100);
                        
                        // Update UI to show download started
                        const statusDiv = document.getElementById('download_status_{download_id}');
                        if (statusDiv) {{
                            setTimeout(function() {{
                                statusDiv.innerHTML = `
                                    <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                                        ✅ <strong>Download Started!</strong><br>
                                        <small>Check your browser's Downloads folder.</small>
                                    </div>
                                    <button onclick="startDownload_{download_id}()" 
                                            style="background: #28a745; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer; margin-top: 0.5rem;">
                                        📥 Download Again
                                    </button>
                                `;
                            }}, 1000);
                        }}
                        
                        console.log('Download process completed successfully');
                    }} catch (error) {{
                        console.error('Download error:', error);
                        if (debugDiv) debugDiv.innerHTML = '❌ Download failed: ' + error.message;
                        alert('Download failed: ' + error.message + '. Please try the fallback download button.');
                    }}
                }};
                
                // Test if function is accessible
                console.log('Download function loaded:', typeof window.startDownload_{download_id});
            </script>
            """, unsafe_allow_html=True)
            
            if show_success:
                st.success("🎉 Debug download setup complete!")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            return False
    
    def create_instant_download_debug_v2(self, file_path, download_name=None, show_success=True):
        """Create an instant download with advanced debugging information"""
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return False
            
        if download_name is None:
            download_name = os.path.basename(file_path)
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Show debug info
        st.markdown("### 🔍 Debug Information")
        st.write(f"**File Path:** {file_path}")
        st.write(f"**File Size:** {file_size_mb:.2f} MB")
        st.write(f"**File Exists:** {os.path.exists(file_path)}")
        
        # Check if file is too large for browser download (>50MB)
        if file_size_mb > 50:
            st.warning(f"⚠️ File is large ({file_size_mb:.1f} MB). Browser download may be slow.")
            
        # Check if file is too large for data URL (>100MB)
        if file_size_mb > 100:
            st.error(f"❌ File is too large ({file_size_mb:.1f} MB) for browser data URL. Using fallback method.")
            return self.create_download_button(file_path, download_name)
        
        try:
            st.info("📖 Reading file content...")
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            st.success(f"✅ File read successfully ({len(file_data)} bytes)")
            
            # Determine MIME type
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                '.tif': 'image/tiff',
                '.tiff': 'image/tiff',
                '.nc': 'application/x-netcdf',
                '.csv': 'text/csv',
                '.zip': 'application/zip',
                '.json': 'application/json'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            st.write(f"**MIME Type:** {mime_type}")
            
            # Create immediate download
            import base64
            try:
                st.info("🔄 Encoding file to base64...")
                b64 = base64.b64encode(file_data).decode()
                st.success(f"✅ Base64 encoding successful ({len(b64)} characters)")
            except Exception as e:
                st.error(f"❌ Error encoding file: {str(e)}")
                return self.create_download_button(file_path, download_name)
            
            # Generate unique ID for this download
            download_id = f"debug_download_v2_{abs(hash(str(file_path) + str(file_size)))}"
            st.write(f"**Download ID:** {download_id}")
            
            # Create enhanced download with JavaScript
            st.markdown("### 🚀 Enhanced Download")
            st.markdown(f"""
            <div id="download_container_{download_id}" style="text-align: center; margin: 2rem 0; padding: 1rem; border: 2px solid #1f77b4; border-radius: 10px; background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);">
                <h3 style="color: #1f77b4; margin-bottom: 1rem;">📥 Enhanced Download Ready!</h3>
                <p style="margin-bottom: 1rem; font-size: 1.1em;">
                    <strong>{download_name}</strong> ({file_size_mb:.1f} MB)
                </p>
                <div id="download_status_{download_id}" style="margin: 1rem 0;">
                    <button id="download_btn_{download_id}" 
                            onclick="startDownload_{download_id}()" 
                            style="background: #1f77b4; 
                                   color: white; 
                                   border: none; 
                                   padding: 1rem 2rem; 
                                   font-size: 1.1em; 
                                   border-radius: 8px; 
                                   cursor: pointer; 
                                   font-weight: bold;
                                   box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        📥 Download Now (Enhanced)
                    </button>
                </div>
                <div id="debug_info_{download_id}" style="font-size: 0.8em; color: #666; margin-top: 1rem;">
                    Click the button to test enhanced download
                </div>
            </div>
            
            <script>
                // Debug logging
                console.log('Setting up download function for {download_name}');
                console.log('Download ID: {download_id}');
                console.log('File size: {file_size_mb:.1f} MB');
                console.log('MIME type: {mime_type}');
                
                // Ensure function is defined in global scope
                window.startDownload_{download_id} = function() {{
                    const debugDiv = document.getElementById('debug_info_{download_id}');
                    try {{
                        console.log('Starting download for {download_name}');
                        if (debugDiv) debugDiv.innerHTML = '🔄 Preparing download...';
                        
                        // Create download link
                        const link = document.createElement('a');
                        link.href = 'data:{mime_type};base64,{b64}';
                        link.download = '{download_name}';
                        link.style.display = 'none';
                        
                        console.log('Download link created');
                        if (debugDiv) debugDiv.innerHTML = '⬇️ Initiating download...';
                        
                        // Add to DOM and trigger click
                        document.body.appendChild(link);
                        link.click();
                        
                        console.log('Download click triggered');
                        if (debugDiv) debugDiv.innerHTML = '✅ Download initiated!';
                        
                        // Clean up
                        setTimeout(function() {{
                            document.body.removeChild(link);
                            console.log('Download link cleaned up');
                        }}, 100);
                        
                        // Update UI to show download started
                        const statusDiv = document.getElementById('download_status_{download_id}');
                        if (statusDiv) {{
                            setTimeout(function() {{
                                statusDiv.innerHTML = `
                                    <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                                        ✅ <strong>Download Started!</strong><br>
                                        <small>Check your browser's Downloads folder.</small>
                                    </div>
                                    <button onclick="startDownload_{download_id}()" 
                                            style="background: #28a745; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer; margin-top: 0.5rem;">
                                        📥 Download Again
                                    </button>
                                `;
                            }}, 1000);
                        }}
                        
                        console.log('Download process completed successfully');
                    }} catch (error) {{
                        console.error('Download error:', error);
                        if (debugDiv) debugDiv.innerHTML = '❌ Download failed: ' + error.message;
                        alert('Download failed: ' + error.message + '. Please try the fallback download button.');
                    }}
                }};
                
                // Test if function is accessible
                console.log('Download function loaded:', typeof window.startDownload_{download_id});
            </script>
            """, unsafe_allow_html=True)
            
            if show_success:
                st.success("🎉 Debug download setup complete!")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            return False

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
                # Get dataset metadata for optimization
                temporal_resolution = config.get('temporal_resolution', 'Daily')
                native_scale = config.get('dataset_native_scale', None)
                scale = 10000  # Default 10km, but may be overridden by native scale

                if config['use_chunking']:
                    df = fetcher.get_time_series_average_chunked(
                        chunk_months=3,
                        export_format='CSV',
                        user_scale=scale,
                        temporal_resolution=temporal_resolution,
                        dataset_native_scale=native_scale
                    )
                else:
                    df = fetcher.get_time_series_average(
                        export_format='CSV',
                        user_scale=scale,
                        dataset_native_scale=native_scale
                    )
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
    
    def show_download_summary(self, output_dir, successful_downloads=0, drive_exports=0):
        """Show download summary with download options for users"""
        st.markdown("---")
        st.markdown("## 📥 Download Your Data")
        
        download_helper = DownloadHelper()
        
        if successful_downloads == 0 and drive_exports == 0:
            st.warning("⚠️ No files were successfully processed for download")
            if drive_exports > 0:
                st.info("📤 Files were sent to Google Drive. Check your Google Drive folder.")
                st.markdown("🔗 [Check Google Drive Tasks](https://code.earthengine.google.com/tasks)")
            return
        
        # Check what files exist in output directory
        if os.path.exists(output_dir):
            files = []
            dirs = []
            
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    files.append(item_path)
                elif os.path.isdir(item_path):
                    # Check if directory has files
                    dir_files = []
                    for root, _, filenames in os.walk(item_path):
                        for filename in filenames:
                            dir_files.append(os.path.join(root, filename))
                    if dir_files:
                        dirs.append((item_path, dir_files))
            
            # Show download options
            if files or dirs:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if files and not dirs:
                        # Files in main directory
                        if len(files) == 1:
                            st.info("📁 Single file ready for download:")
                            download_helper.create_download_button(files[0])
                        else:
                            st.info(f"📁 {len(files)} files ready for download:")
                            
                            # Option to download all as ZIP
                            download_option = st.radio(
                                "Download option:",
                                ["Download all as ZIP", "Download individual files"],
                                key="main_download_option"
                            )
                            
                            if download_option == "Download all as ZIP":
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                zip_name = f"geoclimate_data_{timestamp}.zip"
                                download_helper.create_multi_file_download(files, zip_name)
                            else:
                                st.markdown("**Individual file downloads:**")
                                for file_path in files:
                                    filename = os.path.basename(file_path)
                                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                                    
                                    col_file, col_download = st.columns([3, 1])
                                    with col_file:
                                        st.text(f"📄 {filename} ({file_size:.1f} MB)")
                                    with col_download:
                                        download_helper.create_download_button(file_path)
                    
                    elif dirs:
                        # Directories with files (common for multiple GeoTIFFs)
                        for dir_path, dir_files in dirs:
                            dir_name = os.path.basename(dir_path)
                            st.info(f"📁 {dir_name} ({len(dir_files)} files)")
                            
                            # Calculate total size
                            total_size = sum(os.path.getsize(f) for f in dir_files) / (1024 * 1024)
                            
                            if len(dir_files) == 1:
                                # Single file in directory
                                download_helper.create_download_button(dir_files[0])
                            else:
                                # Multiple files - offer ZIP download
                                st.markdown(f"**Download options for {dir_name}:**")
                                
                                # ZIP download button
                                zip_name = f"{dir_name}.zip"
                                download_helper.create_zip_download(dir_path, zip_name)
                                
                                # Also show individual files in expander
                                with st.expander(f"View individual files in {dir_name}"):
                                    for file_path in dir_files:
                                        filename = os.path.basename(file_path)
                                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                                        
                                        col_file, col_download = st.columns([3, 1])
                                        with col_file:
                                            st.text(f"📄 {filename} ({file_size:.1f} MB)")
                                        with col_download:
                                            download_helper.create_download_button(file_path)
                    
                    # Also include files in main directory if there are dirs
                    if files and dirs:
                        st.markdown("**Additional files:**")
                        for file_path in files:
                            filename = os.path.basename(file_path)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            
                            col_file, col_download = st.columns([3, 1])
                            with col_file:
                                st.text(f"📄 {filename} ({file_size:.1f} MB)")
                            with col_download:
                                download_helper.create_download_button(file_path)
                
                with col2:
                    st.markdown("**📊 Summary:**")
                    if successful_downloads > 0:
                        st.success(f"✅ {successful_downloads} files processed")
                    if drive_exports > 0:
                        st.info(f"📤 {drive_exports} files sent to Google Drive")
                        st.markdown("🔗 [Check Google Drive Tasks](https://code.earthengine.google.com/tasks)")
                    
                    # Show total storage used
                    if files or dirs:
                        all_files = files[:]
                        for _, dir_files in dirs:
                            all_files.extend(dir_files)
                        
                        total_size = sum(os.path.getsize(f) for f in all_files if os.path.exists(f))
                        total_size_mb = total_size / (1024 * 1024)
                        st.metric("Total Size", f"{total_size_mb:.1f} MB")
            
            else:
                st.warning("⚠️ No download files found in output directory")
                if drive_exports > 0:
                    st.info("📤 All files were sent to Google Drive due to size constraints")
                    st.markdown("🔗 [Check Google Drive Tasks](https://code.earthengine.google.com/tasks)")
        else:
            st.error("❌ Output directory not found")
        
        # Add helpful information
        with st.expander("💡 Download Tips"):
            st.markdown("""
            **File Formats:**
            - **GeoTIFF (.tif)**: Raster images for GIS software (QGIS, ArcGIS, etc.)
            - **NetCDF (.nc)**: Scientific data format for analysis (Python, R, MATLAB)
            - **CSV (.csv)**: Tabular data for spreadsheets (Excel, Google Sheets)
            - **ZIP (.zip)**: Compressed archive of multiple files
            
            **Large Downloads:**
            - Files > 50MB are automatically sent to Google Drive
            - Check your Google Drive for large datasets
            - ZIP files compress data for faster downloads
            - Multiple GeoTIFFs are automatically packaged as ZIP
            
            **Using Downloaded Data:**
            - GeoTIFF files can be opened in QGIS, ArcGIS, or Python (rasterio, gdal)
            - NetCDF files work well with xarray, rioxarray in Python
            - CSV files contain statistical summaries and time series data
            """)
        
        # Cleanup
        download_helper.cleanup()