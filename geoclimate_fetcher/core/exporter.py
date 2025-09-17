"""
Module for exporting Google Earth Engine data to local disk or Google Drive.
"""

import os
import ee
import time
import json
import pandas as pd
import xarray as xr
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class GEEExporter:
    """Class for exporting Earth Engine data."""
    
    def __init__(self, max_chunk_size: int = 500000000, timeout: int = 300):
        """
        Initialize the exporter.

        Args:
            max_chunk_size: Maximum file size for direct downloads (bytes)
            timeout: Maximum time to wait for exports to complete (seconds)
        """
        self.max_chunk_size = max_chunk_size
        self.timeout = timeout

        # Smart export configuration
        self.local_size_threshold = 50 * 1024 * 1024  # 50MB in bytes
        self.drive_folder_prefix = "GeoClimate_Exports"

    def smart_export_with_fallback(self, image: ee.Image, filename: str,
                                 region: ee.Geometry, scale: float = 30.0,
                                 export_preference: str = 'auto') -> Dict[str, Any]:
        """
        Smart export method with automatic fallback to Google Drive for large files.

        Args:
            image: Earth Engine Image to export
            filename: Output filename (without extension)
            region: Region to export
            scale: Pixel resolution in meters
            export_preference: 'auto', 'local', or 'drive'

        Returns:
            Dictionary with export results and metadata
        """
        try:
            # Clean filename
            if filename.endswith('.tif'):
                filename = filename[:-4]

            # Create unified result format (no size estimation needed)
            result = {
                'success': False,
                'export_method': 'unknown',
                'estimated_size_mb': None,
                'actual_size_mb': None,
                'filename': filename,
                'reason': '',
                'message': '',
                'file_path': None,
                'file_data': None,
                'drive_folder': None,
                'drive_url': None,
                'task_id': None,
                'task_url': None
            }

            # Handle different export preferences
            if export_preference == 'drive':
                # User explicitly wants Drive
                drive_result = self._export_to_drive_smart(image, filename, region, scale)
                result.update(drive_result)
                result['export_method'] = 'drive'
                result['reason'] = 'user_preference'
                drive_msg = drive_result.get('message', '') or ''
                result['message'] = f"Exported to Google Drive. {drive_msg}"

            elif export_preference == 'local':
                # User explicitly wants local only
                local_result = self._export_to_local_smart(image, filename, region, scale)
                result.update(local_result)
                result['export_method'] = 'local' if local_result['success'] else 'failed'
                result['reason'] = 'user_preference'
                if local_result['success']:
                    local_msg = local_result.get('message', '') or ''
                    result['message'] = f"Downloaded locally ({local_result.get('actual_size_mb', 0):.1f} MB). {local_msg}"
                else:
                    local_msg = local_result.get('message', '') or ''
                    result['message'] = f"Local export failed: {local_msg}"

            else:  # auto - try local first, fallback to Drive
                # Always try local first
                local_result = self._export_to_local_smart(image, filename, region, scale)

                if local_result['success']:
                    # Local worked!
                    result.update(local_result)
                    result['export_method'] = 'local'
                    result['reason'] = 'local_success'
                    local_msg = local_result.get('message', '') or ''
                    result['message'] = f"Downloaded locally ({local_result.get('actual_size_mb', 0):.1f} MB). {local_msg}"
                else:
                    # Local failed, fallback to Drive
                    print(f"Local export failed: {local_result.get('message', '')}")
                    print("Falling back to Google Drive export...")

                    drive_result = self._export_to_drive_smart(image, filename, region, scale)
                    result.update(drive_result)
                    result['export_method'] = 'drive'
                    result['reason'] = 'local_failed'
                    drive_msg = drive_result.get('message', '') or ''
                    local_msg = local_result.get('message', '') or ''
                    result['message'] = f"Local export failed, sent to Google Drive. {drive_msg} (Local error: {local_msg})"

            return result

        except Exception as e:
            return {
                'success': False,
                'export_method': 'unknown',
                'estimated_size_mb': 0,
                'actual_size_mb': None,
                'filename': filename,
                'reason': 'error',
                'message': f"Export failed: {str(e)}",
                'file_path': None,
                'file_data': None,
                'drive_folder': None,
                'drive_url': None,
                'task_id': None,
                'task_url': None,
                'error': str(e)
            }

    def _export_to_local_smart(self, image: ee.Image, filename: str,
                             region: ee.Geometry, scale: float) -> Dict[str, Any]:
        """Helper method for local export with unified result format"""
        import tempfile
        import os

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                temp_path = temp_file.name

            # Export using existing method
            result_path = self.export_image_to_local(image, temp_path, region, scale)

            # Validate that the file was actually created and has content
            if not os.path.exists(result_path):
                raise Exception(f"Export failed: Output file was not created at {result_path}")

            file_size = os.path.getsize(result_path)
            if file_size == 0:
                raise Exception("Export failed: Output file is empty")

            # Validate file size isn't suspiciously small (less than 1KB suggests failure)
            if file_size < 1024:
                raise Exception(f"Export failed: Output file too small ({file_size} bytes), likely corrupted")

            # Read file as bytes for download
            with open(result_path, 'rb') as f:
                file_data = f.read()

            # Double-check we actually read data
            if not file_data:
                raise Exception("Export failed: Could not read file data")

            actual_size_mb = len(file_data) / (1024 * 1024)

            return {
                'success': True,
                'file_path': result_path,
                'file_data': file_data,
                'actual_size_mb': actual_size_mb,
                'message': f"Local export completed successfully"
            }

        except Exception as e:
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
                if 'result_path' in locals() and os.path.exists(result_path):
                    os.unlink(result_path)
            except:
                pass

            return {
                'success': False,
                'file_path': None,
                'file_data': None,
                'actual_size_mb': 0,
                'message': f"Local export failed: {str(e)}",
                'error': str(e)
            }

    def _export_to_drive_smart(self, image: ee.Image, filename: str,
                             region: ee.Geometry, scale: float) -> Dict[str, Any]:
        """Helper method for Drive export with unified result format"""
        # Create timestamped folder name
        timestamp = datetime.now().strftime('%Y_%m_%d')
        drive_folder = f"{self.drive_folder_prefix}_{timestamp}"

        # Export to Drive (non-blocking)
        task_id = self.export_image_to_drive(
            image=image,
            filename=filename,
            folder=drive_folder,
            region=region,
            scale=scale,
            wait=False  # Don't wait, return task ID
        )

        # Create Drive and task URLs
        drive_url = f"https://drive.google.com/drive/folders/"
        task_url = "https://code.earthengine.google.com/tasks"

        return {
            'success': True,
            'drive_folder': drive_folder or 'Unknown',
            'drive_url': drive_url or 'https://drive.google.com/drive/',
            'task_id': task_id or 'Unknown',
            'task_url': task_url or 'https://code.earthengine.google.com/tasks',
            'message': f"Task submitted to Google Drive folder: {drive_folder or 'Unknown'}"
        }

    def export_time_series_to_csv(self, df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
        """
        Export a time series DataFrame to CSV.
        
        Args:
            df: DataFrame containing time series data
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def export_time_series_to_drive_chunked(self, df, filename, folder, 
                                      date_col='date', chunk_months=3):
        """
        Export a time series DataFrame to Google Drive in chunks.
        
        Args:
            df: DataFrame containing time series data
            filename: Base filename for exports
            folder: Google Drive folder name
            date_col: Column name containing dates
            chunk_months: Number of months per chunk
            
        Returns:
            List of task IDs for each chunk
        """
        import ee
        import pandas as pd
        from datetime import datetime, timedelta
        import calendar
        
        if df.empty:
            print("No data to export")
            return []
        
        # Convert date column to datetime if needed
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Get min and max dates
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        # Create chunks
        chunks = []
        current_start = min_date
        
        while current_start <= max_date:
            # Calculate the end of this chunk (adding chunk_months months)
            year = current_start.year
            month = current_start.month + chunk_months
            
            # Handle year overflow
            while month > 12:
                month -= 12
                year += 1
            
            # Get last day of the month
            last_day = calendar.monthrange(year, month)[1]
            
            # Create chunk end date
            chunk_end = datetime(year, month, last_day).date()
            
            # Ensure chunk_end doesn't exceed max_date
            if chunk_end > max_date:
                chunk_end = max_date
            
            chunks.append((current_start, chunk_end))
            
            # Set up the next chunk
            current_start = (chunk_end + timedelta(days=1))
        
        # Process each chunk
        task_ids = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            # Filter the dataframe to this chunk
            chunk_df = df[(df[date_col].dt.date >= chunk_start) & 
                        (df[date_col].dt.date <= chunk_end)]
            
            if chunk_df.empty:
                print(f"Chunk {i+1}: No data for {chunk_start} to {chunk_end}")
                continue
                
            # Create a filename with date range
            chunk_filename = f"{filename}_{chunk_start.strftime('%Y%m%d')}_to_{chunk_end.strftime('%Y%m%d')}"
            
            # Create features for this chunk
            features = []
            for _, row in chunk_df.iterrows():
                properties = {col: val for col, val in row.items() if pd.notna(val)}
                # Convert dates to strings to avoid serialization issues
                if date_col in properties and isinstance(properties[date_col], datetime):
                    properties[date_col] = properties[date_col].strftime('%Y-%m-%d')
                features.append(ee.Feature(None, properties))
                
            fc = ee.FeatureCollection(features)
            
            # Export to Drive
            print(f"Exporting chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end} with {len(chunk_df)} records")
            
            task_id = self.export_table_to_drive(
                fc, chunk_filename, folder, wait=False
            )
            
            if task_id:
                task_ids.append(task_id)
        
        return task_ids
        
    def export_gridded_data_to_netcdf(self, ds: xr.Dataset, output_path: Union[str, Path]) -> Path:
        """
        Export gridded data to NetCDF format.
        
        Args:
            ds: xarray Dataset with gridded data
            output_path: Path to save the NetCDF file
            
        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to NetCDF
        ds.to_netcdf(output_path)
        
        return output_path
    
    def export_time_series_to_geotiff(self, dataset: xr.Dataset, output_dir: Union[str, Path]) -> List[Path]:
        """
        Export a time series xarray Dataset to individual GeoTIFF files, one per date.
        
        Args:
            dataset: xarray Dataset with time, lat, lon dimensions
            output_dir: Directory to save the GeoTIFF files
            
        Returns:
            List of paths to the saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import rasterio
        from rasterio.transform import from_bounds
        from tqdm import tqdm
        
        # Get dimensions and coordinates
        times = dataset.time.values
        lats = dataset.lat.values
        lons = dataset.lon.values
        bands = list(dataset.data_vars.keys())
        
        # Print information
        print(f"Exporting {len(times)} dates with {len(bands)} bands")
        print(f"Time range: {pd.to_datetime(times[0])} to {pd.to_datetime(times[-1])}")
        print(f"Spatial extent: {lons.min()} to {lons.max()}, {lats.min()} to {lats.max()}")
        
        # Get geotransform
        height = len(lats)
        width = len(lons)
        transform = from_bounds(lons.min(), lats.min(), lons.max(), lats.max(), width, height)
        
        # Create list to store output file paths
        output_files = []
        
        # Export each time step as a separate GeoTIFF
        for i, time_value in enumerate(tqdm(times, desc="Exporting time steps")):
            # Convert time to string for filename
            time_str = pd.to_datetime(time_value).strftime('%Y%m%d')
            
            # Create output filename
            output_file = output_dir / f"{time_str}.tif"
            output_files.append(output_file)
            
            # Extract data for this time step
            arrays = []
            for band in bands:
                # Get the data for this time and band
                array = dataset[band].values[i, :, :]
                arrays.append(array)
            
            # Open a new GeoTIFF file
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=len(bands),
                dtype=arrays[0].dtype,
                crs='+proj=longlat +datum=WGS84 +no_defs',
                transform=transform
            ) as dst:
                # Write each band
                for j, array in enumerate(arrays, 1):
                    dst.write(array, j)
                    dst.set_band_description(j, bands[j-1])
        
        print(f"Successfully exported {len(output_files)} GeoTIFF files to {output_dir}")
        return output_files
        
    def estimate_export_size(self, image: ee.Image, region: ee.Geometry, 
                           scale: float) -> float:
        """
        Estimate the size of the exported image in bytes.
        
        Args:
            image: Earth Engine Image to export
            region: Region of interest
            scale: Pixel resolution in meters
            
        Returns:
            Estimated size in bytes
        """
        # Calculate area in square meters
        area = region.area().getInfo()
        
        # Calculate number of pixels
        num_pixels = area / (scale * scale)
        
        # Get number of bands
        bands = len(image.bandNames().getInfo())
        
        # Assume 4 bytes per pixel per band (float32)
        return num_pixels * bands * 4
        
    def _wait_for_task(self, task: ee.batch.Task) -> bool:
        """
        Wait for an Earth Engine task to complete.
        
        Args:
            task: Earth Engine task
            
        Returns:
            True if task completed successfully, False otherwise
        """
        start_time = time.time()
        
        with tqdm(total=100, desc="Export progress") as pbar:
            last_progress = 0
            
            while True:
                task_status = task.status()
                state = task_status['state']
                
                if state == 'COMPLETED':
                    pbar.update(100 - last_progress)
                    return True
                    
                if state == 'FAILED':
                    print(f"Export failed: {task_status.get('error_message', 'Unknown error')}")
                    return False
                    
                if 'progress' in task_status:
                    progress = int(task_status['progress'] * 100)
                    pbar.update(progress - last_progress)
                    last_progress = progress
                    
                # Check timeout
                if time.time() - start_time > self.timeout:
                    print("Export timed out")
                    return False
                    
                # Wait before checking again
                time.sleep(5)
                
    def _sanitize_description(self, description: str) -> str:
        """
        Sanitize a description string for Earth Engine export.
        
        Args:
            description: The input description string
            
        Returns:
            Sanitized string that complies with EE requirements
        """
        import re
        # Replace spaces with underscores
        description = description.replace(' ', '_')
        
        # Keep only allowed characters: a-z, A-Z, 0-9, ".", ",", ":", ";", "_", "-"
        description = re.sub(r'[^a-zA-Z0-9.,;:_\-]', '', description)
        
        # Truncate to 100 characters
        if len(description) > 100:
            description = description[:100]
            
        return description

    def _harmonize_band_types(self, image: ee.Image) -> ee.Image:
        """
        Convert all bands in an image to compatible data types to prevent export errors.
        
        Args:
            image: Earth Engine Image with potentially mixed data types
            
        Returns:
            Image with all bands converted to Float32
        """
        try:
            # Simple approach: convert entire image to float32
            # This handles mixed data types (Int16, Byte, etc.) by converting all to Float32
            harmonized_image = image.toFloat()
            
            print(f"📊 Converted image bands to Float32 to ensure compatibility")
            return harmonized_image
            
        except Exception as e:
            print(f"Warning: Could not harmonize band types ({str(e)}). Using original image.")
            return image

    def export_image_to_drive(self, image: ee.Image, filename: str, 
                            folder: str, region: ee.Geometry, 
                            scale: float = 30.0, crs: str = 'EPSG:4326',
                            wait: bool = True) -> Optional[str]:
        """
        Export an Earth Engine image to Google Drive.
        
        Args:
            image: Earth Engine Image to export
            filename: Output filename
            folder: Google Drive folder name
            region: Region to export
            scale: Pixel resolution in meters
            crs: Coordinate reference system
            wait: If True, wait for the export to complete
            
        Returns:
            Task ID if wait=False, else None
        """
        # Remove file extension if present (EE adds it automatically)
        if filename.endswith('.tif'):
            filename = filename[:-4]
            
        # Sanitize the filename for EE description
        safe_description = self._sanitize_description(filename)
        
        # Harmonize band data types to prevent export errors
        try:
            harmonized_image = self._harmonize_band_types(image)
            print("✅ Harmonized band data types for export")
        except Exception as e:
            print(f"⚠️ Warning: Could not harmonize band types: {str(e)}")
            harmonized_image = image
        
        # Start the export task
        task = ee.batch.Export.image.toDrive(
            image=harmonized_image,
            description=safe_description,
            folder=folder,
            fileNamePrefix=filename,
            region=region.bounds().getInfo()['coordinates'],
            scale=scale,
            crs=crs,
            maxPixels=1e10,
            fileFormat='GeoTIFF'
        )
        
        task.start()
        
        if wait:
            success = self._wait_for_task(task)
            if not success:
                raise Exception("Export to Drive failed")
            return None
        else:
            return task.id
            
    def export_table_to_drive(self, feature_collection: ee.FeatureCollection, 
                            filename: str, folder: str, 
                            wait: bool = True) -> Optional[str]:
        """
        Export an Earth Engine FeatureCollection to Google Drive.
        
        Args:
            feature_collection: Earth Engine FeatureCollection to export
            filename: Output filename
            folder: Google Drive folder name
            wait: If True, wait for the export to complete
            
        Returns:
            Task ID if wait=False, else None
        """
        # Remove file extension if present (EE adds it automatically)
        if filename.endswith('.csv') or filename.endswith('.geojson'):
            filename = filename[:-4]
            
        # Sanitize the filename for EE description
        safe_description = self._sanitize_description(filename)
        
        # Start the export task
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=safe_description,
            folder=folder,
            fileNamePrefix=filename,
            fileFormat='CSV'
        )
        
        task.start()
        
        if wait:
            success = self._wait_for_task(task)
            if not success:
                raise Exception("Export to Drive failed")
            return None
        else:
            return task.id
            
    def export_image_to_local(self, image: ee.Image, output_path: Union[str, Path], 
                        region: ee.Geometry, scale: float = 30.0) -> Path:
        """
        Export an Earth Engine image directly to local disk using geemap.
        
        Args:
            image: Earth Engine Image to export
            output_path: Path to save the GeoTIFF file
            region: Region to export
            scale: Pixel resolution in meters
            
        Returns:
            Path to the saved file
        """
        import geemap
        
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Harmonize band data types to prevent export errors
        try:
            harmonized_image = self._harmonize_band_types(image)
            print("✅ Harmonized band data types for export")
        except Exception as e:
            print(f"⚠️ Warning: Could not harmonize band types: {str(e)}")
            harmonized_image = image
        
        try:
            # Use geemap's export function which is more reliable
            print(f"Exporting image using geemap to {output_path}...")
            
            # First ensure image is clipped to the region
            clipped_image = harmonized_image.clip(region)
            
            # Use geemap to export
            geemap.ee_export_image(
                clipped_image,
                filename=str(output_path),
                scale=scale,
                region=region,
                file_per_band=False
            )
            
            print(f"Export complete: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error exporting image with geemap: {str(e)}")
            print("\nTrying alternative export method...")
            
            try:
                # Fall back to our original method with some improvements
                # Check estimated size
                estimated_size = self.estimate_export_size(image, region, scale)
                
                if estimated_size > self.max_chunk_size:
                    raise ValueError(
                        f"Estimated export size ({estimated_size/1e6:.1f} MB) exceeds "
                        f"maximum direct download size ({self.max_chunk_size/1e6:.1f} MB). "
                        "Use export_to_drive instead."
                    )
                    
                # Get image as numpy arrays using harmonized image
                arrays = {}
                band_names = harmonized_image.bandNames().getInfo()
                
                if not band_names:
                    raise ValueError("No bands found in the image. Please check your band selection.")
                
                # Get the bounds
                bounds = region.bounds().getInfo()['coordinates'][0]
                xs = [p[0] for p in bounds]
                ys = [p[1] for p in bounds]
                
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                
                rect_region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
                
                # Download bands using harmonized image
                with tqdm(total=len(band_names), desc="Downloading bands") as pbar:
                    for band in band_names:
                        try:
                            # Get the data - but don't pass scale parameter to sampleRectangle
                            pixels = harmonized_image.select(band).sampleRectangle(
                                region=rect_region,
                                properties=None,
                                defaultValue=0
                            ).getInfo()
                            
                            if band in pixels and pixels[band] is not None and len(pixels[band]) > 0:
                                arrays[band] = np.array(pixels[band])
                            else:
                                print(f"Warning: No data returned for band '{band}'")
                                
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error downloading band {band}: {str(e)}")
                            pbar.update(1)
                
                # Check if we successfully got any data
                if not arrays:
                    print("No data was returned from Earth Engine. This could be because:")
                    print("1. There is no data available for this region in the selected dataset")
                    print("2. The region might be too large for direct download")
                    print("3. The Earth Engine API request failed")
                    print("\nTry one of the following solutions:")
                    print("- Use 'Export to Google Drive' instead of direct download")
                    print("- Select a smaller region")
                    print("- Try a different dataset that covers this region")
                    raise ValueError("Failed to download any image data")
                
                # Create a simple GeoTIFF
                import rasterio
                from rasterio.transform import from_bounds
                
                # Get dimensions from the first array
                height, width = next(iter(arrays.values())).shape
                
                # Create the geotransform
                transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
                
                # Write the GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=len(arrays),
                    dtype=next(iter(arrays.values())).dtype,
                    crs='+proj=longlat +datum=WGS84 +no_defs',
                    transform=transform
                ) as dst:
                    for i, (band, array) in enumerate(arrays.items(), 1):
                        dst.write(array, i)
                        dst.set_band_description(i, band)
                        
                print(f"Export complete using fallback method: {output_path}")
                return output_path
                
            except Exception as e2:
                print(f"Both export methods failed.")
                print(f"First error: {str(e)}")
                print(f"Second error: {str(e2)}")
                print("Please try using Google Drive export instead.")
                raise ValueError("Failed to export image to local file")
        
    def export_to_cloud_optimized_geotiff(self, image: ee.Image, output_path: Union[str, Path],
                                       region: ee.Geometry, scale: float = 30.0) -> Path:
        """
        Export an Earth Engine image to Cloud Optimized GeoTIFF (COG).
        
        Args:
            image: Earth Engine Image to export
            output_path: Path to save the COG file
            region: Region to export
            scale: Pixel resolution in meters
            
        Returns:
            Path to the saved file
        """
        # First export to regular GeoTIFF
        temp_path = Path(str(output_path) + ".temp.tif")
        self.export_image_to_local(image, temp_path, region, scale)
        
        # Convert to COG
        import rasterio
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        
        # Get COG profile
        output_profile = cog_profiles.get("deflate")
        
        # Convert to COG
        cog_translate(
            str(temp_path),
            str(output_path),
            output_profile,
            quiet=False,
            web_optimized=True
        )
        
        # Remove temporary file
        temp_path.unlink()
        
        return Path(output_path)