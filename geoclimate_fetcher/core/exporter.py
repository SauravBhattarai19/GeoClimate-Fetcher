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
            
        # Start the export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
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
            
        # Start the export task
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=filename,
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
        Export an Earth Engine image directly to local disk.
        
        Args:
            image: Earth Engine Image to export
            output_path: Path to save the GeoTIFF file
            region: Region to export
            scale: Pixel resolution in meters
            
        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check estimated size
        estimated_size = self.estimate_export_size(image, region, scale)
        
        if estimated_size > self.max_chunk_size:
            raise ValueError(
                f"Estimated export size ({estimated_size/1e6:.1f} MB) exceeds "
                f"maximum direct download size ({self.max_chunk_size/1e6:.1f} MB). "
                "Use export_to_drive instead."
            )
            
        # Get image as numpy arrays
        arrays = {}
        band_names = image.bandNames().getInfo()
        
        with tqdm(total=len(band_names), desc="Downloading bands") as pbar:
            for band in band_names:
                # Sample rectangle to get pixel values
                bounds = region.bounds().getInfo()['coordinates'][0]
                xs = [p[0] for p in bounds]
                ys = [p[1] for p in bounds]
                
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                
                rect_region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
                
                # Get the data - but don't pass scale parameter to sampleRectangle
                try:
                    pixels = image.select(band).sampleRectangle(
                        region=rect_region,
                        properties=None,
                        defaultValue=0
                    ).getInfo()
                    
                    if band in pixels:
                        arrays[band] = np.array(pixels[band])
                        
                    pbar.update(1)
                except Exception as e:
                    print(f"Error downloading band {band}: {str(e)}")
                    raise
        
        # Create a simple GeoTIFF
        import rasterio
        from rasterio.transform import from_bounds
        
        # Get dimensions
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
                
        return output_path
        
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