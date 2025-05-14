from typing import Optional, List
import ee
import time
import re

class GEEExporter:
    def __init__(self, max_chunk_size: int = 50331648, timeout: int = 300):
        """
        Initialize the exporter.
        
        Args:
            max_chunk_size: Maximum file size for direct downloads (bytes)
            timeout: Maximum time to wait for exports to complete (seconds)
        """
        self.max_chunk_size = max_chunk_size
        self.timeout = timeout

    def export_image_to_drive(self, image: ee.Image, filename: str, 
                            folder: str, region: ee.Geometry, 
                            scale: float = 30.0, crs: str = 'EPSG:4326',
                            wait: bool = False) -> Optional[ee.batch.Task]:
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
            Task object if wait=False, else None
        """
        # Remove file extension if present (EE adds it automatically)
        if filename.endswith('.tif'):
            filename = filename[:-4]
            
        # Sanitize the filename for EE description
        safe_description = self._sanitize_description(filename)
        
        # Start the export task
        task = ee.batch.Export.image.toDrive(
            image=image,
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
            return task
            
    def export_collection_to_drive(self, collection: ee.ImageCollection, 
                                folder: str, description: str, 
                                region: ee.Geometry, scale: float = 30.0, 
                                crs: str = 'EPSG:4326', bands: List[str] = None,
                                wait: bool = False) -> Optional[ee.batch.Task]:
        """
        Export an Earth Engine ImageCollection to Google Drive.
        
        Args:
            collection: Earth Engine ImageCollection to export
            folder: Google Drive folder name
            description: Base description for the export task
            region: Region to export
            scale: Pixel resolution in meters
            crs: Coordinate reference system
            bands: List of bands to include (if None, all bands are exported)
            wait: If True, wait for the export to complete
            
        Returns:
            Task object if wait=False, else None
        """
        # Sanitize the description for EE
        safe_description = self._sanitize_description(description)
        
        # If bands are specified, select them
        if bands:
            collection = collection.select(bands)
        
        # Create the export task
        task = ee.batch.Export.image.toDrive(
            image=collection.toBands(),  # Convert collection to a single multi-band image
            description=safe_description,
            folder=folder,
            fileNamePrefix=description,
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
            return task
    
    def export_table_to_drive(self, feature_collection: ee.FeatureCollection, 
                            filename: str, folder: str, 
                            wait: bool = True) -> Optional[ee.batch.Task]:
        """
        Export an Earth Engine FeatureCollection to Google Drive.
        
        Args:
            feature_collection: Earth Engine FeatureCollection to export
            filename: Output filename
            folder: Google Drive folder name
            wait: If True, wait for the export to complete
            
        Returns:
            Task object if wait=False, else None
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
            return task

    def _sanitize_description(self, description: str) -> str:
        """
        Sanitize a description string for Earth Engine export.
        
        Args:
            description: The input description string
            
        Returns:
            Sanitized string that complies with EE requirements
        """
        # Replace spaces with underscores
        description = description.replace(' ', '_')
        
        # Keep only allowed characters: a-z, A-Z, 0-9, ".", ",", ":", ";", "_", "-"
        description = re.sub(r'[^a-zA-Z0-9.,;:_\-]', '', description)
        
        # Truncate to 100 characters
        if len(description) > 100:
            description = description[:100]
            
        return description

    def _wait_for_task(self, task: ee.batch.Task) -> bool:
        """
        Wait for an Earth Engine task to complete.
        
        Args:
            task: Earth Engine task to wait for
            
        Returns:
            True if task completed successfully, False otherwise
        """
        start_time = time.time()
        while task.status()['state'] in ['READY', 'RUNNING']:
            # Check if we've exceeded the timeout
            if time.time() - start_time > self.timeout:
                print(f"Task timed out after {self.timeout} seconds")
                return False
                
            # Wait a bit before checking again
            time.sleep(5)
            
        # Return True if the task completed successfully
        return task.status()['state'] == 'COMPLETED'
        
    def export_time_series_to_csv(self, df, output_path):
        """
        Export a time series DataFrame to CSV.
        
        Args:
            df: DataFrame containing time series data
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved file
        """
        import pandas as pd
        from pathlib import Path
        
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
        
    def export_gridded_data_to_netcdf(self, ds, output_path):
        """
        Export a gridded dataset to NetCDF.
        
        Args:
            ds: xarray Dataset containing gridded data
            output_path: Path to save the NetCDF file
            
        Returns:
            Path to the saved file
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to NetCDF
        ds.to_netcdf(output_path)
        
        return output_path 