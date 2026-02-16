"""
ImageCollection fetcher module for retrieving time-series data from Earth Engine.
"""

import ee
import pandas as pd
import xarray as xr
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from datetime import datetime, date
from pathlib import Path
from tqdm import tqdm

class ImageCollectionFetcher:
    """Class for fetching and processing Earth Engine ImageCollections."""
    
    def __init__(self, ee_id: str, bands: List[str], geometry: ee.Geometry):
        """
        Initialize the ImageCollection fetcher.
        
        Args:
            ee_id: Earth Engine ImageCollection ID
            bands: List of band names to use
            geometry: Earth Engine geometry to use as AOI
        """
        self.ee_id = ee_id
        self.bands = bands
        self.geometry = geometry
        self.collection = ee.ImageCollection(ee_id)
        
    def filter_dates(self, start_date: Union[str, datetime, date], 
                    end_date: Union[str, datetime, date]) -> 'ImageCollectionFetcher':
        """
        Filter the collection by date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Self for method chaining
        """
        # Convert Python date/datetime objects to strings if needed
        if isinstance(start_date, (date, datetime)):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = start_date
            
        if isinstance(end_date, (date, datetime)):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = end_date
            
        print(f"Filtering dates from {start_date_str} to {end_date_str}")
        self.collection = self.collection.filterDate(start_date_str, end_date_str)
        
        # Store the date values for later reference
        self.filter_dates_values = [start_date_str, end_date_str]
        
        return self
        
    def select_bands(self, bands: Optional[List[str]] = None) -> 'ImageCollectionFetcher':
        """
        Select specific bands from the collection.
        
        Args:
            bands: List of band names to select. If None, uses the bands
                  provided at initialization.
                  
        Returns:
            Self for method chaining
        """
        if bands is not None:
            self.bands = bands
            
        if self.bands:
            self.collection = self.collection.select(self.bands)
            
        return self
        
    def get_time_series_average(self, export_format: str = 'CSV', user_scale: float = 1000, dataset_native_scale: float = None) -> pd.DataFrame:
        """
        Calculate spatial average time series for the AOI.

        Args:
            export_format: Export format ('CSV' or 'GeoTIFF')
            user_scale: User-specified scale in meters
            dataset_native_scale: Dataset's native resolution in meters

        Returns:
            DataFrame with dates and band averages
        """
        # Optimize scale for CSV downloads (area averaging doesn't need high resolution)
        if export_format.upper() == 'CSV':
            # For area-averaged CSV, use the coarser of 10km or native resolution
            if dataset_native_scale and dataset_native_scale > 10000:
                optimized_scale = dataset_native_scale
                print(f"CSV export: Using dataset native scale {optimized_scale}m (coarser than 10km, more efficient)")
            else:
                optimized_scale = 10000
                print(f"CSV export: Using 10km scale for maximum performance (native: {dataset_native_scale}m)")
        else:
            # For GeoTIFF, honor user's scale choice (spatial detail matters)
            optimized_scale = user_scale
            print(f"GeoTIFF export: Using user scale {optimized_scale}m")

        # Simplify geometry if needed for better performance
        simplified_geometry = self._simplify_geometry_if_needed(self.geometry)

        def extract_date_value(image):
            """Extract date and average values from an image."""
            # Get the timestamp
            date = ee.Date(image.get('system:time_start'))
            date_string = date.format('YYYY-MM-dd')

            # Get the average value for each band
            means = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=simplified_geometry,  # Use simplified geometry
                scale=optimized_scale,  # Use optimized scale
                maxPixels=2e9
            )
            
            # Create properties object with bands
            properties = {'date': date_string}
            for band in self.bands:
                # Use .get() method to get band value from means
                properties[band] = means.get(band)
            
            # Create a feature with properties
            return ee.Feature(None, properties)
            
        # Map over the collection and get average values
        features = self.collection.map(extract_date_value)
        
        # Get the data as a Pandas DataFrame
        print("Fetching feature collection from Earth Engine...")
        feature_collection = features.getInfo()
        
        # Extract the properties we need into a DataFrame
        if 'features' not in feature_collection or not feature_collection.get('features'):
            print("No features returned from Earth Engine. Collection may be empty for the given time range.")
            return pd.DataFrame()
            
        rows = []
        for feature in feature_collection['features']:
            props = feature['properties']
            row = {'date': props['date']}
            for band in self.bands:
                row[band] = props.get(band)
            rows.append(row)
            
        df = pd.DataFrame(rows)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
        return df
    
    def get_time_series_average_chunked(self, chunk_months=3, export_format: str = 'CSV', user_scale: float = 1000, temporal_resolution: str = 'Daily', dataset_native_scale: float = None):
        """
        Calculate spatial average time series in chunks with temporal-resolution-aware chunking.

        Args:
            chunk_months: Number of months per chunk (will be overridden based on temporal_resolution)
            export_format: Export format ('CSV' or 'GeoTIFF')
            user_scale: User-specified scale in meters
            temporal_resolution: Dataset temporal resolution for smart chunking
            dataset_native_scale: Dataset's native resolution in meters

        Returns:
            DataFrame with dates and band averages for the entire period
        """
        # Smart chunking based on temporal resolution
        chunk_strategies = {
            'Static': None,  # No chunking needed
            'Multi-year': None,  # No chunking needed
            'Annual': 60,  # 5 years per chunk
            'Yearly': 60,  # 5 years per chunk
            '30-Year Monthly Average': None,  # Static data
            'Monthly': 36,  # 3 years per chunk
            '16-Day': 24,  # 2 years per chunk
            '8-Day': 12,  # 1 year per chunk
            '6-Day': 12,  # 1 year per chunk
            '2-Day': 12,  # 1 year per chunk
            'Daily': 12,  # 1 year per chunk
            '3-hourly': 6,  # 6 months per chunk
            'Hourly': 3,  # 3 months per chunk
            '30-minute': 1  # 1 month per chunk
        }

        # Determine optimal chunk size
        optimal_chunk_months = chunk_strategies.get(temporal_resolution, chunk_months)

        if optimal_chunk_months is None:
            print(f"No chunking needed for {temporal_resolution} data - processing all at once")
            return self.get_time_series_average(export_format=export_format, user_scale=user_scale, dataset_native_scale=dataset_native_scale)

        print(f"Using {optimal_chunk_months}-month chunks for {temporal_resolution} data (optimized from default {chunk_months})")
        chunk_months = optimal_chunk_months
        import pandas as pd
        import tempfile
        import os
        from datetime import datetime, timedelta, date
        import calendar
        
        # Instead of using the date range from the collection, let's use the filtered dates
        # that were specified when filter_dates() was called
        try:
            # Get date information from the filtered collection
            date_range = self.collection.reduceColumns(
                ee.Reducer.minMax(), 
                ['system:time_start']
            ).getInfo()
            
            # If we can't get date range from the collection, we'll use a safer approach
            if not date_range or 'min' not in date_range or 'max' not in date_range:
                print("Warning: Couldn't determine date range from collection metadata.")
                # Try to get the first and last image to determine date range
                first_img = self.collection.first()
                last_img = self.collection.sort('system:time_start', False).first()
                
                if first_img and last_img:
                    first_date_millis = first_img.get('system:time_start').getInfo()
                    last_date_millis = last_img.get('system:time_start').getInfo()
                    
                    # Use a safer way to convert to dates
                    try:
                        # For dates after 1970
                        if first_date_millis >= 0:
                            start_date = datetime.fromtimestamp(first_date_millis / 1000).date()
                        else:
                            # For dates before 1970, use a different approach
                            # We can approximate by calculating days from epoch
                            days_before_epoch = abs(first_date_millis) // (24 * 60 * 60 * 1000)
                            start_date = date(1970, 1, 1) - timedelta(days=days_before_epoch)
                        
                        if last_date_millis >= 0:
                            end_date = datetime.fromtimestamp(last_date_millis / 1000).date()
                        else:
                            days_before_epoch = abs(last_date_millis) // (24 * 60 * 60 * 1000)
                            end_date = date(1970, 1, 1) - timedelta(days=days_before_epoch)
                    except Exception as e:
                        print(f"Error converting timestamps: {e}")
                        # Fallback to a safe approach - use the filter dates
                        dates_filter = self.collection.get('filter_dates')
                        if dates_filter and len(dates_filter) == 2:
                            start_date = datetime.strptime(dates_filter[0], '%Y-%m-%d').date()
                            end_date = datetime.strptime(dates_filter[1], '%Y-%m-%d').date()
                        else:
                            # If all else fails, ask the user
                            print("Couldn't determine date range automatically.")
                            print("Using fallback range: 30 years from 1980-01-01")
                            start_date = date(1980, 1, 1)
                            end_date = date(2010, 12, 31)
                else:
                    # No images found - use filter dates if available
                    filter_dates = getattr(self, 'filter_dates_values', None)
                    if filter_dates and len(filter_dates) == 2:
                        start_date = datetime.strptime(filter_dates[0], '%Y-%m-%d').date()
                        end_date = datetime.strptime(filter_dates[1], '%Y-%m-%d').date()
                    else:
                        # Last resort fallback
                        print("Warning: No date range information found. Using default range.")
                        start_date = date(1980, 1, 1)
                        end_date = date(2010, 12, 31)
            else:
                # We have date_range from the collection
                # Use a safer conversion method that works for pre-1970 dates
                try:
                    if date_range['min'] >= 0:
                        # For dates after 1970 (Unix epoch)
                        start_date = datetime.fromtimestamp(date_range['min'] / 1000).date()
                    else:
                        # For dates before 1970, we need to handle differently
                        # Calculate days before epoch
                        days_before_epoch = abs(date_range['min']) // (24 * 60 * 60 * 1000)
                        start_date = date(1970, 1, 1) - timedelta(days=days_before_epoch)
                    
                    if date_range['max'] >= 0:
                        end_date = datetime.fromtimestamp(date_range['max'] / 1000).date()
                    else:
                        days_before_epoch = abs(date_range['max']) // (24 * 60 * 60 * 1000)
                        end_date = date(1970, 1, 1) - timedelta(days=days_before_epoch)
                except Exception as e:
                    print(f"Error handling timestamps: {e}")
                    # Fallback - try to use the start and end dates provided in filter_dates
                    try:
                        # Try to access the original filter dates if they were stored
                        filter_dates = getattr(self, 'filter_dates_values', None)
                        if filter_dates and len(filter_dates) == 2:
                            start_date = datetime.strptime(filter_dates[0], '%Y-%m-%d').date()
                            end_date = datetime.strptime(filter_dates[1], '%Y-%m-%d').date()
                        else:
                            # Last resort fallback
                            print("Warning: No date range information found. Using default range.")
                            start_date = date(1980, 1, 1)
                            end_date = date(2010, 12, 31)
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")
                        # Absolutely last resort
                        start_date = date(1980, 1, 1)
                        end_date = date(2010, 12, 31)
        
            print(f"Breaking date range {start_date} to {end_date} into chunks...")
            
            # Create chunks
            chunks = []
            current_start = start_date
            
            while current_start <= end_date:
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
                chunk_end = date(year, month, last_day)
                
                # Ensure chunk_end doesn't exceed end_date
                if chunk_end > end_date:
                    chunk_end = end_date
                
                chunks.append((current_start, chunk_end))
                
                # Set up the next chunk
                current_start = (chunk_end + timedelta(days=1))
            
            # Process each chunk
            chunk_dfs = []
            temp_files = []
            
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
                
                # Filter the collection to this chunk's date range
                chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                chunk_collection = self.collection.filterDate(chunk_start_str, chunk_end_str)
                
                # Create a temporary fetcher for this chunk
                chunk_fetcher = ImageCollectionFetcher(self.ee_id, self.bands, self.geometry)
                chunk_fetcher.collection = chunk_collection
                
                try:
                    # Get time series for this chunk using the optimized method
                    chunk_data = chunk_fetcher.get_time_series_average(
                        export_format=export_format,
                        user_scale=user_scale,
                        dataset_native_scale=dataset_native_scale
                    )
                    
                    if not chunk_data.empty:
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                            temp_path = tmp.name
                            chunk_data.to_csv(temp_path, index=False)
                            temp_files.append(temp_path)
                        
                        chunk_dfs.append(chunk_data)
                        print(f"Chunk {i+1}: Retrieved {len(chunk_data)} records")
                    else:
                        print(f"Chunk {i+1}: No data found")
                        
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
            
            # Combine all chunks
            if chunk_dfs:
                combined_df = pd.concat(chunk_dfs, ignore_index=True)
                combined_df = combined_df.sort_values('date')
                
                # Clean up temp files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
                return combined_df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error in chunked processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _simplify_geometry_if_needed(self, geometry):
        """Simplify geometry if it's too complex for efficient processing"""
        import ee

        try:
            # Check if geometry is too complex by getting coordinate count
            coords_info = geometry.getInfo()

            # Count total coordinates
            total_coords = 0
            if coords_info['type'] == 'Polygon':
                for ring in coords_info['coordinates']:
                    total_coords += len(ring)
            elif coords_info['type'] == 'MultiPolygon':
                for polygon in coords_info['coordinates']:
                    for ring in polygon:
                        total_coords += len(ring)
            else:
                # For other geometry types, use as-is
                return geometry

            print(f"Geometry has {total_coords} coordinate points")

            # If too many coordinates, simplify
            if total_coords > 1000:  # Threshold for simplification
                print(f"Simplifying complex geometry ({total_coords} points)...")

                # Method 1: Buffer and unbuffer to smooth
                simplified = geometry.buffer(100).buffer(-100).simplify(500)

                # If still too complex, use convex hull
                try:
                    simplified_coords = simplified.getInfo()
                    simplified_count = len(simplified_coords['coordinates'][0]) if simplified_coords['type'] == 'Polygon' else 0

                    if simplified_count > 500:
                        print("Using convex hull for maximum simplification...")
                        simplified = geometry.convexHull()

                except:
                    print("Using convex hull as fallback...")
                    simplified = geometry.convexHull()

                return simplified
            else:
                return geometry

        except Exception as e:
            print(f"Warning: Could not analyze geometry complexity ({e}). Using convex hull...")
            # Fallback to convex hull if geometry analysis fails
            return geometry.convexHull()
        
    def get_gridded_data(self, scale: float = 1000.0, crs: str = 'EPSG:4326') -> xr.Dataset:
        """
        Get gridded data for all images in the collection.
        
        Args:
            scale: Pixel resolution in meters
            crs: Coordinate reference system
            
        Returns:
            xarray Dataset with time, lat, lon dimensions
        """
        import xarray as xr
        import numpy as np
        from datetime import datetime
        import pandas as pd
        import os
        import tempfile
        
        print("Fetching gridded data using new unified-export approach...")
        
        # Get the bounds of the region
        bounds = self.geometry.bounds().getInfo()['coordinates'][0]
        xs = [p[0] for p in bounds]
        ys = [p[1] for p in bounds]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
        
        # Create a temporary directory for exporting data
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Get timestamps from the collection
            timestamps = self.collection.aggregate_array('system:time_start').getInfo()
            if not timestamps:
                print("No images found in the collection for the specified time range.")
                return xr.Dataset()
            
            timestamps.sort()  # Make sure they're in order
            
            # Convert to pandas datetime objects for proper NetCDF serialization
            dates = [pd.Timestamp(ts, unit='ms') for ts in timestamps]
            print(f"Found {len(dates)} dates in collection")
            
            # Instead of processing day by day, export the entire collection as a composite
            # For each band
            band_arrays = {}
            
            for band in self.bands:
                print(f"Processing band '{band}'...")
                
                # Create a time series stack of images for this band
                # More efficient approach than day-by-day filtering
                try:
                    # Use ee.ImageCollection.toBands() to convert collection to a multi-band image
                    # where each band represents a different date
                    band_collection = self.collection.select(band)
                    
                    # Add timestamp as a property to use for naming
                    def add_date_to_band_name(img):
                        date_str = ee.Date(img.get('system:time_start')).format('YYYY_MM_dd')
                        return img.rename(ee.String(band).cat('_').cat(date_str))
                    
                    # Map the function to rename bands with dates
                    band_collection = band_collection.map(add_date_to_band_name)
                    
                    # Convert to a single multi-band image
                    stack_img = band_collection.toBands()
                    
                    # Export the multi-band image to a temporary GeoTIFF
                    temp_file = os.path.join(temp_dir, f"{band}_stack.tif")
                    
                    # Use getDownloadURL instead of sampleRectangle
                    url = stack_img.getDownloadURL({
                        'scale': scale,
                        'crs': crs,
                        'region': region,
                        'format': 'GEO_TIFF'
                    })
                    
                    # Download the file
                    import requests
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(temp_file, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded data for band '{band}'")
                    else:
                        print(f"Failed to download data for band '{band}': HTTP {response.status_code}")
                        continue
                    
                    # Read the GeoTIFF with rasterio
                    import rasterio
                    with rasterio.open(temp_file) as src:
                        # Read all bands
                        raster_data = src.read()
                        height, width = raster_data.shape[1], raster_data.shape[2]
                        
                        # Create a 3D array (time, height, width)
                        time_series = np.zeros((len(dates), height, width))
                        
                        # Read the band data - match band names with dates
                        band_names = src.descriptions
                        for i, band_name in enumerate(band_names):
                            if i < len(dates):  # Only process if we have a matching date
                                time_series[i] = raster_data[i]
                        
                        band_arrays[band] = time_series
                        print(f"Successfully processed band '{band}' with shape {time_series.shape}")
                
                except Exception as e:
                    print(f"Error processing band '{band}': {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # If we have any data, create an xarray Dataset
            if band_arrays:
                # Create coordinates for the dataset - use pandas DatetimeIndex for time
                first_band = next(iter(band_arrays))
                coords = {
                    # Use pandas DatetimeIndex which is serializable to NetCDF
                    'time': pd.DatetimeIndex(dates),
                    'lat': np.linspace(ymax, ymin, band_arrays[first_band].shape[1]),
                    'lon': np.linspace(xmin, xmax, band_arrays[first_band].shape[2])
                }
                
                # Create data variables
                data_vars = {
                    band: (['time', 'lat', 'lon'], array)
                    for band, array in band_arrays.items()
                }
                
                # Create the dataset
                ds = xr.Dataset(data_vars=data_vars, coords=coords)
                
                # Add metadata
                ds.attrs['description'] = f"Data from {self.ee_id}"
                ds.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ds.attrs['scale'] = scale
                ds.attrs['crs'] = crs
                ds.attrs['bounds'] = f"[{xmin}, {ymin}, {xmax}, {ymax}]"
                
                return ds
            else:
                print("No data could be retrieved for any band.")
                return xr.Dataset()
        
        finally:
            # Clean up temporary directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {str(e)}")
        
    def aggregate_values(self, reducer: str = 'mean', temporal: bool = False) -> Union[Dict[str, float], pd.DataFrame]:
        """
        Apply a reducer to aggregate values either spatially or temporally.
        
        Args:
            reducer: Type of reducer ('mean', 'min', 'max', 'sum', 'std')
            temporal: If True, aggregate temporally first, then spatially
            
        Returns:
            Dictionary of band values if spatial aggregation, DataFrame if temporal
        """
        if reducer not in ['mean', 'min', 'max', 'sum', 'std']:
            raise ValueError(f"Invalid reducer: {reducer}")
            
        ee_reducer = getattr(ee.Reducer, reducer)()
        
        if temporal:
            # First reduce temporally
            if reducer == 'mean':
                temporal_reduced = self.collection.mean()
            elif reducer == 'min':
                temporal_reduced = self.collection.min()
            elif reducer == 'max':
                temporal_reduced = self.collection.max()
            elif reducer == 'sum':
                temporal_reduced = self.collection.sum()
            else:  # std
                # Standard deviation requires custom calculation
                mean = self.collection.mean()
                def calc_squared_diff(img):
                    return img.subtract(mean).pow(2)
                squared_diffs = self.collection.map(calc_squared_diff)
                variance = squared_diffs.mean()
                temporal_reduced = variance.sqrt()
                
            # Then reduce spatially
            result = temporal_reduced.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.geometry,
                scale=1000,
                maxPixels=2e9
            ).getInfo()
            
            return {band: result.get(band) for band in self.bands}
        else:
            # Get time series with spatial aggregation
            return self.get_time_series_average()