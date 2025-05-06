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
            start_date = start_date.strftime('%Y-%m-%d')
            
        if isinstance(end_date, (date, datetime)):
            end_date = end_date.strftime('%Y-%m-%d')
            
        print(f"Filtering dates from {start_date} to {end_date}")
        self.collection = self.collection.filterDate(start_date, end_date)
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
        
    def get_time_series_average(self) -> pd.DataFrame:
        """
        Calculate spatial average time series for the AOI.
        
        Returns:
            DataFrame with dates and band averages
        """
        def extract_date_value(image):
            """Extract date and average values from an image."""
            # Get the timestamp
            date = ee.Date(image.get('system:time_start'))
            date_string = date.format('YYYY-MM-dd')
            
            # Get the average value for each band
            means = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.geometry,
                scale=1000,  # Can be adjusted based on data resolution
                maxPixels=1e9
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
        
    def get_gridded_data(self, scale: float = 1000.0, crs: str = 'EPSG:4326') -> xr.Dataset:
        """
        Get gridded data for all images in the collection.
        
        Args:
            scale: Pixel resolution in meters
            crs: Coordinate reference system
            
        Returns:
            xarray Dataset with time, lat, lon dimensions
        """
        # Get the time range and bounds
        bounds = self.geometry.bounds().getInfo()['coordinates'][0]
        xs = [p[0] for p in bounds]
        ys = [p[1] for p in bounds]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # Create the dimensions
        width = int((xmax - xmin) / scale * 111000)  # approximate degrees to meters conversion
        height = int((ymax - ymin) / scale * 111000)
        
        # Define the region
        region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
        
        # Get the dates from the collection - use filterDate with our date range
        print(f"Getting available dates from collection...")
        
        # Get explicit date information to debug
        collection_size = self.collection.size().getInfo()
        print(f"Collection contains {collection_size} images")
        
        # Get image information for debugging
        first_image = self.collection.first()
        if first_image:
            first_image_date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            print(f"First image in collection is from: {first_image_date}")
            print(f"Available bands: {first_image.bandNames().getInfo()}")
        
        # Log the full date range
        date_range = self.collection.reduceColumns(
            ee.Reducer.minMax(), 
            ['system:time_start']
        ).getInfo()
        
        if date_range and 'min' in date_range and 'max' in date_range:
            min_date = datetime.fromtimestamp(date_range['min'] / 1000).strftime('%Y-%m-%d')
            max_date = datetime.fromtimestamp(date_range['max'] / 1000).strftime('%Y-%m-%d')
            print(f"Collection date range: {min_date} to {max_date}")
        
        # Get the dates
        dates_info = self.collection.aggregate_array('system:time_start').getInfo()
        if not dates_info:
            print("No dates found in the collection for the specified time range.")
            return xr.Dataset()
        
        # Convert timestamps to datetime objects (normalized to midnight)
        dates = []
        for d in dates_info:
            date_obj = datetime.fromtimestamp(d / 1000)
            # Normalize to midnight
            midnight_date = datetime(date_obj.year, date_obj.month, date_obj.day)
            dates.append(midnight_date)
        
        print(f"Found {len(dates)} dates in collection: {dates[0]} to {dates[-1]}\n")
        
        # Sort dates for consistency
        dates.sort()
        
        # Initialize the dataset arrays
        data_arrays = {}
        for band in self.bands:
            data_arrays[band] = np.zeros((len(dates), height, width))
        
        # Track successful dates and corresponding indices
        successful_indices = []
        successful_dates = []
        
        # Download each image with progress tracking
        from tqdm import tqdm
        for i, img_date in enumerate(tqdm(dates, desc="Processing dates")):
            try:
                # Create date range filters for the exact date (midnight to midnight)
                date_str = img_date.strftime('%Y-%m-%d')
                
                # For dates within the valid range (1980-01-01 onwards)
                if img_date < datetime(1980, 1, 1):
                    print(f"Warning: Date {date_str} is before the dataset's valid range (1980-01-01)")
                    continue
                    
                # Create filter for the specific day
                day_start = ee.Date(date_str)
                day_end = day_start.advance(1, 'day')
                
                # Filter to get image for this day
                filtered = self.collection.filter(ee.Filter.date(day_start, day_end))
                
                # Check if the filtered collection is empty
                count = filtered.size().getInfo()
                if count == 0:
                    print(f"Warning: No image found for date: {date_str}")
                    continue
                
                # Get the first image of that day
                img = filtered.first()
                
                # Get band names from the image to verify
                available_bands = img.bandNames().getInfo()
                
                # Check if all requested bands are available
                missing_bands = [band for band in self.bands if band not in available_bands]
                if missing_bands:
                    print(f"Warning: The following bands are not available for {date_str}: {', '.join(missing_bands)}")
                
                # Track if we got data for at least one band
                got_data_for_date = False
                
                # Process each band
                for j, band in enumerate(self.bands):
                    # Skip missing bands
                    if band not in available_bands:
                        continue
                        
                    try:
                        print(f"  Downloading band '{band}' for {date_str}...")
                        
                        # Get pixel values
                        band_data = img.select(band).sampleRectangle(
                            region=region,
                            properties=None,
                            defaultValue=0,
                            scale=scale
                        ).getInfo()
                        
                        # Extract the array
                        if band in band_data:
                            array = np.array(band_data[band])
                            if array.shape[0] == height and array.shape[1] == width:
                                data_arrays[band][i] = array
                                got_data_for_date = True
                                print(f"  Successfully retrieved {array.shape[0]}x{array.shape[1]} data points")
                            else:
                                print(f"  Warning: Array shape mismatch. Expected {height}x{width}, got {array.shape}")
                        else:
                            print(f"  Warning: No data returned for band {band}")
                            
                    except Exception as band_error:
                        print(f"  Error processing band {band}: {str(band_error)}")
                
                # If we got data for at least one band, mark this date as successful
                if got_data_for_date:
                    print(f"Successfully processed date: {date_str}")
                    successful_indices.append(i)
                    successful_dates.append(img_date)
                else:
                    print(f"No data retrieved for any band on {date_str}")
                    
            except Exception as date_error:
                print(f"Error processing date {img_date.strftime('%Y-%m-%d')}: {str(date_error)}")
        
        # If we have no successful dates, return an empty dataset
        if not successful_dates:
            print("\nNo data could be retrieved for any date. Returning empty dataset.")
            
            # Create minimal valid xarray Dataset - works better than empty
            coords = {
                'time': [datetime.now()],  # Single timestamp
                'lat': [np.mean([ymin, ymax])],  # Single latitude
                'lon': [np.mean([xmin, xmax])]   # Single longitude
            }
            
            data_vars = {
                band: (['time', 'lat', 'lon'], np.zeros((1, 1, 1)))
                for band in self.bands
            }
            
            # Add attributes
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            ds.attrs['description'] = f"Empty dataset - no data found for {self.ee_id}"
            ds.attrs['warning'] = "This dataset contains no valid data"
            ds.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return ds
        
        print(f"\nSuccessfully processed {len(successful_dates)} out of {len(dates)} dates")
        
        # Filter data arrays to only include successful dates
        filtered_data_arrays = {}
        for band in self.bands:
            filtered_data_arrays[band] = data_arrays[band][successful_indices]
        
        # Create the xarray dataset
        coords = {
            'time': successful_dates,
            'lat': np.linspace(ymax, ymin, height),
            'lon': np.linspace(xmin, xmax, width)
        }
        
        data_vars = {
            band: (['time', 'lat', 'lon'], filtered_data_arrays[band])
            for band in self.bands
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
                maxPixels=1e9
            ).getInfo()
            
            return {band: result.get(band) for band in self.bands}
        else:
            # Get time series with spatial aggregation
            return self.get_time_series_average()