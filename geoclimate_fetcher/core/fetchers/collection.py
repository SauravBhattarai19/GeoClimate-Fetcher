"""
ImageCollection fetcher module for retrieving time-series data from Earth Engine.
"""

import os
import ee
import pandas as pd
import xarray as xr
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from datetime import datetime, date
from pathlib import Path

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
            
            # Create a feature with properties
            return ee.Feature(None, {
                'date': date_string,
                **{band: means.get(band) for band in self.bands}
            })
            
        # Map over the collection and get average values
        features = self.collection.map(extract_date_value)
        
        # Get the data as a Pandas DataFrame
        feature_collection = features.getInfo()
        
        # Extract the properties we need into a DataFrame
        if not feature_collection.get('features'):
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
        
        # Get the dates
        dates = self.collection.aggregate_array('system:time_start').getInfo()
        dates = [datetime.fromtimestamp(d / 1000) for d in dates]
        
        # Initialize the dataset arrays
        data_arrays = {}
        for band in self.bands:
            data_arrays[band] = np.zeros((len(dates), height, width))
        
        # Download each image
        for i, img_date in enumerate(dates):
            # Filter to specific date
            img = self.collection.filter(ee.Filter.date(img_date, 
                                                      img_date.replace(hour=23, minute=59, second=59))).first()
            
            # Get the pixel values
            for j, band in enumerate(self.bands):
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
        
        # Create the xarray dataset
        coords = {
            'time': dates,
            'lat': np.linspace(ymax, ymin, height),
            'lon': np.linspace(xmin, xmax, width)
        }
        
        data_vars = {
            band: (['time', 'lat', 'lon'], data_arrays[band])
            for band in self.bands
        }
        
        return xr.Dataset(data_vars=data_vars, coords=coords)
        
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