"""
Static raster fetcher module for retrieving static raster data from Earth Engine.
"""

import os
import ee
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from pathlib import Path

class StaticRasterFetcher:
    """Class for fetching and processing Earth Engine static raster data."""
    
    def __init__(self, ee_id: str, bands: List[str], geometry: ee.Geometry):
        """
        Initialize the static raster fetcher.
        
        Args:
            ee_id: Earth Engine Image ID
            bands: List of band names to use
            geometry: Earth Engine geometry to use as AOI
        """
        self.ee_id = ee_id
        self.bands = bands
        self.geometry = geometry
        self.image = ee.Image(ee_id)
        
    def select_bands(self, bands: Optional[List[str]] = None) -> 'StaticRasterFetcher':
        """
        Select specific bands from the image.
        
        Args:
            bands: List of band names to select. If None, uses the bands
                  provided at initialization.
                  
        Returns:
            Self for method chaining
        """
        if bands is not None:
            self.bands = bands
            
        if self.bands:
            self.image = self.image.select(self.bands)
            
        return self
        
    def get_mean_values(self) -> Dict[str, float]:
        """
        Calculate the mean values of selected bands within the AOI.
        
        Returns:
            Dictionary mapping band names to mean values
        """
        # Calculate mean value for each band in the AOI
        means = self.image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.geometry,
            scale=30,  # Can be adjusted based on data resolution
            maxPixels=1e9
        ).getInfo()
        
        return {band: means.get(band) for band in self.bands}
        
    def get_zonal_statistics(self, statistics: List[str] = ['mean', 'min', 'max', 'stdDev']) -> Dict[str, Dict[str, float]]:
        """
        Calculate zonal statistics for the selected bands within the AOI.
        
        Args:
            statistics: List of statistics to calculate
            
        Returns:
            Nested dictionary mapping band names to statistics
        """
        # Create a reducer with all requested statistics
        reducer = ee.Reducer.mean()
        
        if 'min' in statistics:
            reducer = reducer.combine(ee.Reducer.min(), "", True)
            
        if 'max' in statistics:
            reducer = reducer.combine(ee.Reducer.max(), "", True)
            
        if 'stdDev' in statistics:
            reducer = reducer.combine(ee.Reducer.stdDev(), "", True)
            
        if 'sum' in statistics:
            reducer = reducer.combine(ee.Reducer.sum(), "", True)
            
        # Calculate statistics for each band in the AOI
        stats = self.image.reduceRegion(
            reducer=reducer,
            geometry=self.geometry,
            scale=30,  # Can be adjusted based on data resolution
            maxPixels=1e9
        ).getInfo()
        
        # Organize results by band
        results = {}
        for band in self.bands:
            band_stats = {}
            
            # For each statistic, extract the value
            for stat in statistics:
                key = f"{band}_{stat}" if stat != 'mean' else band
                if key in stats:
                    band_stats[stat] = stats[key]
                    
            results[band] = band_stats
            
        return results
        
    def get_pixel_values(self, scale: float = 30.0) -> Dict[str, np.ndarray]:
        """
        Get raw pixel values within the AOI.
        
        Args:
            scale: Pixel resolution in meters
            
        Returns:
            Dictionary mapping band names to NumPy arrays of pixel values
        """
        # Define the region as the AOI bounds
        bounds = self.geometry.bounds().getInfo()['coordinates'][0]
        xs = [p[0] for p in bounds]
        ys = [p[1] for p in bounds]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
        
        # Sample rectangle to get pixel values
        pixels = self.image.sampleRectangle(
            region=region,
            properties=None,
            defaultValue=0,
        ).getInfo()
        
        # Extract the arrays for each band
        result = {}
        for band in self.bands:
            if band in pixels:
                result[band] = np.array(pixels[band])
                
        return result
        
    def create_hillshade(self, elevation_band: str, azimuth: float = 315.0, 
                        zenith: float = 45.0) -> ee.Image:
        """
        Create a hillshade from a DEM band.
        
        Args:
            elevation_band: Band containing elevation data
            azimuth: Sun azimuth angle (0-360)
            zenith: Sun zenith angle (0-90)
            
        Returns:
            Earth Engine Image with hillshade
        """
        if elevation_band not in self.bands:
            raise ValueError(f"Band '{elevation_band}' not in selected bands")
            
        # Select the elevation band
        dem = self.image.select(elevation_band)
        
        # Convert zenith to radians
        zenith_rad = ee.Number(zenith).multiply(np.pi).divide(180.0)
        
        # Convert azimuth to radians and reverse it (different definition in EE)
        azimuth_rad = ee.Number(azimuth).subtract(90).multiply(np.pi).divide(180.0)
        
        # Calculate slope and aspect
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        
        # Convert slope and aspect to radians
        slope_rad = slope.multiply(np.pi).divide(180.0)
        aspect_rad = aspect.multiply(np.pi).divide(180.0)
        
        # Calculate hillshade
        hillshade = slope_rad.cos().multiply(zenith_rad.cos()).add(
            slope_rad.sin().multiply(zenith_rad.sin()).multiply(
                aspect_rad.subtract(azimuth_rad).cos()
            )
        )
        
        # Scale the hillshade values
        return hillshade.multiply(255).byte()
        
    def get_contours(self, elevation_band: str, interval: float = 100.0) -> ee.FeatureCollection:
        """
        Generate contour lines from a DEM band.
        
        Args:
            elevation_band: Band containing elevation data
            interval: Elevation interval between contour lines
            
        Returns:
            Earth Engine FeatureCollection with contour lines
        """
        if elevation_band not in self.bands:
            raise ValueError(f"Band '{elevation_band}' not in selected bands")
            
        # Select the elevation band
        dem = self.image.select(elevation_band)
        
        # Generate contours
        contours = ee.Algorithms.GeometryConstructors.MultiLineString(
            ee.Terrain.contour(
                image=dem,
                region=self.geometry,
                interval=interval
            )
        )
        
        return ee.FeatureCollection([ee.Feature(contours)])