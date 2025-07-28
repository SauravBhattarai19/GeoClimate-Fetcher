"""
Geometry utilities for handling Areas of Interest (AOI).
"""

import os
import json
import ee
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple

class GeometryHandler:
    """Class to handle Area of Interest (AOI) geometries for Earth Engine."""
    
    def __init__(self):
        """Initialize the geometry handler."""
        self._current_geometry = None
        self._current_geometry_name = None
        
    @property
    def current_geometry(self) -> Optional[ee.Geometry]:
        """
        Get the current geometry.
        
        Returns:
            Current Earth Engine geometry or None if not set
        """
        return self._current_geometry
        
    @property
    def current_geometry_name(self) -> Optional[str]:
        """
        Get the name of the current geometry.
        
        Returns:
            Name of the current geometry or None if not set
        """
        return self._current_geometry_name
        
    def set_geometry_from_geojson(self, geojson_dict: Dict[str, Any], name: str = "custom_aoi") -> ee.Geometry:
        """
        Set the current geometry from a GeoJSON dictionary.
        
        Args:
            geojson_dict: GeoJSON dictionary
            name: Name for the geometry
            
        Returns:
            Earth Engine geometry object
        """
        # Convert to EE geometry
        self._current_geometry = ee.Geometry(geojson_dict)
        self._current_geometry_name = name
        return self._current_geometry
        
    def set_geometry_from_file(self, file_path: Union[str, Path], name: Optional[str] = None) -> ee.Geometry:
        """
        Set the current geometry from a file (Shapefile, GeoJSON).
        
        Args:
            file_path: Path to the geometry file
            name: Optional name for the geometry (defaults to filename)
            
        Returns:
            Earth Engine geometry object
        """
        file_path = Path(file_path)
        
        if name is None:
            name = file_path.stem
            
        self._current_geometry_name = name
        
        # Read with GeoPandas
        gdf = gpd.read_file(str(file_path))
        
        # Check if CRS is WGS84 (EPSG:4326), if not reproject
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
            
        # Convert to GeoJSON
        geojson_dict = json.loads(gdf.geometry.to_json())
        
        # Handle different geometry types
        if geojson_dict["type"] == "FeatureCollection":
            # Grab the first feature for now, could be expanded to handle multiple features
            first_feature = geojson_dict["features"][0]
            geom_dict = first_feature["geometry"]
        else:
            geom_dict = geojson_dict
            
        # Create EE geometry
        self._current_geometry = ee.Geometry(geom_dict)
        return self._current_geometry
        
    def set_geometry_from_drawn(self, geojson_dict: Dict[str, Any]) -> ee.Geometry:
        """
        Set the current geometry from a drawn polygon.
        
        Args:
            geojson_dict: GeoJSON dictionary from drawing
            
        Returns:
            Earth Engine geometry object
        """
        return self.set_geometry_from_geojson(geojson_dict, "drawn_aoi")
        
    def get_geometry_area(self, units: str = 'km2') -> float:
        """
        Calculate the area of the current geometry.
        
        Args:
            units: Area units ('m2' or 'km2')
            
        Returns:
            Area in requested units
        """
        if self._current_geometry is None:
            raise ValueError("No geometry has been set")
            
        area_m2 = self._current_geometry.area().getInfo()
        
        if units.lower() == 'km2':
            return area_m2 / 1e6
        else:
            return area_m2
            
    def get_geometry_bounds(self) -> List[float]:
        """
        Get the bounding box of the current geometry.
        
        Returns:
            List of coordinates [xmin, ymin, xmax, ymax]
        """
        if self._current_geometry is None:
            raise ValueError("No geometry has been set")
            
        bounds = self._current_geometry.bounds().getInfo()
        coordinates = bounds['coordinates'][0]
        
        # Extract min/max coordinates
        xs = [p[0] for p in coordinates]
        ys = [p[1] for p in coordinates]
        
        return [min(xs), min(ys), max(xs), max(ys)]
        
    def geometry_to_ee_feature(self) -> ee.Feature:
        """
        Convert the current geometry to an Earth Engine Feature.
        
        Returns:
            Earth Engine Feature object
        """
        if self._current_geometry is None:
            raise ValueError("No geometry has been set")
            
        return ee.Feature(self._current_geometry)
        
    def validate_geometry(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the current geometry for use with Earth Engine.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self._current_geometry is None:
            return False, "No geometry has been set"
            
        try:
            # Try to execute a computation to see if the geometry is valid
            self._current_geometry.area().getInfo()
            return True, None
        except Exception as e:
            return False, str(e)