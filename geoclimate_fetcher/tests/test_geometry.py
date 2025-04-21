"""
Tests for the geometry module.
"""

import pytest
from unittest.mock import patch, MagicMock
import ee
import json
import geopandas as gpd
from pathlib import Path

from geoclimate_fetcher.core.geometry import GeometryHandler

# Sample GeoJSON for testing
SAMPLE_GEOJSON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.6, 37.2],
            [-122.6, 37.9],
            [-121.8, 37.9],
            [-121.8, 37.2],
            [-122.6, 37.2]
        ]
    ]
}

@pytest.fixture
def mock_ee():
    """Mock Earth Engine API for geometry tests."""
    with patch('geoclimate_fetcher.core.geometry.ee') as mock_ee:
        # Mock Geometry class
        mock_geometry = MagicMock()
        mock_ee.Geometry.return_value = mock_geometry
        
        # Mock area calculation
        mock_area = MagicMock()
        mock_geometry.area.return_value = mock_area
        mock_area.getInfo.return_value = 1000000000  # 1000 km²
        
        # Mock bounds calculation
        mock_bounds = MagicMock()
        mock_geometry.bounds.return_value = mock_bounds
        mock_bounds.getInfo.return_value = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.6, 37.2],
                    [-122.6, 37.9],
                    [-121.8, 37.9],
                    [-121.8, 37.2],
                    [-122.6, 37.2]
                ]
            ]
        }
        
        # Mock Feature creation
        mock_feature = MagicMock()
        mock_ee.Feature.return_value = mock_feature
        
        yield mock_ee

@pytest.fixture
def geometry_handler(mock_ee):
    """Create a GeometryHandler instance with mocked EE."""
    return GeometryHandler()

def test_set_geometry_from_geojson(geometry_handler, mock_ee):
    """Test setting geometry from GeoJSON."""
    result = geometry_handler.set_geometry_from_geojson(SAMPLE_GEOJSON, "test_area")
    
    assert result == mock_ee.Geometry.return_value
    assert geometry_handler.current_geometry == mock_ee.Geometry.return_value
    assert geometry_handler.current_geometry_name == "test_area"
    mock_ee.Geometry.assert_called_once_with(SAMPLE_GEOJSON)

def test_set_geometry_from_file(geometry_handler, mock_ee):
    """Test setting geometry from file."""
    # Mock geopandas.read_file
    with patch('geoclimate_fetcher.core.geometry.gpd.read_file') as mock_read_file:
        # Create mock GeoDataFrame
        mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
        mock_gdf.crs.to_epsg.return_value = 4326  # WGS84
        mock_gdf.geometry.to_json.return_value = json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": SAMPLE_GEOJSON
            }]
        })
        mock_read_file.return_value = mock_gdf
        
        result = geometry_handler.set_geometry_from_file("test.shp", "test_shapefile")
        
        assert result == mock_ee.Geometry.return_value
        assert geometry_handler.current_geometry == mock_ee.Geometry.return_value
        assert geometry_handler.current_geometry_name == "test_shapefile"
        mock_read_file.assert_called_once_with("test.shp")

def test_set_geometry_from_drawn(geometry_handler, mock_ee):
    """Test setting geometry from drawn polygon."""
    result = geometry_handler.set_geometry_from_drawn(SAMPLE_GEOJSON)
    
    assert result == mock_ee.Geometry.return_value
    assert geometry_handler.current_geometry == mock_ee.Geometry.return_value
    assert geometry_handler.current_geometry_name == "drawn_aoi"
    mock_ee.Geometry.assert_called_once_with(SAMPLE_GEOJSON)

def test_get_geometry_area(geometry_handler, mock_ee):
    """Test getting geometry area."""
    # First set a geometry
    geometry_handler.set_geometry_from_geojson(SAMPLE_GEOJSON)
    
    # Test area calculation in km²
    area_km2 = geometry_handler.get_geometry_area(units='km2')
    assert area_km2 == 1000.0
    
    # Test area calculation in m²
    area_m2 = geometry_handler.get_geometry_area(units='m2')
    assert area_m2 == 1000000000

def test_get_geometry_bounds(geometry_handler, mock_ee):
    """Test getting geometry bounds."""
    # First set a geometry
    geometry_handler.set_geometry_from_geojson(SAMPLE_GEOJSON)
    
    bounds = geometry_handler.get_geometry_bounds()
    assert bounds == [-122.6, 37.2, -121.8, 37.9]

def test_geometry_to_ee_feature(geometry_handler, mock_ee):
    """Test converting geometry to EE Feature."""
    # First set a geometry
    geometry_handler.set_geometry_from_geojson(SAMPLE_GEOJSON)
    
    feature = geometry_handler.geometry_to_ee_feature()
    assert feature == mock_ee.Feature.return_value
    mock_ee.Feature.assert_called_once_with(geometry_handler.current_geometry)

def test_validate_geometry_valid(geometry_handler, mock_ee):
    """Test validating a valid geometry."""
    # First set a geometry
    geometry_handler.set_geometry_from_geojson(SAMPLE_GEOJSON)
    
    is_valid, error_msg = geometry_handler.validate_geometry()
    assert is_valid is True
    assert error_msg is None

def test_validate_geometry_invalid(geometry_handler, mock_ee):
    """Test validating an invalid geometry."""
    # First set a geometry
    geometry_handler.set_geometry_from_geojson(SAMPLE_GEOJSON)
    
    # Mock area calculation to raise an exception
    mock_area = geometry_handler.current_geometry.area.return_value
    mock_area.getInfo.side_effect = Exception("Invalid geometry")
    
    is_valid, error_msg = geometry_handler.validate_geometry()
    assert is_valid is False
    assert error_msg == "Invalid geometry"

def test_validate_geometry_no_geometry(geometry_handler):
    """Test validating when no geometry is set."""
    is_valid, error_msg = geometry_handler.validate_geometry()
    assert is_valid is False
    assert error_msg == "No geometry has been set"