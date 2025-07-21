"""
Tests for the fetcher modules.
"""

import pytest
from unittest.mock import patch, MagicMock
import ee
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, date

from geoclimate_fetcher.core.fetchers import ImageCollectionFetcher, StaticRasterFetcher

# Sample data for testing
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
    """Mock Earth Engine API for fetcher tests."""
    with patch('geoclimate_fetcher.core.fetchers.collection.ee') as mock_ee:
        # Mock Geometry
        mock_geometry = MagicMock(spec=ee.Geometry)
        mock_geometry.bounds().getInfo.return_value = {
            "coordinates": [
                [[-122.6, 37.2], [-122.6, 37.9], [-121.8, 37.9], [-121.8, 37.2], [-122.6, 37.2]]
            ]
        }
        
        # Mock ImageCollection
        mock_collection = MagicMock(spec=ee.ImageCollection)
        mock_ee.ImageCollection.return_value = mock_collection
        
        # Mock Image
        mock_image = MagicMock(spec=ee.Image)
        mock_ee.Image.return_value = mock_image
        
        # Mock Feature
        mock_feature = MagicMock(spec=ee.Feature)
        mock_ee.Feature.return_value = mock_feature
        
        # Mock collection methods
        mock_collection.filterDate.return_value = mock_collection
        mock_collection.select.return_value = mock_collection
        mock_collection.map.return_value = mock_collection
        mock_collection.getInfo.return_value = {
            "features": [
                {
                    "properties": {
                        "date": "2021-01-01",
                        "band1": 10.5,
                        "band2": 20.3
                    }
                },
                {
                    "properties": {
                        "date": "2021-01-02",
                        "band1": 11.2,
                        "band2": 19.8
                    }
                }
            ]
        }
        
        # Mock image methods
        mock_image.select.return_value = mock_image
        mock_image.reduceRegion.return_value = ee.Dictionary({
            "band1": 15.3,
            "band2": 22.7
        })
        mock_reduceRegion = MagicMock()
        mock_reduceRegion.getInfo.return_value = {
            "band1": 15.3,
            "band2": 22.7
        }
        mock_image.reduceRegion.return_value = mock_reduceRegion
        
        # Mock Dictionary
        mock_ee.Dictionary.return_value = MagicMock()
        
        # For collection methods that return dates
        mock_collection.aggregate_array.return_value = MagicMock()
        mock_collection.aggregate_array().getInfo.return_value = [
            1609459200000,  # 2021-01-01
            1609545600000   # 2021-01-02
        ]
        
        # For sample rectangle
        mock_sample_result = {"band1": [[1, 2], [3, 4]], "band2": [[5, 6], [7, 8]]}
        mock_image.sampleRectangle.return_value = MagicMock()
        mock_image.sampleRectangle().getInfo.return_value = mock_sample_result
        
        yield mock_ee

@pytest.fixture
def geometry(mock_ee):
    """Create a mock Earth Engine geometry."""
    return MagicMock(spec=ee.Geometry)

@pytest.fixture
def image_collection_fetcher(mock_ee, geometry):
    """Create an ImageCollectionFetcher instance with mocked dependencies."""
    return ImageCollectionFetcher("test/dataset", ["band1", "band2"], geometry)

@pytest.fixture
def static_raster_fetcher(mock_ee, geometry):
    """Create a StaticRasterFetcher instance with mocked dependencies."""
    with patch('geoclimate_fetcher.core.fetchers.static.ee') as static_mock_ee:
        # Reuse the same mock setup
        for key in dir(mock_ee):
            if not key.startswith('__'):
                setattr(static_mock_ee, key, getattr(mock_ee, key))
                
        return StaticRasterFetcher("test/dataset", ["band1", "band2"], geometry)

def test_image_collection_filter_dates(image_collection_fetcher, mock_ee):
    """Test filtering ImageCollection by date range."""
    start_date = "2021-01-01"
    end_date = "2021-01-31"
    
    result = image_collection_fetcher.filter_dates(start_date, end_date)
    
    assert result == image_collection_fetcher
    mock_ee.ImageCollection.return_value.filterDate.assert_called_once_with(start_date, end_date)

def test_image_collection_select_bands(image_collection_fetcher, mock_ee):
    """Test selecting bands from ImageCollection."""
    # Test with default bands
    result = image_collection_fetcher.select_bands()
    
    assert result == image_collection_fetcher
    mock_ee.ImageCollection.return_value.select.assert_called_with(["band1", "band2"])
    
    # Test with new bands
    new_bands = ["band3", "band4"]
    result = image_collection_fetcher.select_bands(new_bands)
    
    assert result == image_collection_fetcher
    assert image_collection_fetcher.bands == new_bands
    mock_ee.ImageCollection.return_value.select.assert_called_with(new_bands)

def test_image_collection_get_time_series_average(image_collection_fetcher, mock_ee):
    """Test getting time series average from ImageCollection."""
    df = image_collection_fetcher.get_time_series_average()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "date" in df.columns
    assert "band1" in df.columns
    assert "band2" in df.columns
    assert df["band1"].iloc[0] == 10.5
    assert df["band2"].iloc[1] == 19.8

def test_static_raster_get_mean_values(static_raster_fetcher, mock_ee):
    """Test getting mean values from static raster."""
    mean_values = static_raster_fetcher.get_mean_values()
    
    assert isinstance(mean_values, dict)
    assert mean_values["band1"] == 15.3
    assert mean_values["band2"] == 22.7
    mock_ee.Image.return_value.reduceRegion.assert_called_once()

def test_static_raster_get_zonal_statistics(static_raster_fetcher, mock_ee):
    """Test getting zonal statistics from static raster."""
    # Mock reduceRegion to return stats
    mock_ee.Reducer.mean.return_value = MagicMock()
    mock_ee.Reducer.min.return_value = MagicMock()
    mock_ee.Reducer.max.return_value = MagicMock()
    mock_ee.Reducer.stdDev.return_value = MagicMock()
    
    # Mock combine method
    mock_ee.Reducer.mean().combine.return_value = MagicMock()
    mock_ee.Reducer.mean().combine().combine.return_value = MagicMock()
    mock_ee.Reducer.mean().combine().combine().combine.return_value = MagicMock()
    
    # Mock reduceRegion getInfo result
    mock_reduce_result = MagicMock()
    mock_reduce_result.getInfo.return_value = {
        "band1": 15.3,
        "band1_min": 5.0,
        "band1_max": 25.0,
        "band1_stdDev": 4.2,
        "band2": 22.7,
        "band2_min": 10.0,
        "band2_max": 30.0,
        "band2_stdDev": 5.1
    }
    static_raster_fetcher.image.reduceRegion.return_value = mock_reduce_result
    
    stats = static_raster_fetcher.get_zonal_statistics()
    
    assert isinstance(stats, dict)
    assert "band1" in stats
    assert "band2" in stats
    assert stats["band1"]["mean"] == 15.3
    assert stats["band1"]["min"] == 5.0
    assert stats["band2"]["max"] == 30.0
    assert stats["band2"]["stdDev"] == 5.1

def test_static_raster_get_pixel_values(static_raster_fetcher, mock_ee):
    """Test getting pixel values from static raster."""
    pixel_values = static_raster_fetcher.get_pixel_values()
    
    assert isinstance(pixel_values, dict)
    assert "band1" in pixel_values
    assert "band2" in pixel_values
    assert isinstance(pixel_values["band1"], np.ndarray)
    assert pixel_values["band1"].shape == (2, 2)
    assert pixel_values["band1"][0, 0] == 1
    assert pixel_values["band2"][1, 1] == 8