"""
Tests for the exporter module.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import ee
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import os

from geoclimate_fetcher.core.exporter import GEEExporter

@pytest.fixture
def exporter():
    """Create a GEEExporter instance."""
    return GEEExporter(max_chunk_size=1000000, timeout=10)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=3),
        'value1': [10.5, 11.2, 9.8],
        'value2': [20.3, 19.8, 21.1]
    })

@pytest.fixture
def sample_ds():
    """Create a sample xarray Dataset for testing."""
    # Create a simple 2D dataset
    data = np.random.rand(3, 4, 5)  # time, lat, lon
    return xr.Dataset(
        data_vars={
            'var1': (['time', 'lat', 'lon'], data),
            'var2': (['time', 'lat', 'lon'], data * 2)
        },
        coords={
            'time': pd.date_range('2021-01-01', periods=3),
            'lat': np.linspace(37.2, 37.9, 4),
            'lon': np.linspace(-122.6, -121.8, 5)
        }
    )

def test_export_time_series_to_csv(exporter, sample_df):
    """Test exporting time series to CSV."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_output.csv"
        
        # Test export
        result = exporter.export_time_series_to_csv(sample_df, output_path)
        
        # Check result
        assert result == output_path
        assert output_path.exists()
        
        # Verify content
        df_read = pd.read_csv(output_path)
        assert len(df_read) == len(sample_df)
        assert 'value1' in df_read.columns
        assert 'value2' in df_read.columns

def test_export_gridded_data_to_netcdf(exporter, sample_ds):
    """Test exporting gridded data to NetCDF."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_output.nc"
        
        # Test export
        result = exporter.export_gridded_data_to_netcdf(sample_ds, output_path)
        
        # Check result
        assert result == output_path
        assert output_path.exists()
        
        # Verify content (basic check)
        ds_read = xr.open_dataset(output_path)
        assert 'var1' in ds_read.data_vars
        assert 'var2' in ds_read.data_vars
        assert ds_read.dims['time'] == 3
        assert ds_read.dims['lat'] == 4
        assert ds_read.dims['lon'] == 5

def test_estimate_export_size(exporter):
    """Test estimating export size."""
    # Mock image and geometry
    image = MagicMock(spec=ee.Image)
    geometry = MagicMock(spec=ee.Geometry)
    
    # Mock methods
    image.bandNames().getInfo.return_value = ['band1', 'band2']
    geometry.area().getInfo.return_value = 100000000  # 100 km²
    
    # Test with 30m resolution
    size = exporter.estimate_export_size(image, geometry, 30.0)
    
    # Expected: 100,000,000 m² / (30m * 30m) * 2 bands * 4 bytes
    expected = 100000000 / (30 * 30) * 2 * 4
    assert size == expected

@patch('geoclimate_fetcher.core.exporter.time')
def test_wait_for_task_completed(mock_time, exporter):
    """Test waiting for a completed task."""
    # Mock task
    task = MagicMock(spec=ee.batch.Task)
    
    # Mock task status
    task.status.side_effect = [
        {'state': 'RUNNING', 'progress': 0.3},
        {'state': 'RUNNING', 'progress': 0.7},
        {'state': 'COMPLETED'}
    ]
    
    # Test wait
    result = exporter._wait_for_task(task)
    
    # Check result
    assert result is True
    assert task.status.call_count == 3
    assert mock_time.sleep.call_count == 2

@patch('geoclimate_fetcher.core.exporter.time')
def test_wait_for_task_failed(mock_time, exporter):
    """Test waiting for a failed task."""
    # Mock task
    task = MagicMock(spec=ee.batch.Task)
    
    # Mock task status
    task.status.side_effect = [
        {'state': 'RUNNING', 'progress': 0.3},
        {'state': 'FAILED', 'error_message': 'Test error'}
    ]
    
    # Test wait
    result = exporter._wait_for_task(task)
    
    # Check result
    assert result is False
    assert task.status.call_count == 2
    assert mock_time.sleep.call_count == 1

@patch('geoclimate_fetcher.core.exporter.ee.batch.Export.image.toDrive')
def test_export_image_to_drive(mock_export_to_drive, exporter):
    """Test exporting image to Google Drive."""
    # Mock image and geometry
    image = MagicMock(spec=ee.Image)
    geometry = MagicMock(spec=ee.Geometry)
    
    # Mock methods
    geometry.bounds().getInfo.return_value = {'coordinates': [[[1, 2], [3, 4]]]}
    
    # Mock task
    mock_task = MagicMock()
    mock_export_to_drive.return_value = mock_task
    
    # Test export without waiting
    task_id = exporter.export_image_to_drive(
        image, "test_file", "test_folder", geometry, 
        scale=30.0, wait=False
    )
    
    # Check result
    assert task_id == mock_task.id
    mock_task.start.assert_called_once()
    mock_export_to_drive.assert_called_once()

@patch('geoclimate_fetcher.core.exporter.ee.batch.Export.table.toDrive')
def test_export_table_to_drive(mock_export_to_drive, exporter):
    """Test exporting table to Google Drive."""
    # Mock feature collection
    fc = MagicMock(spec=ee.FeatureCollection)
    
    # Mock task
    mock_task = MagicMock()
    mock_export_to_drive.return_value = mock_task
    
    # Test export without waiting
    task_id = exporter.export_table_to_drive(
        fc, "test_file", "test_folder", wait=False
    )
    
    # Check result
    assert task_id == mock_task.id
    mock_task.start.assert_called_once()
    mock_export_to_drive.assert_called_once()