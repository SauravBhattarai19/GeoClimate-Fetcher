"""
Tests for the metadata module.
"""

import pytest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import os
from pathlib import Path

from geoclimate_fetcher.core.metadata import MetadataCatalog

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Dataset Name': ['Test Dataset 1', 'Test Dataset 2', 'Test Dataset 3'],
        'Earth Engine ID': ['test/dataset1', 'test/dataset2', 'test/dataset3'],
        'Snippet Type': ['Image', 'ImageCollection', 'Image'],
        'Provider': ['Provider A', 'Provider B', 'Provider A'],
        'Start Date': ['2020-01-01', '2019-01-01', '2018-01-01'],
        'End Date': ['2021-01-01', '2020-01-01', '2019-01-01'],
        'Pixel Size (m)': [30.0, 250.0, 1000.0],
        'Temporal Resolution': ['Daily', 'Monthly', 'Annual'],
        'Band Names': ['band1,band2', 'band3,band4,band5', 'band6'],
        'Band Units': ['m,m', 'mm/day,kg/m^2,m^3', 'degrees'],
        'Description': ['Test description 1', 'Test description 2', 'Test description 3']
    })

@pytest.fixture
def mock_catalog(sample_df):
    """Create a mock catalog with predefined data."""
    with patch('geoclimate_fetcher.core.metadata.pd.read_csv') as mock_read_csv:
        # Mock read_csv to return our sample DataFrame
        mock_read_csv.return_value = sample_df
        
        # Mock Path.glob to return a list of CSV files
        with patch('geoclimate_fetcher.core.metadata.Path.glob') as mock_glob:
            mock_glob.return_value = [
                Path('data/category1.csv'),
                Path('data/category2.csv')
            ]
            
            # Create and return the catalog
            catalog = MetadataCatalog()
            
            # Set up the catalog's internal state
            catalog._catalog = {
                'category1': sample_df.iloc[:2].copy(),
                'category2': sample_df.iloc[2:].copy()
            }
            catalog._all_datasets = sample_df
            
            yield catalog

def test_load_catalogs(mock_catalog):
    """Test that catalogs are loaded correctly."""
    assert len(mock_catalog.categories) == 2
    assert 'category1' in mock_catalog.categories
    assert 'category2' in mock_catalog.categories
    assert len(mock_catalog.all_datasets) == 3

def test_get_dataset_by_name(mock_catalog):
    """Test retrieving a dataset by name."""
    dataset = mock_catalog.get_dataset_by_name('Test Dataset 1')
    assert dataset is not None
    assert dataset['Earth Engine ID'] == 'test/dataset1'
    
    # Test non-existent dataset
    dataset = mock_catalog.get_dataset_by_name('Non-existent Dataset')
    assert dataset is None

def test_get_datasets_by_category(mock_catalog):
    """Test retrieving datasets by category."""
    datasets = mock_catalog.get_datasets_by_category('category1')
    assert len(datasets) == 2
    assert datasets.iloc[0]['Dataset Name'] == 'Test Dataset 1'
    assert datasets.iloc[1]['Dataset Name'] == 'Test Dataset 2'

def test_search_datasets(mock_catalog):
    """Test searching for datasets."""
    # Test search by name
    results = mock_catalog.search_datasets('Dataset 1')
    assert len(results) == 1
    assert results.iloc[0]['Dataset Name'] == 'Test Dataset 1'
    
    # Test search by description
    results = mock_catalog.search_datasets('description 2')
    assert len(results) == 1
    assert results.iloc[0]['Description'] == 'Test description 2'
    
    # Test search with no results
    results = mock_catalog.search_datasets('nonexistent term')
    assert len(results) == 0

def test_get_bands_for_dataset(mock_catalog):
    """Test retrieving bands for a dataset."""
    bands = mock_catalog.get_bands_for_dataset('Test Dataset 2')
    assert bands == ['band3', 'band4', 'band5']
    
    # Test non-existent dataset
    bands = mock_catalog.get_bands_for_dataset('Non-existent Dataset')
    assert bands == []

def test_get_date_range(mock_catalog):
    """Test retrieving date range for a dataset."""
    start_date, end_date = mock_catalog.get_date_range('Test Dataset 1')
    assert start_date == '2020-01-01'
    assert end_date == '2021-01-01'
    
    # Test non-existent dataset
    start_date, end_date = mock_catalog.get_date_range('Non-existent Dataset')
    assert start_date is None
    assert end_date is None

def test_get_ee_id(mock_catalog):
    """Test retrieving Earth Engine ID for a dataset."""
    ee_id = mock_catalog.get_ee_id('Test Dataset 3')
    assert ee_id == 'test/dataset3'
    
    # Test non-existent dataset
    ee_id = mock_catalog.get_ee_id('Non-existent Dataset')
    assert ee_id is None

def test_get_snippet_type(mock_catalog):
    """Test retrieving snippet type for a dataset."""
    snippet_type = mock_catalog.get_snippet_type('Test Dataset 2')
    assert snippet_type == 'ImageCollection'
    
    # Test non-existent dataset
    snippet_type = mock_catalog.get_snippet_type('Non-existent Dataset')
    assert snippet_type is None