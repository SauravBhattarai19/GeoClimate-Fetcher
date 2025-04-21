"""
Tests for the authentication module.
"""

import pytest
from unittest.mock import patch, MagicMock
import ee

from geoclimate_fetcher.core.gee_auth import GEEAuth, authenticate

@pytest.fixture
def mock_ee():
    """Mock Earth Engine API."""
    with patch('geoclimate_fetcher.core.gee_auth.ee') as mock_ee:
        # Mock successful initialization
        mock_ee.Initialize.return_value = None
        
        # Create mock credentials
        mock_credentials = MagicMock()
        mock_ee.ServiceAccountCredentials.return_value = mock_credentials
        
        yield mock_ee

def test_initialize_with_project_id(mock_ee):
    """Test initialization with project ID only."""
    auth = GEEAuth()
    result = auth.initialize("test-project")
    
    assert result is True
    assert auth.is_initialized() is True
    mock_ee.Initialize.assert_called_once_with(project="test-project")

def test_initialize_with_service_account(mock_ee):
    """Test initialization with service account credentials."""
    auth = GEEAuth()
    result = auth.initialize(
        "test-project", 
        service_account="service@example.com",
        key_file="key.json"
    )
    
    assert result is True
    assert auth.is_initialized() is True
    mock_ee.ServiceAccountCredentials.assert_called_once_with(
        "service@example.com", "key.json"
    )
    mock_ee.Initialize.assert_called_once()

def test_initialize_failure(mock_ee):
    """Test initialization failure."""
    mock_ee.Initialize.side_effect = Exception("Auth failed")
    
    auth = GEEAuth()
    result = auth.initialize("test-project")
    
    assert result is False
    assert auth.is_initialized() is False

def test_test_connection_success(mock_ee):
    """Test successful connection test."""
    # Mock successful getInfo call
    mock_image = MagicMock()
    mock_ee.Image.return_value = mock_image
    mock_image.getInfo.return_value = {"bands": []}
    
    auth = GEEAuth()
    auth.initialize("test-project")
    
    result = auth.test_connection()
    assert result is True

def test_test_connection_failure(mock_ee):
    """Test failed connection test."""
    # Mock failed getInfo call
    mock_image = MagicMock()
    mock_ee.Image.return_value = mock_image
    mock_image.getInfo.side_effect = Exception("Connection failed")
    
    auth = GEEAuth()
    auth.initialize("test-project")
    
    result = auth.test_connection()
    assert result is False

def test_authenticate_helper():
    """Test authenticate helper function."""
    with patch('geoclimate_fetcher.core.gee_auth.GEEAuth') as MockGEEAuth:
        # Mock GEEAuth instance
        mock_auth = MagicMock()
        MockGEEAuth.return_value = mock_auth
        
        # Call authenticate
        result = authenticate("test-project")
        
        # Check results
        assert result == mock_auth
        mock_auth.initialize.assert_called_once_with(
            "test-project", None, None
        )