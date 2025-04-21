"""
Authentication module for Google Earth Engine.
"""
import os
from typing import Optional, Dict, Any, Callable
import json
import ee

class GEEAuth:
    """Class to handle Google Earth Engine authentication."""
    
    def __init__(self):
        """Initialize the GEE authentication handler."""
        self._initialized = False
        
    def initialize(self, project_id: str, service_account: Optional[str] = None, 
                  key_file: Optional[str] = None) -> bool:
        """
        Initialize the Earth Engine API with the provided credentials.
        
        Args:
            project_id: The Google Cloud project ID
            service_account: Optional service account email
            key_file: Optional path to service account key file
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            # Initialize with service account if provided
            if service_account and key_file:
                credentials = ee.ServiceAccountCredentials(service_account, key_file)
                ee.Initialize(credentials, project=project_id)
            # Otherwise use user account (requires prior authentication with earthengine-authenticate)
            else:
                ee.Initialize(project=project_id)
                
            self._initialized = True
            return True
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            self._initialized = False
            return False
            
    def is_initialized(self) -> bool:
        """
        Check if Earth Engine has been initialized.
        
        Returns:
            bool: True if Earth Engine is initialized, False otherwise
        """
        return self._initialized
    
    @staticmethod
    def test_connection() -> bool:
        """
        Test the connection to Earth Engine by making a simple API call.
        
        Returns:
            bool: True if the connection is working, False otherwise
        """
        try:
            # Try to get info for a simple image to test connection
            ee.Image("USGS/SRTMGL1_003").getInfo()
            return True
        except Exception:
            return False


def authenticate(project_id: str, service_account: Optional[str] = None, 
                key_file: Optional[str] = None) -> GEEAuth:
    """
    Authenticate with Google Earth Engine.
    
    Args:
        project_id: The Google Cloud project ID
        service_account: Optional service account email
        key_file: Optional path to service account key file
        
    Returns:
        GEEAuth: An instance of the GEEAuth class
    """
    auth = GEEAuth()
    auth.initialize(project_id, service_account, key_file)
    return auth