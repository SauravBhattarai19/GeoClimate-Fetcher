"""
Authentication module for Google Earth Engine.
"""
import os
from typing import Optional, Dict, Any, Callable
import json
import ee
import tempfile

# Define path for storing credentials
CREDENTIALS_FILE = os.path.expanduser("~/.geoclimate-fetcher/credentials.json")

class GEEAuth:
    """Class to handle Google Earth Engine authentication."""
    
    def __init__(self):
        """Initialize the GEE authentication handler."""
        self._initialized = False
        
    def initialize(self, project_id: str, service_account: Optional[str] = None, 
                  key_file: Optional[str] = None, credentials_content: Optional[str] = None) -> bool:
        """
        Initialize the Earth Engine API with the provided credentials.
        
        Args:
            project_id: The Google Cloud project ID
            service_account: Optional service account email
            key_file: Optional path to service account key file
            credentials_content: Optional credentials file content for web apps
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            # Method 1: Service account authentication (preferred for deployed apps)
            if service_account and key_file:
                if os.path.exists(key_file):
                    credentials = ee.ServiceAccountCredentials(service_account, key_file)
                    ee.Initialize(credentials, project=project_id)
                else:
                    raise Exception(f"Service account key file not found: {key_file}")
            
            # Method 2: Credentials file content (for web apps with file upload)
            elif credentials_content:
                # Handle uploaded credentials content (typically the file named 'credentials' created by
                # running `earthengine authenticate` locally). This is **NOT** a service-account JSON; it
                # contains keys like `redirect_uri`, `refresh_token`, and `scopes`.
                try:
                    # Validate that the JSON parses correctly – this also helps catch file upload issues.
                    creds_data = json.loads(credentials_content)

                    # Heuristic: if the uploaded file includes a `private_key`, it's a service-account key
                    # rather than a user credentials file. In that case we fall back to the service-account
                    # method so the existing logic can handle it.
                    if "private_key" in creds_data and "client_email" in creds_data:
                        # Temporarily write to disk and reuse ServiceAccountCredentials logic
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_sa:
                            tmp_sa.write(credentials_content)
                            tmp_sa_path = tmp_sa.name

                        try:
                            credentials = ee.ServiceAccountCredentials(creds_data["client_email"], tmp_sa_path)
                            ee.Initialize(credentials, project=project_id)
                        finally:
                            if os.path.exists(tmp_sa_path):
                                os.unlink(tmp_sa_path)
                    else:
                        # This is a user OAuth credentials file (named just "credentials", created by
                        # `earthengine authenticate`).  We'll do two things:
                        #   1. Persist it to ~/.config/earthengine/ so subsequent sessions work
                        #   2. Build an OAuth2Credentials object from the JSON and pass it explicitly
                        #      to ee.Initialize so we always include the project ID the user provided.

                        # Persist the credentials file to the standard location so future calls that rely
                        # on EE's default discovery behaviour will still succeed.
                        ee_creds_dir = os.path.expanduser("~/.config/earthengine")
                        os.makedirs(ee_creds_dir, exist_ok=True)

                        ee_creds_path = os.path.join(ee_creds_dir, "credentials")

                        try:
                            with open(ee_creds_path, "w", encoding="utf-8") as f:
                                f.write(credentials_content)
                        except Exception as write_err:
                            raise Exception(f"Unable to write credentials file: {write_err}")

                        # Finally, initialize EE; it will discover the credentials file we just wrote.
                        ee.Initialize(project=project_id)

                except json.JSONDecodeError:
                    raise Exception("Invalid credentials file format. Please upload a valid Earth Engine credentials file.")
            
            # Method 3: Try default authentication (works if user has authenticated locally)
            else:
                try:
                    ee.Initialize(project=project_id)
                except Exception as e:
                    # If default auth fails, provide helpful error message
                    raise Exception(
                        "Authentication failed. For web applications, please either:\n"
                        "1. Use service account authentication with a JSON key file, or\n"
                        "2. Upload your Earth Engine credentials file (from ~/.config/earthengine/credentials)\n"
                        "3. For local development: run 'earthengine authenticate' first"
                    )
                
            # Test the connection
            if self.test_connection():
                self._initialized = True
                return True
            else:
                raise Exception("Authentication succeeded but connection test failed")
                
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

    @staticmethod
    def get_auth_url() -> str:
        """
        Get the authentication URL for generating tokens.
        
        Returns:
            str: URL for authentication
        """
        return "https://code.earthengine.google.com/"


def authenticate(project_id: str, service_account: Optional[str] = None, 
                key_file: Optional[str] = None, credentials_content: Optional[str] = None) -> GEEAuth:
    """
    Authenticate with Google Earth Engine.
    
    Args:
        project_id: The Google Cloud project ID
        service_account: Optional service account email
        key_file: Optional path to service account key file
        credentials_content: Optional credentials file content for web apps
        
    Returns:
        GEEAuth: An instance of the GEEAuth class
    """
    auth = GEEAuth()
    auth.initialize(project_id, service_account, key_file, credentials_content)
    return auth


def save_credentials(project_id: str, service_account: Optional[str] = None, 
                   key_file: Optional[str] = None, credentials_content: Optional[str] = None,
                   remember: bool = True) -> None:
    """
    Save credentials to a file for future use.
    
    Args:
        project_id: The Google Cloud project ID
        service_account: Optional service account email
        key_file: Optional path to service account key file
        credentials_content: Optional credentials content (not saved for security)
        remember: Whether to save credentials or remove existing ones
    """
    if not remember:
        # Remove existing credentials if not remembering
        if os.path.exists(CREDENTIALS_FILE):
            try:
                os.remove(CREDENTIALS_FILE)
            except Exception as e:
                print(f"Error removing credentials file: {str(e)}")
        return
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
    
    # Store credentials (excluding sensitive content)
    credentials = {"project_id": project_id}
    if service_account:
        credentials["service_account"] = service_account
    if key_file:
        credentials["key_file"] = key_file
    # Note: We don't save credentials_content for security reasons
        
    # Write to file
    try:
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials, f)
    except Exception as e:
        print(f"Error saving credentials: {str(e)}")


def load_credentials() -> Dict[str, str]:
    """
    Load saved credentials from file.
    
    Returns:
        Dict containing project_id, service_account, key_file, and credentials_content
    """
    credentials = {
        "project_id": None,
        "service_account": None,
        "key_file": None,
        "credentials_content": None
    }
    
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                saved_credentials = json.load(f)
                
            # Update credentials with saved values
            for key in credentials:
                if key in saved_credentials:
                    credentials[key] = saved_credentials[key]
        except Exception as e:
            print(f"Error loading credentials: {str(e)}")
            
    return credentials