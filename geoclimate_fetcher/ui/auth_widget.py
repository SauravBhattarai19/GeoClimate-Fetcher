"""
Authentication widget for Google Earth Engine.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, Dict, Any
import ee
from functools import partial

from geoclimate_fetcher.core.gee_auth import GEEAuth, authenticate

class AuthWidget:
    """Widget for Google Earth Engine authentication."""
    
    def __init__(self, on_auth_success: Optional[Callable] = None):
        """
        Initialize the authentication widget.
        
        Args:
            on_auth_success: Callback function to execute after successful authentication
        """
        self.auth = GEEAuth()
        self.on_auth_success = on_auth_success
        
        # Create UI components
        self.project_id_input = widgets.Text(
            value='',
            description='GEE Project ID:',
            placeholder='Enter your Google Earth Engine project ID',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.service_account_input = widgets.Text(
            value='',
            description='Service Account:',
            placeholder='Optional: service account email',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.key_file_input = widgets.Text(
            value='',
            description='Key File:',
            placeholder='Optional: path to service account key file',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.auth_button = widgets.Button(
            description='Authenticate',
            button_style='primary',
            icon='check'
        )
        
        self.output = widgets.Output()
        
        # Bind button click
        self.auth_button.on_click(self._on_auth_button_click)
        
        # Accordion for optional fields
        advanced_options = widgets.VBox([
            self.service_account_input,
            self.key_file_input
        ])
        
        self.advanced_accordion = widgets.Accordion(
            children=[advanced_options],
            selected_index=None
        )
        self.advanced_accordion.set_title(0, 'Advanced Options')
        
        # Main widget
        self.widget = widgets.VBox([
            widgets.HTML("<h3>Google Earth Engine Authentication</h3>"),
            widgets.HTML("<p>Please enter your GEE Project ID to authenticate:</p>"),
            self.project_id_input,
            self.advanced_accordion,
            self.auth_button,
            self.output
        ])
        
    def display(self):
        """Display the authentication widget."""
        display(self.widget)
        
    def _on_auth_button_click(self, button):
        """Handle authentication button click."""
        with self.output:
            clear_output()
            print("Authenticating with Google Earth Engine...")
            
            project_id = self.project_id_input.value.strip()
            service_account = self.service_account_input.value.strip() or None
            key_file = self.key_file_input.value.strip() or None
            
            if not project_id:
                print("Error: Project ID is required.")
                return
                
            try:
                self.auth = authenticate(project_id, service_account, key_file)
                
                if self.auth.is_initialized():
                    print("Authentication successful!")
                    
                    # Test connection
                    if self.auth.test_connection():
                        print("Connection to Google Earth Engine verified.")
                        
                        if self.on_auth_success:
                            self.on_auth_success(self.auth)
                    else:
                        print("Warning: Authentication successful but connection test failed.")
                else:
                    print("Authentication failed. Please check your credentials.")
                    
            except Exception as e:
                print(f"Error during authentication: {str(e)}")
                
    def get_auth(self) -> GEEAuth:
        """
        Get the current authentication instance.
        
        Returns:
            GEEAuth instance
        """
        return self.auth
        
    def is_authenticated(self) -> bool:
        """
        Check if authentication is successful.
        
        Returns:
            True if authenticated, False otherwise
        """
        return self.auth.is_initialized()