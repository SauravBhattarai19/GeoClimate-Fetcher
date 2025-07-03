import streamlit as st
import os
import json
from pathlib import Path
import sys

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import authenticate

class AuthComponent:
    """Authentication component for Google Earth Engine"""
    
    def __init__(self):
        self.credentials_file = os.path.expanduser("~/.geoclimate-fetcher/credentials.json")
    
    def load_saved_credentials(self):
        """Load previously saved credentials"""
        if os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_credentials(self, credentials):
        """Save credentials for future use"""
        try:
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f)
            return True
        except Exception as e:
            st.warning(f"Could not save credentials: {str(e)}")
            return False
    
    def authenticate_gee(self, project_id, service_account=None, key_file=None, auth_token=None):
        """Authenticate with Google Earth Engine"""
        try:
            auth = authenticate(project_id, service_account, key_file, auth_token)
            if auth.is_initialized():
                return True, "Authentication successful!"
            else:
                return False, "Authentication failed. Please check your credentials."
        except Exception as e:
            return False, f"Authentication failed: {str(e)}"
    
    def render(self):
        """Render the authentication component"""
        st.markdown("## 🔐 Google Earth Engine Authentication")
        
        # Check if already authenticated
        if st.session_state.get('auth_complete', False):
            st.success("✅ Already authenticated with Google Earth Engine!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Continue to Area Selection", type="primary"):
                    return True
            with col2:
                if st.button("Re-authenticate"):
                    st.session_state.auth_complete = False
                    st.rerun()
            return False
        
        # Load saved credentials
        saved_credentials = self.load_saved_credentials()
        
        st.markdown("""
        To use this application, you need to authenticate with Google Earth Engine.
        You'll need a Google Cloud project with Earth Engine enabled.
        """)
        
        # Authentication method selection
        auth_method = st.radio(
            "Choose Authentication Method:",
            ["Token-based (Recommended for Web Apps)", "Service Account", "Default (Local Only)"],
            help="Token-based authentication works best for deployed web applications"
        )
        
        # Main authentication form
        with st.form("auth_form"):
            st.markdown("### 📝 Authentication Details")
            
            # Project ID - always required
            project_id = st.text_input(
                "Google Earth Engine Project ID *",
                value=saved_credentials.get("project_id", ""),
                help="Your Google Cloud project ID with Earth Engine enabled",
                placeholder="my-earth-engine-project"
            )
            
            # Different inputs based on auth method
            service_account = None
            key_file = None
            auth_token = None
            
            if auth_method == "Token-based (Recommended for Web Apps)":
                st.markdown("### 🔑 Authentication Token")
                
                with st.expander("📋 How to Generate Your Authentication Token", expanded=True):
                    st.markdown("""
                    **Step-by-step instructions:**
                    
                    1. 🌐 Go to [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
                    2. 🔐 Sign in with your Google account (the one with Earth Engine access)
                    3. 💻 In the Code Editor, run this command in the console:
                       ```javascript
                       print('Authentication Token:', ee.data.getAuthToken());
                       ```
                    4. 📋 Copy the token that appears in the console
                    5. 📝 Paste it in the field below
                    
                    **Alternative method:**
                    - Click on your profile picture in the Earth Engine Code Editor
                    - Go to "Cloud project" settings
                    - Look for authentication options
                    """)
                
                auth_token = st.text_area(
                    "Authentication Token *",
                    value=saved_credentials.get("auth_token", ""),
                    help="Paste the token generated from Earth Engine Code Editor",
                    placeholder="Paste your authentication token here...",
                    height=100
                )
                
            elif auth_method == "Service Account":
                st.markdown("### 🔧 Service Account Details")
                st.info("💡 Service account authentication is ideal for automated deployments")
                
                service_account = st.text_input(
                    "Service Account Email *",
                    value=saved_credentials.get("service_account", ""),
                    help="Email of the service account (e.g., my-service@project.iam.gserviceaccount.com)",
                    placeholder="service-account@project.iam.gserviceaccount.com"
                )
                
                key_file = st.text_input(
                    "Key File Path *",
                    value=saved_credentials.get("key_file", ""),
                    help="Path to service account JSON key file",
                    placeholder="/path/to/key.json"
                )
                
            else:  # Default method
                st.markdown("### ⚠️ Default Authentication")
                st.warning("""
                **This method only works if you have previously authenticated locally using:**
                ```bash
                earthengine authenticate
                ```
                This won't work in deployed web applications.
                """)
            
            # Options
            with st.expander("💾 Save Settings"):
                remember = st.checkbox("Remember credentials for future use", value=True)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Authenticate", type="primary")
            
            if submitted:
                # Validation
                if not project_id:
                    st.error("❌ Project ID is required!")
                    return False
                
                if auth_method == "Token-based (Recommended for Web Apps)" and not auth_token:
                    st.error("❌ Authentication token is required for token-based authentication!")
                    return False
                
                if auth_method == "Service Account" and (not service_account or not key_file):
                    st.error("❌ Both service account email and key file path are required!")
                    return False
                
                # Attempt authentication
                with st.spinner("🔄 Authenticating with Google Earth Engine..."):
                    success, message = self.authenticate_gee(
                        project_id, service_account, key_file, auth_token
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        
                        # Save credentials if requested
                        if remember:
                            credentials = {"project_id": project_id}
                            if service_account:
                                credentials["service_account"] = service_account
                            if key_file:
                                credentials["key_file"] = key_file
                            if auth_token:
                                credentials["auth_token"] = auth_token
                            
                            if self.save_credentials(credentials):
                                st.info("💾 Credentials saved for future use")
                        
                        st.session_state.auth_complete = True
                        st.session_state.auth_project_id = project_id
                        
                        # Show success message and continue button
                        st.balloons()
                        st.success("🎉 Ready to proceed to area selection!")
                        
                        # Auto-advance after a short delay
                        if st.button("Continue to Area Selection", type="primary"):
                            return True
                        
                    else:
                        st.error(f"❌ {message}")
                        
                        # Provide help for common issues
                        with st.expander("🔍 Troubleshooting"):
                            st.markdown("""
                            **Common authentication issues:**
                            
                            **Token-based authentication:**
                            - Make sure you copied the complete token
                            - Token might be expired - generate a new one
                            - Ensure your Google account has Earth Engine access
                            
                            **Service account authentication:**
                            - Check that the service account email is correct
                            - Ensure the key file path exists and is accessible
                            - Verify the service account has Earth Engine permissions
                            
                            **General issues:**
                            - Project not found: Make sure your project ID is correct
                            - Earth Engine not enabled: Enable Earth Engine API in Google Cloud Console
                            - Insufficient permissions: Ensure your account has Earth Engine access
                            
                            **Need help setting up Earth Engine?**
                            - Visit: https://earthengine.google.com/
                            - Sign up for Earth Engine access
                            - Create a Google Cloud project
                            """)
        
        return False 