import streamlit as st
import os
import json
import time
from pathlib import Path
import sys

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import authenticate

# Import theme utilities
try:
    from .theme_utils import apply_dark_mode_css
except ImportError:
    # Fallback if theme_utils is not available
    def apply_dark_mode_css():
        pass

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
    
    def authenticate_gee(self, project_id, service_account=None, key_file=None, credentials_content=None):
        """Authenticate with Google Earth Engine"""
        try:
            auth = authenticate(project_id, service_account, key_file, credentials_content)
            if auth.is_initialized():
                return True, "Authentication successful!"
            else:
                return False, "Authentication failed. Please check your credentials."
        except Exception as e:
            return False, f"Authentication failed: {str(e)}"
    
    def render(self):
        """Render the authentication component"""
        
        # Apply universal dark mode CSS
        apply_dark_mode_css()
        
        # Additional component-specific styling
        st.markdown("""
        <style>
            .auth-form-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 1rem;
            }
            
            .auth-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .auth-instructions {
                background: rgba(31, 119, 180, 0.1);
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #1f77b4;
                margin-bottom: 1rem;
            }
            
            .credential-upload-info {
                font-size: 0.9rem;
                margin-top: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="auth-form-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-header">', unsafe_allow_html=True)
        st.markdown("## 🔐 Google Earth Engine Authentication")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Check if already authenticated
        if st.session_state.get('auth_complete', False):
            st.success("✅ Already authenticated with Google Earth Engine!")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("🚀 Authentication complete! You can now proceed to the next step.")
            with col2:
                if st.button("Re-authenticate", help="Click to authenticate with different credentials"):
                    st.session_state.auth_complete = False
                    st.rerun()
            
            return True  # This will allow the main app to proceed to next step
        
        
        # Load saved credentials
        saved_credentials = self.load_saved_credentials()
        
        st.markdown("""
        To use this application, you need to authenticate with Google Earth Engine.
        Choose the authentication method that works best for your deployment environment.
        """)
        
        # Authentication method selection
        auth_method = st.radio(
            "Choose Authentication Method:",
            ["Credentials File Upload", "Service Account (Recommended for Web Apps)", "Default (Local Only)"],
            help="Service account authentication is the most reliable method for deployed web applications"
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
            credentials_content = None
            
            if auth_method == "Service Account (Recommended for Web Apps)":
                st.markdown("### 🔧 Service Account Authentication")
                st.info("💡 Service account authentication is ideal for deployed web applications")
                
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
                
            elif auth_method == "Credentials File Upload":
                st.markdown("### 📁 Upload Earth Engine Credentials")
                
                with st.expander("📋 How to Get Your Credentials File", expanded=True):
                    st.markdown("""
                    **For users who have already authenticated locally:**
                    
                    1. 🖥️ On your local machine, run in terminal:
                       ```bash
                       earthengine authenticate
                       ```
                    2. 🌐 This will open a browser for Google OAuth authentication
                    3. ✅ After successful authentication, find your credentials file at:
                       - **Windows**: `C:\\Users\\[USERNAME]\\.config\\earthengine\\credentials`
                       - **Mac/Linux**: `~/.config/earthengine/credentials`
                    4. 📤 Upload that **exact file** (it's named `credentials` with **no file extension**)
                    
                    **Important Notes:**
                    - The file is named just `credentials` (no .json or .txt extension)
                    - It contains JSON content like: `{"redirect_uri": "...", "refresh_token": "...", "scopes": [...]}`
                    - Don't rename the file - upload it as-is
                    
                    **For new users:**
                    1. 🏠 Install Earth Engine API locally: `pip install earthengine-api`
                    2. 🔐 Run `earthengine authenticate` and follow the prompts
                    3. 📄 Upload the generated `credentials` file (no extension) below
                    """)
                
                uploaded_file = st.file_uploader(
                    "Upload Earth Engine Credentials File",
                    type=None,  # Accept any file type since credentials file has no extension
                    help="Upload the file named 'credentials' (no extension) from ~/.config/earthengine/ folder"
                )
                
                if uploaded_file is not None:
                    try:
                        credentials_content = uploaded_file.read().decode('utf-8')
                        st.success("✅ Credentials file uploaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error reading credentials file: {str(e)}")
                        credentials_content = None
                
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
                
                if auth_method == "Service Account (Recommended for Web Apps)" and (not service_account or not key_file):
                    st.error("❌ Both service account email and key file path are required!")
                    return False
                
                if auth_method == "Credentials File Upload" and not credentials_content:
                    st.error("❌ Please upload a credentials file!")
                    return False
                
                # Attempt authentication
                with st.spinner("🔄 Authenticating with Google Earth Engine..."):
                    success, message = self.authenticate_gee(
                        project_id, service_account, key_file, credentials_content
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
                            # Note: We don't save credentials_content for security
                            
                            if self.save_credentials(credentials):
                                st.info("💾 Credentials saved for future use")
                        
                        st.session_state.auth_complete = True
                        st.session_state.auth_project_id = project_id
                        
                        # Show success message - the main app will handle proceeding to next step
                        st.balloons()
                        st.success("🎉 Authentication complete! Proceeding to next step...")
                        
                        return True  # Return True to indicate successful authentication
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close auth-form-container
        return False