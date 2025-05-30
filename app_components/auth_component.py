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
    
    def authenticate_gee(self, project_id, service_account=None, key_file=None):
        """Authenticate with Google Earth Engine"""
        try:
            auth = authenticate(project_id, service_account, key_file)
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
        
        # Main authentication form
        with st.form("auth_form"):
            st.markdown("### 📝 Authentication Details")
            
            # Project ID - required
            project_id = st.text_input(
                "Google Earth Engine Project ID *",
                value=saved_credentials.get("project_id", ""),
                help="Your Google Cloud project ID with Earth Engine enabled",
                placeholder="my-earth-engine-project"
            )
            
            # Advanced options in expander
            with st.expander("🔧 Advanced Authentication Options"):
                st.markdown("""
                **Service Account Authentication** (Optional)
                
                For automated deployments or shared environments, you can use service account authentication.
                """)
                
                service_account = st.text_input(
                    "Service Account Email",
                    value=saved_credentials.get("service_account", ""),
                    help="Email of the service account (e.g., my-service@project.iam.gserviceaccount.com)",
                    placeholder="service-account@project.iam.gserviceaccount.com"
                )
                
                key_file = st.text_input(
                    "Key File Path",
                    value=saved_credentials.get("key_file", ""),
                    help="Path to service account JSON key file",
                    placeholder="/path/to/key.json"
                )
                
                remember = st.checkbox("Remember credentials", value=True)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Authenticate", type="primary")
            
            if submitted:
                if not project_id:
                    st.error("❌ Project ID is required!")
                    return False
                
                with st.spinner("🔄 Authenticating with Google Earth Engine..."):
                    success, message = self.authenticate_gee(project_id, service_account, key_file)
                    
                    if success:
                        st.success(f"✅ {message}")
                        
                        # Save credentials if requested
                        if remember:
                            credentials = {"project_id": project_id}
                            if service_account:
                                credentials["service_account"] = service_account
                            if key_file:
                                credentials["key_file"] = key_file
                            
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
                            
                            1. **Project not found**: Make sure your project ID is correct
                            2. **Earth Engine not enabled**: Enable Earth Engine API in Google Cloud Console
                            3. **Insufficient permissions**: Ensure your account has Earth Engine access
                            4. **Service account issues**: Check that the key file path is correct and accessible
                            
                            **Need help setting up Earth Engine?**
                            - Visit: https://earthengine.google.com/
                            - Sign up for Earth Engine access
                            - Create a Google Cloud project
                            """)
        
        return False 