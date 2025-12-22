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
        # Import cookie manager for per-browser persistence
        try:
            import extra_streamlit_components as stx
            self.cookie_manager = stx.CookieManager()
        except ImportError:
            # Fallback if cookies not available
            self.cookie_manager = None

    def load_saved_credentials(self):
        """
        Load previously saved credentials from browser cookies
        This is PER-BROWSER, not server-side, so it's safe for multi-user apps
        """
        if self.cookie_manager:
            try:
                project_id = self.cookie_manager.get(cookie="gee_project_id_prefill")
                if project_id:
                    return {"project_id": project_id}
            except Exception:
                pass
        return {}

    def save_credentials(self, credentials):
        """
        Save credentials to browser cookies for convenience
        Only stores project_id, NOT the actual credentials (secure)
        """
        if self.cookie_manager and credentials.get("project_id"):
            try:
                from datetime import datetime, timedelta
                self.cookie_manager.set(
                    "gee_project_id_prefill",
                    credentials["project_id"],
                    expires_at=datetime.now() + timedelta(days=365)  # 1 year
                )
                return True
            except Exception:
                pass
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
                project_id_display = st.session_state.get('project_id', 'Unknown')
                st.info(f"🚀 Authenticated as: **{project_id_display}**\n\nYour session is active!")
            with col2:
                if st.button("Re-authenticate", help="Click to authenticate with different credentials"):
                    st.session_state.auth_complete = False
                    if 'gee_credentials_content' in st.session_state:
                        del st.session_state.gee_credentials_content
                    st.rerun()

            return True  # This will allow the main app to proceed to next step

        # Try to auto-authenticate if credentials are cached in session
        if st.session_state.get('gee_credentials_content') and st.session_state.get('project_id'):
            with st.spinner("🔄 Auto-authenticating from cached credentials..."):
                success, message = self.authenticate_gee(
                    st.session_state.project_id,
                    credentials_content=st.session_state.gee_credentials_content
                )
                if success:
                    st.session_state.auth_complete = True
                    st.success("✅ Auto-authenticated successfully!")
                    time.sleep(1)
                    st.rerun()
                    return True
        
        
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

                # Quick help box at top
                import platform
                system = platform.system()
                if system == "Windows":
                    creds_path = f"C:\\Users\\{os.environ.get('USERNAME', '[USERNAME]')}\\.config\\earthengine\\credentials"
                elif system == "Darwin":  # macOS
                    creds_path = f"/Users/{os.environ.get('USER', '[USER]')}/.config/earthengine/credentials"
                else:  # Linux
                    creds_path = f"/home/{os.environ.get('USER', '[USER]')}/.config/earthengine/credentials"

                st.info(f"""
                📂 **Quick Access:** Your credentials file is located at:
                `{creds_path}`

                **Tip:** Copy this path, paste in your file explorer, and upload the `credentials` file below!
                """)

                # Upload button
                uploaded_file = st.file_uploader(
                    "Upload Earth Engine Credentials File",
                    type=None,  # Accept any file type since credentials file has no extension
                    help="Upload the file named 'credentials' (no extension) from the path shown above"
                )

                # Compact instructions in expander
                with st.expander("📋 First Time? How to Get Your Credentials File"):
                    st.markdown("""
                    **📋 Prerequisites:**
                    - **Google Earth Engine Account** → [Sign up FREE](https://earthengine.google.com/signup/) *(for study & research)*
                    - **Python installed** → [python.org](https://python.org) *(if not already installed)*

                    **🆕 First-Time Setup (Only do this ONCE):**
                    1. **Create Google Earth Engine Account** → [earthengine.google.com/signup](https://earthengine.google.com/signup/)
                       - **FREE for academic research and education**
                       - Use your academic email if available

                    2. **Install Python** → [python.org](https://python.org) *(if not already installed)*

                    3. **Open Terminal/Command Prompt**:
                       - **Windows**: Search "cmd" or "Command Prompt"
                       - **Mac**: Search "Terminal"
                       - **Linux**: Ctrl+Alt+T

                    4. **Install Earth Engine API** *(in terminal)*:
                       ```bash
                       pip install earthengine-api
                       ```

                    5. **Authenticate with Google** *(in terminal)*:
                       ```bash
                       earthengine authenticate
                       ```
                       - A **link will appear** in terminal → Click it OR it auto-opens browser
                       - **Select the Google account** you used to register for Google Earth Engine
                       - After successful login, credentials are saved automatically

                    **🔄 Future Use:**
                    - Once set up, you can use this app **forever** until you accidentally delete your credentials
                    - Just upload the same credentials file each time you use this web app

                    **📂 Find Your Credentials File:**
                    - **Windows**: `C:\\Users\\[USERNAME]\\.config\\earthengine\\credentials`
                    - **Mac/Linux**: `~/.config/earthengine/credentials`

                    **⚠️ Important Notes:**
                    - File is named exactly `credentials` (no .json/.txt extension)
                    - Don't rename or modify - upload as-is
                    - **Keep this file safe** - if you delete it from Google, you'll need to re-authenticate
                    - Same credentials work across all Google Earth Engine applications
                    """)
                
                if uploaded_file is not None:
                    try:
                        credentials_content = uploaded_file.read().decode('utf-8')
                        st.info("📄 Credentials file loaded. Click 'Authenticate' to verify.")
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
            
            # Info about session persistence (handled by cookies in app.py)
            st.info("ℹ️ Your session will be remembered in your browser for 30 days via secure cookies.")
            
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

                    # Set session state for this user session
                    st.session_state.auth_complete = True
                    st.session_state.auth_project_id = project_id
                    st.session_state.project_id = project_id  # For display in nav

                    # Cache credentials in session state for this browser session
                    if credentials_content:
                        st.session_state.gee_credentials_content = credentials_content

                    # Save project_id to browser cookies for next visit (1 year)
                    self.save_credentials({"project_id": project_id})

                    # Show single success message
                    st.success("🎉 Authentication complete! Project ID will be remembered.")
                    st.info("ℹ️ Your session stays active while your browser tab is open. You'll need to re-upload credentials if you close the browser or the session expires.")
                    time.sleep(2)
                    st.rerun()  # Rerun to proceed to main app

                    return True  # Return True to indicate successful authentication
                else:
                    # Show clear error message when authentication fails
                    st.error(f"❌ {message}")

                    # Provide helpful guidance based on error type
                    if "project" in message.lower():
                        st.warning("""
                        **Project ID Issue:**
                        - Make sure you entered the correct Google Cloud Project ID
                        - The project must have Earth Engine API enabled
                        - Format: `your-project-id` (not the project name or number)
                        """)
                    elif "credential" in message.lower() or "token" in message.lower():
                        st.warning("""
                        **Credentials Issue:**
                        - Your credentials file may be invalid or expired
                        - Try re-authenticating: Run `earthengine authenticate` in terminal
                        - Make sure you uploaded the correct file from `~/.config/earthengine/credentials`
                        """)
                    elif "permission" in message.lower() or "access" in message.lower():
                        st.warning("""
                        **Permission Issue:**
                        - Your account may not have access to this project
                        - Ensure Earth Engine API is enabled in Google Cloud Console
                        - Your Google account must be registered with Earth Engine
                        """)
                    else:
                        st.warning("""
                        **Troubleshooting:**
                        1. Verify your Project ID is correct
                        2. Check that your credentials file is valid
                        3. Ensure Earth Engine API is enabled for your project
                        4. Try re-running `earthengine authenticate` in terminal
                        """)

                    return False

        st.markdown('</div>', unsafe_allow_html=True)  # Close auth-form-container
        return False