import streamlit as st
import os
import sys
from pathlib import Path

# Add the geoclimate_fetcher directory to Python path
project_root = Path(__file__).parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

# Import app components
from app_components.auth_component import AuthComponent
from app_components.geometry_component import GeometryComponent
from app_components.dataset_component import DatasetComponent
from app_components.download_component import DownloadComponent
from app_components.layout import AppLayout

def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="GeoClimate Fetcher",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize layout
    layout = AppLayout()
    layout.render_header()
    
    # Initialize session state
    if 'app_step' not in st.session_state:
        st.session_state.app_step = 1
    if 'auth_complete' not in st.session_state:
        st.session_state.auth_complete = False
    if 'geometry_complete' not in st.session_state:
        st.session_state.geometry_complete = False
    if 'dataset_complete' not in st.session_state:
        st.session_state.dataset_complete = False
    
    # Sidebar navigation
    layout.render_sidebar()
    
    # Main content area
    if st.session_state.app_step == 1:
        auth_component = AuthComponent()
        if auth_component.render():
            st.session_state.auth_complete = True
            st.session_state.app_step = 2
            st.rerun()
    
    elif st.session_state.app_step == 2:
        geometry_component = GeometryComponent()
        if geometry_component.render():
            st.session_state.geometry_complete = True
            st.session_state.app_step = 3
            st.rerun()
    
    elif st.session_state.app_step == 3:
        dataset_component = DatasetComponent()
        if dataset_component.render():
            st.session_state.dataset_complete = True
            st.session_state.app_step = 4
            st.rerun()
    
    elif st.session_state.app_step == 4:
        download_component = DownloadComponent()
        download_component.render()
    
    # Footer
    layout.render_footer()

if __name__ == "__main__":
    main() 