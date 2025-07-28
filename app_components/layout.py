import streamlit as st
from datetime import datetime

class AppLayout:
    """Main layout and navigation for the GeoClimate Fetcher app"""
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #1f77b4, #2ca02c); color: white; border-radius: 10px; margin-bottom: 2rem;">
            <h1>🌍 GeoClimate Fetcher</h1>
            <p>Fetch and analyze geospatial climate data from Google Earth Engine</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with progress and navigation"""
        st.sidebar.markdown("## 📋 Progress")
        
        # Progress indicators
        steps = [
            ("🔐 Authentication", st.session_state.get('auth_complete', False)),
            ("📍 Area of Interest", st.session_state.get('geometry_complete', False)),
            ("📊 Dataset Selection", st.session_state.get('dataset_complete', False)),
            ("💾 Download", False)  # Always false since it's the final step
        ]
        
        for i, (step_name, completed) in enumerate(steps, 1):
            if completed:
                st.sidebar.success(f"✅ {step_name}")
            elif st.session_state.get('app_step', 1) == i:
                st.sidebar.info(f"🔄 {step_name}")
            else:
                st.sidebar.write(f"⏳ {step_name}")
        
        st.sidebar.markdown("---")
        
        # Navigation buttons
        if st.session_state.get('app_step', 1) > 1:
            if st.sidebar.button("🔙 Previous Step"):
                st.session_state.app_step = max(1, st.session_state.app_step - 1)
                st.rerun()
        
        # Reset button
        if st.sidebar.button("🔄 Reset App"):
            for key in list(st.session_state.keys()):
                if key.startswith(('auth_', 'geometry_', 'dataset_', 'app_')):
                    del st.session_state[key]
            st.session_state.app_step = 1
            st.rerun()
        
        # Help section
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ℹ️ Help")
        
        with st.sidebar.expander("How to use this app"):
            st.markdown("""
            1. **Authenticate** with Google Earth Engine
            2. **Select** your area of interest using the map
            3. **Choose** the dataset and bands you want
            4. **Configure** and download your data
            
            💡 **Tips:**
            - Large files (>50MB) are automatically sent to Google Drive
            - You can draw polygons or upload GeoJSON files
            - Use the search to find specific datasets quickly
            """)
        
        with st.sidebar.expander("Supported Data Sources"):
            st.markdown("""
            - **CHIRPS** - Precipitation data
            - **MODIS** - Satellite imagery & vegetation indices
            - **GLDAS** - Land data assimilation
            - **Landsat** - Multispectral satellite data
            - **Sentinel** - High-resolution satellite data
            - **And many more...**
            """)
    
    def render_footer(self):
        """Render the footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌍 GeoClimate Fetcher**")
            st.markdown("Powered by Google Earth Engine")
        
        with col2:
            st.markdown("**📧 Support**")
            st.markdown("Report issues on GitHub")
        
        with col3:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.markdown(f"**⏰ Session Time**")
            st.markdown(f"{current_time}") 