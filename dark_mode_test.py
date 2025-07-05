"""
Dark Mode Theme Test
This file demonstrates the dark mode improvements made to the GeoClimate Fetcher app.
"""

import streamlit as st
import sys
from pathlib import Path

# Add the app_components directory to the path
app_components_path = Path(__file__).parent / "app_components"
if str(app_components_path) not in sys.path:
    sys.path.insert(0, str(app_components_path))

# Import our components
try:
    from app_components.theme_utils import apply_dark_mode_css
    from app_components.auth_component import AuthComponent
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure the page
st.set_page_config(
    page_title="Dark Mode Theme Test",
    page_icon="🌙",
    layout="wide"
)

# Apply dark mode CSS
apply_dark_mode_css()

# Title
st.title("🌙 Dark Mode Theme Test")

st.markdown("""
## Theme Improvements Summary

This test demonstrates the comprehensive dark mode improvements made to the GeoClimate Fetcher application:

### ✅ **Fixed Issues:**
1. **White spaces in dark mode** - All components now adapt to user's browser theme preference
2. **Authentication flow** - No more double-click requirement after successful authentication
3. **Consistent styling** - Unified theme across all components

### 🎨 **Dark Mode Features:**
- **Automatic detection** - Uses `@media (prefers-color-scheme: dark)` to detect user preference
- **Comprehensive coverage** - Forms, buttons, inputs, alerts, tables, charts all adapt
- **Consistent colors** - Dark backgrounds (#262730), light text (#fafafa), proper borders (#464852)
- **Enhanced UX** - Better contrast, rounded corners, smooth transitions

### 🛠️ **Technical Implementation:**
- **Shared theme utility** - `theme_utils.py` provides consistent styling across components
- **CSS media queries** - Automatically switches between light and dark modes
- **Component isolation** - Each component can add specific styling while maintaining global consistency
""")

st.divider()

# Test various components
st.header("🧪 Component Testing")

# Test the authentication component
st.subheader("Authentication Component")
st.info("The authentication component now has full dark mode support with no white spaces!")

try:
    auth_component = AuthComponent()
    
    # Create a simplified version for testing (without actual authentication)
    with st.expander("🔐 Test Authentication UI"):
        st.markdown("This shows how the authentication component looks in both light and dark modes.")
        
        # Simulate the form elements
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Project ID", value="test-project", help="This input adapts to your theme")
            st.selectbox("Authentication Method", ["Credentials File Upload", "Service Account", "Default"])
            
        with col2:
            st.file_uploader("Upload Credentials", help="File uploader with theme support")
            st.button("🚀 Authenticate", type="primary")
            
        st.success("✅ This success message adapts to dark mode!")
        st.warning("⚠️ Warning messages also have proper dark mode styling")
        st.error("❌ Error messages are clearly visible in both themes")
        st.info("💡 Info messages maintain good contrast")
        
except Exception as e:
    st.error(f"Error testing auth component: {e}")

st.divider()

# Test other UI components
st.subheader("Other UI Components")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Temperature", "25°C", "2°C")
    st.slider("Value", 0, 100, 50)
    
with col2:
    st.checkbox("Enable feature")
    st.radio("Options", ["Option 1", "Option 2", "Option 3"])
    
with col3:
    st.multiselect("Select multiple", ["A", "B", "C", "D"])
    st.date_input("Select date")

# Test data display
st.subheader("Data Display")
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'Temperature': np.random.normal(25, 5, 10),
    'Humidity': np.random.normal(60, 10, 10),
    'Pressure': np.random.normal(1013, 20, 10)
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(data, use_container_width=True)
    
with col2:
    st.line_chart(data)

st.divider()

# Theme detection info
st.subheader("🎯 How It Works")

st.code("""
/* CSS Media Query for Dark Mode Detection */
@media (prefers-color-scheme: dark) {
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* All components get dark styling */
    .stButton > button {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #464852 !important;
    }
}
""", language="css")

st.markdown("""
### 🔧 **To Test:**
1. **Change your browser theme** (or system theme if browser follows system)
2. **Refresh the page** to see the automatic adaptation
3. **All components** should now have appropriate colors for your theme preference

### 🚀 **Benefits:**
- **No more white spaces** in dark mode
- **Better user experience** with theme consistency
- **Accessibility improved** with better contrast
- **Professional appearance** in both light and dark themes
""")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>🌍 GeoClimate Fetcher - Enhanced with Universal Dark Mode Support</p>
    <p>Automatically adapts to your browser's theme preference</p>
</div>
""", unsafe_allow_html=True)
