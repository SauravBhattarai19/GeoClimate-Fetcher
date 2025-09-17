"""
Product Selector Interface Module
Handles the complete interface for the Product Selector tool
"""

import streamlit as st
from app_components.product_selector_component import ProductSelectorComponent


def render_product_selector():
    """Render the complete Product Selector interface"""
    
    # Add home button
    if st.button("🏠 Back to Home"):
        st.session_state.app_mode = None
        st.rerun()
    
    # App title and header
    st.markdown('<h1 class="main-title">🎯 Optimal Product Selector</h1>', unsafe_allow_html=True)
    st.markdown("### Compare meteostat station data with gridded datasets to find optimal data sources")
    
    # Import and initialize the product selector component
    try:
        product_selector = ProductSelectorComponent()
        product_selector.render()
    except Exception as e:
        st.error(f"❌ Error initializing Product Selector: {str(e)}")
        
        # Show helpful information
        st.markdown("""
        ### 🔧 Setup Requirements
        
        The Optimal Product Selector requires additional dependencies:
        
        ```bash
        pip install meteostat scikit-learn pandas numpy plotly
        ```
        
        ### 📋 Features
        
        Once properly set up, the Product Selector provides:
        
        - **📡 Station Discovery & Selection**: Find stations in your area and choose specific ones to use
        - **📊 Dataset Comparison**: Compare multiple gridded climate datasets
        - **📈 Statistical Analysis**: Comprehensive correlation and bias analysis
        - **🎯 Optimal Selection**: Identify the best data source for your needs
        - **📁 Data Export**: Download results in multiple formats
        
        ### 🆘 Troubleshooting
        
        If you continue to see this error:
        1. Check that all required packages are installed
        2. Verify your Python environment
        3. Restart the Streamlit application
        4. Check the console for detailed error messages
        """)
        
        # Show contact information
        st.markdown("""
        ### 📞 Need Help?
        
        If you need assistance with setup or encounter persistent issues:
        - Check the project documentation
        - Review the requirements.txt file
        - Contact the development team
        """)


# Additional utility functions for the Product Selector can be added here
def get_product_selector_status():
    """Check if Product Selector dependencies are available"""
    try:
        from app_components.product_selector_component import ProductSelectorComponent
        return True, "All dependencies available"
    except ImportError as e:
        return False, f"Missing dependencies: {str(e)}"
    except Exception as e:
        return False, f"Setup error: {str(e)}"


def show_product_selector_help():
    """Display help information for Product Selector"""
    st.markdown("""
    ## 🎯 How to Use the Product Selector
    
    ### Step 1: Select Area of Interest
    - Draw on the interactive map
    - Upload a GeoJSON file
    - Enter coordinates manually
    
    ### Step 2: Station Discovery & Selection
    - Automatically discover stations in your area
    - Choose specific stations or use all discovered stations
    - Or upload your own station data with selection capability
    
    ### Step 3: Variable Selection
    - Choose the meteorological variable to analyze
    - Options include temperature, precipitation, humidity, etc.
    
    ### Step 4: Dataset Selection
    - Select gridded datasets to compare
    - Multiple datasets can be analyzed simultaneously
    
    ### Step 5: Time Range
    - Define the analysis period
    - Consider data availability and quality
    
    ### Step 6: Analysis Results
    - View comprehensive statistical comparisons
    - Download results and recommendations
    
    ### 📊 Analysis Outputs
    
    The Product Selector provides:
    - **Correlation Analysis**: How well do datasets match station data?
    - **Bias Assessment**: Systematic over/under-estimation
    - **Temporal Analysis**: Performance across different time periods
    - **Spatial Analysis**: Performance across different locations
    - **Ranking System**: Which dataset performs best overall?
    
    ### 💡 Tips for Best Results
    
    - Use multiple years of data for robust analysis
    - Include stations with good data quality
    - Consider the spatial resolution of gridded datasets
    - Account for local topography and climate patterns
    """)
