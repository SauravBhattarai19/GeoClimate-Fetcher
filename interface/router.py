"""
Interface Router
Routes to appropriate interface modules based on app mode
"""

import streamlit as st


def route_to_interface():
    """Route to the appropriate interface based on app_mode"""
    
    app_mode = st.session_state.get('app_mode')
    
    if app_mode == "data_explorer":
        from interface.geodata_explorer import render_geodata_explorer
        render_geodata_explorer()
    
    elif app_mode == "climate_analytics":
        from interface.climate_analytics import render_climate_analytics
        render_climate_analytics()
    
    elif app_mode == "hydrology":
        from interface.hydrology_analyzer import render_hydrology_analyzer
        render_hydrology_analyzer()
    
    elif app_mode == "product_selector":
        from interface.product_selector import render_product_selector
        render_product_selector()

    elif app_mode == "data_visualizer":
        from interface.data_visualizer import render_data_visualizer
        render_data_visualizer()

    elif app_mode == "multi_geometry_export":
        from interface.multi_geometry_export import render_multi_geometry_export
        render_multi_geometry_export()

    else:
        # Default case - should not happen, but handle gracefully
        st.error(f"❌ Unknown app mode: {app_mode}")
        st.info("Returning to home page...")
        st.session_state.app_mode = None
        st.rerun()
