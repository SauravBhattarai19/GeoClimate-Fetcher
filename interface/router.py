"""
Interface Router
Routes to appropriate interface modules based on app mode

Memory Optimization:
- Handles module transitions with session state cleanup
- Triggers garbage collection when switching between modules
- Cleans up temporary files periodically
"""

import streamlit as st
import gc
import logging

logger = logging.getLogger(__name__)

# Track the previous module for cleanup purposes
_previous_module = None


def _handle_module_transition(new_module: str) -> None:
    """
    Handle transition between modules with memory cleanup.

    This function cleans up session state from the previous module
    to free memory when switching between modules.

    Args:
        new_module: The module being transitioned to
    """
    global _previous_module

    # Get the previous module from session state or global
    previous = st.session_state.get('_previous_app_mode') or _previous_module

    # Only perform cleanup if actually switching modules
    if previous and previous != new_module:
        logger.info(f"Module transition: {previous} -> {new_module}")

        try:
            # Import memory utils and perform cleanup
            from app_components.memory_utils import (
                cleanup_module_state,
                cleanup_temp_files,
                force_garbage_collection
            )

            # Clean up the previous module's session state
            cleaned = cleanup_module_state(previous)
            logger.info(f"Cleaned {cleaned} session state keys from {previous}")

            # Clean up old temporary files (older than 1 hour)
            temp_cleaned = cleanup_temp_files(max_age_hours=1)
            if temp_cleaned > 0:
                logger.info(f"Cleaned {temp_cleaned} temporary files")

            # Force garbage collection
            gc_collected = force_garbage_collection()
            logger.debug(f"Garbage collection freed {gc_collected} objects")

        except ImportError:
            # If memory_utils not available, do basic cleanup
            logger.warning("memory_utils not available, performing basic cleanup")
            gc.collect()
        except Exception as e:
            logger.error(f"Error during module transition cleanup: {e}")
            # Still try garbage collection
            gc.collect()

    # Update tracking
    _previous_module = new_module
    st.session_state['_previous_app_mode'] = new_module


def route_to_interface():
    """
    Route to the appropriate interface based on app_mode.

    This function handles:
    1. Module transition cleanup (memory optimization)
    2. Lazy importing of interface modules
    3. Rendering the appropriate interface
    """

    app_mode = st.session_state.get('app_mode')

    # Handle module transition for memory cleanup
    _handle_module_transition(app_mode)

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
        st.error(f"Unknown app mode: {app_mode}")
        st.info("Returning to home page...")
        st.session_state.app_mode = None
        st.rerun()
