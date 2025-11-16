"""
Global Navigation Component
Provides persistent navigation across all platform tools
"""

import streamlit as st


def render_global_navigation(clear_auth_func=None):
    """Render the global navigation bar for all platform tools"""

    # Get current app mode for highlighting
    current_mode = st.session_state.get('app_mode', None)

    # Navigation items with icons, labels, and modes
    nav_items = [
        ("🏠", "Home", None),
        ("🔍", "Data Explorer", "data_explorer"),
        ("🗺️", "Multi-Geo", "multi_geometry_export"),
        ("🧠", "Climate Analytics", "climate_analytics"),
        ("💧", "Hydrology", "hydrology"),
        ("🎯", "Product Selector", "product_selector"),
        ("📊", "Visualizer", "data_visualizer"),
        ("🚪", "Logout", "logout")
    ]

    # Custom CSS for navigation bar
    st.markdown("""
    <style>
    .global-nav {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .global-nav {
            background: linear-gradient(90deg, #262730 0%, #343a46 100%);
            border: 1px solid #464852;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
    }

    .nav-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0 0.25rem;
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        cursor: pointer;
    }

    .nav-button:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }

    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .nav-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }

    .nav-label {
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .nav-button {
            padding: 0.4rem 0.6rem;
            margin: 0 0.1rem;
        }

        .nav-label {
            display: none;
        }

        .nav-icon {
            margin-right: 0;
            font-size: 1.1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Render navigation header
    st.markdown('<div class="global-nav">', unsafe_allow_html=True)

    # Create columns for navigation items
    cols = st.columns(len(nav_items))

    for i, (icon, label, mode) in enumerate(nav_items):
        with cols[i]:
            # Determine if this is the active tool
            is_active = (current_mode == mode)

            # Create button with appropriate styling
            button_class = "active" if is_active else ""
            button_key = f"nav_{mode or 'home'}_{i}"

            # Special handling for logout button
            if mode == "logout":
                # Show project info and logout button
                if st.session_state.get('project_id'):
                    st.caption(f"🔐 {st.session_state.project_id}")
                if st.button(f"{icon} {label}", key=button_key):
                    handle_logout(clear_auth_func)
            else:
                # Use custom button styling for better visual feedback
                if is_active:
                    st.markdown(f"""
                    <div class="nav-button active">
                        <span class="nav-icon">{icon}</span>
                        <span class="nav-label">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Use Streamlit button for interactivity
                    if st.button(f"{icon} {label}", key=button_key):
                        handle_navigation_click(mode, label)

    st.markdown('</div>', unsafe_allow_html=True)

    # Add subtle separator
    st.markdown("---")


def handle_navigation_click(target_mode, tool_name):
    """Handle navigation click with intelligent state management"""

    current_mode = st.session_state.get('app_mode')

    # If already on the target tool, don't do anything
    if current_mode == target_mode:
        return

    # Special handling for different navigation scenarios
    if target_mode is None:
        # Going to Home - clear app mode
        st.session_state.app_mode = None
        # Clear any post-download states
        if 'post_download_active' in st.session_state:
            st.session_state.post_download_active = False
        if 'post_download_results' in st.session_state:
            st.session_state.post_download_results = []
    else:
        # Going to a specific tool
        st.session_state.app_mode = target_mode

        # Clear post-download state when switching tools
        if 'post_download_active' in st.session_state:
            st.session_state.post_download_active = False
        if 'post_download_results' in st.session_state:
            st.session_state.post_download_results = []

        # Clear direct visualization data when switching away from visualizer
        if current_mode == "data_visualizer" and 'direct_visualization_data' in st.session_state:
            st.session_state.direct_visualization_data = None

    # Provide user feedback
    if target_mode:
        st.success(f"🚀 Switching to {tool_name}...")
    else:
        st.success("🏠 Returning to Home...")

    # Trigger rerun to navigate
    st.rerun()


def handle_logout(clear_auth_func):
    """Handle logout functionality"""
    if clear_auth_func:
        clear_auth_func()
        st.success("👋 Logged out successfully!")
        st.rerun()
    else:
        st.error("❌ Logout function not available")


def get_current_tool_info():
    """Get information about the currently active tool"""

    current_mode = st.session_state.get('app_mode')

    tool_info = {
        None: {"name": "Home", "icon": "🏠", "description": "Platform Overview"},
        "data_explorer": {"name": "Data Explorer", "icon": "🔍", "description": "Download Earth Engine Datasets"},
        "multi_geometry_export": {"name": "Multi-Geometry Export", "icon": "🗺️", "description": "Export Data for Multiple Regions"},
        "climate_analytics": {"name": "Climate Analytics", "icon": "🧠", "description": "Calculate Climate Indices"},
        "hydrology": {"name": "Hydrology Analyzer", "icon": "💧", "description": "Precipitation Analysis"},
        "product_selector": {"name": "Product Selector", "icon": "🎯", "description": "Compare Data Sources"},
        "data_visualizer": {"name": "Data Visualizer", "icon": "📊", "description": "Visualize Your Data"}
    }

    return tool_info.get(current_mode, {"name": "Unknown", "icon": "❓", "description": "Unknown Tool"})


def render_breadcrumb():
    """Render a subtle breadcrumb showing current location"""

    tool_info = get_current_tool_info()

    if st.session_state.get('app_mode') is not None:
        st.caption(f"{tool_info['icon']} **{tool_info['name']}** - {tool_info['description']}")