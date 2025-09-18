"""
Post-Download Integration Component
Provides universal download success handling and visualization integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import visualization utilities
from app_components.visualization_utils import (
    create_time_series_plot, create_statistical_summary,
    create_data_summary_card, detect_time_series_patterns
)


class PostDownloadHandler:
    """Handles post-download integration and visualization options"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.download_results = {}
        # Button configuration with defaults for backward compatibility
        self.show_data_visualizer_button = True
        self.show_continue_working_button = True
        self.show_back_home_button = True

    def configure_buttons(self,
                         show_data_visualizer: bool = True,
                         show_continue_working: bool = True,
                         show_back_home: bool = True):
        """
        Configure which buttons to show in the visualization options

        Args:
            show_data_visualizer: Show "Open Data Visualizer" button
            show_continue_working: Show "Continue in Module" button
            show_back_home: Show "Back to Home" button
        """
        self.show_data_visualizer_button = show_data_visualizer
        self.show_continue_working_button = show_continue_working
        self.show_back_home_button = show_back_home

    def register_download_result(self,
                               file_name: str,
                               data: Any,
                               data_type: str,
                               metadata: Optional[Dict] = None,
                               file_path: Optional[str] = None) -> str:
        """
        Register a download result for potential visualization

        Args:
            file_name: Name of the downloaded file
            data: The actual data (DataFrame for CSV, file path for TIFF, etc.)
            data_type: Type of data ('csv', 'tiff', 'netcdf', 'zip')
            metadata: Optional metadata about the file
            file_path: Optional file path for files saved to disk

        Returns:
            Unique ID for the registered result
        """
        result_id = f"{self.module_name}_{len(self.download_results)}_{datetime.now().strftime('%H%M%S')}"

        self.download_results[result_id] = {
            'file_name': file_name,
            'data': data,
            'data_type': data_type,
            'metadata': metadata or {},
            'file_path': file_path,
            'module': self.module_name,
            'timestamp': datetime.now()
        }

        # Store in session state for cross-module access
        if 'global_download_results' not in st.session_state:
            st.session_state.global_download_results = {}

        st.session_state.global_download_results[result_id] = self.download_results[result_id]

        return result_id

    def render_download_success(self, result_ids: List[str]) -> bool:
        """
        Render download success interface with visualization options

        Args:
            result_ids: List of result IDs to display

        Returns:
            True if user chose to visualize, False otherwise
        """
        if not result_ids:
            return False

        st.markdown("### 🎉 Download Complete!")

        # Summary of downloaded files
        total_files = len(result_ids)
        csv_files = sum(1 for rid in result_ids
                       if self.download_results.get(rid, {}).get('data_type') == 'csv')
        tiff_files = sum(1 for rid in result_ids
                        if self.download_results.get(rid, {}).get('data_type') == 'tiff')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Total Files", total_files)
        with col2:
            st.metric("📊 CSV Files", csv_files)
        with col3:
            st.metric("🗺️ Spatial Files", tiff_files)

        # Show file previews and options
        return self._render_file_previews_and_options(result_ids)

    def _render_file_previews_and_options(self, result_ids: List[str]) -> bool:
        """Render file previews and visualization options"""

        st.markdown("#### 📋 Downloaded Files:")

        visualization_requested = False
        quick_viz_data = []

        for result_id in result_ids:
            result = self.download_results.get(result_id)
            if not result:
                continue

            with st.expander(f"📄 {result['file_name']}", expanded=len(result_ids) == 1):

                if result['data_type'] == 'csv':
                    viz_requested = self._render_csv_preview(result)
                    if viz_requested:
                        quick_viz_data.append(result)

                elif result['data_type'] == 'tiff':
                    viz_requested = self._render_tiff_preview(result)
                    if viz_requested:
                        visualization_requested = True

                elif result['data_type'] == 'zip':
                    viz_requested = self._render_zip_preview(result)
                    if viz_requested:
                        visualization_requested = True
                else:
                    viz_requested = self._render_generic_preview(result)
                    if viz_requested:
                        visualization_requested = True

                if viz_requested:
                    visualization_requested = True

        # Render visualization options
        return self._render_visualization_options(result_ids, quick_viz_data) or visualization_requested

    def _render_csv_preview(self, result: Dict) -> bool:
        """Render CSV file preview with quick visualization option"""
        data = result['data']

        if isinstance(data, pd.DataFrame):
            # Show data summary
            summary = create_data_summary_card(data)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{summary['total_rows']:,}")
            with col2:
                st.metric("Columns", summary['total_columns'])
            with col3:
                st.metric("Size", summary['memory_usage'])

            # Show data preview
            st.markdown("**Preview:**")
            st.dataframe(data.head(5), use_container_width=True)

            # Detect visualization potential
            date_cols = []
            numeric_cols = []

            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    date_cols.append(col)
                elif data[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)

            if date_cols and numeric_cols:
                st.success(f"🎯 **Visualization Ready!** Found date column(s): {', '.join(date_cols)} and {len(numeric_cols)} numeric columns")

                # Quick visualization option
                if st.button(f"📈 Quick Plot: {result['file_name']}", key=f"quick_viz_{id(result)}"):
                    self._create_quick_csv_plot(data, date_cols[0], numeric_cols[:3])
                    return True
            else:
                st.info("ℹ️ CSV detected but no time series data found")

        return False

    def _render_tiff_preview(self, result: Dict) -> bool:
        """Render TIFF file preview with spatial visualization option"""
        metadata = result.get('metadata', {})

        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", f"{metadata.get('width', 'N/A'):,} px")
            with col2:
                st.metric("Height", f"{metadata.get('height', 'N/A'):,} px")
            with col3:
                st.metric("Bands", metadata.get('bands', 'N/A'))

            stats = metadata.get('statistics', {})
            if stats and 'min' in stats and stats['min'] is not None:
                st.info(f"📊 **Data Range:** {stats['min']:.3f} to {stats['max']:.3f} (Mean: {stats['mean']:.3f})")
                st.success("🗺️ **Spatial visualization available!**")

                if st.button(f"🗺️ View Map: {result['file_name']}", key=f"spatial_viz_{id(result)}"):
                    return True
            else:
                st.warning("⚠️ Could not read spatial data statistics")

        return False

    def _render_zip_preview(self, result: Dict) -> bool:
        """Render ZIP archive preview"""
        metadata = result.get('metadata', {})

        st.info(f"📦 **Archive contains:** {metadata.get('total_files', 0)} files")

        csv_files = metadata.get('csv_files', [])
        tiff_files = metadata.get('tiff_files', [])

        if csv_files:
            st.success(f"📊 {len(csv_files)} CSV files found")
        if tiff_files:
            st.success(f"🗺️ {len(tiff_files)} spatial files found")

        if csv_files or tiff_files:
            if st.button(f"📦 Explore Archive: {result['file_name']}", key=f"archive_viz_{id(result)}"):
                return True

        return False

    def _render_generic_preview(self, result: Dict) -> bool:
        """Render preview for other file types"""
        st.info(f"📄 **File type:** {result['data_type'].upper()}")
        st.info(f"💾 **Size:** {result.get('file_size', 'Unknown')}")

        if st.button(f"🔍 Analyze: {result['file_name']}", key=f"generic_viz_{id(result)}"):
            return True

        return False

    def _create_quick_csv_plot(self, data: pd.DataFrame, date_col: str, value_cols: List[str]):
        """Create quick CSV plot without leaving the module"""
        st.markdown("#### 📈 Quick Visualization")

        try:
            # Create time series plot
            fig = create_time_series_plot(
                data, date_col, value_cols,
                title=f"Time Series: {', '.join(value_cols)}",
                x_title=date_col,
                y_title="Values"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show pattern analysis for single variable
            if len(value_cols) == 1:
                patterns = detect_time_series_patterns(data, date_col, value_cols[0])

                if patterns:
                    pattern_cols = st.columns(3)
                    with pattern_cols[0]:
                        if 'trend' in patterns:
                            trend_icon = "📈" if patterns['trend'] == 'increasing' else "📉" if patterns['trend'] == 'decreasing' else "➡️"
                            st.metric("Trend", f"{trend_icon} {patterns['trend'].title()}")

                    with pattern_cols[1]:
                        if 'outliers_percentage' in patterns:
                            st.metric("Outliers", f"{patterns['outliers_percentage']:.1f}%")

                    with pattern_cols[2]:
                        if 'seasonal_variation' in patterns:
                            st.metric("Seasonal Var", f"{patterns['seasonal_variation']:.2f}")

        except Exception as e:
            st.error(f"Error creating quick plot: {str(e)}")

    def _render_visualization_options(self, result_ids: List[str], quick_viz_data: List[Dict]) -> bool:
        """Render main visualization options"""

        # Only show visualization options if at least one button is enabled
        buttons_to_show = []
        if self.show_data_visualizer_button:
            buttons_to_show.append("data_visualizer")
        if self.show_continue_working_button:
            buttons_to_show.append("continue_working")

        # Add direct tool navigation buttons
        buttons_to_show.append("data_explorer")
        buttons_to_show.append("climate_analytics")

        if self.show_back_home_button:
            buttons_to_show.append("back_home")

        if not buttons_to_show:
            return False

        st.markdown("---")
        st.markdown("#### 🚀 Navigation Options")

        # Create columns only for enabled buttons
        num_columns = len(buttons_to_show)
        columns = st.columns(num_columns)
        col_index = 0

        if self.show_data_visualizer_button:
            with columns[col_index]:
                st.markdown("**📊 Advanced Analysis**")
                st.info("Upload to Data Visualizer for full analysis features, comparisons, and export options.")

                if st.button("🚀 Open Data Visualizer", type="primary", use_container_width=True):
                    st.session_state.app_mode = "data_visualizer"
                    # Store data for direct access
                    st.session_state.direct_visualization_data = {
                        'results': [self.download_results[rid] for rid in result_ids],
                        'source_module': self.module_name
                    }
                    # Clear post-download state since user is navigating away
                    st.session_state.post_download_active = False
                    st.session_state.post_download_results = []
                    st.rerun()
            col_index += 1

        if self.show_continue_working_button:
            with columns[col_index]:
                st.markdown("**💾 Continue Working**")
                st.info("Stay in current module to continue analysis or download more data.")

                if st.button("↩️ Continue in Module", use_container_width=True):
                    # Clear post-download state and continue in module
                    st.session_state.post_download_active = False
                    st.session_state.post_download_results = []
                    st.rerun()
            col_index += 1

        # Data Explorer navigation button
        with columns[col_index]:
            st.markdown("**🔍 Data Explorer**")
            st.info("Go directly to GeoData Explorer to download more datasets.")

            if st.button("🔍 Data Explorer", use_container_width=True):
                st.session_state.app_mode = "data_explorer"
                # Clear post-download state when navigating to tool
                st.session_state.post_download_active = False
                st.session_state.post_download_results = []
                st.rerun()
        col_index += 1

        # Climate Analytics navigation button
        with columns[col_index]:
            st.markdown("**🧠 Climate Analytics**")
            st.info("Go directly to Climate Intelligence Hub for climate analysis.")

            if st.button("🧠 Climate Analytics", use_container_width=True):
                st.session_state.app_mode = "climate_analytics"
                # Clear post-download state when navigating to tool
                st.session_state.post_download_active = False
                st.session_state.post_download_results = []
                st.rerun()
        col_index += 1

        if self.show_back_home_button:
            with columns[col_index]:
                st.markdown("**🏠 Return Home**")
                st.info("Go back to main platform to access other tools.")

                if st.button("🏠 Back to Home", use_container_width=True):
                    st.session_state.app_mode = None
                    # Clear post-download state when going home
                    st.session_state.post_download_active = False
                    st.session_state.post_download_results = []
                    st.rerun()

        return False

    def create_download_summary_widget(self, result_ids: List[str]):
        """Create a compact download summary widget"""
        if not result_ids:
            return

        total_files = len(result_ids)
        latest_result = self.download_results.get(result_ids[-1], {})

        with st.container():
            st.markdown("##### 📥 Recent Downloads")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"✅ {total_files} file(s) downloaded from {self.module_name.title()}")
                if latest_result:
                    st.caption(f"Latest: {latest_result['file_name']}")

            with col2:
                if st.button("📊 Visualize", key=f"summary_viz_{self.module_name}"):
                    return self.render_download_success(result_ids)

    def clear_results(self):
        """Clear stored download results"""
        self.download_results = {}


# Global handlers for each module
_handlers = {}

def get_download_handler(module_name: str) -> PostDownloadHandler:
    """Get or create download handler for a module"""
    if module_name not in _handlers:
        _handlers[module_name] = PostDownloadHandler(module_name)
    return _handlers[module_name]


def render_post_download_integration(module_name: str,
                                   result_ids: List[str],
                                   show_data_visualizer: bool = True,
                                   show_continue_working: bool = True,
                                   show_back_home: bool = True) -> bool:
    """
    Universal function to render post-download integration

    Args:
        module_name: Name of the calling module
        result_ids: List of download result IDs
        show_data_visualizer: Show "Open Data Visualizer" button
        show_continue_working: Show "Continue in Module" button
        show_back_home: Show "Back to Home" button

    Returns:
        True if visualization was requested, False otherwise
    """
    handler = get_download_handler(module_name)
    handler.configure_buttons(
        show_data_visualizer=show_data_visualizer,
        show_continue_working=show_continue_working,
        show_back_home=show_back_home
    )
    return handler.render_download_success(result_ids)


def register_csv_download(module_name: str,
                         file_name: str,
                         data: pd.DataFrame,
                         metadata: Optional[Dict] = None) -> str:
    """Quick function to register CSV download"""
    handler = get_download_handler(module_name)
    return handler.register_download_result(file_name, data, 'csv', metadata)


def register_tiff_download(module_name: str,
                          file_name: str,
                          file_path: str,
                          metadata: Optional[Dict] = None) -> str:
    """Quick function to register TIFF download"""
    handler = get_download_handler(module_name)
    return handler.register_download_result(file_name, file_path, 'tiff', metadata, file_path)


def register_zip_download(module_name: str,
                         file_name: str,
                         file_path: str,
                         metadata: Optional[Dict] = None) -> str:
    """Quick function to register ZIP download"""
    handler = get_download_handler(module_name)
    return handler.register_download_result(file_name, file_path, 'zip', metadata, file_path)