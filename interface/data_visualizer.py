"""
Universal Data Visualizer Interface
Handles visualization of CSV and TIFF files from all modules
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Optional
import plotly.graph_objects as go
from streamlit_folium import folium_static

# Import our custom utilities
from app_components.data_processors import data_detector
from app_components.visualization_utils import (
    create_time_series_plot, create_statistical_summary, create_distribution_plot,
    create_correlation_heatmap, create_spatial_map, create_comparison_plot,
    export_plot_as_html, create_data_summary_card, detect_time_series_patterns
)


def render_data_visualizer():
    """Main data visualizer interface"""

    # Add home button
    if st.button("🏠 Back to Home"):
        st.session_state.app_mode = None
        st.rerun()

    # Header
    st.markdown('<h1 class="main-title">📊 Universal Data Visualizer</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'visualizer_data' not in st.session_state:
        st.session_state.visualizer_data = {}
    if 'visualizer_files' not in st.session_state:
        st.session_state.visualizer_files = []

    # Check for direct data passing from other modules
    if 'direct_visualization_data' in st.session_state and st.session_state.direct_visualization_data:
        render_direct_data_visualization()
        return

    # Standard file upload interface
    st.markdown("### Upload and visualize your downloaded data from any module")

    # File upload section
    render_file_upload_section()

    # Process uploaded files
    if st.session_state.visualizer_files:
        # Check if all files are zip files (likely climate data)
        all_zip_files = all(file.name.lower().endswith('.zip') for file in st.session_state.visualizer_files)

        if all_zip_files and len(st.session_state.visualizer_files) == 1:
            # For single zip files, skip detailed analysis and go straight to processing
            st.markdown("## 🌍 Climate Data Processing")
            st.info("📦 **ZIP file detected** - Processing climate data directly for visualization")
            render_file_processing_section_simplified()
        else:
            # Standard detailed file analysis for non-zip or multiple files
            render_file_processing_section()

        # Show visualization options if data is loaded
        if st.session_state.visualizer_data:
            render_visualization_section()


def render_direct_data_visualization():
    """Handle direct data visualization from other modules"""

    direct_data = st.session_state.direct_visualization_data
    source_module = direct_data.get('source_module', 'Unknown Module')
    results = direct_data.get('results', [])

    # Header for direct visualization
    st.markdown(f"### 📊 Data from {source_module.title().replace('_', ' ')}")
    st.info(f"🔄 **Data automatically loaded from {source_module.replace('_', ' ').title()}** - No upload required!")

    # Show return option
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("↩️ Return to Upload", type="secondary"):
            # Clear direct data to return to upload interface
            st.session_state.direct_visualization_data = None
            st.rerun()

    # Process and display the direct data
    processed_data = {}

    for i, result in enumerate(results):
        file_name = result.get('file_name', f'Data_{i}')
        data = result.get('data')
        data_type = result.get('data_type', 'unknown')
        metadata = result.get('metadata', {})
        transfer_method = result.get('transfer_method', 'data_object')

        try:
            if data_type == 'csv':
                # Handle CSV data (should already be processed)
                if isinstance(data, pd.DataFrame):
                    processed_data[file_name] = {
                        'type': 'csv',
                        'data': data,
                        'column_suggestions': result.get('column_suggestions'),
                        'quality_report': result.get('quality_report'),
                        'summary': result.get('summary'),
                        'metadata': metadata,
                        'source_module': source_module
                    }

            elif data_type == 'tiff':
                # Handle TIFF files based on transfer method
                if transfer_method == 'file_path':
                    # Process TIFF from file path (from climate analytics)
                    tiff_result = process_tiff_file_from_path(result.get('file_path'), metadata)
                    if tiff_result:
                        processed_data[file_name] = {
                            **tiff_result,
                            'source_module': source_module
                        }
                else:
                    # Standard TIFF processing (from geodata explorer temp files)
                    processed_data[file_name] = {
                        'type': 'tiff',
                        'file_path': result.get('file_path'),
                        'metadata': metadata,
                        'source_module': source_module
                    }

            elif data_type == 'zip':
                # Handle ZIP files from file path (from climate analytics or geodata explorer)
                if transfer_method in ['file_path', 'temp_file']:
                    zip_results = process_zip_file_from_path(result.get('file_path'), metadata)
                    processed_data.update(zip_results)

        except Exception as e:
            st.warning(f"⚠️ Could not process {file_name}: {str(e)}")
            continue

    # Store in session state for visualization
    st.session_state.visualizer_data = processed_data

    # Check if any data was successfully processed
    if not processed_data:
        st.warning("⚠️ No data could be processed from the source module.")
        st.info("💡 Please use the file upload below to visualize your data manually.")
        # Clear direct data and show upload interface
        st.session_state.direct_visualization_data = None
        return

    # Show file previews
    st.markdown("#### 📋 Available Data:")

    for file_name, file_data in processed_data.items():
        with st.expander(f"📄 {file_name}", expanded=len(processed_data) == 1):

            if file_data['type'] == 'csv':
                _render_direct_csv_preview(file_data)
            elif file_data['type'] == 'tiff':
                _render_direct_tiff_preview(file_data)

    # Show visualization section if data is available
    if processed_data:
        render_visualization_section()

    # Option to clear and start fresh
    st.markdown("---")
    if st.button("🔄 Start Fresh Visualization Session", type="secondary"):
        st.session_state.direct_visualization_data = None
        st.session_state.visualizer_data = {}
        st.rerun()


def _render_direct_csv_preview(file_data: Dict):
    """Render preview for directly passed CSV data"""
    data = file_data['data']
    metadata = file_data.get('metadata', {})
    source_module = file_data.get('source_module', 'Unknown')

    # Show metadata from source module
    if metadata:
        info_items = []
        for key, value in metadata.items():
            if key not in ['size_kb', 'file_size']:  # Skip technical details
                info_items.append(f"**{key.replace('_', ' ').title()}:** {value}")

        if info_items:
            st.info(" • ".join(info_items))

    # Show data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(data):,}")
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.1f} MB")

    # Show data preview
    st.markdown("**Data Preview:**")
    st.dataframe(data.head(5), use_container_width=True)

    # Show column information
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()

    if date_cols and numeric_cols:
        st.success(f"🎯 **Ready for time series analysis!** {len(date_cols)} date column(s), {len(numeric_cols)} numeric column(s)")
    elif numeric_cols:
        st.info(f"📊 **Numeric data available:** {len(numeric_cols)} columns for statistical analysis")
    else:
        st.info("📄 **Text/categorical data detected**")


def _render_direct_tiff_preview(file_data: Dict):
    """Render simplified preview for directly passed TIFF data"""
    metadata = file_data.get('metadata', {})

    if metadata:
        # Simple one-line info
        band_count = metadata.get('band_count', metadata.get('bands', 'N/A'))
        st.info(f"🗺️ **{metadata.get('width', 'N/A')} x {metadata.get('height', 'N/A')} pixels** • **{band_count} bands** • Ready to visualize!")

        # Direct visualization - no preview, jump straight to map


def render_file_upload_section():
    """File upload interface with drag and drop"""

    st.markdown("## 📤 Upload Your Data")
    st.markdown("Supported formats: **CSV** (time series data), **TIFF** (spatial data), **ZIP** (multiple files)")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to visualize",
        accept_multiple_files=True,
        type=['csv', 'tiff', 'tif', 'zip'],
        help="Upload CSV files for time series analysis or TIFF files for spatial visualization"
    )

    if uploaded_files:
        st.session_state.visualizer_files = uploaded_files

        # Show basic file information
        with st.expander("📋 Uploaded Files Summary", expanded=True):
            cols = st.columns(4)

            total_size = sum(f.size for f in uploaded_files)
            csv_count = len([f for f in uploaded_files if f.name.endswith('.csv')])
            tiff_count = len([f for f in uploaded_files if f.name.endswith(('.tif', '.tiff'))])
            zip_count = len([f for f in uploaded_files if f.name.endswith('.zip')])

            with cols[0]:
                st.metric("Total Files", len(uploaded_files))
            with cols[1]:
                st.metric("CSV Files", csv_count)
            with cols[2]:
                st.metric("TIFF Files", tiff_count)
            with cols[3]:
                st.metric("Total Size", f"{total_size/1024:.1f} KB")


def render_file_processing_section_simplified():
    """Simplified processing for zip files - skip detailed analysis"""

    processed_data = {}

    for uploaded_file in st.session_state.visualizer_files:
        # Detect file format
        format_info = data_detector.detect_file_format(uploaded_file)

        if format_info['valid']:
            # Process file directly without showing detailed analysis
            if format_info['type'] == 'zip':
                result = process_zip_file_silent(uploaded_file, format_info)
                if result:
                    processed_data.update(result)

    # Store processed data
    st.session_state.visualizer_data = processed_data


def render_file_processing_section():
    """Process and analyze uploaded files"""

    st.markdown("## 🔍 File Analysis")

    processed_data = {}

    for uploaded_file in st.session_state.visualizer_files:
        with st.expander(f"📄 {uploaded_file.name}", expanded=False):

            # Detect file format
            format_info = data_detector.detect_file_format(uploaded_file)

            # Display format information
            col1, col2, col3 = st.columns(3)
            with col1:
                status_color = "🟢" if format_info['valid'] else "🔴"
                st.write(f"**Status:** {status_color} {'Valid' if format_info['valid'] else 'Invalid'}")

            with col2:
                st.write(f"**Type:** {format_info['type'].upper()}")

            with col3:
                st.write(f"**Format:** {format_info['format'].replace('_', ' ').title()}")

            # Show error if any
            if format_info.get('error'):
                st.error(f"❌ {format_info['error']}")
                continue

            # Process based on file type
            if format_info['type'] == 'csv' and format_info['valid']:
                processed_data[uploaded_file.name] = process_csv_file(uploaded_file, format_info)

            elif format_info['type'] == 'tiff' and format_info['valid']:
                processed_data[uploaded_file.name] = process_tiff_file(uploaded_file, format_info)

            elif format_info['type'] == 'zip' and format_info['valid']:
                zip_data = process_zip_file(uploaded_file, format_info)
                processed_data.update(zip_data)

    # Store processed data
    st.session_state.visualizer_data = processed_data


def process_csv_file(uploaded_file, format_info: Dict) -> Dict:
    """Process CSV file and extract visualization data"""

    try:
        # Load and process data
        df = data_detector.process_csv_data(uploaded_file, format_info)

        # Get column suggestions
        column_suggestions = data_detector.get_column_suggestions(df)

        # Validate data quality
        quality_report = data_detector.validate_data_quality(df)

        # Create summary
        summary = create_data_summary_card(df)

        # Display basic information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{summary['total_rows']:,}")
        with col2:
            st.metric("Columns", summary['total_columns'])
        with col3:
            st.metric("Memory", summary['memory_usage'])

        # Show data preview
        st.markdown("**Data Preview:**")
        st.dataframe(df.head(10), use_container_width=True)

        # Show column information
        st.markdown("**Column Information:**")
        col_info_cols = st.columns(2)

        with col_info_cols[0]:
            if column_suggestions['date_columns']:
                st.success(f"📅 Date columns: {', '.join(column_suggestions['date_columns'])}")
            if column_suggestions['numeric_columns']:
                st.info(f"🔢 Numeric columns: {', '.join(column_suggestions['numeric_columns'][:5])}{'...' if len(column_suggestions['numeric_columns']) > 5 else ''}")

        with col_info_cols[1]:
            if column_suggestions['categorical_columns']:
                st.info(f"🏷️ Categorical columns: {', '.join(column_suggestions['categorical_columns'][:3])}{'...' if len(column_suggestions['categorical_columns']) > 3 else ''}")

        # Show quality score
        quality_color = "🟢" if quality_report['quality_score'] > 80 else "🟡" if quality_report['quality_score'] > 60 else "🔴"
        st.write(f"**Data Quality:** {quality_color} {quality_report['quality_score']}/100")

        if quality_report['issues']:
            with st.expander("⚠️ Data Quality Issues", expanded=False):
                for issue in quality_report['issues']:
                    st.warning(f"• {issue}")
                for suggestion in quality_report['suggestions']:
                    st.info(f"💡 {suggestion}")

        return {
            'type': 'csv',
            'data': df,
            'format_info': format_info,
            'column_suggestions': column_suggestions,
            'quality_report': quality_report,
            'summary': summary
        }

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None


def process_tiff_file_silent(uploaded_file, format_info: Dict) -> Dict:
    """Process TIFF file silently without displaying metadata"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            # Handle different file object types
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
            else:
                # If it's already bytes
                content = uploaded_file
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Reset file pointer if possible
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)

        return {
            'type': 'tiff',
            'file_path': tmp_file_path,
            'format_info': format_info,
            'metadata': format_info['metadata'],
            'cleanup_files': [tmp_file_path]  # Files to clean up later
        }

    except Exception as e:
        return None


def process_tiff_file(uploaded_file, format_info: Dict) -> Dict:
    """Process TIFF file and extract spatial data"""

    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            # Handle different file object types
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
            else:
                # If it's already bytes
                content = uploaded_file
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Reset file pointer if possible
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)

        # Display metadata
        metadata = format_info['metadata']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Width", f"{metadata['width']:,} px")
        with col2:
            st.metric("Height", f"{metadata['height']:,} px")
        with col3:
            st.metric("Bands", metadata['bands'])
        with col4:
            st.metric("Data Type", metadata['dtype'])

        # Show spatial information
        st.markdown("**Spatial Information:**")
        spatial_cols = st.columns(2)

        with spatial_cols[0]:
            st.info(f"**CRS:** {metadata['crs']}")
            bounds = metadata['bounds']
            st.info(f"**Bounds:** {bounds['left']:.6f}, {bounds['bottom']:.6f} to {bounds['right']:.6f}, {bounds['top']:.6f}")

        with spatial_cols[1]:
            stats = metadata['statistics']
            st.info(f"**Value Range:** {stats['min']:.3f} to {stats['max']:.3f}")
            st.info(f"**Mean:** {stats['mean']:.3f} ± {stats['std']:.3f}")

        return {
            'type': 'tiff',
            'file_path': tmp_file_path,
            'format_info': format_info,
            'metadata': metadata,
            'cleanup_files': [tmp_file_path]  # Files to clean up later
        }

    except Exception as e:
        st.error(f"Error processing TIFF file: {str(e)}")
        return None


def process_zip_file_silent(uploaded_file, format_info: Dict) -> Dict:
    """Process ZIP file contents silently without showing archive details"""

    try:
        # Extract contents silently
        extracted_files = data_detector.extract_zip_contents(uploaded_file)
        processed_files = {}

        for extracted_file in extracted_files:
            file_name = extracted_file['name']

            if extracted_file['format']['type'] == 'tiff':
                display_name = f"[ZIP] {file_name}"
                tiff_result = process_tiff_file_silent(
                    extracted_file['data'],
                    extracted_file['format']
                )
                if tiff_result:
                    # Add temporal information for climate data
                    tiff_result['original_filename'] = file_name
                    tiff_result['temporal_info'] = _extract_temporal_info(file_name)
                    processed_files[display_name] = tiff_result

        return processed_files

    except Exception as e:
        return {}


def process_zip_file(uploaded_file, format_info: Dict) -> Dict:
    """Process ZIP file contents"""

    try:
        # Extract contents
        extracted_files = data_detector.extract_zip_contents(uploaded_file)

        st.markdown("**Archive Contents:**")
        metadata = format_info['metadata']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CSV Files", len(metadata['csv_files']))
        with col2:
            st.metric("TIFF Files", len(metadata['tiff_files']))
        with col3:
            st.metric("Total Files", metadata['total_files'])

        # Process each extracted file with improved climate data handling
        processed_files = {}
        tiff_files = []
        csv_files = []

        for extracted_file in extracted_files:
            file_name = extracted_file['name']

            if extracted_file['format']['type'] == 'csv':
                display_name = f"[ZIP] {file_name}"
                csv_result = process_csv_file(
                    extracted_file['data'],
                    extracted_file['format']
                )
                if csv_result:
                    processed_files[display_name] = csv_result
                    csv_files.append(file_name)

            elif extracted_file['format']['type'] == 'tiff':
                display_name = f"[ZIP] {file_name}"
                tiff_result = process_tiff_file(
                    extracted_file['data'],
                    extracted_file['format']
                )
                if tiff_result:
                    # Add temporal information for climate data
                    tiff_result['original_filename'] = file_name
                    tiff_result['temporal_info'] = _extract_temporal_info(file_name)
                    processed_files[display_name] = tiff_result
                    tiff_files.append(file_name)

        # Show summary of climate data if detected
        if tiff_files and _is_climate_time_series(tiff_files):
            st.success(f"🌍 **Climate Time Series Detected!** Found {len(tiff_files)} temporal TIFF files")
            _show_climate_time_series_summary(tiff_files)

        return processed_files

    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return {}


def render_visualization_section():
    """Main visualization interface"""

    st.markdown("## 📊 Data Visualization")

    # File selector
    available_files = list(st.session_state.visualizer_data.keys())

    if len(available_files) == 0:
        st.warning("No valid data files found for visualization.")
        return

    # Check if we have TIFF files from zip archives
    zip_tiff_files = [name for name in available_files
                      if name.startswith('[ZIP]') and
                      st.session_state.visualizer_data[name].get('type') == 'tiff']

    if zip_tiff_files:
        # For zip TIFF files, show simple selection and direct spatial visualization
        if len(zip_tiff_files) > 1:
            selected_file = st.selectbox(
                "Select TIFF file to visualize:",
                zip_tiff_files,
                help="Select a TIFF file to view its spatial map"
            )
        else:
            selected_file = zip_tiff_files[0]
            st.info(f"📍 **Showing spatial map for:** {selected_file.replace('[ZIP] ', '')}")
    else:
        # Standard file selection for non-zip files
        selected_file = st.selectbox(
            "Select file to visualize:",
            available_files,
            help="Choose which file you want to visualize"
        )

    if selected_file:
        file_data = st.session_state.visualizer_data[selected_file]

        if file_data is None:
            st.error("Selected file could not be processed.")
            return

        # Render based on data type
        if file_data['type'] == 'csv':
            render_csv_visualization(file_data, selected_file)
        elif file_data['type'] == 'tiff':
            render_tiff_visualization(file_data, selected_file)


def _generate_fallback_column_suggestions(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Generate column suggestions when metadata is missing"""
    import pandas as pd
    import numpy as np

    suggestions = {
        'date_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'text_columns': []
    }

    for col in df.columns:
        # Numeric columns
        if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
            suggestions['numeric_columns'].append(col)
        # Date columns - basic heuristic
        elif 'date' in str(col).lower() or 'time' in str(col).lower() or df[col].dtype == 'datetime64[ns]':
            suggestions['date_columns'].append(col)
        # Try to detect dates by parsing a sample
        elif df[col].dtype == 'object':
            try:
                # Try to parse a few non-null values
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    pd.to_datetime(sample_values.iloc[0])
                    # If first value parses as date, assume it's a date column
                    suggestions['date_columns'].append(col)
                else:
                    suggestions['text_columns'].append(col)
            except:
                # Categorical columns (limited unique values)
                if df[col].nunique() < len(df) * 0.5 and df[col].nunique() < 20:
                    suggestions['categorical_columns'].append(col)
                else:
                    suggestions['text_columns'].append(col)
        else:
            suggestions['text_columns'].append(col)

    return suggestions


def _generate_fallback_summary(df: pd.DataFrame) -> Dict:
    """Generate basic summary when metadata is missing"""
    import numpy as np

    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
        'has_missing_values': df.isnull().any().any(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'date_columns': len([col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }


def _generate_fallback_quality_report(df: pd.DataFrame) -> Dict:
    """Generate basic quality report when metadata is missing"""

    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    duplicate_percentage = (df.duplicated().sum() / len(df)) * 100

    # Simple quality score based on missing data and duplicates
    quality_score = max(0, 100 - missing_percentage - duplicate_percentage)

    issues = []
    suggestions = []

    if missing_percentage > 5:
        issues.append(f"High missing data: {missing_percentage:.1f}%")
        suggestions.append("Consider data cleaning for missing values")

    if duplicate_percentage > 5:
        issues.append(f"High duplicate rows: {duplicate_percentage:.1f}%")
        suggestions.append("Consider removing duplicate entries")

    return {
        'quality_score': int(quality_score),
        'issues': issues,
        'suggestions': suggestions
    }


def render_csv_visualization(file_data: Dict, file_name: str):
    """Render CSV data visualizations"""

    df = file_data['data']

    # Get column suggestions with fallback if missing
    column_suggestions = file_data.get('column_suggestions')
    if column_suggestions is None:
        # Generate column suggestions on the fly
        column_suggestions = _generate_fallback_column_suggestions(df)

    # Visualization options
    viz_tabs = st.tabs(["📈 Time Series", "📊 Distribution", "🔗 Correlation", "📋 Summary"])

    with viz_tabs[0]:  # Time Series
        render_time_series_viz(df, column_suggestions)

    with viz_tabs[1]:  # Distribution
        render_distribution_viz(df, column_suggestions)

    with viz_tabs[2]:  # Correlation
        render_correlation_viz(df, column_suggestions)

    with viz_tabs[3]:  # Summary
        render_summary_viz(df, file_data)


def render_time_series_viz(df: pd.DataFrame, column_suggestions: Dict):
    """Render time series visualizations"""

    if not column_suggestions['date_columns']:
        st.warning("No date columns detected. Time series analysis requires a date/time column.")
        return

    if not column_suggestions['numeric_columns']:
        st.warning("No numeric columns detected. Time series analysis requires numeric data.")
        return

    # Column selection
    col1, col2 = st.columns(2)

    with col1:
        date_col = st.selectbox(
            "Date Column:",
            column_suggestions['date_columns'],
            help="Select the column containing dates/times"
        )

    with col2:
        value_cols = st.multiselect(
            "Value Columns:",
            column_suggestions['numeric_columns'],
            default=column_suggestions['numeric_columns'][:3],  # Select first 3 by default
            help="Select numeric columns to plot"
        )

    if date_col and value_cols:
        # Create time series plot
        fig = create_time_series_plot(
            df, date_col, value_cols,
            title=f"Time Series Analysis",
            x_title=date_col,
            y_title="Values"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pattern detection
        if len(value_cols) == 1:
            st.markdown("#### 🔍 Pattern Analysis")
            patterns = detect_time_series_patterns(df, date_col, value_cols[0])

            pattern_cols = st.columns(3)
            with pattern_cols[0]:
                if 'trend' in patterns:
                    trend_icon = "📈" if patterns['trend'] == 'increasing' else "📉" if patterns['trend'] == 'decreasing' else "➡️"
                    st.metric("Trend", f"{trend_icon} {patterns['trend'].title()}")

            with pattern_cols[1]:
                if 'outliers_percentage' in patterns:
                    outlier_color = "🔴" if patterns['outliers_percentage'] > 5 else "🟡" if patterns['outliers_percentage'] > 2 else "🟢"
                    st.metric("Outliers", f"{outlier_color} {patterns['outliers_percentage']:.1f}%")

            with pattern_cols[2]:
                if 'seasonal_variation' in patterns:
                    st.metric("Seasonal Variation", f"{patterns['seasonal_variation']:.2f}")

        # Export options
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("📥 Download Plot as HTML"):
                html_str = fig.to_html()
                st.download_button(
                    label="Download HTML",
                    data=html_str,
                    file_name=f"time_series_{date_col}_{'-'.join(value_cols)}.html",
                    mime="text/html"
                )

        with export_col2:
            if st.button("📊 Download Data as CSV"):
                csv_data = df[[date_col] + value_cols].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"filtered_data_{date_col}.csv",
                    mime="text/csv"
                )


def render_distribution_viz(df: pd.DataFrame, column_suggestions: Dict):
    """Render distribution visualizations"""

    if not column_suggestions['numeric_columns']:
        st.warning("No numeric columns detected for distribution analysis.")
        return

    # Column selection
    col1, col2 = st.columns(2)

    with col1:
        selected_col = st.selectbox(
            "Select Column:",
            column_suggestions['numeric_columns'],
            help="Choose a numeric column to analyze"
        )

    with col2:
        plot_type = st.selectbox(
            "Plot Type:",
            ['histogram', 'box', 'violin'],
            help="Choose the type of distribution plot"
        )

    if selected_col:
        # Create distribution plot
        fig = create_distribution_plot(df, selected_col, plot_type)
        st.plotly_chart(fig, use_container_width=True)

        # Show statistics
        st.markdown("#### 📊 Statistical Summary")
        stats = create_statistical_summary(df, [selected_col])
        st.dataframe(stats, use_container_width=True)


def render_correlation_viz(df: pd.DataFrame, column_suggestions: Dict):
    """Render correlation analysis"""

    if len(column_suggestions['numeric_columns']) < 2:
        st.warning("At least 2 numeric columns are required for correlation analysis.")
        return

    # Column selection
    selected_cols = st.multiselect(
        "Select Columns for Correlation:",
        column_suggestions['numeric_columns'],
        default=column_suggestions['numeric_columns'][:5],  # First 5 columns
        help="Select numeric columns to include in correlation analysis"
    )

    if len(selected_cols) >= 2:
        # Create correlation heatmap
        fig = create_correlation_heatmap(df, selected_cols)
        st.plotly_chart(fig, use_container_width=True)

        # Show correlation table
        st.markdown("#### 📊 Correlation Coefficients")
        corr_matrix = df[selected_cols].corr()
        st.dataframe(corr_matrix, use_container_width=True)


def render_summary_viz(df: pd.DataFrame, file_data: Dict):
    """Render data summary and quality information"""

    # Get summary with fallback
    summary = file_data.get('summary')
    if summary is None:
        summary = _generate_fallback_summary(df)

    # Get quality report with fallback
    quality_report = file_data.get('quality_report')
    if quality_report is None:
        quality_report = _generate_fallback_quality_report(df)

    # Summary metrics
    st.markdown("#### 📋 Dataset Overview")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Rows", f"{summary['total_rows']:,}")
    with metric_cols[1]:
        st.metric("Total Columns", summary['total_columns'])
    with metric_cols[2]:
        st.metric("Quality Score", f"{quality_report['quality_score']}/100")
    with metric_cols[3]:
        st.metric("Memory Usage", summary['memory_usage'])

    # Data types breakdown
    st.markdown("#### 🔍 Data Types")
    dtype_summary = df.dtypes.value_counts()
    dtype_cols = st.columns(len(dtype_summary))

    for i, (dtype, count) in enumerate(dtype_summary.items()):
        with dtype_cols[i]:
            st.metric(str(dtype), count)

    # Missing values analysis
    st.markdown("#### ❓ Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df)) * 100
        }).round(2)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values detected!")

    # Full statistical summary
    st.markdown("#### 📊 Statistical Summary")
    full_stats = create_statistical_summary(df, df.select_dtypes(include=[np.number]).columns.tolist())
    if not full_stats.empty:
        st.dataframe(full_stats, use_container_width=True)


def render_tiff_visualization(file_data: Dict, file_name: str):
    """Render simplified TIFF spatial visualization focused on the map"""

    file_path = file_data['file_path']
    metadata = file_data['metadata']

    # Simple header without extra information
    if file_name.startswith('[ZIP]'):
        # For zip files, just show the spatial map directly
        pass
    else:
        st.markdown("#### 🗺️ Spatial Map")

    # Simple band selection (minimal UI for zip files)
    band_count = metadata.get('band_count', metadata.get('bands', 1))
    if band_count > 1 and not file_name.startswith('[ZIP]'):
        # Only show band selector for non-zip files
        band_options = [f"Band {i}" for i in range(1, band_count + 1)]
        selected_band_label = st.radio(
            f"📡 **Bands ({band_count} available):**",
            band_options,
            horizontal=True,
            key=f"band_selector_{file_name}"
        )
        selected_band = int(selected_band_label.split()[-1])
    else:
        selected_band = 1

    try:
        # Colormap selection - simplified for zip files, full options for others
        if file_name.startswith('[ZIP]'):
            # Simplified colormap options for climate data
            colormap_options = {
                'viridis': 'Climate (Viridis)',
                'RdYlBu_r': 'Temperature (Blue-Red)',
                'Blues': 'Precipitation (Blues)',
                'plasma': 'General (Plasma)'
            }
            selected_colormap = st.selectbox(
                "🎨 Color Scheme:",
                options=list(colormap_options.keys()),
                format_func=lambda x: colormap_options[x],
                index=0,
                key=f"colormap_{file_name}"
            )
        else:
            # Full colormap selection for other files
            temporal_info = file_data.get('temporal_info', {})
            if temporal_info.get('has_temporal'):
                colormap_options = {
                    'RdYlBu_r': 'Temperature (Blue-White-Red)',
                    'Blues': 'Precipitation (Blue)',
                    'viridis': 'General Climate (Viridis)',
                    'plasma': 'Anomalies (Plasma)',
                    'coolwarm': 'Deviations (Cool-Warm)'
                }
                selected_colormap = st.selectbox(
                    "🎨 **Color Scheme:**",
                    options=list(colormap_options.keys()),
                    format_func=lambda x: colormap_options[x],
                    index=0
                )
            else:
                selected_colormap = 'viridis'

        # Create and display spatial map
        with st.spinner("🗺️ Loading map..."):
            # Simple title for zip files
            if file_name.startswith('[ZIP]'):
                map_title = file_name.replace('[ZIP] ', '').replace('.tif', '')
            else:
                temporal_info = file_data.get('temporal_info', {})
                map_title = f"{file_name}"
                if temporal_info.get('has_temporal'):
                    map_title += f" - {temporal_info['datetime'].strftime('%Y-%m-%d')}"
                else:
                    map_title += f" - Band {selected_band}"

            spatial_map = create_spatial_map(
                file_path,
                band=selected_band,
                colormap=selected_colormap,
                title=map_title
            )

            # Display map with full width
            folium_static(spatial_map, width=800, height=600)

        # Show basic info only for non-zip files
        if not file_name.startswith('[ZIP]'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", f"{metadata['width']} px")
            with col2:
                st.metric("Height", f"{metadata['height']} px")
            with col3:
                st.metric("Data Type", metadata.get('dtype', 'Unknown'))

    except Exception as e:
        st.error(f"❌ Error loading map: {str(e)}")
        st.info("💡 Try refreshing the page if the error persists.")


# Cleanup function (called when app closes)
def _extract_temporal_info(filename: str) -> Dict:
    """Extract temporal information from climate data filenames"""
    import re
    from datetime import datetime

    # Pattern for image_YYYY_MM_DD_HH_MM_SS.tif
    pattern = r'image_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.tif'
    match = re.match(pattern, filename)

    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        try:
            dt = datetime(year, month, day, hour, minute, second)
            return {
                'has_temporal': True,
                'datetime': dt,
                'year': year,
                'month': month,
                'day': day,
                'iso_string': dt.isoformat()
            }
        except ValueError:
            pass

    return {'has_temporal': False}


def _is_climate_time_series(filenames: List[str]) -> bool:
    """Check if the files appear to be climate time series data"""
    temporal_count = 0
    for filename in filenames:
        if _extract_temporal_info(filename)['has_temporal']:
            temporal_count += 1

    # Consider it time series if more than 50% have temporal info
    return temporal_count > len(filenames) * 0.5


def _show_climate_time_series_summary(filenames: List[str]):
    """Show summary information for climate time series data"""
    temporal_data = []
    for filename in filenames:
        temp_info = _extract_temporal_info(filename)
        if temp_info['has_temporal']:
            temporal_data.append(temp_info)

    if temporal_data:
        # Sort by date
        temporal_data.sort(key=lambda x: x['datetime'])

        # Show temporal range
        start_date = temporal_data[0]['datetime']
        end_date = temporal_data[-1]['datetime']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Start Date", start_date.strftime("%Y-%m-%d"))
        with col2:
            st.metric("📅 End Date", end_date.strftime("%Y-%m-%d"))
        with col3:
            years = end_date.year - start_date.year + 1
            st.metric("📊 Years", years)

        # Show yearly distribution
        years_counts = {}
        for td in temporal_data:
            year = td['year']
            years_counts[year] = years_counts.get(year, 0) + 1

        if len(years_counts) > 1:
            st.markdown("**📈 Temporal Distribution:**")
            year_df = pd.DataFrame(list(years_counts.items()), columns=['Year', 'Files'])
            st.bar_chart(year_df.set_index('Year'))


def process_tiff_file_from_path(file_path: str, metadata: Dict = None) -> Dict:
    """Process TIFF file from file path (for direct transfers from other modules)"""
    try:
        import rasterio

        # Read metadata from TIFF file
        with rasterio.open(file_path) as src:
            # Get basic raster information
            raster_metadata = {
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs) if src.crs else 'Unknown',
                'bounds': {
                    'left': src.bounds.left,
                    'bottom': src.bounds.bottom,
                    'right': src.bounds.right,
                    'top': src.bounds.top
                }
            }

            # Get basic statistics from first band
            band_data = src.read(1, masked=True)
            valid_data = band_data.compressed()  # Remove masked/nodata values

            if len(valid_data) > 0:
                raster_metadata['statistics'] = {
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std())
                }
            else:
                raster_metadata['statistics'] = {
                    'min': 0, 'max': 0, 'mean': 0, 'std': 0
                }

        return {
            'type': 'tiff',
            'file_path': file_path,
            'metadata': {**raster_metadata, **(metadata or {})},
            'cleanup_files': []  # No cleanup needed for existing files
        }

    except Exception as e:
        st.error(f"❌ Error processing TIFF file {file_path}: {str(e)}")
        return None


def process_zip_file_from_path(file_path: str, metadata: Dict = None) -> Dict:
    """Process ZIP file from file path (for direct transfers from other modules)"""
    try:
        import zipfile
        import tempfile

        processed_files = {}

        # Extract and process ZIP contents
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            # Filter for TIFF files
            tiff_files = [f for f in file_list if f.lower().endswith(('.tif', '.tiff'))]

            for tiff_name in tiff_files[:10]:  # Limit to first 10 files for performance
                try:
                    # Extract to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                        temp_file.write(zip_ref.read(tiff_name))
                        temp_path = temp_file.name

                    # Process the extracted TIFF
                    tiff_metadata = {
                        'original_filename': tiff_name,
                        'source_zip': file_path,
                        **(metadata or {})
                    }

                    tiff_result = process_tiff_file_from_path(temp_path, tiff_metadata)
                    if tiff_result:
                        # Add temporal information for climate data
                        tiff_result['temporal_info'] = _extract_temporal_info(tiff_name)
                        tiff_result['cleanup_files'] = [temp_path]  # Mark for cleanup

                        display_name = f"[ZIP] {tiff_name}"
                        processed_files[display_name] = tiff_result

                except Exception as e:
                    continue  # Skip problematic files

        return processed_files

    except Exception as e:
        st.error(f"❌ Error processing ZIP file {file_path}: {str(e)}")
        return {}


def cleanup_temp_files():
    """Clean up temporary files"""
    if 'visualizer_data' in st.session_state:
        for file_data in st.session_state.visualizer_data.values():
            if file_data and file_data.get('cleanup_files'):
                for temp_file in file_data['cleanup_files']:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass  # Ignore cleanup errors