"""
Quick Visualization Component
Provides instant visualization without full upload process
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import rasterio
import folium
from streamlit_folium import folium_static

# Import visualization utilities
from app_components.visualization_utils import (
    create_time_series_plot, create_statistical_summary,
    create_spatial_map, detect_time_series_patterns
)


class QuickVisualizer:
    """Handles instant visualization for downloaded data"""

    def __init__(self):
        self.supported_quick_viz = ['csv', 'tiff', 'dataframe']

    def render_quick_csv_analysis(self,
                                 data: pd.DataFrame,
                                 title: str = "Quick Data Analysis",
                                 max_columns: int = 5) -> bool:
        """
        Render quick CSV analysis with automatic column detection

        Args:
            data: DataFrame to analyze
            title: Analysis title
            max_columns: Maximum number of columns to analyze

        Returns:
            True if visualization was successful
        """
        try:
            st.markdown(f"#### 📊 {title}")

            # Auto-detect column types
            column_analysis = self._analyze_columns(data)

            # Quick data summary
            self._render_quick_summary(data, column_analysis)

            # Automatic visualization based on data structure
            viz_created = False

            # Time series visualization (priority)
            if column_analysis['date_columns'] and column_analysis['numeric_columns']:
                viz_created = self._render_auto_timeseries(data, column_analysis, max_columns)

            # Multi-variable analysis
            elif len(column_analysis['numeric_columns']) >= 2:
                viz_created = self._render_multi_numeric_analysis(data, column_analysis, max_columns)

            # Single variable analysis
            elif len(column_analysis['numeric_columns']) == 1:
                viz_created = self._render_single_variable_analysis(data, column_analysis)

            # Categorical analysis
            elif column_analysis['categorical_columns']:
                viz_created = self._render_categorical_analysis(data, column_analysis)

            else:
                st.info("ℹ️ Data structure detected but no suitable visualization found")
                self._render_data_preview(data)
                viz_created = True

            return viz_created

        except Exception as e:
            st.error(f"Error in quick analysis: {str(e)}")
            return False

    def render_quick_spatial_analysis(self,
                                    file_path: str,
                                    title: str = "Quick Spatial Analysis") -> bool:
        """
        Render quick spatial analysis for TIFF files

        Args:
            file_path: Path to TIFF file
            title: Analysis title

        Returns:
            True if visualization was successful
        """
        try:
            st.markdown(f"#### 🗺️ {title}")

            # Read spatial metadata
            with rasterio.open(file_path) as src:
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'crs': str(src.crs) if src.crs else 'Unknown',
                    'bounds': src.bounds
                }

                # Quick metadata display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Width", f"{metadata['width']:,} px")
                with col2:
                    st.metric("Height", f"{metadata['height']:,} px")
                with col3:
                    st.metric("Bands", metadata['bands'])
                with col4:
                    area_km2 = ((metadata['bounds'].right - metadata['bounds'].left) *
                               (metadata['bounds'].top - metadata['bounds'].bottom)) / 1e6
                    st.metric("Area", f"{area_km2:.1f} km²")

                # Quick spatial visualization
                self._render_quick_spatial_viz(file_path, metadata)

            return True

        except Exception as e:
            st.error(f"Error in spatial analysis: {str(e)}")
            return False

    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze DataFrame columns and categorize them"""
        analysis = {
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'text_columns': [],
            'boolean_columns': []
        }

        for col in data.columns:
            # Date columns
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                analysis['date_columns'].append(col)
            # Try to detect date strings
            elif data[col].dtype == 'object':
                try:
                    pd.to_datetime(data[col].dropna().iloc[:10])
                    analysis['date_columns'].append(col)
                    continue
                except:
                    pass

            # Numeric columns
            if pd.api.types.is_numeric_dtype(data[col]):
                analysis['numeric_columns'].append(col)

            # Boolean columns
            elif pd.api.types.is_bool_dtype(data[col]):
                analysis['boolean_columns'].append(col)

            # Categorical columns (limited unique values)
            elif data[col].nunique() < min(20, len(data) * 0.5):
                analysis['categorical_columns'].append(col)

            # Text columns
            else:
                analysis['text_columns'].append(col)

        return analysis

    def _render_quick_summary(self, data: pd.DataFrame, column_analysis: Dict):
        """Render quick data summary"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Rows", f"{len(data):,}")

        with col2:
            st.metric("📋 Columns", len(data.columns))

        with col3:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("❓ Missing", f"{missing_pct:.1f}%")

        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memory", f"{memory_mb:.1f} MB")

        # Column types summary
        if any(column_analysis.values()):
            col_info = []
            if column_analysis['date_columns']:
                col_info.append(f"📅 {len(column_analysis['date_columns'])} date")
            if column_analysis['numeric_columns']:
                col_info.append(f"🔢 {len(column_analysis['numeric_columns'])} numeric")
            if column_analysis['categorical_columns']:
                col_info.append(f"🏷️ {len(column_analysis['categorical_columns'])} categorical")

            if col_info:
                st.info(" • ".join(col_info))

    def _render_auto_timeseries(self, data: pd.DataFrame, column_analysis: Dict, max_columns: int) -> bool:
        """Render automatic time series visualization"""
        date_col = column_analysis['date_columns'][0]
        numeric_cols = column_analysis['numeric_columns'][:max_columns]

        st.markdown("##### 📈 Time Series Analysis")

        # Convert date column if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])

        # Create time series plot
        fig = create_time_series_plot(
            data, date_col, numeric_cols,
            title="Automatic Time Series Detection",
            x_title=date_col,
            y_title="Values"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pattern analysis for single variable
        if len(numeric_cols) == 1:
            patterns = detect_time_series_patterns(data, date_col, numeric_cols[0])
            self._render_pattern_summary(patterns)

        # Show data range info
        date_range = data[date_col].max() - data[date_col].min()
        st.info(f"📅 **Data period:** {data[date_col].min().strftime('%Y-%m-%d')} to {data[date_col].max().strftime('%Y-%m-%d')} ({date_range.days} days)")

        return True

    def _render_multi_numeric_analysis(self, data: pd.DataFrame, column_analysis: Dict, max_columns: int) -> bool:
        """Render multi-variable numeric analysis"""
        numeric_cols = column_analysis['numeric_columns'][:max_columns]

        st.markdown("##### 📊 Multi-Variable Analysis")

        # Create correlation heatmap if enough variables
        if len(numeric_cols) >= 2:
            corr_matrix = data[numeric_cols].corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(3).values,
                    texttemplate="%{text}",
                    hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
                )
            )
            fig.update_layout(title="Variable Correlations", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        stats = create_statistical_summary(data, numeric_cols)
        st.dataframe(stats, use_container_width=True)

        return True

    def _render_single_variable_analysis(self, data: pd.DataFrame, column_analysis: Dict) -> bool:
        """Render single variable analysis"""
        var_col = column_analysis['numeric_columns'][0]

        st.markdown(f"##### 📊 Analysis: {var_col}")

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data[var_col], nbinsx=30, opacity=0.7))
            fig.update_layout(title=f"Distribution of {var_col}", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot
            fig = go.Figure()
            fig.add_trace(go.Box(y=data[var_col], name=var_col, boxpoints='outliers'))
            fig.update_layout(title=f"Box Plot: {var_col}", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Statistics
        stats = data[var_col].describe()
        stat_cols = st.columns(4)

        with stat_cols[0]:
            st.metric("Mean", f"{stats['mean']:.3f}")
        with stat_cols[1]:
            st.metric("Median", f"{stats['50%']:.3f}")
        with stat_cols[2]:
            st.metric("Std Dev", f"{stats['std']:.3f}")
        with stat_cols[3]:
            outlier_threshold = 1.5 * (stats['75%'] - stats['25%'])
            outliers = len(data[(data[var_col] < stats['25%'] - outlier_threshold) |
                             (data[var_col] > stats['75%'] + outlier_threshold)])
            st.metric("Outliers", outliers)

        return True

    def _render_categorical_analysis(self, data: pd.DataFrame, column_analysis: Dict) -> bool:
        """Render categorical data analysis"""
        cat_col = column_analysis['categorical_columns'][0]

        st.markdown(f"##### 🏷️ Categorical Analysis: {cat_col}")

        # Value counts
        value_counts = data[cat_col].value_counts()

        # Bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values))
        fig.update_layout(title=f"Distribution of {cat_col}", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", data[cat_col].nunique())
        with col2:
            st.metric("Most Common", value_counts.index[0])
        with col3:
            st.metric("Missing", data[cat_col].isnull().sum())

        return True

    def _render_pattern_summary(self, patterns: Dict):
        """Render pattern analysis summary"""
        if not patterns:
            return

        st.markdown("**🔍 Pattern Analysis:**")
        pattern_cols = st.columns(3)

        with pattern_cols[0]:
            if 'trend' in patterns:
                trend_icon = "📈" if patterns['trend'] == 'increasing' else "📉" if patterns['trend'] == 'decreasing' else "➡️"
                st.metric("Trend", f"{trend_icon} {patterns['trend'].title()}")

        with pattern_cols[1]:
            if 'outliers_percentage' in patterns:
                outlier_color = "🔴" if patterns['outliers_percentage'] > 5 else "🟡" if patterns['outliers_percentage'] > 2 else "🟢"
                st.metric("Outliers", f"{patterns['outliers_percentage']:.1f}%")

        with pattern_cols[2]:
            if 'seasonal_variation' in patterns:
                st.metric("Seasonal Var", f"{patterns['seasonal_variation']:.2f}")

    def _render_data_preview(self, data: pd.DataFrame):
        """Render basic data preview"""
        st.markdown("##### 👀 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)

        # Column info
        col_info = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            unique = data[col].nunique()
            col_info.append({'Column': col, 'Type': dtype, 'Unique Values': unique})

        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True)

    def _render_quick_spatial_viz(self, file_path: str, metadata: Dict):
        """Render quick spatial visualization"""
        try:
            # Create simple spatial map
            spatial_map = create_spatial_map(
                file_path,
                band=1,
                colormap='viridis',
                title="Quick Spatial Preview"
            )

            # Display map
            folium_static(spatial_map, width=700, height=400)

            st.success("🗺️ Interactive spatial visualization created!")
            st.info("💡 Use the full Data Visualizer for advanced spatial analysis, band selection, and custom colormaps.")

        except Exception as e:
            st.error(f"Could not create spatial visualization: {str(e)}")
            st.info("📋 Spatial file detected but visualization failed. You can still upload to the Data Visualizer for analysis.")

    def render_comparison_quick_viz(self,
                                  datasets: List[Tuple[str, pd.DataFrame]],
                                  title: str = "Quick Dataset Comparison") -> bool:
        """
        Render quick comparison visualization for multiple datasets

        Args:
            datasets: List of (name, dataframe) tuples
            title: Comparison title

        Returns:
            True if visualization was successful
        """
        if len(datasets) < 2:
            return self.render_quick_csv_analysis(datasets[0][1], title)

        st.markdown(f"#### 📊 {title}")

        try:
            # Find common patterns across datasets
            common_columns = self._find_common_columns(datasets)

            if common_columns['date_columns'] and common_columns['numeric_columns']:
                self._render_multi_dataset_timeseries(datasets, common_columns)
            else:
                self._render_multi_dataset_summary(datasets)

            return True

        except Exception as e:
            st.error(f"Error in comparison analysis: {str(e)}")
            return False

    def _find_common_columns(self, datasets: List[Tuple[str, pd.DataFrame]]) -> Dict:
        """Find common column patterns across datasets"""
        all_analyses = []

        for name, df in datasets:
            analysis = self._analyze_columns(df)
            all_analyses.append(analysis)

        # Find intersection of column types
        common = {
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': []
        }

        if all_analyses:
            # Find columns that appear in most datasets
            for col_type in common.keys():
                all_cols = []
                for analysis in all_analyses:
                    all_cols.extend(analysis[col_type])

                # Keep columns that appear in at least half the datasets
                from collections import Counter
                col_counts = Counter(all_cols)
                threshold = len(datasets) // 2
                common[col_type] = [col for col, count in col_counts.items() if count > threshold]

        return common

    def _render_multi_dataset_timeseries(self,
                                       datasets: List[Tuple[str, pd.DataFrame]],
                                       common_columns: Dict):
        """Render time series comparison across datasets"""
        date_col = common_columns['date_columns'][0]
        numeric_col = common_columns['numeric_columns'][0]

        fig = go.Figure()

        for name, df in datasets:
            if date_col in df.columns and numeric_col in df.columns:
                # Convert date column
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])

                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[numeric_col],
                    mode='lines',
                    name=name,
                    hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ))

        fig.update_layout(
            title=f"Dataset Comparison: {numeric_col}",
            xaxis_title=date_col,
            yaxis_title=numeric_col,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_multi_dataset_summary(self, datasets: List[Tuple[str, pd.DataFrame]]):
        """Render summary comparison for datasets"""
        summary_data = []

        for name, df in datasets:
            summary_data.append({
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Missing %': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%",
                'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)


# Global instance
quick_visualizer = QuickVisualizer()