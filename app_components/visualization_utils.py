"""
Shared Visualization Utilities
Reusable visualization functions for all modules in the GeoClimate Intelligence Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import rasterio
import rasterio.plot
from datetime import datetime, timedelta
import base64
import io
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


def create_time_series_plot(df: pd.DataFrame,
                           x_col: str,
                           y_cols: List[str],
                           title: str = "Time Series Analysis",
                           x_title: str = "Date",
                           y_title: str = "Value",
                           colors: Optional[List[str]] = None) -> go.Figure:
    """
    Create an interactive time series plot with multiple y variables

    Args:
        df: DataFrame with time series data
        x_col: Column name for x-axis (usually date/time)
        y_cols: List of column names for y-axis values
        title: Plot title
        x_title: X-axis label
        y_title: Y-axis label
        colors: Optional list of colors for each line

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Default colors if none provided
    if colors is None:
        colors = px.colors.qualitative.Set1

    # Add traces for each y column
    for i, y_col in enumerate(y_cols):
        if y_col in df.columns:
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=color),
                    marker=dict(size=4, color=color),
                    hovertemplate=f'<b>{y_col}</b><br>' +
                                f'{x_title}: %{{x}}<br>' +
                                f'{y_title}: %{{y:.2f}}<extra></extra>'
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode='x unified',
        showlegend=len(y_cols) > 1,
        height=500,
        template='plotly_white'
    )

    # Add range selector for time series
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

    return fig


def create_statistical_summary(df: pd.DataFrame,
                             numeric_cols: List[str]) -> pd.DataFrame:
    """
    Create statistical summary for numeric columns

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names

    Returns:
        DataFrame with statistical summary
    """
    summary_stats = []

    for col in numeric_cols:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            stats = {
                'Variable': col,
                'Count': len(df[col].dropna()),
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Std Dev': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Missing': df[col].isna().sum()
            }
            summary_stats.append(stats)

    return pd.DataFrame(summary_stats).round(4)


def create_distribution_plot(df: pd.DataFrame,
                           col: str,
                           plot_type: str = 'histogram') -> go.Figure:
    """
    Create distribution plots (histogram, box plot, etc.)

    Args:
        df: Input DataFrame
        col: Column name to plot
        plot_type: Type of plot ('histogram', 'box', 'violin')

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    if plot_type == 'histogram':
        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=30,
                name=col,
                opacity=0.7,
                marker_color='skyblue'
            )
        )
        fig.update_layout(
            title=f'Distribution of {col}',
            xaxis_title=col,
            yaxis_title='Frequency'
        )

    elif plot_type == 'box':
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                marker_color='lightblue',
                boxpoints='outliers'
            )
        )
        fig.update_layout(
            title=f'Box Plot of {col}',
            yaxis_title=col
        )

    elif plot_type == 'violin':
        fig.add_trace(
            go.Violin(
                y=df[col],
                name=col,
                box_visible=True,
                meanline_visible=True
            )
        )
        fig.update_layout(
            title=f'Violin Plot of {col}',
            yaxis_title=col
        )

    fig.update_layout(
        height=400,
        template='plotly_white'
    )

    return fig


def create_correlation_heatmap(df: pd.DataFrame,
                             numeric_cols: List[str]) -> go.Figure:
    """
    Create correlation heatmap for numeric variables

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names

    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_data = df[numeric_cols].corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        )
    )

    fig.update_layout(
        title='Correlation Matrix',
        height=500,
        template='plotly_white'
    )

    return fig


def create_spatial_map(tiff_path: str,
                      band: int = 1,
                      colormap: str = 'viridis',
                      title: str = "Spatial Data") -> folium.Map:
    """
    Create simple, fast spatial map from TIFF file

    Args:
        tiff_path: Path to TIFF file
        band: Band number to display (1-indexed)
        colormap: Colormap name (fixed to viridis for performance)
        title: Map title

    Returns:
        Folium map object
    """
    try:
        # Open raster file
        with rasterio.open(tiff_path) as src:
            # Read the specified band with downsampling for large files
            data = src.read(band, out_shape=(
                min(src.height, 1000),
                min(src.width, 1000)
            ), resampling=rasterio.enums.Resampling.bilinear)

            bounds = src.bounds

            # Calculate center point
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2

            # Create base map with simplified styling
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap'
            )

            # Get data statistics
            data_min, data_max = np.nanmin(data), np.nanmax(data)

            # Create colormap based on the selected colormap
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from PIL import Image

            # Normalize data
            if data_min != data_max:
                data_normalized = (data - data_min) / (data_max - data_min)
            else:
                data_normalized = np.zeros_like(data)

            # Apply colormap to create RGB image
            if colormap == 'RdYlBu_r':
                cmap = plt.cm.RdYlBu_r
            elif colormap == 'Blues':
                cmap = plt.cm.Blues
            elif colormap == 'plasma':
                cmap = plt.cm.plasma
            elif colormap == 'coolwarm':
                cmap = plt.cm.coolwarm
            else:  # Default to viridis
                cmap = plt.cm.viridis

            # Convert to RGB
            colored_data = cmap(data_normalized)
            # Convert to 0-255 range and remove alpha channel
            colored_data_rgb = (colored_data[:, :, :3] * 255).astype(np.uint8)

            # Add raster overlay with colored data
            folium.raster_layers.ImageOverlay(
                image=colored_data_rgb,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.8,
                interactive=False,
                cross_origin=False,
                zindex=1
            ).add_to(m)

            # Create color bar legend
            from branca.colormap import LinearColormap

            # Create colormap based on the selected colormap
            colormap_colors = {
                'viridis': ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
                'RdYlBu_r': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],
                'Blues': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
                'plasma': ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
                'coolwarm': ['#3b4cc0', '#688aef', '#99baff', '#c9d7f0', '#e8e8e8', '#f7d7d7', '#f9b2b2', '#ed7c7c', '#dc2624']
            }

            colors = colormap_colors.get(colormap, colormap_colors['viridis'])

            # Create the color bar
            colorbar = LinearColormap(
                colors=colors,
                vmin=data_min,
                vmax=data_max,
                caption=f'{title} - Data Range'
            )

            # Add color bar to map
            colorbar.add_to(m)

            # Add enhanced info box with better styling
            info_html = f'''
            <div style="position: fixed;
                        top: 10px; right: 10px; width: 220px; height: 90px;
                        background-color: rgba(255,255,255,0.95); border:2px solid #333; z-index:9999;
                        font-size:13px; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <b style="color: #333; font-size: 14px;">{title}</b><br>
                <div style="margin-top: 5px; color: #666;">
                    <strong>Min:</strong> {data_min:.3f}<br>
                    <strong>Max:</strong> {data_max:.3f}<br>
                    <strong>Range:</strong> {data_max - data_min:.3f}
                </div>
            </div>'''
            m.get_root().html.add_child(folium.Element(info_html))

            return m

    except Exception as e:
        st.error(f"Error creating spatial map: {str(e)}")
        # Return basic map on error
        return folium.Map(location=[40, -100], zoom_start=4)


def create_comparison_plot(df: pd.DataFrame,
                          x_col: str,
                          y_cols: List[str],
                          comparison_type: str = 'overlay') -> go.Figure:
    """
    Create comparison plots for multiple variables

    Args:
        df: Input DataFrame
        x_col: X-axis column
        y_cols: List of Y-axis columns to compare
        comparison_type: 'overlay', 'subplot', or 'difference'

    Returns:
        Plotly figure object
    """
    if comparison_type == 'overlay':
        return create_time_series_plot(df, x_col, y_cols, title="Variable Comparison")

    elif comparison_type == 'subplot':
        fig = make_subplots(
            rows=len(y_cols), cols=1,
            subplot_titles=y_cols,
            shared_xaxes=True,
            vertical_spacing=0.08
        )

        colors = px.colors.qualitative.Set1

        for i, y_col in enumerate(y_cols, 1):
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines',
                    name=y_col,
                    line=dict(color=colors[i-1 % len(colors)])
                ),
                row=i, col=1
            )

        fig.update_layout(
            height=300 * len(y_cols),
            title_text="Variable Comparison (Subplots)",
            template='plotly_white'
        )

        return fig

    elif comparison_type == 'difference' and len(y_cols) == 2:
        # Calculate difference
        diff_col = f"{y_cols[0]} - {y_cols[1]}"
        df[diff_col] = df[y_cols[0]] - df[y_cols[1]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[diff_col],
                mode='lines+markers',
                name=diff_col,
                line=dict(color='red'),
                fill='tonexty'
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="black")

        fig.update_layout(
            title=f"Difference Plot: {diff_col}",
            xaxis_title=x_col,
            yaxis_title="Difference",
            height=400,
            template='plotly_white'
        )

        return fig


def export_plot_as_html(fig: go.Figure, filename: str) -> str:
    """
    Export plotly figure as HTML string for download

    Args:
        fig: Plotly figure object
        filename: Output filename

    Returns:
        Base64 encoded HTML string for download
    """
    html_str = fig.to_html()
    b64 = base64.b64encode(html_str.encode()).decode()
    return b64


def create_data_summary_card(df: pd.DataFrame) -> Dict:
    """
    Create summary statistics card for DataFrame

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'date_columns': len(date_cols),
        'missing_values': df.isnull().sum().sum(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
    }

    if date_cols:
        date_col = date_cols[0]
        summary['date_range'] = {
            'start': df[date_col].min(),
            'end': df[date_col].max(),
            'duration': (df[date_col].max() - df[date_col].min()).days
        }

    return summary


def detect_time_series_patterns(df: pd.DataFrame,
                               date_col: str,
                               value_col: str) -> Dict:
    """
    Detect patterns in time series data

    Args:
        df: Input DataFrame
        date_col: Date column name
        value_col: Value column name

    Returns:
        Dictionary with detected patterns
    """
    patterns = {}

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df_sorted = df.sort_values(date_col)

    # Basic trend detection
    if len(df_sorted) > 1:
        slope = np.polyfit(range(len(df_sorted)), df_sorted[value_col], 1)[0]
        patterns['trend'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        patterns['slope'] = slope

    # Seasonality detection (simplified)
    if len(df_sorted) > 12:
        df_sorted['month'] = df_sorted[date_col].dt.month
        monthly_means = df_sorted.groupby('month')[value_col].mean()
        patterns['seasonal_variation'] = monthly_means.max() - monthly_means.min()
        patterns['peak_month'] = monthly_means.idxmax()
        patterns['low_month'] = monthly_means.idxmin()

    # Outlier detection (simple IQR method)
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[value_col] < Q1 - 1.5 * IQR) | (df[value_col] > Q3 + 1.5 * IQR)]
    patterns['outliers_count'] = len(outliers)
    patterns['outliers_percentage'] = (len(outliers) / len(df)) * 100

    return patterns


def mann_kendall_test(data: pd.Series) -> Dict:
    """
    Perform Mann-Kendall trend test on time series data

    Args:
        data: Time series data as pandas Series

    Returns:
        Dictionary with test results
    """
    try:
        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) < 3:
            return {'error': 'Insufficient data for Mann-Kendall test (need at least 3 points)'}

        n = len(clean_data)

        # Calculate S statistic
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if clean_data.iloc[j] > clean_data.iloc[i]:
                    S += 1
                elif clean_data.iloc[j] < clean_data.iloc[i]:
                    S -= 1

        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18

        # Calculate Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0

        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

        # Determine trend
        alpha = 0.05  # significance level
        if p_value < alpha:
            if S > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            significant = True
        else:
            trend = 'no significant trend'
            significant = False

        return {
            'S': S,
            'Z': Z,
            'p_value': p_value,
            'trend': trend,
            'significant': significant,
            'alpha': alpha,
            'n': n
        }

    except Exception as e:
        return {'error': f'Error in Mann-Kendall test: {str(e)}'}


def sens_slope(data: pd.Series) -> Dict:
    """
    Calculate Sen's slope for trend magnitude estimation

    Args:
        data: Time series data as pandas Series

    Returns:
        Dictionary with slope results
    """
    try:
        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) < 2:
            return {'error': 'Insufficient data for Sen\'s slope (need at least 2 points)'}

        n = len(clean_data)
        slopes = []

        # Calculate all possible slopes
        for i in range(n-1):
            for j in range(i+1, n):
                if j != i:  # Avoid division by zero
                    slope = (clean_data.iloc[j] - clean_data.iloc[i]) / (j - i)
                    slopes.append(slope)

        if not slopes:
            return {'error': 'Could not calculate slopes'}

        # Sen's slope is the median of all slopes
        sens_slope_value = np.median(slopes)

        # Calculate confidence interval (simplified)
        slopes_sorted = np.sort(slopes)
        n_slopes = len(slopes)

        # 95% confidence interval indices (simplified)
        if n_slopes >= 3:
            ci_low_idx = max(0, int(0.025 * n_slopes))
            ci_high_idx = min(n_slopes - 1, int(0.975 * n_slopes))
            ci_low = slopes_sorted[ci_low_idx]
            ci_high = slopes_sorted[ci_high_idx]
        else:
            ci_low, ci_high = sens_slope_value, sens_slope_value

        return {
            'slope': sens_slope_value,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_slopes': n_slopes,
            'slope_unit': 'units per time step'
        }

    except Exception as e:
        return {'error': f'Error in Sen\'s slope calculation: {str(e)}'}


def detect_spatial_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect spatial coordinate columns in DataFrame

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with detected spatial columns
    """
    spatial_cols = {
        'latitude': [],
        'longitude': [],
        'coordinate_pairs': []
    }

    # Common latitude column names
    lat_patterns = ['lat', 'latitude', 'y', 'north', 'northing']
    # Common longitude column names
    lon_patterns = ['lon', 'lng', 'long', 'longitude', 'x', 'east', 'easting']

    for col in df.columns:
        col_lower = col.lower()

        # Check for latitude
        if any(pattern in col_lower for pattern in lat_patterns):
            # Verify values are in reasonable latitude range
            if df[col].dtype in ['float64', 'int64']:
                if df[col].min() >= -90 and df[col].max() <= 90:
                    spatial_cols['latitude'].append(col)

        # Check for longitude
        elif any(pattern in col_lower for pattern in lon_patterns):
            # Verify values are in reasonable longitude range
            if df[col].dtype in ['float64', 'int64']:
                if df[col].min() >= -180 and df[col].max() <= 180:
                    spatial_cols['longitude'].append(col)

    return spatial_cols


def create_spatial_scatter_map(df: pd.DataFrame,
                              lat_col: str,
                              lon_col: str,
                              value_col: str = None,
                              title: str = "Spatial Data") -> folium.Map:
    """
    Create spatial scatter plot map from CSV data

    Args:
        df: DataFrame with spatial data
        lat_col: Latitude column name
        lon_col: Longitude column name
        value_col: Optional value column for color coding
        title: Map title

    Returns:
        Folium map object
    """
    try:
        # Remove rows with NaN coordinates
        clean_df = df.dropna(subset=[lat_col, lon_col])

        if len(clean_df) == 0:
            st.warning("No valid coordinate pairs found")
            return folium.Map(location=[40, -100], zoom_start=4)

        # Calculate map center
        center_lat = clean_df[lat_col].mean()
        center_lon = clean_df[lon_col].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )

        # Determine color scheme
        if value_col and value_col in clean_df.columns:
            # Color code by values
            values = clean_df[value_col]
            min_val, max_val = values.min(), values.max()

            # Create colormap
            from branca.colormap import LinearColormap
            colormap = LinearColormap(
                colors=['blue', 'green', 'yellow', 'red'],
                vmin=min_val,
                vmax=max_val,
                caption=f'{value_col} Values'
            )
            colormap.add_to(m)

            # Add colored markers
            for idx, row in clean_df.iterrows():
                if pd.notna(row[value_col]):
                    color = colormap(row[value_col])
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=6,
                        popup=f"{lat_col}: {row[lat_col]:.4f}<br>{lon_col}: {row[lon_col]:.4f}<br>{value_col}: {row[value_col]:.3f}",
                        color='black',
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=1
                    ).add_to(m)
        else:
            # Simple markers without color coding
            for idx, row in clean_df.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=f"{lat_col}: {row[lat_col]:.4f}<br>{lon_col}: {row[lon_col]:.4f}",
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)

        # Add info box
        info_html = f'''
        <div style="position: fixed;
                    top: 10px; right: 10px; width: 200px; height: 80px;
                    background-color: rgba(255,255,255,0.95); border:2px solid #333; z-index:9999;
                    font-size:12px; padding: 8px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <b style="color: #333; font-size: 13px;">{title}</b><br>
            <div style="margin-top: 3px; color: #666;">
                <strong>Points:</strong> {len(clean_df)}<br>
                <strong>Lat Range:</strong> {clean_df[lat_col].min():.2f} to {clean_df[lat_col].max():.2f}<br>
                <strong>Lon Range:</strong> {clean_df[lon_col].min():.2f} to {clean_df[lon_col].max():.2f}
            </div>
        </div>'''
        m.get_root().html.add_child(folium.Element(info_html))

        return m

    except Exception as e:
        st.error(f"Error creating spatial map: {str(e)}")
        return folium.Map(location=[40, -100], zoom_start=4)