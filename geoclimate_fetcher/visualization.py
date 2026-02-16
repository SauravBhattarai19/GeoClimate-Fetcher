"""
Visualization module for GeoClimate Intelligence Platform
Provides functions to visualize climate data in various formats
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import rasterio
from rasterio.plot import show
import folium
from folium import plugins
import streamlit as st
from datetime import datetime


class DataVisualizer:
    """Handles visualization of climate data in various formats"""
    
    def __init__(self):
        self.colorscales = {
            'temperature': 'RdBu_r',
            'precipitation': 'Blues',
            'ndvi': 'RdYlGn',
            'default': 'Viridis'
        }
    
    def visualize_netcdf(self, file_path):
        """Visualize NetCDF file with temporal and spatial plots"""
        try:
            # Load the dataset
            ds = xr.open_dataset(file_path)
            
            # Display dataset info
            st.write("### 📊 NetCDF Dataset Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Variables", len(ds.data_vars))
                st.write("**Available variables:**")
                for var in ds.data_vars:
                    st.write(f"• {var}")
            
            with col2:
                dims_info = {dim: len(ds[dim]) for dim in ds.dims}
                st.metric("Dimensions", len(ds.dims))
                for dim, size in dims_info.items():
                    st.write(f"• {dim}: {size}")
            
            with col3:
                # Show coordinate ranges
                st.write("**Coordinate ranges:**")
                for coord in ds.coords:
                    if len(ds[coord]) > 0:
                        st.write(f"• {coord}: [{ds[coord].min().values:.2f}, {ds[coord].max().values:.2f}]")
            
            # Variable selection for visualization
            st.write("---")
            st.write("### 🎨 Visualization Options")
            
            selected_var = st.selectbox("Select variable to visualize:", list(ds.data_vars))
            
            # Check if data has time dimension
            has_time = 'time' in ds.dims
            
            if has_time:
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🗺️ Spatial Map", "🎬 Animation", "📊 Statistics"])
                
                with tab1:
                    self._plot_time_series(ds, selected_var)
                
                with tab2:
                    self._plot_spatial_map(ds, selected_var)
                
                with tab3:
                    self._create_animation(ds, selected_var)
                
                with tab4:
                    self._show_statistics(ds, selected_var)
            else:
                # For non-temporal data
                tab1, tab2 = st.tabs(["🗺️ Spatial Map", "📊 Statistics"])
                
                with tab1:
                    self._plot_spatial_map(ds, selected_var)
                
                with tab2:
                    self._show_statistics(ds, selected_var)
            
            # Close the dataset
            ds.close()
            
        except Exception as e:
            st.error(f"Error visualizing NetCDF file: {str(e)}")
    
    def _plot_time_series(self, ds, var_name):
        """Plot time series for NetCDF data"""
        try:
            # Get the data
            data = ds[var_name]
            
            # Calculate spatial mean for each time step
            if 'lat' in data.dims and 'lon' in data.dims:
                ts_mean = data.mean(dim=['lat', 'lon'])
                ts_min = data.min(dim=['lat', 'lon'])
                ts_max = data.max(dim=['lat', 'lon'])
            else:
                # Handle different dimension names
                spatial_dims = [d for d in data.dims if d != 'time']
                ts_mean = data.mean(dim=spatial_dims)
                ts_min = data.min(dim=spatial_dims)
                ts_max = data.max(dim=spatial_dims)
            
            # Create the plot
            fig = go.Figure()
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=data.time.values,
                y=ts_mean.values,
                mode='lines',
                name='Mean',
                line=dict(color='blue', width=2)
            ))
            
            # Add range (min-max)
            fig.add_trace(go.Scatter(
                x=data.time.values,
                y=ts_max.values,
                mode='lines',
                name='Max',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=data.time.values,
                y=ts_min.values,
                mode='lines',
                name='Min',
                line=dict(color='green', width=1, dash='dash'),
                fill='tonexty'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Time Series: {var_name}",
                xaxis_title="Time",
                yaxis_title=f"{var_name} {data.attrs.get('units', '')}",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend analysis
            if st.checkbox("Show trend analysis"):
                self._plot_trend_analysis(data.time.values, ts_mean.values, var_name)
                
        except Exception as e:
            st.error(f"Error plotting time series: {str(e)}")
    
    def _plot_spatial_map(self, ds, var_name):
        """Plot spatial map for NetCDF data"""
        try:
            data = ds[var_name]
            
            # Handle time dimension
            if 'time' in data.dims:
                # Let user select time step
                time_steps = data.time.values
                time_idx = st.slider(
                    "Select time step:",
                    0, len(time_steps) - 1,
                    len(time_steps) // 2
                )
                data_slice = data.isel(time=time_idx)
                title_suffix = f" at {pd.Timestamp(time_steps[time_idx]).strftime('%Y-%m-%d')}"
            else:
                data_slice = data
                title_suffix = ""
            
            # Determine colorscale
            colorscale = self._get_colorscale(var_name)
            
            # Create the heatmap
            fig = px.imshow(
                data_slice.values,
                labels=dict(x="Longitude", y="Latitude", color=var_name),
                title=f"Spatial Distribution: {var_name}{title_suffix}",
                color_continuous_scale=colorscale,
                aspect='auto'
            )
            
            # Update axes with actual coordinates
            if 'lon' in data_slice.coords and 'lat' in data_slice.coords:
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=np.linspace(0, len(data_slice.lon) - 1, 5),
                    ticktext=[f"{v:.1f}" for v in np.linspace(
                        data_slice.lon.min().values,
                        data_slice.lon.max().values,
                        5
                    )]
                )
                fig.update_yaxes(
                    tickmode='array',
                    tickvals=np.linspace(0, len(data_slice.lat) - 1, 5),
                    ticktext=[f"{v:.1f}" for v in np.linspace(
                        data_slice.lat.min().values,
                        data_slice.lat.max().values,
                        5
                    )]
                )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting spatial map: {str(e)}")
    
    def _create_animation(self, ds, var_name):
        """Create animation for temporal data"""
        try:
            data = ds[var_name]
            
            if 'time' not in data.dims:
                st.warning("No time dimension found for animation")
                return
            
            # Subsample time if too many frames
            time_steps = data.time.values
            max_frames = 50
            
            if len(time_steps) > max_frames:
                st.warning(f"Dataset has {len(time_steps)} time steps. Subsampling to {max_frames} frames for performance.")
                indices = np.linspace(0, len(time_steps) - 1, max_frames, dtype=int)
                data_subset = data.isel(time=indices)
                time_subset = time_steps[indices]
            else:
                data_subset = data
                time_subset = time_steps
            
            # Create frames
            frames = []
            for i, t in enumerate(time_subset):
                frame_data = data_subset.isel(time=i).values
                frames.append(go.Frame(
                    data=[go.Heatmap(
                        z=frame_data,
                        colorscale=self._get_colorscale(var_name),
                        zmin=data.min().values,
                        zmax=data.max().values
                    )],
                    name=str(i),
                    layout=go.Layout(
                        title=f"{var_name} - {pd.Timestamp(t).strftime('%Y-%m-%d')}"
                    )
                ))
            
            # Create figure
            fig = go.Figure(
                data=[go.Heatmap(
                    z=data_subset.isel(time=0).values,
                    colorscale=self._get_colorscale(var_name),
                    zmin=data.min().values,
                    zmax=data.max().values
                )],
                layout=go.Layout(
                    title=f"{var_name} Animation",
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {'label': 'Play', 'method': 'animate', 'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True
                            }]},
                            {'label': 'Pause', 'method': 'animate', 'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]}
                        ]
                    }],
                    sliders=[{
                        'steps': [
                            {'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}}],
                             'label': pd.Timestamp(t).strftime('%Y-%m-%d'),
                             'method': 'animate'}
                            for i, t in enumerate(time_subset)
                        ]
                    }]
                ),
                frames=frames
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating animation: {str(e)}")
    
    def _show_statistics(self, ds, var_name):
        """Show statistical summary of the data"""
        try:
            data = ds[var_name]
            
            # Basic statistics
            st.write("#### Basic Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{float(data.mean().values):.4f}")
            with col2:
                st.metric("Std Dev", f"{float(data.std().values):.4f}")
            with col3:
                st.metric("Min", f"{float(data.min().values):.4f}")
            with col4:
                st.metric("Max", f"{float(data.max().values):.4f}")
            
            # Distribution plot
            st.write("#### Value Distribution")
            
            # Flatten the data for histogram
            flat_data = data.values.flatten()
            # Remove NaN values
            flat_data = flat_data[~np.isnan(flat_data)]
            
            # Sample if too large
            if len(flat_data) > 100000:
                flat_data = np.random.choice(flat_data, 100000, replace=False)
            
            fig = px.histogram(
                x=flat_data,
                nbins=50,
                title=f"Distribution of {var_name}",
                labels={'x': var_name, 'y': 'Count'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Percentiles
            if st.checkbox("Show percentiles"):
                percentiles = [0, 10, 25, 50, 75, 90, 100]
                perc_values = np.percentile(flat_data, percentiles)
                
                perc_df = pd.DataFrame({
                    'Percentile': [f"{p}%" for p in percentiles],
                    'Value': perc_values
                })
                
                st.dataframe(perc_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error showing statistics: {str(e)}")
    
    def visualize_geotiff(self, file_path):
        """Visualize GeoTIFF file"""
        try:
            # Show helpful info about visualization improvements
            st.info("🎨 **Visualization Enhanced**: Fixed compatibility issues and improved coordinate display")
            
            with rasterio.open(file_path) as src:
                # Display metadata
                st.write("### 📊 GeoTIFF Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bands", src.count)
                    st.metric("Width", f"{src.width} px")
                    st.metric("Height", f"{src.height} px")
                
                with col2:
                    st.metric("CRS", str(src.crs))
                    res = src.res
                    st.metric("Resolution", f"{res[0]:.2f} x {res[1]:.2f}")
                
                with col3:
                    bounds = src.bounds
                    st.write("**Bounds:**")
                    st.write(f"• Left: {bounds.left:.4f}")
                    st.write(f"• Right: {bounds.right:.4f}")
                    st.write(f"• Top: {bounds.top:.4f}")
                    st.write(f"• Bottom: {bounds.bottom:.4f}")
                
                # Band selection
                st.write("---")
                if src.count > 1:
                    band_idx = st.selectbox(
                        "Select band to visualize:",
                        range(1, src.count + 1),
                        format_func=lambda x: f"Band {x}"
                    )
                else:
                    band_idx = 1
                
                # Read the band
                band_data = src.read(band_idx)
                
                # Create visualization tabs
                tab1, tab2, tab3 = st.tabs(["🗺️ Raster View", "📊 Statistics", "🎨 3D View"])
                
                with tab1:
                    self._plot_raster(band_data, src, band_idx)
                
                with tab2:
                    self._show_raster_statistics(band_data, band_idx)
                
                with tab3:
                    if st.checkbox("Generate 3D visualization (may be slow for large rasters)"):
                        self._plot_3d_surface(band_data, src)
                        
        except Exception as e:
            st.error(f"Error visualizing GeoTIFF: {str(e)}")
    
    def _plot_raster(self, band_data, src, band_idx):
        """Plot raster data"""
        try:
            # Handle nodata values first
            if src.nodata is not None:
                band_data = np.ma.masked_equal(band_data, src.nodata)
            
            # Handle potential data type issues
            if band_data.dtype == 'uint8':
                band_data = band_data.astype(np.float32)
            
            # Determine colorscale based on data range
            valid_data = band_data[~np.isnan(band_data) if isinstance(band_data, np.ma.MaskedArray) else ~np.isnan(band_data)]
            if len(valid_data) == 0:
                st.error("No valid data found in this band.")
                return
                
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
            
            # Create the plot
            fig = px.imshow(
                band_data,
                title=f"Band {band_idx} Visualization (Range: {vmin:.2f} to {vmax:.2f})",
                color_continuous_scale='viridis',
                aspect='equal'
            )
            
            # Add proper coordinates
            transform = src.transform
            height, width = band_data.shape
            
            # Calculate extent more robustly
            try:
                left = transform[2]
                top = transform[5]
                right = left + width * transform[0]
                bottom = top + height * transform[4]
                
                # Only update axes if we have valid coordinates
                if not (np.isnan(left) or np.isnan(top) or np.isnan(right) or np.isnan(bottom)):
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=np.linspace(0, width - 1, 5),
                        ticktext=[f"{v:.4f}" for v in np.linspace(left, right, 5)],
                        title="Longitude"
                    )
                    
                    fig.update_yaxes(
                        tickmode='array',
                        tickvals=np.linspace(0, height - 1, 5),
                        ticktext=[f"{v:.4f}" for v in np.linspace(top, bottom, 5)],
                        title="Latitude"
                    )
                else:
                    # Fallback to pixel coordinates
                    fig.update_xaxes(title="Pixel X")
                    fig.update_yaxes(title="Pixel Y")
            except Exception as coord_error:
                st.warning(f"Could not set geographic coordinates: {str(coord_error)}")
                fig.update_xaxes(title="Pixel X")
                fig.update_yaxes(title="Pixel Y")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Color scale adjustment
            col1, col2 = st.columns(2)
            with col1:
                new_vmin = st.number_input("Min value", value=float(vmin), key="vmin")
            with col2:
                new_vmax = st.number_input("Max value", value=float(vmax), key="vmax")
            
            if st.button("Update color scale"):
                fig.update_coloraxes(cmin=new_vmin, cmax=new_vmax)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error plotting raster: {str(e)}")
            
            # Provide helpful troubleshooting info
            with st.expander("🔍 Troubleshooting Information"):
                st.write("**Common solutions:**")
                st.write("• Check if the GeoTIFF file is valid and not corrupted")
                st.write("• Ensure the file has proper coordinate reference system")
                st.write("• Try a different band if multiple bands are available")
                st.write("• Verify the file has valid data (not all nodata values)")
                
                st.write("**Technical Details:**")
                st.code(f"Error: {str(e)}", language="text")
    
    def _show_raster_statistics(self, band_data, band_idx):
        """Show statistics for raster band"""
        try:
            # Remove nodata values more robustly
            if isinstance(band_data, np.ma.MaskedArray):
                valid_data = band_data.compressed()  # Get unmasked data
            else:
                valid_data = band_data[~np.isnan(band_data)]
            
            # Ensure we have valid data
            if len(valid_data) == 0:
                st.warning(f"No valid data found in Band {band_idx}")
                return
            
            st.write(f"#### Band {band_idx} Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{np.mean(valid_data):.4f}")
            with col2:
                st.metric("Std Dev", f"{np.std(valid_data):.4f}")
            with col3:
                st.metric("Min", f"{np.min(valid_data):.4f}")
            with col4:
                st.metric("Max", f"{np.max(valid_data):.4f}")
            
            # Histogram
            st.write("#### Value Distribution")
            
            # Sample if too large
            if len(valid_data) > 100000:
                sample_data = np.random.choice(valid_data.flatten(), 100000, replace=False)
            else:
                sample_data = valid_data.flatten()
            
            fig = px.histogram(
                x=sample_data,
                nbins=50,
                title=f"Band {band_idx} Value Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error showing raster statistics: {str(e)}")
    
    def _plot_3d_surface(self, band_data, src):
        """Create 3D surface plot of raster"""
        try:
            # Downsample for performance
            max_size = 100
            if band_data.shape[0] > max_size or band_data.shape[1] > max_size:
                step_y = max(1, band_data.shape[0] // max_size)
                step_x = max(1, band_data.shape[1] // max_size)
                data_subset = band_data[::step_y, ::step_x]
                st.info(f"Downsampled to {data_subset.shape} for 3D visualization")
            else:
                data_subset = band_data
            
            # Create coordinate arrays
            y = np.arange(data_subset.shape[0])
            x = np.arange(data_subset.shape[1])
            
            fig = go.Figure(data=[go.Surface(
                z=data_subset,
                x=x,
                y=y,
                colorscale='viridis'
            )])
            
            fig.update_layout(
                title='3D Surface Plot',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Value'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating 3D plot: {str(e)}")
    
    def visualize_csv(self, file_path):
        """Visualize CSV time series data"""
        try:
            # Load the CSV
            df = pd.read_csv(file_path)
            
            st.write("### 📊 CSV Data Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory", f"{df.memory_usage().sum() / 1024:.1f} KB")
            
            # Show data preview
            st.write("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check for date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                # Parse dates
                date_col = st.selectbox("Select date column:", date_cols)
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=date_col)
                
                # Select columns to plot
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    st.write("---")
                    st.write("### 📈 Time Series Visualization")
                    
                    selected_cols = st.multiselect(
                        "Select columns to plot:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    )
                    
                    if selected_cols:
                        # Create time series plot
                        fig = go.Figure()
                        
                        for col in selected_cols:
                            fig.add_trace(go.Scatter(
                                x=df[date_col],
                                y=df[col],
                                mode='lines+markers',
                                name=col
                            ))
                        
                        fig.update_layout(
                            title="Time Series Plot",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional analysis options
                        if st.checkbox("Show correlation matrix"):
                            corr_matrix = df[selected_cols].corr()
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                labels=dict(color="Correlation"),
                                x=selected_cols,
                                y=selected_cols,
                                color_continuous_scale='RdBu',
                                zmin=-1,
                                zmax=1
                            )
                            
                            fig_corr.update_layout(title="Correlation Matrix")
                            st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("No date/time column found. Showing basic statistics only.")
                
                # Show basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write("#### Numeric Column Statistics")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error visualizing CSV: {str(e)}")
    
    def _get_colorscale(self, var_name):
        """Get appropriate colorscale based on variable name"""
        var_lower = var_name.lower()
        
        for key in self.colorscales:
            if key in var_lower:
                return self.colorscales[key]
        
        return self.colorscales['default']
    
    def _plot_trend_analysis(self, time_values, data_values, var_name):
        """Plot trend analysis with linear regression"""
        try:
            # Convert time to numeric for regression
            time_numeric = np.arange(len(time_values))
            
            # Calculate linear regression
            z = np.polyfit(time_numeric, data_values, 1)
            p = np.poly1d(z)
            
            # Create the plot
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=time_values,
                y=data_values,
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1)
            ))
            
            # Trend line
            fig.add_trace(go.Scatter(
                x=time_values,
                y=p(time_numeric),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add trend info
            slope = z[0]
            if hasattr(time_values[0], 'year'):
                # If time is datetime, calculate per year
                time_span_years = (time_values[-1] - time_values[0]).days / 365.25
                trend_per_year = slope * (len(time_values) / time_span_years)
                trend_text = f"Trend: {trend_per_year:.4f} per year"
            else:
                trend_text = f"Slope: {slope:.4f}"
            
            fig.update_layout(
                title=f"Trend Analysis: {var_name}<br><sub>{trend_text}</sub>",
                xaxis_title="Time",
                yaxis_title=var_name,
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}") 