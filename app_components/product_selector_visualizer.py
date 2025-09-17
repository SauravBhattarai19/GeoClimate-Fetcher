"""
Visualization module for Optimal Product Selector results
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class ProductSelectorVisualizer:
    """Create visualizations for product selector analysis results"""
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def create_scatter_plot(self, merged_data: pd.DataFrame, station_id: str, dataset_name: str, variable: str) -> go.Figure:
        """Create scatter plot comparing station vs gridded data
        
        Args:
            merged_data: DataFrame with merged station and gridded data
            station_id: Station identifier
            dataset_name: Dataset name
            variable: Variable name
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=merged_data['value_station'],
                y=merged_data['value_gridded'],
                mode='markers',
                name='Data Points',
                marker=dict(
                    size=6,
                    color='blue',
                    opacity=0.6
                ),
                hovertemplate='<b>Station:</b> %{x}<br><b>Gridded:</b> %{y}<extra></extra>'
            ))
            
            # Add 1:1 line
            min_val = min(merged_data['value_station'].min(), merged_data['value_gridded'].min())
            max_val = max(merged_data['value_station'].max(), merged_data['value_gridded'].max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='1:1 Line',
                line=dict(color='red', dash='dash'),
                hoverinfo='skip'
            ))
            
            # Add trend line
            z = np.polyfit(merged_data['value_station'], merged_data['value_gridded'], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[p(min_val), p(max_val)],
                mode='lines',
                name='Trend Line',
                line=dict(color='green'),
                hoverinfo='skip'
            ))
            
            # Update layout
            units_map = {'prcp': 'mm', 'tmax': '°C', 'tmin': '°C'}
            unit = units_map.get(variable, 'units')
            
            fig.update_layout(
                title=f'Station vs Gridded Data Comparison<br>{station_id} - {dataset_name}',
                xaxis_title=f'Station Data ({unit})',
                yaxis_title=f'Gridded Data ({unit})',
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating scatter plot: {str(e)}")
            return go.Figure()
    
    def create_time_series_plot(self, merged_data: pd.DataFrame, station_id: str, dataset_name: str, variable: str) -> go.Figure:
        """Create time series plot comparing station vs gridded data
        
        Args:
            merged_data: DataFrame with merged station and gridded data
            station_id: Station identifier
            dataset_name: Dataset name
            variable: Variable name
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add station data
            fig.add_trace(go.Scatter(
                x=merged_data['date'],
                y=merged_data['value_station'],
                mode='lines',
                name=f'Station ({station_id})',
                line=dict(color='blue'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Station:</b> %{y}<extra></extra>'
            ))
            
            # Add gridded data
            fig.add_trace(go.Scatter(
                x=merged_data['date'],
                y=merged_data['value_gridded'],
                mode='lines',
                name=f'Gridded ({dataset_name})',
                line=dict(color='red'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Gridded:</b> %{y}<extra></extra>'
            ))
            
            # Update layout
            units_map = {'prcp': 'mm', 'tmax': '°C', 'tmin': '°C'}
            unit = units_map.get(variable, 'units')
            
            fig.update_layout(
                title=f'Time Series Comparison<br>{station_id} - {dataset_name}',
                xaxis_title='Date',
                yaxis_title=f'{variable.upper()} ({unit})',
                showlegend=True,
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating time series plot: {str(e)}")
            return go.Figure()
    
    def create_seasonal_boxplot(self, merged_data: pd.DataFrame, station_id: str, dataset_name: str, variable: str) -> go.Figure:
        """Create seasonal boxplot comparison
        
        Args:
            merged_data: DataFrame with merged station and gridded data
            station_id: Station identifier
            dataset_name: Dataset name
            variable: Variable name
            
        Returns:
            Plotly figure
        """
        try:
            # Add season column
            merged_data = merged_data.copy()
            merged_data['month'] = merged_data['date'].dt.month
            merged_data['season'] = merged_data['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Prepare data for plotting
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']
            fig = go.Figure()
            
            for i, season in enumerate(seasons):
                season_data = merged_data[merged_data['season'] == season]
                
                if len(season_data) > 0:
                    # Station data
                    fig.add_trace(go.Box(
                        y=season_data['value_station'],
                        name=f'{season} - Station',
                        boxpoints='outliers',
                        marker_color='blue',
                        offsetgroup=i,
                        legendgroup='station',
                        legendgrouptitle_text='Station Data'
                    ))
                    
                    # Gridded data
                    fig.add_trace(go.Box(
                        y=season_data['value_gridded'],
                        name=f'{season} - Gridded',
                        boxpoints='outliers',
                        marker_color='red',
                        offsetgroup=i,
                        legendgroup='gridded',
                        legendgrouptitle_text='Gridded Data'
                    ))
            
            # Update layout
            units_map = {'prcp': 'mm', 'tmax': '°C', 'tmin': '°C'}
            unit = units_map.get(variable, 'units')
            
            fig.update_layout(
                title=f'Seasonal Distribution Comparison<br>{station_id} - {dataset_name}',
                yaxis_title=f'{variable.upper()} ({unit})',
                boxmode='group',
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating seasonal boxplot: {str(e)}")
            return go.Figure()
    
    def create_statistics_table(self, stats: Dict, station_id: str, dataset_name: str) -> go.Figure:
        """Create a table with statistical metrics
        
        Args:
            stats: Dictionary with statistical metrics
            station_id: Station identifier
            dataset_name: Dataset name
            
        Returns:
            Plotly figure with table
        """
        try:
            # Prepare table data
            metrics = [
                ('Number of Observations', stats.get('n_observations', 'N/A')),
                ('RMSE', f"{stats.get('rmse', 0):.3f}"),
                ('MAE', f"{stats.get('mae', 0):.3f}"),
                ('R²', f"{stats.get('r2', 0):.3f}"),
                ('Correlation', f"{stats.get('correlation', 0):.3f}"),
                ('Bias', f"{stats.get('bias', 0):.3f}"),
                ('Station Mean', f"{stats.get('station_mean', 0):.3f}"),
                ('Gridded Mean', f"{stats.get('gridded_mean', 0):.3f}"),
                ('Station Std Dev', f"{stats.get('station_std', 0):.3f}"),
                ('Gridded Std Dev', f"{stats.get('gridded_std', 0):.3f}")
            ]
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='lightblue',
                    align='left',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[[metric[0] for metric in metrics], 
                           [metric[1] for metric in metrics]],
                    fill_color='white',
                    align='left',
                    font=dict(size=11)
                )
            )])
            
            fig.update_layout(
                title=f'Statistical Summary<br>{station_id} - {dataset_name}',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating statistics table: {str(e)}")
            return go.Figure()
    
    def create_seasonal_stats_table(self, seasonal_stats: Dict, station_id: str, dataset_name: str) -> go.Figure:
        """Create a table with seasonal statistical metrics
        
        Args:
            seasonal_stats: Dictionary with seasonal statistics
            station_id: Station identifier
            dataset_name: Dataset name
            
        Returns:
            Plotly figure with table
        """
        try:
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']
            metrics = ['n_observations', 'rmse', 'mae', 'r2', 'correlation', 'bias']
            
            # Prepare table data
            header_values = ['Season'] + [metric.upper().replace('_', ' ').title() for metric in metrics]
            
            table_data = []
            for season in seasons:
                row = [season]
                season_data = seasonal_stats.get(season, {})
                
                for metric in metrics:
                    value = season_data.get(metric, 'N/A')
                    if isinstance(value, (int, float)) and metric != 'n_observations':
                        row.append(f"{value:.3f}")
                    else:
                        row.append(str(value))
                
                table_data.append(row)
            
            # Transpose for table format
            table_values = list(map(list, zip(*table_data)))
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=header_values,
                    fill_color='lightgreen',
                    align='left',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=table_values,
                    fill_color='white',
                    align='left',
                    font=dict(size=11)
                )
            )])
            
            fig.update_layout(
                title=f'Seasonal Statistics<br>{station_id} - {dataset_name}',
                height=300
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating seasonal stats table: {str(e)}")
            return go.Figure()
    
    def create_comparison_overview(self, all_results: Dict, variable: str) -> go.Figure:
        """Create overview comparison of all datasets for all stations
        
        Args:
            all_results: Dictionary with all analysis results
            variable: Variable name
            
        Returns:
            Plotly figure
        """
        try:
            # Prepare data for comparison
            comparison_data = []
            
            for station_id, station_results in all_results.items():
                for dataset_name, dataset_results in station_results.items():
                    if 'stats' in dataset_results:
                        stats = dataset_results['stats']
                        comparison_data.append({
                            'Station': station_id,
                            'Dataset': dataset_name,
                            'RMSE': stats.get('rmse', np.nan),
                            'R²': stats.get('r2', np.nan),
                            'Correlation': stats.get('correlation', np.nan),
                            'Bias': abs(stats.get('bias', np.nan))
                        })
            
            if not comparison_data:
                return go.Figure()
            
            df = pd.DataFrame(comparison_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('RMSE (Lower is Better)', 'R² (Higher is Better)', 
                               'Correlation (Higher is Better)', 'Absolute Bias (Lower is Better)')
            )
            
            # RMSE plot
            for dataset in df['Dataset'].unique():
                dataset_data = df[df['Dataset'] == dataset]
                fig.add_trace(
                    go.Bar(x=dataset_data['Station'], y=dataset_data['RMSE'], 
                           name=f'{dataset} - RMSE', legendgroup=dataset),
                    row=1, col=1
                )
            
            # R² plot
            for dataset in df['Dataset'].unique():
                dataset_data = df[df['Dataset'] == dataset]
                fig.add_trace(
                    go.Bar(x=dataset_data['Station'], y=dataset_data['R²'], 
                           name=f'{dataset} - R²', legendgroup=dataset, showlegend=False),
                    row=1, col=2
                )
            
            # Correlation plot
            for dataset in df['Dataset'].unique():
                dataset_data = df[df['Dataset'] == dataset]
                fig.add_trace(
                    go.Bar(x=dataset_data['Station'], y=dataset_data['Correlation'], 
                           name=f'{dataset} - Corr', legendgroup=dataset, showlegend=False),
                    row=2, col=1
                )
            
            # Bias plot
            for dataset in df['Dataset'].unique():
                dataset_data = df[df['Dataset'] == dataset]
                fig.add_trace(
                    go.Bar(x=dataset_data['Station'], y=dataset_data['Bias'], 
                           name=f'{dataset} - Bias', legendgroup=dataset, showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=f'Dataset Performance Comparison - {variable.upper()}',
                height=700,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating comparison overview: {str(e)}")
            return go.Figure()
    
    def create_station_map(self, stations_df: pd.DataFrame, analysis_results: Dict = None) -> go.Figure:
        """Create a map showing station locations with optional performance coloring
        
        Args:
            stations_df: DataFrame with station information
            analysis_results: Optional dictionary with analysis results for coloring
            
        Returns:
            Plotly figure with map
        """
        try:
            fig = go.Figure()
            
            # Base map with stations
            if analysis_results:
                # Color by best performing dataset (highest R²)
                colors = []
                hover_text = []
                
                for _, station in stations_df.iterrows():
                    station_id = station['id']
                    if station_id in analysis_results:
                        # Find best dataset
                        best_r2 = -1
                        best_dataset = 'None'
                        
                        for dataset_name, results in analysis_results[station_id].items():
                            if 'stats' in results:
                                r2 = results['stats'].get('r2', -1)
                                if r2 > best_r2:
                                    best_r2 = r2
                                    best_dataset = dataset_name
                        
                        colors.append(best_r2)
                        hover_text.append(f"Station: {station_id}<br>Best Dataset: {best_dataset}<br>R²: {best_r2:.3f}")
                    else:
                        colors.append(0)
                        hover_text.append(f"Station: {station_id}<br>No analysis results")
                
                fig.add_trace(go.Scattermapbox(
                    lat=stations_df['latitude'],
                    lon=stations_df['longitude'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors,
                        colorscale='RdYlBu',
                        cmin=0,
                        cmax=1,
                        colorbar=dict(title="Best R²")
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name='Stations'
                ))
            else:
                fig.add_trace(go.Scattermapbox(
                    lat=stations_df['latitude'],
                    lon=stations_df['longitude'],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    text=stations_df['id'],
                    hovertemplate='Station: %{text}<extra></extra>',
                    name='Stations'
                ))
            
            # Calculate map center
            center_lat = stations_df['latitude'].mean()
            center_lon = stations_df['longitude'].mean()
            
            fig.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=8
                ),
                title='Station Locations',
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating station map: {str(e)}")
            return go.Figure()
