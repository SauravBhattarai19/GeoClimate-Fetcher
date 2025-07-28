"""
Hydrology Analysis Module
Provides comprehensive precipitation analysis tools for hydrology education and research
Includes return period analysis, frequency analysis, and hydrological statistics
"""

import ee
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import streamlit as st


class HydrologyAnalyzer:
    """
    Comprehensive hydrology analysis tool for precipitation data
    Designed for undergraduate and graduate hydrology education
    """
    
    def __init__(self, geometry: ee.Geometry):
        """
        Initialize the hydrology analyzer
        
        Args:
            geometry: ee.Geometry object defining the area of interest
        """
        self.geometry = geometry
        self.precipitation_data = None
        self.analysis_results = {}
        
    def fetch_precipitation_data(self, dataset_info: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch precipitation data from Google Earth Engine
        Uses chunking to handle large date ranges and avoid 5000 element limit
        
        Args:
            dataset_info: Dictionary containing dataset information
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with precipitation time series
        """
        from geoclimate_fetcher.core.fetchers.collection import ImageCollectionFetcher
        from datetime import datetime
        
        # Initialize fetcher
        fetcher = ImageCollectionFetcher(
            ee_id=dataset_info['ee_id'],
            bands=[dataset_info['precipitation_band']],
            geometry=self.geometry
        )
        
        # Filter dates
        fetcher.filter_dates(start_date, end_date)
        
        # Calculate the time span to determine if we need chunking
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        years_span = (end_dt - start_dt).days / 365.25
        
        print(f"Fetching {years_span:.1f} years of precipitation data...")
        
        # Use chunking for large datasets to avoid 5000 element limit
        if years_span > 5:  # More than 5 years, use chunking
            print("Using chunked approach to handle large dataset...")
            # Use smaller chunks for very large datasets
            if years_span > 20:
                chunk_months = 6  # 6 month chunks for very large datasets
            elif years_span > 10:
                chunk_months = 12  # 1 year chunks for large datasets  
            else:
                chunk_months = 24  # 2 year chunks for medium datasets
                
            df = fetcher.get_time_series_average_chunked(chunk_months=chunk_months)
        else:
            print("Using standard approach for smaller dataset...")
            df = fetcher.get_time_series_average()
        
        if not df.empty:
            # Rename precipitation column for consistency
            precip_col = dataset_info['precipitation_band']
            df = df.rename(columns={precip_col: 'precipitation'})
            
            # Convert units if necessary
            if dataset_info.get('unit') == 'm':
                df['precipitation'] = df['precipitation'] * 1000  # Convert m to mm
            elif dataset_info.get('unit') == 'kg/m^2/s':
                # Convert kg/m^2/s to mm/day (1 kg/m^2/s = 86400 mm/day)
                df['precipitation'] = df['precipitation'] * 86400
            elif dataset_info.get('unit') == '0.1 mm':
                df['precipitation'] = df['precipitation'] * 0.1  # Convert 0.1 mm to mm
            elif dataset_info.get('unit') == 'mm/hr':
                # For monthly data, assume it's already in mm for the month
                pass
            elif dataset_info.get('unit') == 'mm/day':
                # Already in correct units
                pass
            
            # Handle any remaining unit issues - check if values are too small
            mean_precip = df['precipitation'].mean()
            if mean_precip < 0.1:  # Suspiciously low values
                print(f"Warning: Low precipitation values detected (mean: {mean_precip:.6f}). Checking units...")
                # Might need scaling
                if mean_precip < 0.001:
                    df['precipitation'] = df['precipitation'] * 1000  # Try scaling by 1000
                    print("Applied scaling factor of 1000 to precipitation values")
            
            # Ensure non-negative values
            df['precipitation'] = df['precipitation'].clip(lower=0)
            
            # Remove any NaN or infinite values
            df = df.dropna()
            df = df[np.isfinite(df['precipitation'])]
            
            print(f"Successfully fetched {len(df)} precipitation records")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Precipitation stats: mean={df['precipitation'].mean():.2f}mm, max={df['precipitation'].max():.2f}mm")
            
        else:
            print("Warning: No precipitation data returned from Earth Engine")
            
        self.precipitation_data = df
        return df
    
    def calculate_annual_maxima(self) -> pd.DataFrame:
        """
        Calculate annual maximum precipitation values
        Essential for extreme value analysis and return period calculations
        
        Returns:
            DataFrame with annual maximum values
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return pd.DataFrame()
        
        df = self.precipitation_data.copy()
        df['year'] = df['date'].dt.year
        
        # Calculate annual maxima
        annual_maxima = df.groupby('year')['precipitation'].max().reset_index()
        annual_maxima.columns = ['year', 'annual_max_precipitation']
        
        return annual_maxima
    
    def calculate_return_periods(self, annual_maxima: pd.DataFrame) -> Dict:
        """
        Calculate return periods using multiple statistical distributions
        
        Args:
            annual_maxima: DataFrame with annual maximum precipitation values
            
        Returns:
            Dictionary containing return period analysis results
        """
        if annual_maxima.empty:
            return {}
        
        values = annual_maxima['annual_max_precipitation'].values
        
        # Define return periods to calculate
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
        
        results = {
            'return_periods': return_periods,
            'distributions': {}
        }
        
        # Fit multiple distributions
        distributions = {
            'Gumbel': stats.gumbel_r,
            'GEV': stats.genextreme,
            'Log-Normal': stats.lognorm,
            'Gamma': stats.gamma,
            'Weibull': stats.weibull_min
        }
        
        for dist_name, distribution in distributions.items():
            try:
                # Fit distribution
                params = distribution.fit(values)
                
                # Calculate return period values
                probabilities = 1 - (1/np.array(return_periods))
                return_values = distribution.ppf(probabilities, *params)
                
                # Calculate goodness of fit (Kolmogorov-Smirnov test)
                ks_stat, ks_pvalue = stats.kstest(values, lambda x: distribution.cdf(x, *params))
                
                results['distributions'][dist_name] = {
                    'params': params,
                    'return_values': return_values,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'aic': self._calculate_aic(values, distribution, params)
                }
                
            except Exception as e:
                print(f"Error fitting {dist_name}: {str(e)}")
                continue
        
        return results
    
    def _calculate_aic(self, data: np.ndarray, distribution, params) -> float:
        """Calculate Akaike Information Criterion"""
        try:
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)  # number of parameters
            n = len(data)    # sample size
            aic = 2*k - 2*log_likelihood
            return aic
        except:
            return np.inf
    
    def calculate_intensity_duration_frequency(self) -> Dict:
        """
        Calculate Intensity-Duration-Frequency (IDF) curves
        Note: This is a simplified version using available data
        """
        if self.precipitation_data is None:
            return {}
        
        df = self.precipitation_data.copy()
        
        # For demonstration, we'll calculate different accumulation periods
        # In practice, you'd need sub-daily data for proper IDF curves
        
        # Calculate rolling sums for different durations (in days)
        durations = [1, 3, 7, 14, 30]  # days
        
        idf_data = {}
        
        for duration in durations:
            # Calculate rolling sum
            df[f'precip_{duration}d'] = df['precipitation'].rolling(window=duration, min_periods=1).sum()
            
            # Get annual maxima for this duration
            df['year'] = df['date'].dt.year
            annual_max = df.groupby('year')[f'precip_{duration}d'].max()
            
            # Fit Gumbel distribution (commonly used for IDF)
            if len(annual_max) > 2:
                try:
                    params = stats.gumbel_r.fit(annual_max.values)
                    
                    # Calculate values for different return periods
                    return_periods = [2, 5, 10, 25, 50, 100]
                    probabilities = 1 - (1/np.array(return_periods))
                    intensities = stats.gumbel_r.ppf(probabilities, *params) / duration  # mm/day
                    
                    idf_data[duration] = {
                        'return_periods': return_periods,
                        'intensities': intensities,
                        'annual_maxima': annual_max.values
                    }
                except:
                    continue
        
        return idf_data
    
    def calculate_precipitation_statistics(self) -> Dict:
        """
        Calculate comprehensive precipitation statistics
        """
        if self.precipitation_data is None:
            return {}
        
        precip = self.precipitation_data['precipitation']
        
        # Basic statistics
        stats_dict = {
            'count': len(precip),
            'mean': precip.mean(),
            'median': precip.median(),
            'std': precip.std(),
            'min': precip.min(),
            'max': precip.max(),
            'skewness': precip.skew(),
            'kurtosis': precip.kurtosis(),
            'cv': precip.std() / precip.mean() if precip.mean() > 0 else 0
        }
        
        # Percentiles
        percentiles = [5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats_dict[f'p{p}'] = precip.quantile(p/100)
        
        # Wet day statistics (precipitation > 1mm)
        wet_days = precip[precip > 1]
        stats_dict['wet_days_count'] = len(wet_days)
        stats_dict['wet_days_percentage'] = (len(wet_days) / len(precip)) * 100
        stats_dict['wet_days_mean'] = wet_days.mean() if len(wet_days) > 0 else 0
        
        # Extreme events
        stats_dict['days_heavy_rain'] = len(precip[precip > 10])  # >10mm
        stats_dict['days_very_heavy_rain'] = len(precip[precip > 20])  # >20mm
        stats_dict['days_extreme_rain'] = len(precip[precip > 50])  # >50mm
        
        return stats_dict
    
    def calculate_drought_analysis(self) -> Dict:
        """
        Calculate drought-related indices and statistics
        """
        if self.precipitation_data is None:
            return {}
        
        df = self.precipitation_data.copy()
        
        # Standardized Precipitation Index (SPI) - simplified version
        # Calculate for different time scales
        spi_results = {}
        
        for months in [3, 6, 12]:
            # Calculate rolling sum
            rolling_sum = df['precipitation'].rolling(window=months*30, min_periods=1).sum()
            
            # Fit gamma distribution and calculate SPI
            try:
                # Remove zeros for gamma fitting
                non_zero = rolling_sum[rolling_sum > 0]
                if len(non_zero) > 10:
                    shape, loc, scale = stats.gamma.fit(non_zero, floc=0)
                    
                    # Calculate SPI
                    spi_values = []
                    for value in rolling_sum:
                        if value > 0:
                            prob = stats.gamma.cdf(value, shape, loc, scale)
                            spi = stats.norm.ppf(prob)
                            spi_values.append(spi)
                        else:
                            spi_values.append(-2.0)  # Very dry
                    
                    spi_results[f'SPI_{months}'] = np.array(spi_values)
            except:
                continue
        
        # Consecutive dry days analysis
        dry_days = (df['precipitation'] < 1).astype(int)
        
        # Find consecutive dry periods
        dry_periods = []
        current_period = 0
        
        for is_dry in dry_days:
            if is_dry:
                current_period += 1
            else:
                if current_period > 0:
                    dry_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            dry_periods.append(current_period)
        
        drought_stats = {
            'max_consecutive_dry_days': max(dry_periods) if dry_periods else 0,
            'mean_dry_period_length': np.mean(dry_periods) if dry_periods else 0,
            'number_of_dry_periods': len(dry_periods),
            'spi_results': spi_results
        }
        
        return drought_stats
    
    def create_return_period_plot(self, return_period_results: Dict) -> go.Figure:
        """
        Create return period plot comparing different distributions
        """
        fig = go.Figure()
        
        if not return_period_results or 'distributions' not in return_period_results:
            return fig
        
        return_periods = return_period_results['return_periods']
        
        # Plot each distribution
        for dist_name, dist_data in return_period_results['distributions'].items():
            fig.add_trace(go.Scatter(
                x=return_periods,
                y=dist_data['return_values'],
                mode='lines+markers',
                name=f"{dist_name} (AIC: {dist_data['aic']:.1f})",
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Precipitation Return Period Analysis",
            xaxis_title="Return Period (years)",
            yaxis_title="Precipitation (mm)",
            xaxis_type="log",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified'
        )
        
        return fig
    
    def create_idf_curves_plot(self, idf_data: Dict) -> go.Figure:
        """
        Create Intensity-Duration-Frequency curves plot
        """
        fig = go.Figure()
        
        if not idf_data:
            return fig
        
        # Colors for different return periods
        colors = px.colors.qualitative.Set1
        
        return_periods = [2, 5, 10, 25, 50, 100]
        
        for i, rp in enumerate(return_periods):
            durations = []
            intensities = []
            
            for duration, data in idf_data.items():
                if rp in data['return_periods']:
                    idx = data['return_periods'].index(rp)
                    durations.append(duration)
                    intensities.append(data['intensities'][idx])
            
            if durations:
                fig.add_trace(go.Scatter(
                    x=durations,
                    y=intensities,
                    mode='lines+markers',
                    name=f"{rp}-year",
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Intensity-Duration-Frequency (IDF) Curves",
            xaxis_title="Duration (days)",
            yaxis_title="Intensity (mm/day)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_precipitation_time_series_plot(self) -> go.Figure:
        """
        Create comprehensive precipitation time series plot
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "Daily Precipitation Time Series",
                "Monthly Precipitation Totals", 
                "Annual Precipitation Totals"
            ],
            vertical_spacing=0.08
        )
        
        if self.precipitation_data is None:
            return fig
        
        df = self.precipitation_data.copy()
        
        # Daily precipitation
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['precipitation'],
            mode='lines',
            name='Daily Precipitation',
            line=dict(color='blue', width=1)
        ), row=1, col=1)
        
        # Monthly totals
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month')['precipitation'].sum().reset_index()
        monthly['date'] = monthly['year_month'].dt.to_timestamp()
        
        fig.add_trace(go.Bar(
            x=monthly['date'],
            y=monthly['precipitation'],
            name='Monthly Total',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # Annual totals
        df['year'] = df['date'].dt.year
        annual = df.groupby('year')['precipitation'].sum().reset_index()
        
        fig.add_trace(go.Bar(
            x=annual['year'],
            y=annual['precipitation'],
            name='Annual Total',
            marker_color='darkblue'
        ), row=3, col=1)
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=3, col=1)
        
        fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
        fig.update_yaxes(title_text="Precipitation (mm)", row=2, col=1)
        fig.update_yaxes(title_text="Precipitation (mm)", row=3, col=1)
        
        fig.update_layout(
            height=800,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def create_frequency_analysis_plot(self) -> go.Figure:
        """
        Create precipitation frequency analysis plot
        """
        if self.precipitation_data is None:
            return go.Figure()
        
        precip = self.precipitation_data['precipitation']
        
        # Create histogram and probability density
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Precipitation Histogram",
                "Probability Density Function",
                "Cumulative Distribution Function", 
                "Q-Q Plot (Normal Distribution)"
            ]
        )
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=precip,
            nbinsx=50,
            name='Frequency',
            marker_color='lightblue'
        ), row=1, col=1)
        
        # PDF comparison
        x_range = np.linspace(precip.min(), precip.max(), 100)
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(precip)
        pdf_normal = stats.norm.pdf(x_range, mu, sigma)
        
        # Fit gamma distribution (better for precipitation)
        try:
            shape, loc, scale = stats.gamma.fit(precip, floc=0)
            pdf_gamma = stats.gamma.pdf(x_range, shape, loc, scale)
            
            fig.add_trace(go.Scatter(
                x=x_range, y=pdf_gamma,
                mode='lines', name='Gamma Distribution',
                line=dict(color='red', width=2)
            ), row=1, col=2)
        except:
            pass
        
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf_normal,
            mode='lines', name='Normal Distribution',
            line=dict(color='blue', width=2)
        ), row=1, col=2)
        
        # CDF
        sorted_data = np.sort(precip)
        y_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        
        fig.add_trace(go.Scatter(
            x=sorted_data, y=y_cdf,
            mode='lines', name='Empirical CDF',
            line=dict(color='green', width=2)
        ), row=2, col=1)
        
        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(precip)))
        sample_quantiles = np.sort(precip)
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles, y=sample_quantiles,
            mode='markers', name='Q-Q Plot',
            marker=dict(color='orange', size=4)
        ), row=2, col=2)
        
        # Add perfect fit line for Q-Q plot
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ), row=2, col=2)
        
        fig.update_layout(
            height=600,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_drought_analysis_plot(self, drought_results: Dict) -> go.Figure:
        """
        Create drought analysis visualization
        """
        if not drought_results or self.precipitation_data is None:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                "Standardized Precipitation Index (SPI)",
                "Consecutive Dry Days Analysis"
            ],
            vertical_spacing=0.15
        )
        
        df = self.precipitation_data.copy()
        
        # Plot SPI if available
        if 'spi_results' in drought_results:
            for spi_name, spi_values in drought_results['spi_results'].items():
                fig.add_trace(go.Scatter(
                    x=df['date'][:len(spi_values)],
                    y=spi_values,
                    mode='lines',
                    name=spi_name,
                    line=dict(width=2)
                ), row=1, col=1)
            
            # Add drought thresholds
            fig.add_hline(y=-1.0, line_dash="dash", line_color="orange", 
                         annotation_text="Moderate Drought", row=1, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="red", 
                         annotation_text="Severe Drought", row=1, col=1)
        
        # Consecutive dry days
        dry_days = (df['precipitation'] < 1).astype(int)
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=dry_days,
            mode='lines',
            name='Dry Days (< 1mm)',
            line=dict(color='brown', width=1)
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            template="plotly_white"
        )
        
        return fig
