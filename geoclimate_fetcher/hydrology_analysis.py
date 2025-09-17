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
        Fetch precipitation data from Google Earth Engine with server-side daily aggregation
        Handles high-resolution data (like IMERG 30-minute) by aggregating to daily on server

        Args:
            dataset_info: Dictionary containing dataset information
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with daily precipitation time series
        """
        from datetime import datetime
        import ee
        import pandas as pd

        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            years_span = (end_dt - start_dt).days / 365.25
            total_days = (end_dt - start_dt).days

            print(f"Fetching {years_span:.1f} years ({total_days} days) of precipitation data...")
            print(f"Dataset: {dataset_info['name']} ({dataset_info.get('ee_id')})")

            # Load the image collection
            print(f"Loading collection: {dataset_info['ee_id']}")
            print(f"Looking for band: {dataset_info['precipitation_band']}")

            # First, let's check what bands are available
            try:
                sample_collection = ee.ImageCollection(dataset_info['ee_id']) \
                                     .filterDate(start_date, end_date) \
                                     .filterBounds(self.geometry) \
                                     .limit(1)

                # Get band names from first image
                first_image = sample_collection.first()
                band_names = first_image.bandNames().getInfo()
                print(f"Available bands in dataset: {band_names}")

                # Check if the expected band exists
                if dataset_info['precipitation_band'] not in band_names:
                    print(f"Warning: Band '{dataset_info['precipitation_band']}' not found!")
                    print("Trying common precipitation band names...")

                    # Try common precipitation band names
                    common_precip_bands = ['precipitation', 'precipitationCal', 'total_precipitation', 'tp', 'precip']
                    found_band = None

                    for band in common_precip_bands:
                        if band in band_names:
                            found_band = band
                            print(f"Found precipitation band: {band}")
                            break

                    if found_band:
                        dataset_info['precipitation_band'] = found_band
                    else:
                        print(f"No common precipitation bands found. Available: {band_names}")
                        return pd.DataFrame()

            except Exception as e:
                print(f"Could not check band names: {e}")

            collection = ee.ImageCollection(dataset_info['ee_id']) \
                          .select(dataset_info['precipitation_band']) \
                          .filterDate(start_date, end_date) \
                          .filterBounds(self.geometry)

            # Check temporal resolution and apply appropriate aggregation
            ee_id = dataset_info['ee_id']
            is_high_resolution = any(x in ee_id.upper() for x in ['IMERG', 'GPM']) or 'mm/hr' in dataset_info.get('unit', '')

            if is_high_resolution:
                print("Detected high-resolution dataset - applying server-side daily aggregation...")

                # For IMERG/GPM: aggregate sub-daily data to daily totals on server
                def aggregate_daily(date):
                    """Aggregate all images for a single day"""
                    date = ee.Date(date)
                    start_of_day = date
                    end_of_day = date.advance(1, 'day')

                    daily_images = collection.filterDate(start_of_day, end_of_day)

                    # Convert mm/hr to mm/day by summing all intervals in the day
                    if 'mm/hr' in dataset_info.get('unit', ''):
                        # For half-hourly data: multiply by 0.5 hours then sum
                        daily_total = daily_images.map(lambda img: img.multiply(0.5)).sum()
                    else:
                        # For other units, just sum
                        daily_total = daily_images.sum()

                    return daily_total.set('system:time_start', start_of_day.millis())

                # Generate date list and aggregate
                date_list = ee.List.sequence(0, total_days - 1).map(
                    lambda days: ee.Date(start_date).advance(days, 'day')
                )

                daily_collection = ee.ImageCollection.fromImages(
                    date_list.map(aggregate_daily)
                )

                print(f"Aggregated to {total_days} daily images")

            else:
                print("Using daily dataset directly...")
                daily_collection = collection

            # Use reduceRegion approach for area-averaged time series (more efficient)
            print("Extracting area-averaged time series...")

            # Simplify geometry if it's too complex
            simplified_geometry = self._simplify_geometry_if_needed(self.geometry)

            # Get area-averaged precipitation for each day
            def get_daily_mean(image):
                """Get area-averaged precipitation for a single day"""
                # Reduce region to get mean precipitation over the area
                reduction = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=simplified_geometry,
                    scale=5000,  # 5km resolution for efficiency
                    maxPixels=1e9
                )

                # Get the precipitation value
                precip_value = reduction.get(dataset_info['precipitation_band'])

                # Return feature with date and precipitation
                return ee.Feature(None, {
                    'date': image.get('system:time_start'),
                    'precipitation': precip_value
                })

            # Apply to all images and get results
            if total_days > 1000:  # Use chunking for very large requests
                chunk_days = 365
                print(f"Using chunked approach with {chunk_days}-day chunks...")

                all_data = []
                for chunk_start_days in range(0, total_days, chunk_days):
                    chunk_end_days = min(chunk_start_days + chunk_days, total_days)

                    chunk_start_date = start_dt + pd.Timedelta(days=chunk_start_days)
                    chunk_end_date = start_dt + pd.Timedelta(days=chunk_end_days)

                    chunk_start_str = chunk_start_date.strftime('%Y-%m-%d')
                    chunk_end_str = chunk_end_date.strftime('%Y-%m-%d')

                    print(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")

                    chunk_collection = daily_collection.filterDate(chunk_start_str, chunk_end_str)

                    # Convert to feature collection and get data
                    features = chunk_collection.map(get_daily_mean)
                    data = features.getInfo()

                    if data and data['features']:
                        chunk_df = self._process_feature_collection_data(data, dataset_info)
                        if not chunk_df.empty:
                            all_data.append(chunk_df)

                # Combine all chunks
                if all_data:
                    df = pd.concat(all_data, ignore_index=True).sort_values('date').reset_index(drop=True)
                else:
                    df = pd.DataFrame()

            else:
                print("Processing all data at once...")
                # Process all at once for smaller datasets
                features = daily_collection.map(get_daily_mean)
                data = features.getInfo()
                df = self._process_feature_collection_data(data, dataset_info)

            if not df.empty:
                print(f"Successfully fetched {len(df)} daily precipitation records")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"Precipitation stats: mean={df['precipitation'].mean():.2f}mm, max={df['precipitation'].max():.2f}mm")
            else:
                print("Warning: No precipitation data returned from Earth Engine")

            self.precipitation_data = df
            return df

        except Exception as e:
            print(f"Error fetching precipitation data: {str(e)}")
            return pd.DataFrame()

    def _process_time_series_data(self, time_series: list, dataset_info: Dict) -> pd.DataFrame:
        """Process raw time series data from Earth Engine into a clean DataFrame"""
        if not time_series or len(time_series) <= 1:
            return pd.DataFrame()

        # Convert to DataFrame
        headers = time_series[0]
        data = time_series[1:]

        df = pd.DataFrame(data, columns=headers)

        # Convert time to datetime
        df['date'] = pd.to_datetime(df['time'], unit='ms')

        # Get precipitation column
        precip_col = dataset_info['precipitation_band']
        if precip_col not in df.columns:
            print(f"Warning: Precipitation band '{precip_col}' not found in data")
            return pd.DataFrame()

        # Rename and clean
        df = df.rename(columns={precip_col: 'precipitation'})
        df = df[['date', 'precipitation']].dropna()

        # Convert units if necessary
        unit = dataset_info.get('unit', '')
        if unit == 'm':
            df['precipitation'] = df['precipitation'] * 1000  # Convert m to mm
        elif unit == 'kg/m^2/s':
            df['precipitation'] = df['precipitation'] * 86400  # Convert to mm/day
        elif unit == '0.1 mm':
            df['precipitation'] = df['precipitation'] * 0.1  # Convert to mm
        elif unit == 'mm/hr':
            # Already converted during aggregation
            pass

        # Ensure non-negative values
        df['precipitation'] = df['precipitation'].clip(lower=0)

        # Remove any NaN or infinite values
        df = df.dropna()
        df = df[np.isfinite(df['precipitation'])]

        return df

    def _simplify_geometry_if_needed(self, geometry):
        """Simplify geometry if it's too complex for efficient processing"""
        import ee

        try:
            # Check if geometry is too complex by getting coordinate count
            coords_info = geometry.getInfo()

            # Count total coordinates
            total_coords = 0
            if coords_info['type'] == 'Polygon':
                for ring in coords_info['coordinates']:
                    total_coords += len(ring)
            elif coords_info['type'] == 'MultiPolygon':
                for polygon in coords_info['coordinates']:
                    for ring in polygon:
                        total_coords += len(ring)
            else:
                # For other geometry types, use as-is
                return geometry

            print(f"Geometry has {total_coords} coordinate points")

            # If too many coordinates, simplify
            if total_coords > 1000:  # Threshold for simplification
                print(f"Simplifying complex geometry ({total_coords} points)...")

                # Method 1: Buffer and unbuffer to smooth
                simplified = geometry.buffer(100).buffer(-100).simplify(500)

                # If still too complex, use convex hull
                try:
                    simplified_coords = simplified.getInfo()
                    simplified_count = len(simplified_coords['coordinates'][0]) if simplified_coords['type'] == 'Polygon' else 0

                    if simplified_count > 500:
                        print("Using convex hull for maximum simplification...")
                        simplified = geometry.convexHull()

                except:
                    print("Using convex hull as fallback...")
                    simplified = geometry.convexHull()

                return simplified
            else:
                return geometry

        except Exception as e:
            print(f"Warning: Could not analyze geometry complexity ({e}). Using convex hull...")
            # Fallback to convex hull if geometry analysis fails
            return geometry.convexHull()

    def _process_feature_collection_data(self, data: dict, dataset_info: Dict) -> pd.DataFrame:
        """Process feature collection data into a clean DataFrame"""
        if not data or 'features' not in data or not data['features']:
            return pd.DataFrame()

        # Extract data from features
        records = []
        for feature in data['features']:
            if 'properties' in feature:
                props = feature['properties']
                if 'date' in props and 'precipitation' in props:
                    # Skip null values
                    if props['precipitation'] is not None:
                        records.append({
                            'date': pd.to_datetime(props['date'], unit='ms'),
                            'precipitation': props['precipitation']
                        })

        if not records:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(records)

        # Convert units if necessary
        unit = dataset_info.get('unit', '')
        if unit == 'm':
            df['precipitation'] = df['precipitation'] * 1000  # Convert m to mm
        elif unit == 'kg/m^2/s':
            df['precipitation'] = df['precipitation'] * 86400  # Convert to mm/day
        elif unit == '0.1 mm':
            df['precipitation'] = df['precipitation'] * 0.1  # Convert to mm

        # Ensure non-negative values
        df['precipitation'] = df['precipitation'].clip(lower=0)

        # Remove any NaN or infinite values
        df = df.dropna()
        df = df[np.isfinite(df['precipitation'])]

        return df.sort_values('date').reset_index(drop=True)

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
