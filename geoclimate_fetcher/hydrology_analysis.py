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
        self.dataset_info = None  # Store dataset info for IDF calculations
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
            self.dataset_info = dataset_info  # Store for later use in IDF calculations
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

    def _validate_data_for_yearly_analysis(self) -> tuple[bool, str]:
        """
        Validate precipitation data for yearly analysis

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return False, "No precipitation data available"

        df = self.precipitation_data

        # Check required columns
        if 'date' not in df.columns:
            return False, "Date column missing from precipitation data"

        if 'precipitation' not in df.columns:
            return False, "Precipitation column missing from precipitation data"

        # Check data types and convertibility
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                pd.to_datetime(df['date'])  # Test conversion
        except Exception as e:
            return False, f"Date column cannot be converted to datetime: {str(e)}"

        # Check for NaN values
        nan_dates = df['date'].isna().sum()
        nan_precip = df['precipitation'].isna().sum()

        if nan_dates > 0:
            return False, f"Found {nan_dates} NaN values in date column"

        # Check temporal coverage
        try:
            temp_df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(temp_df['date']):
                temp_df['date'] = pd.to_datetime(temp_df['date'])

            temp_df['year'] = temp_df['date'].dt.year
            unique_years = len(temp_df['year'].unique())

            if unique_years < 1:
                return False, "No complete years of data available"

        except Exception as e:
            return False, f"Error processing temporal data: {str(e)}"

        return True, "Data validation passed"

    def calculate_yearly_statistics(self) -> Dict:
        """
        Calculate comprehensive yearly statistics including max, mean, median, and trends

        Returns:
            Dictionary with yearly statistics and trend analysis
        """
        try:
            # Validate data first
            is_valid, validation_message = self._validate_data_for_yearly_analysis()

            if not is_valid:
                return {}

            df = self.precipitation_data.copy()

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            # Filter out NaN precipitation values
            df = df.dropna(subset=['precipitation'])

            # Check for negative precipitation (data quality issue)
            negative_precip = (df['precipitation'] < 0).sum()
            if negative_precip > 0:
                df.loc[df['precipitation'] < 0, 'precipitation'] = 0

            df['year'] = df['date'].dt.year

            # Calculate yearly statistics
            yearly_stats = df.groupby('year')['precipitation'].agg([
                ('max', 'max'),
                ('mean', 'mean'),
                ('median', 'median'),
                ('total', 'sum'),
                ('wet_days', lambda x: (x > 1.0).sum()),
                ('dry_days', lambda x: (x <= 1.0).sum())
            ]).reset_index()

            # Calculate trends for each metric
            years = yearly_stats['year'].values
            trends = {}

            for metric in ['max', 'mean', 'median', 'total']:
                values = yearly_stats[metric].values

                if len(values) >= 3:  # Need at least 3 years for trend
                    try:
                        # Simple linear trend using least squares
                        n = len(years)
                        numerator = n * np.sum(years * values) - np.sum(years) * np.sum(values)
                        denominator = n * np.sum(years**2) - np.sum(years)**2

                        if abs(denominator) < 1e-10:  # Prevent division by zero
                            slope = 0
                            trend = 'stable'
                        else:
                            slope = numerator / denominator

                            # Classify trend
                            if abs(slope) < 0.01:  # Threshold for "no trend"
                                trend = 'stable'
                            elif slope > 0:
                                trend = 'increasing'
                            else:
                                trend = 'decreasing'

                        trends[metric] = {
                            'slope': slope,
                            'trend': trend,
                            'direction': '📈' if slope > 0 else '📉' if slope < 0 else '➡️'
                        }

                    except Exception as e:
                        trends[metric] = {
                            'slope': 0,
                            'trend': 'calculation_error',
                            'direction': '❓'
                        }
                else:
                    trends[metric] = {
                        'slope': 0,
                        'trend': 'insufficient_data',
                        'direction': '❓'
                    }

            # Calculate summary statistics
            try:
                summary = {
                    'years_analyzed': len(yearly_stats),
                    'start_year': yearly_stats['year'].min(),
                    'end_year': yearly_stats['year'].max(),
                    'max_year': yearly_stats.loc[yearly_stats['max'].idxmax(), 'year'],
                    'max_value': yearly_stats['max'].max(),
                    'min_year': yearly_stats.loc[yearly_stats['max'].idxmin(), 'year'],
                    'min_value': yearly_stats['max'].min(),
                    'mean_annual_total': yearly_stats['total'].mean(),
                    'wettest_year': yearly_stats.loc[yearly_stats['total'].idxmax(), 'year'],
                    'driest_year': yearly_stats.loc[yearly_stats['total'].idxmin(), 'year']
                }
            except Exception as e:
                summary = {
                    'years_analyzed': len(yearly_stats),
                    'start_year': yearly_stats['year'].min() if len(yearly_stats) > 0 else 0,
                    'end_year': yearly_stats['year'].max() if len(yearly_stats) > 0 else 0,
                    'max_year': 0,
                    'max_value': 0,
                    'min_year': 0,
                    'min_value': 0,
                    'mean_annual_total': 0,
                    'wettest_year': 0,
                    'driest_year': 0
                }

            result = {
                'yearly_data': yearly_stats,
                'trends': trends,
                'summary': summary
            }
            return result

        except Exception as e:
            return {}
    
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
    
    def calculate_intensity_duration_frequency(self, start_date: str = None, end_date: str = None,
                                              return_periods: list = None) -> Dict:
        """
        Enhanced IDF calculation with intelligent temporal resolution handling

        Automatically detects dataset temporal resolution:
        - Sub-daily data (IMERG 30-min): Uses server-side GEE aggregation for efficiency
        - Daily data (CHIRPS, ERA5-Land): Uses local rolling window approach

        Args:
            start_date: Start date (if None, uses self.precipitation_data dates)
            end_date: End date (if None, uses self.precipitation_data dates)
            return_periods: List of return periods in years (default: [2, 5, 10, 25, 50, 100])

        Returns:
            Dictionary with IDF results for each duration
        """
        if return_periods is None:
            return_periods = [2, 5, 10, 25, 50, 100]

        # Check if we have dataset info
        if self.dataset_info is None:
            # Fall back to basic daily method using existing data
            print("No dataset info available, using basic daily approach...")
            return self._calculate_idf_daily(return_periods)

        # Determine temporal resolution
        temporal_resolution = self.dataset_info.get('temporal_resolution', 'daily')

        # Check if sub-daily aggregation info is available
        if temporal_resolution == 'half-hourly' and 'sub_daily_aggregation' in self.dataset_info:
            print(f"Detected sub-daily data ({temporal_resolution})")
            print("Using server-side GEE aggregation for optimal performance...")

            # Use server-side approach for sub-daily data
            if start_date is None or end_date is None:
                # Try to get from existing data
                if self.precipitation_data is not None and not self.precipitation_data.empty:
                    start_date = self.precipitation_data['date'].min().strftime('%Y-%m-%d')
                    end_date = self.precipitation_data['date'].max().strftime('%Y-%m-%d')
                else:
                    print("Error: No date range specified and no precipitation data available")
                    return {}

            return self._calculate_idf_server_side(start_date, end_date, return_periods)

        else:
            # Use daily approach for daily datasets
            print(f"Using daily rolling window approach (temporal resolution: {temporal_resolution})")
            return self._calculate_idf_daily(return_periods)

    def _calculate_idf_daily(self, return_periods: list) -> Dict:
        """
        Calculate IDF curves from daily data using rolling windows
        (Original method for daily datasets)
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return {}

        df = self.precipitation_data.copy()

        # Calculate rolling sums for different durations (in days)
        durations_days = [1, 2, 3, 7, 14, 30]

        idf_data = {}

        for duration in durations_days:
            # Calculate rolling sum
            df[f'precip_{duration}d'] = df['precipitation'].rolling(window=duration, min_periods=1).sum()

            # Get annual maxima for this duration
            df['year'] = df['date'].dt.year
            annual_max = df.groupby('year')[f'precip_{duration}d'].max()

            # Fit Gumbel distribution (commonly used for IDF)
            if len(annual_max) > 2:
                try:
                    params = stats.gumbel_r.fit(annual_max.values)

                    # Calculate depths for different return periods
                    probabilities = 1 - (1/np.array(return_periods))
                    depths = stats.gumbel_r.ppf(probabilities, *params)
                    intensities = depths / duration  # mm/day

                    idf_data[duration] = {
                        'duration_days': duration,
                        'duration_hours': duration * 24,
                        'return_periods': return_periods,
                        'return_depths': depths.tolist(),
                        'return_intensities_mm_per_day': intensities.tolist(),
                        'return_intensities_mm_per_hour': (intensities / 24).tolist(),
                        'annual_maxima': annual_max.values.tolist(),
                        'distribution': 'Gumbel',
                        'distribution_params': params
                    }
                except Exception as e:
                    print(f"Error fitting distribution for {duration}-day duration: {e}")
                    continue

        return idf_data

    def _calculate_idf_server_side(self, start_date: str, end_date: str, return_periods: list) -> Dict:
        """
        Calculate IDF curves using server-side GEE aggregation for sub-daily data

        This method minimizes data transfer by:
        1. Aggregating to various durations on GEE server
        2. Extracting annual maxima on GEE server
        3. Downloading only annual maxima values (~10-20 values per duration)

        For IMERG 30-min data: Downloads ~60 values instead of 175,000!
        """
        from datetime import datetime

        dataset_info = self.dataset_info

        # Get sub-daily parameters
        sub_daily_info = dataset_info.get('sub_daily_aggregation', {})
        time_interval_hrs = sub_daily_info.get('time_interval_hours', 0.5)
        images_per_day = sub_daily_info.get('images_per_day', 48)

        print(f"Sub-daily resolution: {time_interval_hrs} hours ({images_per_day} images/day)")

        # Define durations based on temporal resolution
        if time_interval_hrs == 0.5:  # 30-minute data (IMERG)
            durations_hours = [0.5, 1, 2, 3, 6, 12, 24]
        elif time_interval_hrs == 1.0:  # Hourly data
            durations_hours = [1, 2, 3, 6, 12, 24]
        else:
            durations_hours = [1, 3, 6, 12, 24]

        print(f"Calculating IDF for durations: {durations_hours} hours")

        # Load the image collection (don't download yet!)
        collection = ee.ImageCollection(dataset_info['ee_id']) \
                      .select(dataset_info['precipitation_band']) \
                      .filterDate(start_date, end_date) \
                      .filterBounds(self.geometry)

        # Simplify geometry if needed
        simplified_geometry = self._simplify_geometry_if_needed(self.geometry)

        # Get year range
        start_year = datetime.strptime(start_date, '%Y-%m-%d').year
        end_year = datetime.strptime(end_date, '%Y-%m-%d').year
        years = list(range(start_year, end_year + 1))

        idf_data = {}

        # Process each duration
        for duration_hrs in durations_hours:
            print(f"  Processing {duration_hrs}-hour duration...")

            try:
                # Calculate annual maxima for this duration on server
                annual_maxima = self._get_annual_maxima_for_duration_server_side(
                    collection, simplified_geometry, dataset_info,
                    duration_hrs, time_interval_hrs, start_date, end_date, years
                )

                if len(annual_maxima) < 3:
                    print(f"    Insufficient data for {duration_hrs}-hour duration (need at least 3 years)")
                    continue

                # Fit distribution locally (fast! only ~10-20 values)
                params = stats.gumbel_r.fit(annual_maxima)

                # Calculate return period values
                probabilities = 1 - (1/np.array(return_periods))
                return_depths = stats.gumbel_r.ppf(probabilities, *params)
                return_intensities = return_depths / duration_hrs  # mm/hr

                idf_data[duration_hrs] = {
                    'duration_hours': duration_hrs,
                    'return_periods': return_periods,
                    'return_depths': return_depths.tolist(),  # mm
                    'return_intensities': return_intensities.tolist(),  # mm/hr
                    'annual_maxima': annual_maxima,
                    'distribution': 'Gumbel',
                    'distribution_params': params,
                    'num_years': len(annual_maxima)
                }

                print(f"    ✓ Completed {duration_hrs}-hour duration ({len(annual_maxima)} years)")

            except Exception as e:
                print(f"    ✗ Error processing {duration_hrs}-hour duration: {e}")
                continue

        return idf_data

    def _get_annual_maxima_for_duration_server_side(self, collection, geometry, dataset_info,
                                                     duration_hrs, time_interval_hrs,
                                                     start_date, end_date, years):
        """
        Extract annual maxima for a specific duration using server-side processing

        This minimizes data transfer by computing everything on GEE servers
        """
        from datetime import datetime, timedelta

        # Calculate window size (number of images to aggregate)
        window_size = int(duration_hrs / time_interval_hrs)

        print(f"      Window size: {window_size} images ({duration_hrs} hours)")

        # For each year, find the maximum aggregated value
        annual_maxima = []

        for year in years:
            try:
                # Define year boundaries
                year_start = f"{year}-01-01"
                year_end = f"{year}-12-31"

                # Filter collection to this year
                year_collection = collection.filterDate(year_start, year_end)

                # Get count to check if data exists
                count = year_collection.size().getInfo()
                if count == 0:
                    print(f"        No data for year {year}, skipping")
                    continue

                # Create rolling aggregates for this duration
                # We'll use a simpler approach: aggregate to duration intervals and find max

                if duration_hrs >= 24:
                    # For daily or longer durations, aggregate to daily first
                    daily_collection = self._aggregate_to_daily_server_side(
                        year_collection, year_start, year_end, dataset_info
                    )

                    # Then apply rolling window if duration > 1 day
                    duration_days = int(duration_hrs / 24)
                    if duration_days == 1:
                        max_image = daily_collection.max()
                    else:
                        # For multi-day durations, we'll use a simplified approach
                        # Calculate sum over the duration
                        max_image = daily_collection.max()  # Simplified for now
                else:
                    # For sub-daily durations, aggregate directly
                    max_image = self._get_max_for_subdaily_duration(
                        year_collection, window_size, time_interval_hrs, dataset_info
                    )

                # Extract the maximum value over the geometry
                reduction = max_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=5000,
                    maxPixels=1e9
                )

                max_value = reduction.get(dataset_info['precipitation_band']).getInfo()

                if max_value is not None and max_value > 0:
                    annual_maxima.append(max_value)
                    print(f"        Year {year}: {max_value:.2f} mm")
                else:
                    print(f"        Year {year}: No valid data")

            except Exception as e:
                print(f"        Error processing year {year}: {e}")
                continue

        return annual_maxima

    def _aggregate_to_daily_server_side(self, collection, start_date, end_date, dataset_info):
        """Aggregate sub-daily collection to daily on server"""
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        num_days = (end_dt - start_dt).days + 1

        def aggregate_daily(date):
            date = ee.Date(date)
            start_of_day = date
            end_of_day = date.advance(1, 'day')

            daily_images = collection.filterDate(start_of_day, end_of_day)

            # Convert mm/hr to mm/day
            if 'mm/hr' in dataset_info.get('unit', ''):
                time_interval_hrs = dataset_info.get('sub_daily_aggregation', {}).get('time_interval_hours', 0.5)
                daily_total = daily_images.map(lambda img: img.multiply(time_interval_hrs)).sum()
            else:
                daily_total = daily_images.sum()

            return daily_total.set('system:time_start', start_of_day.millis())

        # Generate date list
        date_list = ee.List.sequence(0, num_days - 1).map(
            lambda days: ee.Date(start_date).advance(days, 'day')
        )

        daily_collection = ee.ImageCollection.fromImages(date_list.map(aggregate_daily))
        return daily_collection

    def _get_max_for_subdaily_duration(self, collection, window_size, time_interval_hrs, dataset_info):
        """
        Get maximum aggregated value for sub-daily duration
        Uses a simplified approach suitable for server-side processing
        """
        # For sub-daily durations, we'll use a moving window approach
        # Convert to list and process

        # Simplified approach: Get all images and find maximum sum over window
        # This is a placeholder - full implementation would use proper rolling windows

        # For now, aggregate all and multiply by window size as approximation
        total = collection.sum()

        # Convert rate to total if needed
        if 'mm/hr' in dataset_info.get('unit', ''):
            total = total.multiply(time_interval_hrs * window_size)

        return total
    
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

    def calculate_mann_kendall_trends(self, aggregation='annual') -> Dict:
        """
        Perform Mann-Kendall trend test on precipitation data

        The Mann-Kendall test is a non-parametric test for detecting monotonic trends
        in time series data. It's more robust than linear regression as it doesn't
        assume normality or linearity.

        Args:
            aggregation: 'annual', 'monthly', or 'seasonal'
                - 'annual': Tests trend in annual total precipitation
                - 'monthly': Tests trend in monthly totals (seasonal variant)
                - 'seasonal': Tests seasonal patterns

        Returns:
            Dictionary with trend results including:
            - trend: 'increasing', 'decreasing', or 'no trend'
            - p_value: statistical significance (< 0.05 = significant)
            - tau: Kendall's tau correlation coefficient (-1 to 1)
            - z_score: standardized test statistic
            - slope: Sen's slope estimator (magnitude of trend)
            - significance: interpretation of p-value
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return {'error': 'No precipitation data available'}

        try:
            import pymannkendall as mk
        except ImportError:
            print("Warning: pymannkendall not installed. Run: pip install pymannkendall")
            return {'error': 'pymannkendall library not installed'}

        df = self.precipitation_data.copy()

        results = {}

        if aggregation == 'annual':
            # Annual total precipitation
            annual_series = df.groupby(df['date'].dt.year)['precipitation'].sum()

            if len(annual_series) < 3:
                return {'error': 'Insufficient data (need at least 3 years)'}

            # Perform original Mann-Kendall test
            mk_result = mk.original_test(annual_series.values)

            results = {
                'aggregation': 'annual',
                'series_length': len(annual_series),
                'trend': mk_result.trend,
                'p_value': mk_result.p,
                'tau': mk_result.Tau,
                'z_score': mk_result.z,
                'slope': mk_result.slope,  # Sen's slope (mm/year)
                'slope_unit': 'mm/year',
                'significance': 'significant (α=0.05)' if mk_result.p < 0.05 else 'not significant',
                'interpretation': self._interpret_mann_kendall(mk_result.trend, mk_result.p, mk_result.slope),
                'years': annual_series.index.tolist(),
                'values': annual_series.values.tolist()
            }

        elif aggregation == 'monthly':
            # Monthly totals with seasonal Mann-Kendall
            monthly_series = df.resample('M', on='date')['precipitation'].sum()

            if len(monthly_series) < 12:
                return {'error': 'Insufficient data (need at least 12 months)'}

            # Seasonal Mann-Kendall (accounts for seasonality)
            mk_result = mk.seasonal_test(monthly_series.values, period=12)

            # Convert timestamps to strings for JSON compatibility
            dates_list = [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d)
                         for d in monthly_series.index]

            results = {
                'aggregation': 'monthly',
                'series_length': len(monthly_series),
                'trend': mk_result.trend,
                'p_value': mk_result.p,
                'tau': mk_result.Tau,
                'z_score': mk_result.z,
                'slope': mk_result.slope,  # mm/month
                'slope_unit': 'mm/month',
                'significance': 'significant (α=0.05)' if mk_result.p < 0.05 else 'not significant',
                'interpretation': self._interpret_mann_kendall(mk_result.trend, mk_result.p, mk_result.slope),
                'dates': dates_list,
                'values': monthly_series.values.tolist()
            }

        elif aggregation == 'seasonal':
            # Seasonal analysis (DJF, MAM, JJA, SON)
            df['season'] = df['date'].dt.quarter
            seasonal_trends = {}

            season_names = {1: 'DJF (Winter)', 2: 'MAM (Spring)', 3: 'JJA (Summer)', 4: 'SON (Autumn)'}

            for season, season_name in season_names.items():
                season_data = df[df['season'] == season].groupby(df['date'].dt.year)['precipitation'].sum()

                if len(season_data) >= 3:
                    mk_result = mk.original_test(season_data.values)
                    seasonal_trends[season_name] = {
                        'trend': mk_result.trend,
                        'p_value': mk_result.p,
                        'slope': mk_result.slope,
                        'significance': 'significant' if mk_result.p < 0.05 else 'not significant'
                    }

            results = {
                'aggregation': 'seasonal',
                'seasonal_trends': seasonal_trends
            }

        return results

    def _interpret_mann_kendall(self, trend, p_value, slope):
        """Generate human-readable interpretation of Mann-Kendall test"""
        if p_value >= 0.05:
            return f"No significant trend detected (p={p_value:.4f})"

        if trend == 'increasing':
            return f"Significant increasing trend detected (p={p_value:.4f}, slope={slope:.2f})"
        elif trend == 'decreasing':
            return f"Significant decreasing trend detected (p={p_value:.4f}, slope={slope:.2f})"
        else:
            return f"No monotonic trend (p={p_value:.4f})"

    def calculate_pettitt_test(self, aggregation='annual') -> Dict:
        """
        Pettitt test for detecting change points in precipitation time series

        The Pettitt test is a non-parametric test for detecting a single change point
        in a time series. It identifies the location where the statistical properties
        of the series change significantly.

        Args:
            aggregation: 'annual' or 'monthly'

        Returns:
            Dictionary with:
            - change_point_index: index of detected change point
            - change_point_date: date/year of change point
            - p_value: statistical significance (< 0.05 = significant)
            - U_statistic: Pettitt's U statistic
            - mean_before: mean before change point
            - mean_after: mean after change point
            - change_magnitude: percentage change
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return {'error': 'No precipitation data available'}

        df = self.precipitation_data.copy()

        # Aggregate data
        if aggregation == 'annual':
            df_agg = df.groupby(df['date'].dt.year).agg({
                'precipitation': 'sum',
                'date': 'first'
            }).reset_index(drop=True)
            time_col = 'year'
        elif aggregation == 'monthly':
            df_agg = df.resample('M', on='date').agg({
                'precipitation': 'sum'
            }).reset_index()
            time_col = 'date'
        else:
            return {'error': 'Invalid aggregation type'}

        series = df_agg['precipitation'].values
        n = len(series)

        if n < 10:
            return {'error': 'Insufficient data (need at least 10 periods)'}

        # Calculate Pettitt's U statistic
        U = np.zeros(n)
        for t in range(1, n):
            for i in range(t):
                for j in range(t, n):
                    U[t] += np.sign(series[j] - series[i])

        # Find maximum absolute U
        K = np.argmax(np.abs(U))
        U_K = np.abs(U[K])

        # Calculate p-value (approximation for large samples)
        p_value = 2 * np.exp(-6 * U_K**2 / (n**3 + n**2))

        # Calculate statistics before and after change point
        mean_before = np.mean(series[:K])
        mean_after = np.mean(series[K:])
        std_before = np.std(series[:K])
        std_after = np.std(series[K:])

        change_magnitude = ((mean_after - mean_before) / mean_before) * 100 if mean_before > 0 else 0

        # Get change point date
        if aggregation == 'annual':
            change_point_date = int(df_agg.iloc[K]['date'].year)
        else:
            # Convert timestamp to string
            date_val = df_agg.iloc[K]['date']
            change_point_date = date_val.strftime('%Y-%m-%d') if isinstance(date_val, pd.Timestamp) else str(date_val)

        results = {
            'aggregation': aggregation,
            'change_point_index': int(K),
            'change_point_date': change_point_date,
            'p_value': float(p_value),
            'U_statistic': float(U_K),
            'is_significant': p_value < 0.05,
            'mean_before': float(mean_before),
            'mean_after': float(mean_after),
            'std_before': float(std_before),
            'std_after': float(std_after),
            'change_magnitude_percent': float(change_magnitude),
            'series_values': series.tolist(),
            'U_values': U.tolist(),
            'interpretation': self._interpret_pettitt(p_value, change_magnitude, change_point_date)
        }

        return results

    def _interpret_pettitt(self, p_value, change_magnitude, change_date):
        """Generate human-readable interpretation of Pettitt test"""
        if p_value >= 0.05:
            return f"No significant change point detected (p={p_value:.4f})"

        direction = "increase" if change_magnitude > 0 else "decrease"
        return (f"Significant change point detected at {change_date} (p={p_value:.4f}). "
                f"Precipitation shows a {abs(change_magnitude):.1f}% {direction} after this point.")

    def calculate_flow_duration_curve(self) -> Dict:
        """
        Calculate Flow Duration Curve (Exceedance Probability Analysis)

        FDC shows the percentage of time that a given value is exceeded.
        Very useful for understanding precipitation variability and extremes.

        Returns:
            Dictionary with:
            - values: sorted precipitation values (descending)
            - exceedance_probability: percentage of time each value is exceeded
            - percentiles: key percentile values (Q10, Q50, Q90, etc.)
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return {'error': 'No precipitation data available'}

        precip = self.precipitation_data['precipitation'].values

        # Remove zeros for better visualization (optional)
        precip_nonzero = precip[precip > 0.1]  # Only values > 0.1mm

        # Sort in descending order
        sorted_values = np.sort(precip_nonzero)[::-1]

        # Calculate exceedance probability
        n = len(sorted_values)
        exceedance_prob = (np.arange(1, n + 1) / n) * 100

        # Calculate key percentiles
        percentiles = {
            'Q10': np.percentile(precip, 90),  # Exceeded 10% of time (high flow)
            'Q25': np.percentile(precip, 75),
            'Q50': np.percentile(precip, 50),  # Median
            'Q75': np.percentile(precip, 25),
            'Q90': np.percentile(precip, 10),  # Exceeded 90% of time (low flow)
            'Q95': np.percentile(precip, 5),
            'Q99': np.percentile(precip, 1)
        }

        results = {
            'values': sorted_values.tolist(),
            'exceedance_probability': exceedance_prob.tolist(),
            'percentiles': percentiles,
            'num_wet_days': int(np.sum(precip > 0.1)),
            'num_dry_days': int(np.sum(precip <= 0.1)),
            'total_days': len(precip)
        }

        return results

    def calculate_wet_dry_spell_analysis(self, wet_threshold=1.0) -> Dict:
        """
        Analyze consecutive wet and dry periods

        Args:
            wet_threshold: Minimum precipitation (mm) to define a wet day (default: 1.0mm)

        Returns:
            Dictionary with:
            - Dry spell statistics (max, mean, distribution)
            - Wet spell statistics
            - Drought risk metrics
        """
        if self.precipitation_data is None or self.precipitation_data.empty:
            return {'error': 'No precipitation data available'}

        df = self.precipitation_data.copy()
        precip = df['precipitation'].values

        # Classify days as wet (1) or dry (0)
        is_wet = (precip >= wet_threshold).astype(int)

        # Find spell changes
        spell_changes = np.diff(is_wet, prepend=is_wet[0])
        spell_starts = np.where(spell_changes != 0)[0]

        # Extract spell lengths
        dry_spells = []
        wet_spells = []

        for i in range(len(spell_starts) - 1):
            start = spell_starts[i]
            end = spell_starts[i + 1]
            spell_length = end - start

            if is_wet[start] == 0:  # Dry spell
                dry_spells.append(spell_length)
            else:  # Wet spell
                wet_spells.append(spell_length)

        # Handle last spell
        if len(spell_starts) > 0:
            start = spell_starts[-1]
            spell_length = len(is_wet) - start
            if is_wet[start] == 0:
                dry_spells.append(spell_length)
            else:
                wet_spells.append(spell_length)

        # Calculate statistics
        dry_spells = np.array(dry_spells) if dry_spells else np.array([0])
        wet_spells = np.array(wet_spells) if wet_spells else np.array([0])

        results = {
            'wet_threshold': wet_threshold,
            'dry_spells': {
                'max_length': int(np.max(dry_spells)),
                'mean_length': float(np.mean(dry_spells)),
                'median_length': float(np.median(dry_spells)),
                'std_length': float(np.std(dry_spells)),
                'num_spells': len(dry_spells),
                'distribution': dry_spells.tolist(),
                'percentile_90': float(np.percentile(dry_spells, 90)),
                'percentile_95': float(np.percentile(dry_spells, 95))
            },
            'wet_spells': {
                'max_length': int(np.max(wet_spells)),
                'mean_length': float(np.mean(wet_spells)),
                'median_length': float(np.median(wet_spells)),
                'std_length': float(np.std(wet_spells)),
                'num_spells': len(wet_spells),
                'distribution': wet_spells.tolist(),
                'percentile_90': float(np.percentile(wet_spells, 90)),
                'percentile_95': float(np.percentile(wet_spells, 95))
            },
            'drought_risk': {
                'probability_dry_spell_gt_7days': float(np.sum(dry_spells > 7) / len(dry_spells)),
                'probability_dry_spell_gt_14days': float(np.sum(dry_spells > 14) / len(dry_spells)),
                'probability_dry_spell_gt_30days': float(np.sum(dry_spells > 30) / len(dry_spells))
            }
        }

        return results

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
        Create enhanced Intensity-Duration-Frequency curves plot

        Handles both hourly and daily durations automatically
        """
        fig = go.Figure()

        if not idf_data:
            return fig

        # Colors for different return periods
        colors = px.colors.qualitative.Set1

        # Determine if we have hourly or daily data
        first_duration = list(idf_data.keys())[0]
        first_data = idf_data[first_duration]

        # Get return periods from first entry
        return_periods = first_data.get('return_periods', [2, 5, 10, 25, 50, 100])

        # Check if we have hourly or daily durations
        is_hourly = 'duration_hours' in first_data and first_data.get('duration_hours', 0) < 24

        for i, rp in enumerate(return_periods):
            durations = []
            intensities = []

            for duration_key, data in idf_data.items():
                if rp in data['return_periods']:
                    idx = data['return_periods'].index(rp)

                    # Get duration in appropriate units
                    if 'duration_hours' in data:
                        dur = data['duration_hours']
                    else:
                        # Legacy format: key is the duration
                        dur = duration_key

                    durations.append(dur)

                    # Get intensity in appropriate units
                    if 'return_intensities' in data:
                        # Hourly data (mm/hr)
                        intensities.append(data['return_intensities'][idx])
                    elif 'return_intensities_mm_per_hour' in data:
                        # Daily data converted to hourly
                        intensities.append(data['return_intensities_mm_per_hour'][idx])
                    else:
                        # Legacy format
                        intensities.append(data.get('intensities', [0])[idx])

            if durations:
                fig.add_trace(go.Scatter(
                    x=durations,
                    y=intensities,
                    mode='lines+markers',
                    name=f"{rp}-year",
                    line=dict(width=2.5, color=colors[i % len(colors)]),
                    marker=dict(size=8)
                ))

        # Set labels based on data type
        if is_hourly:
            xaxis_title = "Duration (hours)"
            title = "Intensity-Duration-Frequency (IDF) Curves (Sub-daily Analysis)"
        else:
            xaxis_title = "Duration (hours)"
            title = "Intensity-Duration-Frequency (IDF) Curves (Daily Analysis)"

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Intensity (mm/hr)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98, title="Return Period"),
            hovermode='x unified',
            height=500
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

    def create_mann_kendall_plot(self, mk_results: Dict) -> go.Figure:
        """
        Create visualization for Mann-Kendall trend analysis
        """
        fig = go.Figure()

        if not mk_results or 'error' in mk_results:
            return fig

        aggregation = mk_results.get('aggregation', 'annual')

        if aggregation in ['annual', 'monthly']:
            # Get time series data
            if aggregation == 'annual':
                x_values = mk_results.get('years', [])
                x_title = 'Year'
            else:
                # Convert timestamps to strings for plotting
                x_values_raw = mk_results.get('dates', [])
                x_values = [pd.to_datetime(d).strftime('%Y-%m') if isinstance(d, (str, pd.Timestamp)) else str(d)
                           for d in x_values_raw]
                x_title = 'Date'

            y_values = mk_results.get('values', [])

            # Plot actual data
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name='Actual Data',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            # Add Sen's slope trend line
            slope = mk_results.get('slope', 0)
            if slope != 0 and len(x_values) > 0:
                # Calculate trend line
                if aggregation == 'annual':
                    x_numeric = np.array(x_values)
                    mid_year = np.mean(x_numeric)
                    trend_y = slope * (x_numeric - mid_year) + np.median(y_values)
                else:
                    # For monthly, use index
                    x_numeric = np.arange(len(x_values))
                    mid_idx = len(x_values) / 2
                    trend_y = slope * (x_numeric - mid_idx) + np.median(y_values)

                # Determine color based on trend
                trend = mk_results.get('trend', 'no trend')
                trend_color = 'red' if trend == 'increasing' else 'green' if trend == 'decreasing' else 'gray'

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=trend_y,
                    mode='lines',
                    name=f"Sen's Slope ({slope:.2f} {mk_results.get('slope_unit', 'mm/yr')})",
                    line=dict(color=trend_color, width=2, dash='dash')
                ))

            # Update layout
            title_text = f"Mann-Kendall Trend Analysis ({aggregation.capitalize()})"
            if 'interpretation' in mk_results:
                title_text += f"<br><sub>{mk_results['interpretation']}</sub>"

            fig.update_layout(
                title=title_text,
                xaxis_title=x_title,
                yaxis_title='Precipitation (mm)',
                template='plotly_white',
                hovermode='x unified',
                height=500
            )

        return fig

    def create_pettitt_plot(self, pettitt_results: Dict) -> go.Figure:
        """
        Create visualization for Pettitt change point test
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Time Series with Change Point', "Pettitt's U Statistic"],
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )

        if not pettitt_results or 'error' in pettitt_results:
            return fig

        series_values = pettitt_results.get('series_values', [])
        change_point_idx = pettitt_results.get('change_point_index', 0)
        U_values = pettitt_results.get('U_values', [])

        # Plot 1: Time series with change point
        x_values = list(range(len(series_values)))

        # Before change point
        fig.add_trace(go.Scatter(
            x=x_values[:change_point_idx],
            y=series_values[:change_point_idx],
            mode='lines+markers',
            name='Before Change',
            line=dict(color='blue', width=2),
            marker=dict(size=5)
        ), row=1, col=1)

        # After change point
        fig.add_trace(go.Scatter(
            x=x_values[change_point_idx:],
            y=series_values[change_point_idx:],
            mode='lines+markers',
            name='After Change',
            line=dict(color='red', width=2),
            marker=dict(size=5)
        ), row=1, col=1)

        # Add vertical line at change point
        fig.add_vline(
            x=change_point_idx,
            line_dash="dash",
            line_color="orange",
            line_width=2,
            annotation_text=f"Change Point: {pettitt_results.get('change_point_date', 'N/A')}",
            row=1, col=1
        )

        # Add mean lines
        mean_before = pettitt_results.get('mean_before', 0)
        mean_after = pettitt_results.get('mean_after', 0)

        fig.add_hline(y=mean_before, line_dash="dot", line_color="blue",
                     annotation_text=f"Mean: {mean_before:.1f}", row=1, col=1)
        fig.add_hline(y=mean_after, line_dash="dot", line_color="red",
                     annotation_text=f"Mean: {mean_after:.1f}", row=1, col=1)

        # Plot 2: U statistic
        fig.add_trace(go.Scatter(
            x=x_values,
            y=U_values,
            mode='lines',
            name="U Statistic",
            line=dict(color='purple', width=2),
            showlegend=False
        ), row=2, col=1)

        # Mark maximum U
        fig.add_trace(go.Scatter(
            x=[change_point_idx],
            y=[U_values[change_point_idx]],
            mode='markers',
            name='Maximum |U|',
            marker=dict(color='red', size=12, symbol='star')
        ), row=2, col=1)

        # Update layout
        interpretation = pettitt_results.get('interpretation', '')
        fig.update_layout(
            title=f"Pettitt Change Point Test<br><sub>{interpretation}</sub>",
            template='plotly_white',
            height=700,
            showlegend=True
        )

        fig.update_xaxes(title_text="Time Index", row=2, col=1)
        fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
        fig.update_yaxes(title_text="U Statistic", row=2, col=1)

        return fig

    def create_flow_duration_curve_plot(self, fdc_results: Dict) -> go.Figure:
        """
        Create Flow Duration Curve visualization
        """
        fig = go.Figure()

        if not fdc_results or 'error' in fdc_results:
            return fig

        values = fdc_results.get('values', [])
        exceedance_prob = fdc_results.get('exceedance_probability', [])
        percentiles = fdc_results.get('percentiles', {})

        # Main FDC curve
        fig.add_trace(go.Scatter(
            x=exceedance_prob,
            y=values,
            mode='lines',
            name='FDC',
            line=dict(color='blue', width=2.5)
        ))

        # Add percentile markers
        percentile_colors = {
            'Q10': 'red',
            'Q50': 'green',
            'Q90': 'orange'
        }

        for pct_name, pct_value in percentiles.items():
            if pct_name in percentile_colors:
                # Find corresponding exceedance probability
                prob = int(pct_name[1:])
                fig.add_trace(go.Scatter(
                    x=[prob],
                    y=[pct_value],
                    mode='markers',
                    name=f'{pct_name} ({pct_value:.1f} mm)',
                    marker=dict(size=12, color=percentile_colors[pct_name])
                ))

        fig.update_layout(
            title='Flow Duration Curve (Precipitation Exceedance)',
            xaxis_title='Exceedance Probability (%)',
            yaxis_title='Precipitation (mm)',
            yaxis_type='log',
            template='plotly_white',
            hovermode='closest',
            height=500
        )

        return fig

    def create_wet_dry_spell_plot(self, spell_results: Dict) -> go.Figure:
        """
        Create visualization for wet/dry spell analysis
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Dry Spell Distribution', 'Wet Spell Distribution']
        )

        if not spell_results or 'error' in spell_results:
            return fig

        dry_spells = spell_results.get('dry_spells', {}).get('distribution', [])
        wet_spells = spell_results.get('wet_spells', {}).get('distribution', [])

        # Dry spell histogram
        fig.add_trace(go.Histogram(
            x=dry_spells,
            name='Dry Spells',
            marker_color='brown',
            nbinsx=30
        ), row=1, col=1)

        # Wet spell histogram
        fig.add_trace(go.Histogram(
            x=wet_spells,
            name='Wet Spells',
            marker_color='blue',
            nbinsx=30
        ), row=1, col=2)

        # Add statistics annotations
        dry_stats = spell_results.get('dry_spells', {})
        wet_stats = spell_results.get('wet_spells', {})

        dry_annotation = (f"Max: {dry_stats.get('max_length', 0)} days<br>"
                         f"Mean: {dry_stats.get('mean_length', 0):.1f} days<br>"
                         f"Median: {dry_stats.get('median_length', 0):.1f} days")

        wet_annotation = (f"Max: {wet_stats.get('max_length', 0)} days<br>"
                         f"Mean: {wet_stats.get('mean_length', 0):.1f} days<br>"
                         f"Median: {wet_stats.get('median_length', 0):.1f} days")

        fig.add_annotation(
            text=dry_annotation,
            xref="x", yref="y",
            x=0.95, y=0.95,
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1
        )

        fig.add_annotation(
            text=wet_annotation,
            xref="x2", yref="y2",
            x=0.95, y=0.95,
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=2
        )

        fig.update_xaxes(title_text="Spell Length (days)", row=1, col=1)
        fig.update_xaxes(title_text="Spell Length (days)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_layout(
            title=f'Wet/Dry Spell Analysis (Threshold: {spell_results.get("wet_threshold", 1.0)} mm)',
            template='plotly_white',
            height=500,
            showlegend=False
        )

        return fig
