"""
Climate Indices Calculation Module
Based on ETCCDI (Expert Team on Climate Change Detection and Indices) standards
Implements ClimPACT indices for temperature and precipitation analysis
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import xarray as xr
from .core.dataset_config import get_dataset_config


class ClimateIndicesCalculator:
    """
    Calculates climate indices based on ETCCDI standards
    All calculations are performed server-side using Google Earth Engine
    Enhanced with automatic unit conversion and JSON configuration support
    """

    def __init__(self, geometry: ee.Geometry, dataset_id: Optional[str] = None):
        """
        Initialize the calculator with area of interest

        Args:
            geometry: ee.Geometry object defining the area of interest
            dataset_id: Earth Engine dataset ID for automatic band mapping and unit conversion
        """
        self.geometry = geometry
        self.dataset_id = dataset_id
        self.dataset_config = get_dataset_config()
        self.indices_metadata = self._get_indices_metadata()
    
    def _get_indices_metadata(self) -> Dict:
        """Get indices metadata from JSON configuration"""
        # Get from JSON config if available, otherwise use defaults
        try:
            return self.dataset_config._config.get('climate_indices', {})
        except:
            # Fallback to basic metadata
            return {
                'TXx': {
                    'name': 'Maximum Temperature',
                    'description': 'Monthly maximum value of daily maximum temperature',
                    'unit': '°C',
                    'category': 'temperature',
                    'complexity': 'simple'
                },
            'TNx': {
                'name': 'Max Tmin',
                'description': 'Monthly maximum value of daily minimum temperature',
                'unit': '°C',
                'category': 'temperature'
            },
            'TXn': {
                'name': 'Min Tmax',
                'description': 'Monthly minimum value of daily maximum temperature',
                'unit': '°C',
                'category': 'temperature'
            },
            'TNn': {
                'name': 'Min Tmin', 
                'description': 'Monthly minimum value of daily minimum temperature',
                'unit': '°C',
                'category': 'temperature'
            },
            'TN10p': {
                'name': 'Cool nights',
                'description': 'Percentage of days when TN < 10th percentile',
                'unit': '%',
                'category': 'temperature'
            },
            'TX10p': {
                'name': 'Cool days',
                'description': 'Percentage of days when TX < 10th percentile',
                'unit': '%',
                'category': 'temperature'
            },
            'TN90p': {
                'name': 'Warm nights',
                'description': 'Percentage of days when TN > 90th percentile',
                'unit': '%',
                'category': 'temperature'
            },
            'TX90p': {
                'name': 'Warm days',
                'description': 'Percentage of days when TX > 90th percentile',
                'unit': '%',
                'category': 'temperature'
            },
            'WSDI': {
                'name': 'Warm spell duration index',
                'description': 'Annual count of days with at least 6 consecutive days when TX > 90th percentile',
                'unit': 'days',
                'category': 'temperature'
            },
            'CSDI': {
                'name': 'Cold spell duration index',
                'description': 'Annual count of days with at least 6 consecutive days when TN < 10th percentile',
                'unit': 'days',
                'category': 'temperature'
            },
            'DTR': {
                'name': 'Diurnal temperature range',
                'description': 'Monthly mean difference between TX and TN',
                'unit': '°C',
                'category': 'temperature'
            },
            'FD': {
                'name': 'Frost days',
                'description': 'Annual count of days when TN < 0°C',
                'unit': 'days',
                'category': 'temperature'
            },
            'SU': {
                'name': 'Summer days',
                'description': 'Annual count of days when TX > 25°C',
                'unit': 'days',
                'category': 'temperature'
            },
            'ID': {
                'name': 'Ice days',
                'description': 'Annual count of days when TX < 0°C',
                'unit': 'days',
                'category': 'temperature'
            },
            'TR': {
                'name': 'Tropical nights',
                'description': 'Annual count of days when TN > 20°C',
                'unit': 'days',
                'category': 'temperature'
            },
            'GSL': {
                'name': 'Growing season length',
                'description': 'Annual count between first span of 6 days with TG > 5°C and first span after July 1 of 6 days with TG < 5°C',
                'unit': 'days',
                'category': 'temperature'
            },
            # Precipitation indices
            'RX1day': {
                'name': 'Max 1-day precipitation',
                'description': 'Monthly maximum 1-day precipitation',
                'unit': 'mm',
                'category': 'precipitation'
            },
            'RX5day': {
                'name': 'Max 5-day precipitation',
                'description': 'Monthly maximum consecutive 5-day precipitation',
                'unit': 'mm',
                'category': 'precipitation'
            },
            'SDII': {
                'name': 'Simple daily intensity index',
                'description': 'Annual total precipitation divided by the number of wet days',
                'unit': 'mm/day',
                'category': 'precipitation'
            },
            'R10mm': {
                'name': 'Heavy precipitation days',
                'description': 'Annual count of days when precipitation ≥ 10mm',
                'unit': 'days',
                'category': 'precipitation'
            },
            'R20mm': {
                'name': 'Very heavy precipitation days',
                'description': 'Annual count of days when precipitation ≥ 20mm',
                'unit': 'days',
                'category': 'precipitation'
            },
            'CDD': {
                'name': 'Consecutive dry days',
                'description': 'Maximum length of dry spell (precipitation < 1mm)',
                'unit': 'days',
                'category': 'precipitation'
            },
            'CWD': {
                'name': 'Consecutive wet days',
                'description': 'Maximum length of wet spell (precipitation ≥ 1mm)',
                'unit': 'days',
                'category': 'precipitation'
            },
            'R95p': {
                'name': 'Very wet days',
                'description': 'Annual total precipitation when RR > 95th percentile',
                'unit': 'mm',
                'category': 'precipitation'
            },
            'R99p': {
                'name': 'Extremely wet days',
                'description': 'Annual total precipitation when RR > 99th percentile',
                'unit': 'mm',
                'category': 'precipitation'
            },
            'PRCPTOT': {
                'name': 'Annual total wet-day precipitation',
                'description': 'Annual total precipitation in wet days (RR ≥ 1mm)',
                'unit': 'mm',
                'category': 'precipitation'
            }
        }
    
    def calculate_simple_percentile(self, collection: ee.ImageCollection,
                                  percentile: int = 90) -> ee.Image:
        """
        Calculate simple percentile from collection (simplified approach)

        Args:
            collection: Temperature or precipitation collection
            percentile: Percentile to calculate (10, 90, etc.)

        Returns:
            ee.Image with percentile values
        """
        return collection.reduce(ee.Reducer.percentile([percentile])).rename([f"p{percentile}"])

    def calculate_base_period_percentiles(self, collection: ee.ImageCollection,
                                        percentiles: List[float],
                                        base_start: str = "1980-01-01",
                                        base_end: str = "2000-12-31") -> Dict[str, ee.Image]:
        """
        Calculate percentiles for the fixed 1980-2000 base period

        Args:
            collection: ee.ImageCollection to calculate percentiles for
            percentiles: List of percentile values (e.g., [10, 75, 90, 95, 99])
            base_start: Start date of base period (default: 1980-01-01)
            base_end: End date of base period (default: 2000-12-31)

        Returns:
            Dictionary mapping percentile names to ee.Image objects
        """
        # Filter collection to base period
        base_collection = collection.filterDate(base_start, base_end)

        # Calculate percentiles for the entire base period
        percentile_images = {}

        try:
            for percentile in percentiles:
                percentile_image = base_collection.reduce(ee.Reducer.percentile([percentile]))
                percentile_images[f'p{int(percentile)}'] = percentile_image
        except Exception as e:
            error_details = str(e)
            # Check for common Earth Engine errors
            if "Collection is empty" in error_details:
                raise ValueError(f"No data found in base period ({base_start} to {base_end}). The dataset may not cover this time range.")
            elif "Image.reduce" in error_details:
                raise ValueError(f"Earth Engine error during percentile calculation for base period ({base_start} to {base_end}). Raw error: {error_details}")
            else:
                raise ValueError(f"Failed to calculate percentiles for base period ({base_start} to {base_end}). Raw error: {error_details}")

        return percentile_images

    # Temperature Indices Calculations
    
    def calculate_TXx(self, tmax_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      temporal_resolution: str = 'yearly',
                      climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate maximum value of daily maximum temperature

        Formula: TXx = max(TX) for each period (monthly or yearly)

        Args:
            tmax_collection: Daily maximum temperature collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Filter collection to date range
        filtered = tmax_collection.filterDate(start_date, end_date)

        # Create year-month sequence for ALL months in the date range
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Calculate monthly maxima for ALL months in the date range
            def calculate_monthly_max(month_year):
                month = ee.Number(month_year).int()
                year = month.divide(100).int()
                month = month.mod(100)

                # Filter to specific month and year
                monthly_data = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Return monthly maximum
                max_img = monthly_data.max()

                return max_img.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TXx',
                    'unit': 'celsius'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            # Create ALL year-month combinations (encode as YYYYMM)
            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            # Calculate monthly maxima for ALL combinations
            monthly_results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_max)
            )

            return monthly_results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_txx_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Return annual maximum
                max_img = yearly.max()

                return max_img.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TXx',
                    'unit': 'celsius'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_txx_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TXx',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_txx(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Return annual maximum
                max_img = yearly.max()

                return max_img.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TXx',
                    'unit': 'celsius'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_txx)
            )

            return results
    
    def calculate_TNn(self, tmin_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      temporal_resolution: str = 'yearly',
                      climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate minimum value of daily minimum temperature

        Formula: TNn = min(TN) for each period (monthly or yearly)

        Args:
            tmin_collection: Daily minimum temperature collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Filter collection to date range
        filtered = tmin_collection.filterDate(start_date, end_date)

        # Create year-month sequence for ALL months in the date range
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Calculate monthly minima for ALL months in the date range
            def calculate_monthly_min(month_year):
                month = ee.Number(month_year).int()
                year = month.divide(100).int()
                month = month.mod(100)

                # Filter to specific month and year
                monthly_data = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Return monthly minimum
                min_img = monthly_data.min()

                return min_img.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TNn',
                    'unit': 'celsius'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            # Create ALL year-month combinations (encode as YYYYMM)
            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            # Calculate monthly minima for ALL combinations
            monthly_results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_min)
            )

            return monthly_results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_tnn_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Return annual minimum
                min_img = yearly.min()

                return min_img.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TNn',
                    'unit': 'celsius'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tnn_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TNn',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_tnn(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Return annual minimum
                min_img = yearly.min()

                return min_img.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TNn',
                    'unit': 'celsius'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tnn)
            )

            return results
    
    def calculate_TX90p(self, tmax_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1980-01-01",
                        base_end: str = "2000-12-31",
                        percentile: float = 90.0,
                        temporal_resolution: str = 'yearly',
                        climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate count of days when Tmax > percentile threshold of base period

        Formula: TXXXp = count(TX > TX_percentile) count in days

        Args:
            tmax_collection: Daily maximum temperature collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 90.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = tmax_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_tx90p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Count exceedances
                exceedances = monthly.map(lambda img: img.gt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TX90p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_tx90p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_tx90p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.gt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TX90p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tx90p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TX90p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_tx90p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.gt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TX90p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tx90p)
            )

            return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_TX10p(self, tmax_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1980-01-01",
                        base_end: str = "2000-12-31",
                        percentile: float = 10.0,
                        temporal_resolution: str = 'yearly',
                        climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate count of days when Tmax < percentile threshold of base period

        Formula: TXXXp = count(TX < TX_percentile) count in days

        Args:
            tmax_collection: Daily maximum temperature collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 10.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = tmax_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_tx10p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Count exceedances
                exceedances = monthly.map(lambda img: img.lt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TX10p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_tx10p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_tx10p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.lt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TX10p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tx10p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TX10p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_tx10p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.lt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TX10p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tx10p)
            )

            return results

    def calculate_TN90p(self, tmin_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1980-01-01",
                        base_end: str = "2000-12-31",
                        percentile: float = 90.0,
                        temporal_resolution: str = 'yearly',
                        climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate count of days when Tmin > percentile threshold of base period

        Formula: TNXXXp = count(TN > TN_percentile) count in days

        Args:
            tmin_collection: Daily minimum temperature collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 90.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = tmin_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = tmin_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_tn90p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Count exceedances
                exceedances = monthly.map(lambda img: img.gt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TN90p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_tn90p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_tn90p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.gt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TN90p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tn90p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TN90p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_tn90p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.gt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TN90p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tn90p)
            )

            return results

    def calculate_TN10p(self, tmin_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1980-01-01",
                        base_end: str = "2000-12-31",
                        percentile: float = 10.0,
                        temporal_resolution: str = 'yearly',
                        climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate count of days when Tmin < percentile threshold of base period

        Formula: TNXXXp = count(TN < TN_percentile) count in days

        Args:
            tmin_collection: Daily minimum temperature collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 10.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = tmin_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = tmin_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_tn10p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Count exceedances
                exceedances = monthly.map(lambda img: img.lt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TN10p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_tn10p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_tn10p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.lt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TN10p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tn10p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TN10p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_tn10p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Count exceedances
                exceedances = yearly.map(lambda img: img.lt(percentile_threshold))
                exceedance_count = exceedances.sum()

                return exceedance_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TN10p',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_tn10p)
            )

            return results

    def calculate_TXn(self, tmax_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      temporal_resolution: str = 'yearly',
                      climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate minimum value of daily maximum temperature

        Formula: TXn = min(TX) minimum in °C

        Args:
            tmax_collection: Daily maximum temperature collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date)

        # Create year-month sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_txn(month_year):
                month = ee.Number(month_year).int()
                year = month.divide(100).int()
                month = month.mod(100)

                # Get month data
                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Calculate minimum
                monthly_min = monthly.min()

                return monthly_min.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TXn',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_txn)
            )

            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_yearly_txn_clim(year):
                year = ee.Number(year)
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Calculate minimum for year
                yearly_min = yearly.min()

                return yearly_min.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TXn',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_txn_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TXn',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_yearly_txn(year):
                year = ee.Number(year)
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Calculate minimum for year
                yearly_min = yearly.min()

                return yearly_min.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TXn',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_txn)
            )
            return results

    def calculate_TNx(self, tmin_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      temporal_resolution: str = 'yearly',
                      climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate maximum value of daily minimum temperature

        Formula: TNx = max(TN) maximum in °C

        Args:
            tmin_collection: Daily minimum temperature collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Filter to analysis period
        filtered = tmin_collection.filterDate(start_date, end_date)

        # Create year-month sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_tnx(month_year):
                month = ee.Number(month_year).int()
                year = month.divide(100).int()
                month = month.mod(100)

                # Get month data
                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Calculate maximum
                monthly_max = monthly.max()

                return monthly_max.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'TNx',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_tnx)
            )

            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_yearly_tnx_clim(year):
                year = ee.Number(year)
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Calculate maximum for year
                yearly_max = yearly.max()

                return yearly_max.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TNx',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_tnx_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'TNx',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_yearly_tnx(year):
                year = ee.Number(year)
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Calculate maximum for year
                yearly_max = yearly.max()

                return yearly_max.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'TNx',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_tnx)
            )
            return results

    def calculate_SU(self, tmax_collection: ee.ImageCollection,
                     start_date: str, end_date: str,
                     threshold: float = 25.0) -> ee.ImageCollection:
        """
        Calculate annual count of days when daily maximum temperature > 25°C

        Formula: SU = count(TX > 25°C) annual count in days
        """
        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'temperature_max')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        # Convert threshold from user units (e.g., °C) to raw units (e.g., Kelvin for ERA5)
        # For ERA5: raw = (user - offset) / scaling_factor
        # If dataset uses Kelvin: threshold_kelvin = threshold_celsius + 273.15
        # But we use the dataset's scaling info for consistency
        raw_threshold = (threshold - offset) / scaling_factor if scaling_factor else threshold

        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        def calculate_annual_su(year):
            year = ee.Number(year)
            yearly = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            # Count summer days using raw threshold
            summer_days = yearly.map(lambda img: img.gt(raw_threshold))
            summer_day_count = summer_days.sum()
            return summer_day_count.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                'index_name': 'SU',
                'unit': 'days'
            })

        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        years = ee.List.sequence(start_year, end_year)

        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_su)
        )
        return results

    def calculate_R20mm(self, precip_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        threshold: float = 20.0,
                        temporal_resolution: str = 'yearly',
                        climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate count of days when daily precipitation ≥ 20mm

        Formula: R20mm = count(precip ≥ 20mm) count in days

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            threshold: Precipitation threshold (default 20.0 mm)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', etc.
            climatology_reducer: 'mean', 'median', 'min', or 'max'
        """
        threshold_mm = threshold  # User-provided threshold in mm
        
        # ── YEARLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            # Convert threshold from mm to raw units (e.g., meters for ERA5)
            raw_threshold = (threshold_mm - offset) / scaling_factor if scaling_factor else threshold_mm

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_coll = raw_filtered.filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                heavy_days = year_coll.map(lambda img: img.gte(raw_threshold))
                total_heavy = heavy_days.sum()
                yearly_images.append(total_heavy.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'R20mm',
                    'unit': 'days'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_threshold = (threshold_mm - offset) / scaling_factor if scaling_factor else threshold_mm

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_coll = raw_filtered.filterDate(month_start, month_end)
                    heavy_days = month_coll.map(lambda img: img.gte(raw_threshold))
                    total_heavy = heavy_days.sum()
                    monthly_images.append(total_heavy.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'R20mm',
                        'unit': 'days'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'precipitation')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        raw_threshold = (threshold_mm - offset) / scaling_factor if scaling_factor else threshold_mm

        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            def calculate_annual_r20_clim(year):
                year = ee.Number(year)
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )
                heavy_rain_days = yearly.map(lambda img: img.gte(raw_threshold))
                heavy_rain_count = heavy_rain_days.sum()
                return heavy_rain_count.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R20mm',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r20_clim)
            )

            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'R20mm',
                'start_year': start_year,
                'end_year': end_year
            })

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_WSDI(self, tmax_collection: ee.ImageCollection,
                       start_date: str, end_date: str,
                       base_start: str = "1981-01-01", 
                       base_end: str = "2010-12-31") -> ee.ImageCollection:
        """
        Calculate Warm Spell Duration Index (simplified)
        Annual count of days with at least 6 consecutive days when TX > 90th percentile
        
        Note: This is a simplified implementation counting warm days instead of consecutive spells
        """
        # Calculate 90th percentile from base period
        base_collection = tmax_collection.filterDate(base_start, base_end)
        percentile_90 = self.calculate_simple_percentile(base_collection, 90)
        
        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date)
        
        def calculate_annual_wsdi(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Count warm days (simplified approach - should be consecutive spells in full implementation)
            warm_days = annual.map(lambda img: img.gt(percentile_90))
            warm_day_count = warm_days.sum()
            
            return warm_day_count.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_wsdi)
        )
        
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_GSL(self, tmean_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      threshold: float = 5.0) -> ee.ImageCollection:
        """
        Calculate Growing Season Length (simplified)
        Annual count between first span of 6 days with TG > threshold°C and first span after July 1 of 6 days with TG < threshold°C

        Note: This is a simplified implementation counting days above threshold°C instead of identifying specific spans
        Default threshold: 5°C
        """
        # Get scaling info for threshold conversion
        # Try temperature_mean first, fallback to temperature_max or temperature_min
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = None
            for band_type in ['temperature_mean', 'temperature_max', 'temperature_min']:
                try:
                    band_info = self.dataset_config.get_band_scaling_info(
                        self.dataset_id, band_type)
                    break
                except (KeyError, AttributeError, ValueError):
                    continue
            if band_info:
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

        # Convert threshold from user units (e.g., °C) to raw units
        raw_threshold = (threshold - offset) / scaling_factor if scaling_factor else threshold

        filtered = tmean_collection.filterDate(start_date, end_date).filterBounds(self.geometry)
        
        def calculate_annual_gsl(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            # Count days above threshold using raw threshold
            growing_days = annual.map(lambda img: img.gt(raw_threshold))
            growing_day_count = growing_days.sum()
            return growing_day_count.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                'index_name': 'GSL',
                'unit': 'days'
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_gsl)
        )
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_FD(self, tmin_collection: ee.ImageCollection,
                     start_date: str, end_date: str,
                     threshold: float = 0.0) -> ee.ImageCollection:
        """
        Calculate annual count of frost days (TN < threshold°C)

        Formula: FD = count(TN < threshold°C)
        Default threshold: 0°C
        """
        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'temperature_min')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        # Convert threshold from user units (e.g., °C) to raw units
        raw_threshold = (threshold - offset) / scaling_factor if scaling_factor else threshold

        filtered = tmin_collection.filterDate(start_date, end_date).filterBounds(self.geometry)
        
        def calculate_annual_fd(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            # Count days below threshold using raw threshold
            frost_days = annual.map(lambda img: img.lt(raw_threshold)).sum()
            return frost_days.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                'index_name': 'FD',
                'unit': 'days'
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_fd)
        )
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_DTR(self, tmax_collection: ee.ImageCollection,
                      tmin_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      temporal_resolution: str = 'yearly',
                      climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate mean diurnal temperature range

        Formula: DTR = mean(TX - TN)

        Args:
            tmax_collection: Daily maximum temperature collection
            tmin_collection: Daily minimum temperature collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Filter collections to date range
        tmax_filtered = tmax_collection.filterDate(start_date, end_date)
        tmin_filtered = tmin_collection.filterDate(start_date, end_date)

        # Create year-month sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_dtr(month_year):
                month = ee.Number(month_year).int()
                year = month.divide(100).int()
                month = month.mod(100)

                # Get monthly data
                monthly_tmax = tmax_filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                monthly_tmin = tmin_filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Calculate mean temperatures for the month
                mean_tmax = monthly_tmax.mean()
                mean_tmin = monthly_tmin.mean()

                # Calculate DTR
                dtr = mean_tmax.subtract(mean_tmin)

                return dtr.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'DTR',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_dtr)
            )

            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_yearly_dtr_clim(year):
                year = ee.Number(year)

                # Get yearly data
                yearly_tmax = tmax_filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                yearly_tmin = tmin_filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Calculate mean temperatures for the year
                mean_tmax = yearly_tmax.mean()
                mean_tmin = yearly_tmin.mean()

                # Calculate DTR
                dtr = mean_tmax.subtract(mean_tmin)

                return dtr.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'DTR',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_dtr_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'DTR',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_yearly_dtr(year):
                year = ee.Number(year)

                # Get yearly data
                yearly_tmax = tmax_filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                yearly_tmin = tmin_filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Calculate mean temperatures for the year
                mean_tmax = yearly_tmax.mean()
                mean_tmin = yearly_tmin.mean()

                # Calculate DTR
                dtr = mean_tmax.subtract(mean_tmin)

                return dtr.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'DTR',
                    'unit': '°C'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_dtr)
            )

            return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    # Precipitation Indices Calculations
    
    def calculate_RX1day(self, precip_collection: ee.ImageCollection,
                         start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate monthly maximum 1-day precipitation
        
        Formula: RX1day = max(RR) for each month
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def monthly_max_precip(month_year):
            month = ee.Number(month_year).int()
            year = month.divide(100).int()
            month = month.mod(100)
            
            monthly = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            return monthly.max().set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })
        
        # Create year-month sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        
        year_months = years.map(
            lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
        ).flatten()
        
        results = ee.ImageCollection.fromImages(
            year_months.map(monthly_max_precip)
        )
        
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_RX5day(self, precip_collection: ee.ImageCollection,
                         start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate monthly maximum consecutive 5-day precipitation
        
        Formula: RX5day = max(sum(RR_i to RR_i+4)) for each month
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_5day_sum(image):
            # Get the date of current image
            date = ee.Date(image.get('system:time_start'))
            
            # Get 5-day window
            window = filtered.filterDate(
                date, date.advance(5, 'day')
            )
            
            # Sum precipitation
            sum_5day = window.sum()
            
            return sum_5day.set({
                'system:time_start': date.millis()
            })
        
        # Calculate 5-day sums
        sums_5day = filtered.map(calculate_5day_sum)
        
        # Get monthly maximum
        def monthly_max_5day(month_year):
            month = ee.Number(month_year).int()
            year = month.divide(100).int()
            month = month.mod(100)
            
            monthly = sums_5day.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            return monthly.max().set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })
        
        # Create year-month sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        
        year_months = years.map(
            lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
        ).flatten()
        
        results = ee.ImageCollection.fromImages(
            year_months.map(monthly_max_5day)
        )
        
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_CDD(self, precip_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      threshold: float = 1.0) -> ee.ImageCollection:
        """
        Calculate annual maximum consecutive dry days (simplified)
        
        Formula: CDD = max(consecutive days with RR < threshold)
        Note: This is simplified to count total dry days instead of consecutive runs
        """
        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'precipitation')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        # Convert threshold from mm to raw units (e.g., meters for ERA5)
        raw_threshold = (threshold - offset) / scaling_factor if scaling_factor else threshold

        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)
        
        def calculate_annual_cdd(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            # Count dry days using raw threshold
            dry_days = annual.map(lambda img: img.lt(raw_threshold))
            total_dry = dry_days.sum()
            return total_dry.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                'index_name': 'CDD',
                'unit': 'days'
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_cdd)
        )
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_R10mm(self, precip_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        temporal_resolution: str = 'yearly',
                        climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate count of heavy precipitation days (≥ 10mm)

        Formula: R10mm = count(RR ≥ 10mm)

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', etc.
            climatology_reducer: 'mean', 'median', 'min', or 'max'
        """
        threshold_mm = 10.0  # Fixed threshold for R10mm
        
        # ── YEARLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            # Convert threshold from mm to raw units (e.g., meters for ERA5)
            raw_threshold = (threshold_mm - offset) / scaling_factor if scaling_factor else threshold_mm

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_coll = raw_filtered.filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                heavy_days = year_coll.map(lambda img: img.gte(raw_threshold))
                total_heavy = heavy_days.sum()
                yearly_images.append(total_heavy.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'R10mm',
                    'unit': 'days'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_threshold = (threshold_mm - offset) / scaling_factor if scaling_factor else threshold_mm

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_coll = raw_filtered.filterDate(month_start, month_end)
                    heavy_days = month_coll.map(lambda img: img.gte(raw_threshold))
                    total_heavy = heavy_days.sum()
                    monthly_images.append(total_heavy.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'R10mm',
                        'unit': 'days'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'precipitation')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        raw_threshold = (threshold_mm - offset) / scaling_factor if scaling_factor else threshold_mm

        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            def calculate_annual_r10(year):
                annual = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )
                heavy_days = annual.map(lambda img: img.gte(raw_threshold)).sum()
                return heavy_days.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R10mm',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r10)
            )

            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'R10mm',
                'start_year': start_year,
                'end_year': end_year
            })

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def calculate_SDII(self, precip_collection: ee.ImageCollection,
                       start_date: str, end_date: str,
                       wet_threshold: float = 1.0,
                       temporal_resolution: str = 'yearly',
                       climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate Simple Daily Intensity Index

        Formula: SDII = sum(RR on wet days) / count(wet days)

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            wet_threshold: Threshold for wet day (default 1.0 mm)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', etc.
            climatology_reducer: 'mean', 'median', 'min', or 'max'
        """
        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'precipitation')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        # Convert threshold from mm to raw units (e.g., meters for ERA5)
        raw_threshold = (wet_threshold - offset) / scaling_factor if scaling_factor else wet_threshold

        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_sdii(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Sum precipitation on wet days (using raw threshold)
                wet_precip = monthly.map(
                    lambda img: img.updateMask(img.gte(raw_threshold))
                ).sum()

                # Count wet days (using raw threshold)
                wet_days = monthly.map(lambda img: img.gte(raw_threshold)).sum()

                # Calculate SDII
                sdii = wet_precip.divide(wet_days)

                return sdii.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'SDII',
                    'unit': 'mm/day'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_sdii)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_sdii_clim(year):
                year = ee.Number(year)
                annual = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation on wet days (using raw threshold)
                wet_precip = annual.map(
                    lambda img: img.updateMask(img.gte(raw_threshold))
                ).sum()

                # Count wet days (using raw threshold)
                wet_days = annual.map(lambda img: img.gte(raw_threshold)).sum()

                # Calculate SDII
                sdii = wet_precip.divide(wet_days)

                return sdii.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'SDII',
                    'unit': 'mm/day'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_sdii_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'SDII',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_sdii(year):
                annual = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation on wet days (using raw threshold)
                wet_precip = annual.map(
                    lambda img: img.updateMask(img.gte(raw_threshold))
                ).sum()

                # Count wet days (using raw threshold)
                wet_days = annual.map(lambda img: img.gte(raw_threshold)).sum()

                # Calculate SDII
                sdii = wet_precip.divide(wet_days)

                return sdii.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'SDII',
                    'unit': 'mm/day'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_sdii)
            )

            return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats

    def calculate_R95p(self, precip_collection: ee.ImageCollection,
                       start_date: str, end_date: str,
                       base_start: str = "1980-01-01",
                       base_end: str = "2000-12-31",
                       percentile: float = 95.0,
                       temporal_resolution: str = 'yearly',
                       climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate total precipitation when daily precipitation > percentile threshold of base period

        Formula: RXXXp = sum(precip > precip_percentile) total in mm

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 95.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = precip_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = precip_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_r95p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Sum precipitation above percentile threshold
                wet_days = monthly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'R95p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_r95p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_r95p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation above percentile threshold
                wet_days = yearly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R95p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r95p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'R95p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_r95p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation above percentile threshold
                wet_days = yearly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R95p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r95p)
            )

            return results

    def calculate_R99p(self, precip_collection: ee.ImageCollection,
                       start_date: str, end_date: str,
                       base_start: str = "1980-01-01",
                       base_end: str = "2000-12-31",
                       percentile: float = 99.0,
                       temporal_resolution: str = 'yearly',
                       climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate total precipitation when daily precipitation > percentile threshold of base period

        Formula: RXXXp = sum(precip > precip_percentile) total in mm

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 99.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = precip_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = precip_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_r99p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Sum precipitation above percentile threshold
                wet_days = monthly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'R99p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_r99p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_r99p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation above percentile threshold
                wet_days = yearly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R99p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r99p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'R99p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_r99p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation above percentile threshold
                wet_days = yearly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R99p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r99p)
            )

            return results

    def calculate_R75p(self, precip_collection: ee.ImageCollection,
                       start_date: str, end_date: str,
                       base_start: str = "1980-01-01",
                       base_end: str = "2000-12-31",
                       percentile: float = 75.0,
                       temporal_resolution: str = 'yearly',
                       climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate total precipitation when daily precipitation > percentile threshold of base period

        Formula: RXXXp = sum(precip > precip_percentile) total in mm

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            base_start: Base period start date
            base_end: Base period end date
            percentile: Percentile threshold (default 75.0)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Calculate specified percentile from base period
        base_collection = precip_collection.filterDate(base_start, base_end)
        percentiles = self.calculate_base_period_percentiles(base_collection, [percentile])
        percentile_threshold = percentiles[f'p{int(percentile)}']

        # Filter to analysis period
        filtered = precip_collection.filterDate(start_date, end_date)

        # Create year sequence
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_r75p(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Sum precipitation above percentile threshold
                wet_days = monthly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'R75p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_r75p)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The export function will aggregate to single image based on metadata
            def calculate_annual_r75p_clim(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation above percentile threshold
                wet_days = yearly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R75p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r75p_clim)
            )

            # Set metadata for climatology mode
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'R75p',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_r75p(year):
                year = ee.Number(year)

                # Get year data
                yearly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation above percentile threshold
                wet_days = yearly.map(lambda img: img.multiply(img.gt(percentile_threshold)))
                total_precip = wet_days.sum()

                return total_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'R75p',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_r75p)
            )

            return results

    def calculate_PRCPTOT(self, precip_collection: ee.ImageCollection,
                          start_date: str, end_date: str,
                          wet_threshold: float = 1.0) -> ee.ImageCollection:
        """
        Calculate annual total precipitation on wet days
        
        Formula: PRCPTOT = sum(RR on days where RR ≥ wet_threshold)
        """
        # Get scaling info for threshold conversion
        scaling_factor, offset = 1.0, 0.0
        if self.dataset_id:
            band_info = self.dataset_config.get_band_scaling_info(
                self.dataset_id, 'precipitation')
            scaling_factor = band_info.get('scaling_factor', 1.0)
            offset         = band_info.get('offset', 0.0)

        # Convert threshold from mm to raw units (e.g., meters for ERA5)
        raw_threshold = (wet_threshold - offset) / scaling_factor if scaling_factor else wet_threshold

        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)
        
        def calculate_annual_prcptot(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            # Sum precipitation on wet days only (using raw threshold)
            wet_total = annual.map(
                lambda img: img.updateMask(img.gte(raw_threshold))
            ).sum()
            return wet_total.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                'index_name': 'PRCPTOT',
                'unit': 'mm'
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_prcptot)
        )
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats
    
    def extract_time_series(self, image_collection: ee.ImageCollection,
                           scale: int = 1000) -> pd.DataFrame:
        """
        Extract time series data from image collection
        
        Args:
            image_collection: Collection with calculated indices
            scale: Scale in meters for reduction
            
        Returns:
            pandas DataFrame with time series
        """
        # Reduce to mean over geometry
        def reduce_image(image):
            reduced = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.geometry,
                scale=scale,
                maxPixels=1e9
            )
            
            return ee.Feature(None, reduced).set({
                'system:time_start': image.get('system:time_start')
            })
        
        # Convert to feature collection
        fc = ee.FeatureCollection(image_collection.map(reduce_image))
        
        # Export to pandas
        data = fc.getInfo()
        
        # Convert to DataFrame
        records = []
        for feature in data['features']:
            record = feature['properties']
            if 'system:time_start' in record:
                record['date'] = pd.to_datetime(record['system:time_start'], unit='ms')
            records.append(record)
        
        df = pd.DataFrame(records)
        if 'date' in df.columns:
            df = df.sort_values('date')
            df = df.set_index('date')
        
        return df
    
    def extract_time_series_optimized(self, image_collection: ee.ImageCollection,
                                    scale: int = 5000, max_pixels: int = 1e9) -> pd.DataFrame:
        """Extract time series using getDownloadURL — the same approach that works in
        GeoData Explorer.

        WHY getDownloadURL instead of getInfo():
        ─────────────────────────────────────────
        getInfo() routes through GEE's *interactive* API, which evaluates the full
        computation graph for ALL images simultaneously in a single worker.  For
        complex climate-index expressions (e.g. RX1day = yearly max of 365 daily
        images) mapped over 30 years, this exhausts the interactive memory budget
        and raises "User memory limit exceeded".

        getDownloadURL() routes through GEE's *batch/export* API.  The computation
        is queued and executed with higher memory limits and proper server-side
        streaming, exactly as in GeoData Explorer's get_time_series_average().

        Falls back to a sequential per-image approach if getDownloadURL also fails.
        """
        import requests
        import io as _io

        # ── Primary: server-side CSV via getDownloadURL ──────────────────────
        try:
            def to_feature(image):
                date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
                # Flatten reduced values directly into feature properties
                # (do NOT nest inside 'value' — getDownloadURL needs flat columns)
                reduced = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.geometry,
                    scale=scale,
                    maxPixels=max_pixels,
                    bestEffort=True,
                    tileScale=4
                )
                return ee.Feature(None, reduced.set('date', date_str))

            fc  = ee.FeatureCollection(image_collection.map(to_feature))
            url = fc.getDownloadURL(filetype='csv')
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()

            df = pd.read_csv(_io.StringIO(resp.text))
            if df.empty:
                return pd.DataFrame(columns=['date', 'value'])

            # Normalise date column name
            if 'date' not in df.columns:
                date_cols = [c for c in df.columns
                             if 'date' in c.lower() or 'time' in c.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: 'date'})

            # Pick the first real numeric column as the value
            skip = {'date', 'system:index', '.geo'}
            value_col = next(
                (c for c in df.columns
                 if c not in skip and pd.api.types.is_numeric_dtype(df[c])),
                None
            )
            if value_col is None:
                return pd.DataFrame(columns=['date', 'value'])

            result = df[['date', value_col]].rename(columns={value_col: 'value'}).copy()
            result['date'] = pd.to_datetime(result['date'], errors='coerce')
            result = (result.dropna(subset=['date', 'value'])
                           .sort_values('date')
                           .reset_index(drop=True))
            print(f"Successfully extracted {len(result)} points via getDownloadURL")
            return result

        except Exception as primary_err:
            print(f"getDownloadURL extraction failed: {primary_err}. "
                  f"Falling back to sequential per-image extraction…")

        # ── Fallback: one image at a time (slow but avoids OOM) ─────────────
        return self._extract_time_series_sequential(image_collection, scale, max_pixels)

    def _extract_time_series_sequential(self, image_collection: ee.ImageCollection,
                                        scale: int = 10000,
                                        max_pixels: float = 1e9) -> pd.DataFrame:
        """Process one image at a time to avoid GEE expression-graph memory limits.

        Slower than getDownloadURL but robust for any collection size because each
        reduceRegion call is an independent, lightweight GEE computation.
        """
        try:
            collection_size = image_collection.size().getInfo()
            images_list     = image_collection.toList(min(collection_size, 500))
            records         = []

            print(f"Sequential extraction: {collection_size} images at {scale}m…")
            for i in range(min(collection_size, 500)):
                try:
                    img      = ee.Image(images_list.get(i))
                    date_str = img.date().format('YYYY-MM-dd').getInfo()
                    stats    = img.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=self.geometry,
                        scale=scale,
                        maxPixels=max_pixels,
                        bestEffort=True,
                        tileScale=4
                    ).getInfo()

                    # First valid numeric value across all bands
                    value = next(
                        (v for v in stats.values()
                         if v is not None and isinstance(v, (int, float))),
                        None
                    )
                    if date_str and value is not None:
                        records.append({'date': date_str, 'value': value})

                except Exception as img_err:
                    print(f"Sequential: image {i + 1} failed: {img_err}")
                    continue

            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = (df.dropna(subset=['date', 'value'])
                        .sort_values('date')
                        .reset_index(drop=True))
                print(f"Sequential extraction: {len(df)} points retrieved")
                return df

            return pd.DataFrame(columns=['date', 'value'])

        except Exception as seq_err:
            print(f"Sequential extraction also failed: {seq_err}")
            return pd.DataFrame(columns=['date', 'value'])

    def apply_unit_conversion(self, collection: ee.ImageCollection,
                            band_type: str) -> ee.ImageCollection:
        """
        Apply unit conversion based on dataset configuration

        Args:
            collection: Earth Engine ImageCollection
            band_type: Band type identifier (e.g., 'temperature_max', 'precipitation')

        Returns:
            Collection with converted units
        """
        if not self.dataset_id:
            return collection

        band_info = self.dataset_config.get_band_scaling_info(self.dataset_id, band_type)

        scaling_factor = band_info.get('scaling_factor', 1.0)
        offset = band_info.get('offset', 0.0)

        if scaling_factor != 1.0 or offset != 0.0:
            # Apply conversion: new_value = (original * scaling_factor) + offset
            def convert_image(image):
                converted = image.multiply(scaling_factor).add(offset)
                return converted.copyProperties(image, ['system:time_start', 'system:id'])

            collection = collection.map(convert_image)

        return collection

    def calculate_simple_TXx(self, tmax_collection: ee.ImageCollection,
                           start_date: str, end_date: str,
                           temporal_resolution: str = 'monthly') -> ee.ImageCollection:
        """
        Calculate monthly maximum of daily maximum temperature (simplified implementation)
        """
        # ── YEARLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_max')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_filtered = (tmax_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_max = (raw_filtered
                            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                            .max())
                if scaling_factor != 1.0 or offset != 0.0:
                    year_max = year_max.multiply(scaling_factor).add(offset)
                yearly_images.append(year_max.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'TXx',
                    'unit': '°C'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────────
        # Same memory optimization as yearly: use raw collection, Python loops,
        # apply scaling to individual monthly result images.
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_max')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_filtered = (tmax_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    # Skip months before start or after end
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    # Calculate next month for date range
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_max = (raw_filtered
                                 .filterDate(month_start, month_end)
                                 .max())
                    if scaling_factor != 1.0 or offset != 0.0:
                        month_max = month_max.multiply(scaling_factor).add(offset)
                    monthly_images.append(month_max.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'TXx',
                        'unit': '°C'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        # Apply unit conversion if dataset is specified
        if self.dataset_id:
            tmax_collection = self.apply_unit_conversion(tmax_collection, 'temperature_max')

        # Filter collection to date range
        filtered = tmax_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        # Handle temporal resolution
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
        # Default to monthly if not yearly (already handled above)
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)

        year_months = years.map(
            lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
        ).flatten()

        def calculate_monthly_max(year_month):
            year = ee.Number(year_month).divide(100).int()
            month = ee.Number(year_month).mod(100).int()
            monthly_data = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            max_img = monthly_data.max().clip(self.geometry)
            return max_img.set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                'index_name': 'TXx',
                'unit': '°C'
            })

        monthly_results = ee.ImageCollection.fromImages(
            year_months.map(calculate_monthly_max)
        )
        return monthly_results

    def calculate_simple_TNn(self, tmin_collection: ee.ImageCollection,
                           start_date: str, end_date: str,
                           temporal_resolution: str = 'monthly') -> ee.ImageCollection:
        """
        Calculate monthly minimum of daily minimum temperature (simplified implementation)
        """
        # ── YEARLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_min')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_filtered = (tmin_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_min = (raw_filtered
                            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                            .min())
                if scaling_factor != 1.0 or offset != 0.0:
                    year_min = year_min.multiply(scaling_factor).add(offset)
                yearly_images.append(year_min.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'TNn',
                    'unit': '°C'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────────
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_min')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_filtered = (tmin_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_min = (raw_filtered
                                 .filterDate(month_start, month_end)
                                 .min())
                    if scaling_factor != 1.0 or offset != 0.0:
                        month_min = month_min.multiply(scaling_factor).add(offset)
                    monthly_images.append(month_min.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'TNn',
                        'unit': '°C'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        # Apply unit conversion if dataset is specified
        if self.dataset_id:
            tmin_collection = self.apply_unit_conversion(tmin_collection, 'temperature_min')

        # Filter collection to date range
        filtered = tmin_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        # Handle temporal resolution
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        year_months = years.map(
            lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
        ).flatten()

        def calculate_monthly_min(year_month):
            year = ee.Number(year_month).divide(100).int()
            month = ee.Number(year_month).mod(100).int()
            monthly_data = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            min_img = monthly_data.min().clip(self.geometry)
            return min_img.set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                'index_name': 'TNn',
                'unit': '°C'
            })

        monthly_results = ee.ImageCollection.fromImages(
            year_months.map(calculate_monthly_min)
        )
        return monthly_results

    def calculate_simple_DTR(self, tmax_collection: ee.ImageCollection,
                           tmin_collection: ee.ImageCollection,
                           start_date: str, end_date: str,
                           temporal_resolution: str = 'monthly') -> ee.ImageCollection:
        """
        Calculate monthly mean diurnal temperature range (simplified implementation)
        """
        # ── YEARLY PATH: handle before collection-level .map() ───────────────
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            tmax_scaling, tmax_offset = 1.0, 0.0
            tmin_scaling, tmin_offset = 1.0, 0.0
            if self.dataset_id:
                tmax_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_max')
                tmax_scaling = tmax_info.get('scaling_factor', 1.0)
                tmax_offset  = tmax_info.get('offset', 0.0)
                tmin_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_min')
                tmin_scaling = tmin_info.get('scaling_factor', 1.0)
                tmin_offset  = tmin_info.get('offset', 0.0)

            raw_tmax_filtered = (tmax_collection
                                 .filterDate(start_date, end_date)
                                 .filterBounds(self.geometry))
            raw_tmin_filtered = (tmin_collection
                                 .filterDate(start_date, end_date)
                                 .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_tmax = (raw_tmax_filtered
                             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                             .mean())
                year_tmin = (raw_tmin_filtered
                             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                             .mean())
                # Apply scaling to each temperature, then compute DTR
                if tmax_scaling != 1.0 or tmax_offset != 0.0:
                    year_tmax = year_tmax.multiply(tmax_scaling).add(tmax_offset)
                if tmin_scaling != 1.0 or tmin_offset != 0.0:
                    year_tmin = year_tmin.multiply(tmin_scaling).add(tmin_offset)
                dtr = year_tmax.subtract(year_tmin)
                yearly_images.append(dtr.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'DTR',
                    'unit': '°C'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────────
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            tmax_scaling, tmax_offset = 1.0, 0.0
            tmin_scaling, tmin_offset = 1.0, 0.0
            if self.dataset_id:
                tmax_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_max')
                tmax_scaling = tmax_info.get('scaling_factor', 1.0)
                tmax_offset  = tmax_info.get('offset', 0.0)
                tmin_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'temperature_min')
                tmin_scaling = tmin_info.get('scaling_factor', 1.0)
                tmin_offset  = tmin_info.get('offset', 0.0)

            raw_tmax_filtered = (tmax_collection
                                 .filterDate(start_date, end_date)
                                 .filterBounds(self.geometry))
            raw_tmin_filtered = (tmin_collection
                                 .filterDate(start_date, end_date)
                                 .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_tmax = (raw_tmax_filtered
                                   .filterDate(month_start, month_end)
                                   .mean())
                    month_tmin = (raw_tmin_filtered
                                   .filterDate(month_start, month_end)
                                   .mean())
                    if tmax_scaling != 1.0 or tmax_offset != 0.0:
                        month_tmax = month_tmax.multiply(tmax_scaling).add(tmax_offset)
                    if tmin_scaling != 1.0 or tmin_offset != 0.0:
                        month_tmin = month_tmin.multiply(tmin_scaling).add(tmin_offset)
                    dtr = month_tmax.subtract(month_tmin)
                    monthly_images.append(dtr.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'DTR',
                        'unit': '°C'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        # Apply unit conversions if dataset is specified
        if self.dataset_id:
            tmax_collection = self.apply_unit_conversion(tmax_collection, 'temperature_max')
            tmin_collection = self.apply_unit_conversion(tmin_collection, 'temperature_min')

        # Filter collections to date range
        tmax_filtered = tmax_collection.filterDate(start_date, end_date).filterBounds(self.geometry)
        tmin_filtered = tmin_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        # Handle temporal resolution
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
        years = ee.List.sequence(start_year, end_year)
        months = ee.List.sequence(1, 12)
        year_months = years.map(
            lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
        ).flatten()

        def calculate_monthly_dtr(year_month):
            year = ee.Number(year_month).divide(100).int()
            month = ee.Number(year_month).mod(100).int()
            monthly_tmax = tmax_filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            monthly_tmin = tmin_filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            mean_tmax = monthly_tmax.mean().clip(self.geometry)
            mean_tmin = monthly_tmin.mean().clip(self.geometry)
            dtr = mean_tmax.subtract(mean_tmin)
            return dtr.set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                'index_name': 'DTR',
                'unit': '°C'
            })

        results = ee.ImageCollection.fromImages(
            year_months.map(calculate_monthly_dtr)
        )
        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats

    def calculate_simple_RX1day(self, precip_collection: ee.ImageCollection,
                              start_date: str, end_date: str,
                              temporal_resolution: str = 'monthly',
                              climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate maximum 1-day precipitation (simplified implementation)

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # ── YEARLY PATH: handle before any collection-level .map() ──────────
        # apply_unit_conversion() maps a multiply/add over the ENTIRE raw collection
        # (e.g. 10 958 images).  ee.ImageCollection([img0..img29]) where every img
        # references that mapped collection creates a JSON expression so large that
        # GEE rejects even lightweight metadata calls (aggregate_array, getInfo).
        #
        # Fix: filter the RAW collection per year with concrete string dates, then
        # apply the scaling factor to the single result image.  This is exactly
        # what the GEE Code Editor does and it is trivially light for GEE.
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            # Get scaling without mapping over the full collection
            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_max = (raw_filtered
                            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                            .max())
                # Apply unit conversion to the single aggregated image, not the
                # full collection — keeps the expression tree tiny and fast.
                if scaling_factor != 1.0 or offset != 0.0:
                    year_max = year_max.multiply(scaling_factor).add(offset)
                yearly_images.append(year_max.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'RX1day',
                    'unit': 'mm'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────────
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_max = (raw_filtered
                                 .filterDate(month_start, month_end)
                                 .max())
                    if scaling_factor != 1.0 or offset != 0.0:
                        month_max = month_max.multiply(scaling_factor).add(offset)
                    monthly_images.append(month_max.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'RX1day',
                        'unit': 'mm'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        # Apply unit conversion if dataset is specified
        if self.dataset_id:
            precip_collection = self.apply_unit_conversion(precip_collection, 'precipitation')

        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        # Handle temporal resolution
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            # The spatial export will aggregate to single image based on metadata
            def calculate_yearly_max(year):
                yearly_data = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )
                max_precip = yearly_data.max().clip(self.geometry)
                return max_precip.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'RX1day',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_yearly_max)
            )

            # Add climatology metadata to the collection
            # This tells the export function to aggregate before exporting
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'

            # Return yearly collection with climatology metadata
            # Time series plot will show yearly values (informative)
            # Spatial export will check metadata and aggregate to single image
            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'RX1day',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Monthly aggregation (default)
            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(monthly_max_precip)
            )

            return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats

    def calculate_simple_CDD(self, precip_collection: ee.ImageCollection,
                           start_date: str, end_date: str,
                           threshold: float = 1.0,
                           temporal_resolution: str = 'yearly',
                           climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate consecutive dry days (simplified - counts total dry days)

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            threshold: Precipitation threshold for dry day (default 1.0 mm)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', etc.
            climatology_reducer: 'mean', 'median', 'min', or 'max'
        """
        # ── YEARLY PATH: bypass collection-level .map() (same reason as RX1day) ─
        # For CDD, the threshold must be expressed in the SAME units as the raw
        # collection band (e.g. meters for ERA5-Land total_precipitation_sum).
        # scaling_factor converts raw→user units:  user_value = raw * factor + offset
        # Therefore: raw_threshold = (threshold - offset) / factor
        if temporal_resolution == 'yearly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            # Convert the user-facing threshold (mm) to raw collection units
            raw_threshold = (threshold - offset) / scaling_factor if scaling_factor else threshold

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_coll = raw_filtered.filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                dry_days  = year_coll.map(lambda img: img.lt(raw_threshold))
                total_dry = dry_days.sum()
                yearly_images.append(total_dry.set({
                    'year': year,
                    'system:time_start': ee.Date(f'{year}-01-01').millis(),
                    'index_name': 'CDD',
                    'unit': 'days'
                }))

            return (ee.ImageCollection(yearly_images)
                    if yearly_images else ee.ImageCollection([]))

        # ── MONTHLY PATH: handle before collection-level .map() ───────────────────
        if temporal_resolution == 'monthly':
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            scaling_factor, offset = 1.0, 0.0
            if self.dataset_id:
                band_info = self.dataset_config.get_band_scaling_info(
                    self.dataset_id, 'precipitation')
                scaling_factor = band_info.get('scaling_factor', 1.0)
                offset         = band_info.get('offset', 0.0)

            # Convert threshold to raw units
            raw_threshold = (threshold - offset) / scaling_factor if scaling_factor else threshold

            raw_filtered = (precip_collection
                            .filterDate(start_date, end_date)
                            .filterBounds(self.geometry))

            monthly_images = []
            for year in range(_start.year, _end.year + 1):
                for month in range(1, 13):
                    if year == _start.year and month < _start.month:
                        continue
                    if year == _end.year and month > _end.month:
                        continue
                    
                    if month == 12:
                        month_start = f'{year}-12-01'
                        month_end = f'{year+1}-01-01'
                    else:
                        month_start = f'{year}-{month:02d}-01'
                        month_end = f'{year}-{month+1:02d}-01'
                    
                    month_coll = raw_filtered.filterDate(month_start, month_end)
                    dry_days = month_coll.map(lambda img: img.lt(raw_threshold))
                    total_dry = dry_days.sum()
                    monthly_images.append(total_dry.set({
                        'month': month,
                        'year': year,
                        'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                        'index_name': 'CDD',
                        'unit': 'days'
                    }))

            return (ee.ImageCollection(monthly_images)
                    if monthly_images else ee.ImageCollection([]))

        # ── Other temporal resolutions (climatology, etc.) ───────────────────────
        # Apply unit conversion if dataset is specified
        if self.dataset_id:
            precip_collection = self.apply_unit_conversion(precip_collection, 'precipitation')

        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology mode: Return YEARLY collection for time series plotting
            def calculate_annual_cdd(year):
                annual = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )
                dry_days = annual.map(lambda img: img.lt(threshold).clip(self.geometry))
                total_dry = dry_days.sum()
                return total_dry.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'CDD',
                    'unit': 'days'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_cdd)
            )

            # Add climatology metadata to the collection
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'

            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'CDD',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation — Python loop with concrete dates (see RX1day comment).
            from datetime import datetime as _dt
            _start = _dt.strptime(str(start_date)[:10], '%Y-%m-%d')
            _end   = _dt.strptime(str(end_date)[:10],   '%Y-%m-%d')

            yearly_images = []
            for year in range(_start.year, _end.year + 1):
                year_coll = filtered.filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                # Dry day = precipitation below threshold (binary 0/1 image per day)
                dry_days  = year_coll.map(lambda img: img.lt(threshold))
                total_dry = dry_days.sum()
                yearly_images.append(
                    total_dry.set({
                        'year': year,
                        'system:time_start': ee.Date(f'{year}-01-01').millis(),
                        'index_name': 'CDD',
                        'unit': 'days'
                    })
                )

            if not yearly_images:
                return ee.ImageCollection([])
            return ee.ImageCollection(yearly_images)

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats

    def calculate_simple_PRCPTOT(self, precip_collection: ee.ImageCollection,
                               start_date: str, end_date: str,
                               wet_threshold: float = 1.0,
                               temporal_resolution: str = 'yearly',
                               climatology_reducer: str = 'mean') -> ee.ImageCollection:
        """
        Calculate total precipitation on wet days (simplified implementation)

        Args:
            precip_collection: Daily precipitation collection
            start_date: Start date string
            end_date: End date string
            wet_threshold: Threshold for wet day (default 1.0 mm)
            temporal_resolution: 'monthly', 'yearly', 'climatology_mean', 'climatology_median', 'climatology_min', or 'climatology_max' aggregation
            climatology_reducer: 'mean', 'median', 'min', or 'max' for climatology calculations
        """
        # Apply unit conversion if dataset is specified
        if self.dataset_id:
            precip_collection = self.apply_unit_conversion(precip_collection, 'precipitation')

        filtered = precip_collection.filterDate(start_date, end_date).filterBounds(self.geometry)

        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')

        if temporal_resolution == 'monthly':
            # Monthly aggregation
            def calculate_monthly_prcptot(year_month):
                year = ee.Number(year_month).divide(100).int()
                month = ee.Number(year_month).mod(100).int()

                monthly = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                ).filter(
                    ee.Filter.calendarRange(month, month, 'month')
                )

                # Sum precipitation on wet days only
                wet_total = monthly.map(
                    lambda img: img.updateMask(img.gte(wet_threshold)).clip(self.geometry)
                ).sum()

                return wet_total.set({
                    'month': month,
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, month, 1).millis(),
                    'index_name': 'PRCPTOT',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            months = ee.List.sequence(1, 12)

            year_months = years.map(
                lambda y: months.map(lambda m: ee.Number(y).multiply(100).add(m))
            ).flatten()

            results = ee.ImageCollection.fromImages(
                year_months.map(calculate_monthly_prcptot)
            )
            return results

        elif temporal_resolution in ['climatology_mean', 'climatology_median', 'climatology_min', 'climatology_max']:
            # Climatology: single image representing mean/median across all years
            def calculate_annual_prcptot(year):
                annual = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )
                wet_total = annual.map(
                    lambda img: img.updateMask(img.gte(wet_threshold)).clip(self.geometry)
                ).sum()
                return wet_total.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'PRCPTOT',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)
            yearly_collection = ee.ImageCollection.fromImages(
                years.map(calculate_annual_prcptot)
            )

            # Add climatology metadata to the collection
            if temporal_resolution == 'climatology_median':
                climatology_type = 'median'
            elif temporal_resolution == 'climatology_min':
                climatology_type = 'min'
            elif temporal_resolution == 'climatology_max':
                climatology_type = 'max'
            else:
                climatology_type = 'mean'

            return yearly_collection.set({
                'climatology_mode': True,
                'climatology_type': climatology_type,
                'temporal_resolution': temporal_resolution,
                'index_name': 'PRCPTOT',
                'start_year': start_year,
                'end_year': end_year
            })

        else:
            # Yearly aggregation (default)
            def calculate_annual_prcptot(year):
                annual = filtered.filter(
                    ee.Filter.calendarRange(year, year, 'year')
                )

                # Sum precipitation on wet days only
                wet_total = annual.map(
                    lambda img: img.updateMask(img.gte(wet_threshold)).clip(self.geometry)
                ).sum()

                return wet_total.set({
                    'year': year,
                    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis(),
                    'index_name': 'PRCPTOT',
                    'unit': 'mm'
                })

            years = ee.List.sequence(start_year, end_year)

            results = ee.ImageCollection.fromImages(
                years.map(calculate_annual_prcptot)
            )

            return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats

    def calculate_simple_index(self, index_name: str,
                             collection_dict: Dict[str, ee.ImageCollection],
                             start_date: str, end_date: str,
                             temporal_resolution: str = 'monthly',
                             **kwargs) -> ee.ImageCollection:
        """
        Calculate simple climate indices with automatic unit conversion

        Args:
            index_name: Name of the index (TXx, TNn, DTR, RX1day, CDD, PRCPTOT, and all 14 new indices)
            collection_dict: Dictionary with collections mapped by band type
            start_date: Start date string
            end_date: End date string
            **kwargs: Additional parameters (e.g., thresholds)

        Returns:
            ee.ImageCollection with calculated index
        """
        if index_name == 'TXx':
            return self.calculate_simple_TXx(
                collection_dict.get('temperature_max'), start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'TNn':
            return self.calculate_simple_TNn(
                collection_dict.get('temperature_min'), start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'DTR':
            return self.calculate_simple_DTR(
                collection_dict.get('temperature_max'),
                collection_dict.get('temperature_min'),
                start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'RX1day':
            return self.calculate_simple_RX1day(
                collection_dict.get('precipitation'), start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'CDD':
            threshold = kwargs.get('threshold', 1.0)
            return self.calculate_simple_CDD(
                collection_dict.get('precipitation'), start_date, end_date, threshold,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'PRCPTOT':
            wet_threshold = kwargs.get('wet_threshold', 1.0)
            return self.calculate_simple_PRCPTOT(
                collection_dict.get('precipitation'), start_date, end_date, wet_threshold,
                temporal_resolution=temporal_resolution
            )
        # New percentile-based indices
        elif index_name == 'TX90p':
            return self.calculate_TX90p(
                collection_dict.get('temperature_max'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        elif index_name == 'TX10p':
            return self.calculate_TX10p(
                collection_dict.get('temperature_max'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        elif index_name == 'TN90p':
            return self.calculate_TN90p(
                collection_dict.get('temperature_min'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        elif index_name == 'TN10p':
            return self.calculate_TN10p(
                collection_dict.get('temperature_min'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        elif index_name == 'R95p':
            return self.calculate_R95p(
                collection_dict.get('precipitation'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        elif index_name == 'R99p':
            return self.calculate_R99p(
                collection_dict.get('precipitation'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        elif index_name == 'R75p':
            return self.calculate_R75p(
                collection_dict.get('precipitation'), start_date, end_date,
                temporal_resolution=temporal_resolution, **kwargs
            )
        # New threshold-based indices
        elif index_name == 'TXn':
            return self.calculate_TXn(
                collection_dict.get('temperature_max'), start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'TNx':
            return self.calculate_TNx(
                collection_dict.get('temperature_min'), start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'SU':
            threshold = kwargs.get('threshold', 25.0)
            return self.calculate_SU(
                collection_dict.get('temperature_max'), start_date, end_date, threshold
            )
        elif index_name == 'FD':
            threshold = kwargs.get('threshold', 0.0)
            return self.calculate_FD(
                collection_dict.get('temperature_min'), start_date, end_date, threshold
            )
        elif index_name == 'R10mm':
            return self.calculate_R10mm(
                collection_dict.get('precipitation'), start_date, end_date,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'R20mm':
            threshold = kwargs.get('threshold', 20.0)
            return self.calculate_R20mm(
                collection_dict.get('precipitation'), start_date, end_date, threshold,
                temporal_resolution=temporal_resolution
            )
        elif index_name == 'SDII':
            wet_threshold = kwargs.get('wet_threshold', 1.0)
            return self.calculate_SDII(
                collection_dict.get('precipitation'), start_date, end_date, wet_threshold,
                temporal_resolution=temporal_resolution
            )
        else:
            raise ValueError(f"Simple index '{index_name}' not implemented")

    def calculate_index(self, index_name: str, 
                       tmax_collection: Optional[ee.ImageCollection] = None,
                       tmin_collection: Optional[ee.ImageCollection] = None,
                       precip_collection: Optional[ee.ImageCollection] = None,
                       start_date: str = None,
                       end_date: str = None,
                       **kwargs) -> ee.ImageCollection:
        """
        Calculate specified climate index
        
        Args:
            index_name: Name of the index to calculate
            tmax_collection: Maximum temperature collection
            tmin_collection: Minimum temperature collection
            precip_collection: Precipitation collection
            start_date: Start date for calculation
            end_date: End date for calculation
            **kwargs: Additional parameters for specific indices
            
        Returns:
            ee.ImageCollection with calculated index
        """
        if index_name not in self.indices_metadata:
            raise ValueError(f"Unknown index: {index_name}")
        
        # Temperature indices
        if index_name == 'TXx':
            return self.calculate_TXx(tmax_collection, start_date, end_date)
        elif index_name == 'TNn':
            return self.calculate_TNn(tmin_collection, start_date, end_date)
        elif index_name == 'TX90p':
            return self.calculate_TX90p(tmax_collection, start_date, end_date, **kwargs)
        elif index_name == 'TX10p':
            return self.calculate_TX10p(tmax_collection, start_date, end_date, **kwargs)
        elif index_name == 'TN90p':
            return self.calculate_TN90p(tmin_collection, start_date, end_date, **kwargs)
        elif index_name == 'TN10p':
            return self.calculate_TN10p(tmin_collection, start_date, end_date, **kwargs)
        elif index_name == 'TXn':
            return self.calculate_TXn(tmax_collection, start_date, end_date)
        elif index_name == 'TNx':
            return self.calculate_TNx(tmin_collection, start_date, end_date)
        elif index_name == 'SU':
            return self.calculate_SU(tmax_collection, start_date, end_date, **kwargs)
        elif index_name == 'WSDI':
            return self.calculate_WSDI(tmax_collection, start_date, end_date, **kwargs)
        elif index_name == 'GSL':
            # For GSL, use tmax as mean temperature if tmean not available
            tmean_collection = tmax_collection  # Simplified approach
            threshold = kwargs.get('threshold', 5.0)
            return self.calculate_GSL(tmean_collection, start_date, end_date, threshold)
        elif index_name == 'FD':
            threshold = kwargs.get('threshold', 0.0)
            return self.calculate_FD(tmin_collection, start_date, end_date, threshold)
        elif index_name == 'DTR':
            return self.calculate_DTR(tmax_collection, tmin_collection, start_date, end_date)
        
        # Precipitation indices
        elif index_name == 'RX1day':
            return self.calculate_RX1day(precip_collection, start_date, end_date)
        elif index_name == 'RX5day':
            return self.calculate_RX5day(precip_collection, start_date, end_date)
        elif index_name == 'CDD':
            return self.calculate_CDD(precip_collection, start_date, end_date, **kwargs)
        elif index_name == 'R10mm':
            return self.calculate_R10mm(precip_collection, start_date, end_date)
        elif index_name == 'R20mm':
            return self.calculate_R20mm(precip_collection, start_date, end_date, **kwargs)
        elif index_name == 'R95p':
            return self.calculate_R95p(precip_collection, start_date, end_date, **kwargs)
        elif index_name == 'R99p':
            return self.calculate_R99p(precip_collection, start_date, end_date, **kwargs)
        elif index_name == 'R75p':
            return self.calculate_R75p(precip_collection, start_date, end_date, **kwargs)
        elif index_name == 'SDII':
            return self.calculate_SDII(precip_collection, start_date, end_date, **kwargs)
        elif index_name == 'PRCPTOT':
            return self.calculate_PRCPTOT(precip_collection, start_date, end_date, **kwargs)
        
        else:
            raise NotImplementedError(f"Index {index_name} not yet implemented")
    
    def batch_calculate_indices(self, indices: List[str],
                               tmax_collection: Optional[ee.ImageCollection] = None,
                               tmin_collection: Optional[ee.ImageCollection] = None,
                               precip_collection: Optional[ee.ImageCollection] = None,
                               start_date: str = None,
                               end_date: str = None,
                               **kwargs) -> Dict[str, ee.ImageCollection]:
        """
        Calculate multiple indices at once
        
        Returns:
            Dictionary mapping index names to their results
        """
        results = {}
        
        for index in indices:
            try:
                result = self.calculate_index(
                    index, tmax_collection, tmin_collection, 
                    precip_collection, start_date, end_date, **kwargs
                )
                results[index] = result
            except Exception as e:
                print(f"Error calculating {index}: {str(e)}")
                results[index] = None

        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats

    def aggregate_to_yearly(self, monthly_collection: ee.ImageCollection,
                           index_name: str, aggregation_type: str = 'auto') -> ee.ImageCollection:
        """
        Aggregate monthly climate index values to yearly values

        Args:
            monthly_collection: Monthly climate index collection
            index_name: Name of the climate index for intelligent aggregation
            aggregation_type: Type of aggregation ('auto', 'mean', 'max', 'min', 'sum')

        Returns:
            ee.ImageCollection with yearly aggregated values
        """
        # Determine aggregation method based on index characteristics
        if aggregation_type == 'auto':
            # Use metadata to determine best aggregation method
            index_info = self.indices_metadata.get(index_name, {})

            # Temperature extremes - use max/min
            if index_name in ['TXx', 'TNx', 'RX1day', 'RX5day']:
                aggregation_type = 'max'
            elif index_name in ['TXn', 'TNn']:
                aggregation_type = 'min'
            # Annual totals - use sum
            elif index_name in ['PRCPTOT', 'R10mm', 'R20mm', 'FD', 'WSDI', 'CSDI']:
                aggregation_type = 'sum'
            # Percentile-based indices - use mean
            elif index_name in ['TX90p', 'TN10p', 'TN90p', 'TX10p']:
                aggregation_type = 'mean'
            # Default to mean for other indices
            else:
                aggregation_type = 'mean'

        # Get unique years from the collection
        def get_years(collection):
            def extract_year(img):
                return img.set('year', ee.Date(img.get('system:time_start')).get('year'))
            return collection.map(extract_year).aggregate_array('year').distinct()

        years = get_years(monthly_collection)

        def aggregate_yearly(year):
            year = ee.Number(year)

            # Filter to specific year
            yearly_data = monthly_collection.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )

            # Apply aggregation
            if aggregation_type == 'mean':
                result = yearly_data.mean()
            elif aggregation_type == 'max':
                result = yearly_data.max()
            elif aggregation_type == 'min':
                result = yearly_data.min()
            elif aggregation_type == 'sum':
                result = yearly_data.sum()
            else:
                result = yearly_data.mean()  # fallback

            # Set proper time properties
            return result.set({
                'year': year,
                'index_name': index_name,
                'aggregation': aggregation_type,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })

        # Map over years and return as collection
        yearly_results = ee.ImageCollection(years.map(aggregate_yearly))

        return yearly_results

    def calculate_index_with_temporal_resolution(self, index_name: str,
                                               temporal_resolution: str = 'monthly',
                                               tmax_collection: Optional[ee.ImageCollection] = None,
                                               tmin_collection: Optional[ee.ImageCollection] = None,
                                               precip_collection: Optional[ee.ImageCollection] = None,
                                               start_date: str = None,
                                               end_date: str = None,
                                               **kwargs) -> ee.ImageCollection:
        """
        Calculate climate index with specified temporal resolution

        Args:
            index_name: Name of the index to calculate
            temporal_resolution: 'monthly' or 'yearly'
            Other args: Same as calculate_index

        Returns:
            ee.ImageCollection with requested temporal resolution
        """
        # First calculate at monthly resolution
        monthly_result = self.calculate_index(
            index_name, tmax_collection, tmin_collection,
            precip_collection, start_date, end_date, **kwargs
        )

        # If yearly resolution requested, aggregate monthly results
        if temporal_resolution == 'yearly':
            return self.aggregate_to_yearly(monthly_result, index_name)
        else:
            return monthly_result

    def batch_calculate_indices_with_temporal_resolution(self, indices: List[str],
                                                       temporal_resolution: str = 'monthly',
                                                       tmax_collection: Optional[ee.ImageCollection] = None,
                                                       tmin_collection: Optional[ee.ImageCollection] = None,
                                                       precip_collection: Optional[ee.ImageCollection] = None,
                                                       start_date: str = None,
                                                       end_date: str = None,
                                                       **kwargs) -> Dict[str, ee.ImageCollection]:
        """
        Calculate multiple indices with specified temporal resolution

        Returns:
            Dictionary mapping index names to their results
        """
        results = {}

        for index in indices:
            try:
                result = self.calculate_index_with_temporal_resolution(
                    index, temporal_resolution, tmax_collection, tmin_collection,
                    precip_collection, start_date, end_date, **kwargs
                )
                results[index] = result
            except Exception as e:
                print(f"Error calculating {index}: {str(e)}")
                results[index] = None

        return results

    def calculate_mann_kendall_trend(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Mann-Kendall trend test statistic for time series

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Mann-Kendall statistics (S, tau, p-value approximation)
        """
        # Convert collection to list for pairwise comparisons
        images_list = time_series_collection.sort('system:time_start').toList(time_series_collection.size())
        n = time_series_collection.size()

        def calculate_mk_statistic(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))

            def compare_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))

                # Sign of difference (later - current)
                diff = later_img.subtract(current_img)
                return diff.gt(0).subtract(diff.lt(0))  # +1 if increasing, -1 if decreasing, 0 if equal

            # Compare with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            comparisons = later_indices.map(compare_with_later)

            # Sum all comparisons for this time point
            return ee.ImageCollection(comparisons).sum()

        # Calculate S statistic (sum of all pairwise comparisons)
        all_indices = ee.List.sequence(0, n.subtract(2))
        s_components = all_indices.map(calculate_mk_statistic)
        s_statistic = ee.ImageCollection(s_components).sum()

        # Calculate variance for normal approximation
        # Var(S) = n(n-1)(2n+5)/18 (simplified, ignoring ties)
        variance = n.multiply(n.subtract(1)).multiply(n.multiply(2).add(5)).divide(18)

        # Calculate standardized test statistic Z
        # Z = (S-1)/sqrt(Var(S)) if S > 0, (S+1)/sqrt(Var(S)) if S < 0, 0 if S = 0
        z_stat = s_statistic.where(
            s_statistic.gt(0),
            s_statistic.subtract(1).divide(variance.sqrt())
        ).where(
            s_statistic.lt(0),
            s_statistic.add(1).divide(variance.sqrt())
        ).where(
            s_statistic.eq(0),
            0
        )

        # Calculate Kendall's tau
        tau = s_statistic.multiply(2).divide(n.multiply(n.subtract(1)))

        # Approximate p-value using normal distribution
        # p ≈ 2 * (1 - Φ(|Z|)) where Φ is standard normal CDF
        # For Earth Engine, we'll use a rough approximation
        abs_z = z_stat.abs()
        p_value_approx = abs_z.multiply(-0.5).exp().multiply(2).min(1.0)

        return s_statistic.addBands([tau, z_stat, p_value_approx]).rename([
            'mann_kendall_s', 'kendall_tau', 'z_statistic', 'p_value_approx'
        ])

    def calculate_sens_slope(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Calculate Sen's slope estimator for trend magnitude

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with Sen's slope estimate
        """
        # Convert collection to list and get time stamps
        sorted_collection = time_series_collection.sort('system:time_start')
        images_list = sorted_collection.toList(sorted_collection.size())
        n = sorted_collection.size()

        # Get time stamps for slope calculation
        def get_time_stamps(img):
            return ee.Date(img.get('system:time_start')).millis()

        times_list = sorted_collection.map(get_time_stamps).toList(n)

        def calculate_pairwise_slopes(i):
            i = ee.Number(i)
            current_img = ee.Image(images_list.get(i))
            current_time = ee.Number(times_list.get(i))

            def slope_with_later(j):
                j = ee.Number(j)
                later_img = ee.Image(images_list.get(j))
                later_time = ee.Number(times_list.get(j))

                # Calculate slope: (y2 - y1) / (t2 - t1)
                # Convert time difference from milliseconds to years
                time_diff_years = later_time.subtract(current_time).divide(1000 * 60 * 60 * 24 * 365.25)
                value_diff = later_img.subtract(current_img)

                return value_diff.divide(time_diff_years)

            # Calculate slopes with all later time points
            later_indices = ee.List.sequence(i.add(1), n.subtract(1))
            return later_indices.map(slope_with_later)

        # Get all pairwise slopes
        all_indices = ee.List.sequence(0, n.subtract(2))
        all_slopes_nested = all_indices.map(calculate_pairwise_slopes)

        # Flatten the nested list
        all_slopes = ee.List([]).cat(all_slopes_nested.get(0) or ee.List([]))

        def add_slopes(current_list, slopes_list):
            return ee.List(current_list).cat(slopes_list or ee.List([]))

        remaining_slopes = ee.List.sequence(1, all_slopes_nested.size().subtract(1))
        all_slopes_flat = remaining_slopes.iterate(
            lambda i, acc: add_slopes(acc, all_slopes_nested.get(i)),
            all_slopes
        )

        # Convert to ImageCollection and calculate median (Sen's slope)
        slopes_collection = ee.ImageCollection(all_slopes_flat)
        sens_slope = slopes_collection.median()

        return sens_slope.rename(['sens_slope'])

    def analyze_time_series_trends(self, time_series_collection: ee.ImageCollection) -> ee.Image:
        """
        Perform comprehensive trend analysis on time series data

        Args:
            time_series_collection: Time series of climate index values

        Returns:
            ee.Image with trend statistics including Mann-Kendall and Sen's slope
        """
        # Calculate Mann-Kendall trend test
        mk_stats = self.calculate_mann_kendall_trend(time_series_collection)

        # Calculate Sen's slope
        sens_slope = self.calculate_sens_slope(time_series_collection)

        # Calculate basic statistics
        mean_value = time_series_collection.mean().rename(['mean'])
        std_value = time_series_collection.reduce(ee.Reducer.stdDev()).rename(['std_dev'])
        min_value = time_series_collection.min().rename(['min'])
        max_value = time_series_collection.max().rename(['max'])

        # Calculate linear trend for comparison
        # Use least squares regression
        n = time_series_collection.size()

        # Create time index (0, 1, 2, ..., n-1)
        def add_time_index(img):
            return img.set('time_index', ee.Number(time_series_collection.distance(img)).int())

        indexed_collection = time_series_collection.map(add_time_index)

        # Simple linear regression slope
        def calculate_linear_slope():
            # Get time indices and values
            def extract_data(img):
                time_idx = ee.Number(img.get('time_index'))
                return img.addBands(ee.Image.constant(time_idx).rename('time'))

            data_collection = indexed_collection.map(extract_data)

            # Calculate correlation and linear trend
            regression_result = data_collection.select(['time', ee.String(data_collection.first().bandNames().get(0))]).reduce(
                ee.Reducer.linearRegression(1, 1)
            )

            return regression_result.select('scale').rename(['linear_slope'])

        linear_slope = calculate_linear_slope()

        # Combine all trend statistics
        trend_stats = mk_stats.addBands([
            sens_slope,
            linear_slope,
            mean_value,
            std_value,
            min_value,
            max_value
        ])

        return trend_stats 