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


class ClimateIndicesCalculator:
    """
    Calculates climate indices based on ETCCDI standards
    All calculations are performed server-side using Google Earth Engine
    """
    
    def __init__(self, geometry: ee.Geometry):
        """
        Initialize the calculator with area of interest
        
        Args:
            geometry: ee.Geometry object defining the area of interest
        """
        self.geometry = geometry
        self.indices_metadata = self._get_indices_metadata()
    
    def _get_indices_metadata(self) -> Dict:
        """Define metadata for all available indices"""
        return {
            # Temperature indices
            'TXx': {
                'name': 'Max Tmax',
                'description': 'Monthly maximum value of daily maximum temperature',
                'unit': '°C',
                'category': 'temperature'
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
    
    # Temperature Indices Calculations
    
    def calculate_TXx(self, tmax_collection: ee.ImageCollection, 
                      start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate monthly maximum value of daily maximum temperature
        
        Formula: TXx = max(TX) for each month
        """
        # Filter collection to date range
        filtered = tmax_collection.filterDate(start_date, end_date)
        
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
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })
        
        # Create year-month sequence for ALL months in the date range
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
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
    
    def calculate_TNn(self, tmin_collection: ee.ImageCollection, 
                      start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate monthly minimum value of daily minimum temperature
        
        Formula: TNn = min(TN) for each month
        """
        # Filter collection to date range
        filtered = tmin_collection.filterDate(start_date, end_date)
        
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
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })
        
        # Create year-month sequence for ALL months in the date range
        start_year = ee.Date(start_date).get('year')
        end_year = ee.Date(end_date).get('year')
        
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
    
    def calculate_TX90p(self, tmax_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1981-01-01", 
                        base_end: str = "2010-12-31") -> ee.ImageCollection:
        """
        Calculate percentage of days when TX > 90th percentile (simplified approach)
        
        Formula: TX90p = (count(TX > TX_90th) / total_days) * 100
        """
        # Calculate 90th percentile from base period (simplified)
        base_collection = tmax_collection.filterDate(base_start, base_end)
        percentile_90 = self.calculate_simple_percentile(base_collection, 90)
        
        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date)
        
        def calculate_monthly_tx90p(month_year):
            month = ee.Number(month_year).int()
            year = month.divide(100).int()
            month = month.mod(100)
            
            # Get month data
            monthly = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            # Count exceedances and total days
            exceedances = monthly.map(lambda img: img.gt(percentile_90))
            exceedance_count = exceedances.sum()
            total_days = monthly.size()
            
            # Calculate percentage
            percentage = exceedance_count.divide(total_days).multiply(100)
            
            return percentage.set({
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
            year_months.map(calculate_monthly_tx90p)
        )
        
        return results
    
    def calculate_TN10p(self, tmin_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1981-01-01", 
                        base_end: str = "2010-12-31") -> ee.ImageCollection:
        """
        Calculate percentage of days when TN < 10th percentile
        
        Formula: TN10p = (count(TN < TN_10th) / total_days) * 100
        """
        # Calculate 10th percentile from base period
        base_collection = tmin_collection.filterDate(base_start, base_end)
        percentile_10 = self.calculate_simple_percentile(base_collection, 10)
        
        # Filter to analysis period
        filtered = tmin_collection.filterDate(start_date, end_date)
        
        def calculate_monthly_tn10p(month_year):
            month = ee.Number(month_year).int()
            year = month.divide(100).int()
            month = month.mod(100)
            
            # Get month data
            monthly = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            # Count exceedances and total days
            exceedances = monthly.map(lambda img: img.lt(percentile_10))
            exceedance_count = exceedances.sum()
            total_days = monthly.size()
            
            # Calculate percentage
            percentage = exceedance_count.divide(total_days).multiply(100)
            
            return percentage.set({
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
            year_months.map(calculate_monthly_tn10p)
        )
        
        return results
    
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
    
    def calculate_GSL(self, tmean_collection: ee.ImageCollection,
                      start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate Growing Season Length (simplified)
        Annual count between first span of 6 days with TG > 5°C and first span after July 1 of 6 days with TG < 5°C
        
        Note: This is a simplified implementation counting days above 5°C instead of identifying specific spans
        """
        # Filter to analysis period
        filtered = tmean_collection.filterDate(start_date, end_date)
        
        def calculate_annual_gsl(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Count days above 5°C (273.15 + 5 = 278.15K) - simplified approach
            growing_days = annual.map(lambda img: img.gt(278.15))
            growing_day_count = growing_days.sum()
            
            return growing_day_count.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_gsl)
        )
        
        return results
    
    def calculate_FD(self, tmin_collection: ee.ImageCollection,
                     start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate annual count of frost days (TN < 0°C)
        
        Formula: FD = count(TN < 0°C)
        """
        filtered = tmin_collection.filterDate(start_date, end_date)
        
        def calculate_annual_fd(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Count days below 0°C (273.15K)
            frost_days = annual.map(lambda img: img.lt(273.15)).sum()
            
            return frost_days.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_fd)
        )
        
        return results
    
    def calculate_DTR(self, tmax_collection: ee.ImageCollection,
                      tmin_collection: ee.ImageCollection,
                      start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate monthly mean diurnal temperature range (simplified approach)
        
        Formula: DTR = mean(TX - TN)
        """
        # Filter collections to date range
        tmax_filtered = tmax_collection.filterDate(start_date, end_date)
        tmin_filtered = tmin_collection.filterDate(start_date, end_date)
        
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
            year_months.map(calculate_monthly_dtr)
        )
        
        return results
    
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
    
    def calculate_CDD(self, precip_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      threshold: float = 1.0) -> ee.ImageCollection:
        """
        Calculate annual maximum consecutive dry days (simplified)
        
        Formula: CDD = max(consecutive days with RR < threshold)
        Note: This is simplified to count total dry days instead of consecutive runs
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_annual_cdd(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Count dry days (simplified approach)
            dry_days = annual.map(lambda img: img.lt(threshold))
            total_dry = dry_days.sum()
            
            return total_dry.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_cdd)
        )
        
        return results
    
    def calculate_R10mm(self, precip_collection: ee.ImageCollection,
                        start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Calculate annual count of heavy precipitation days (≥ 10mm)
        
        Formula: R10mm = count(RR ≥ 10mm)
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_annual_r10(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Count days >= 10mm
            heavy_days = annual.map(lambda img: img.gte(10)).sum()
            
            return heavy_days.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_r10)
        )
        
        return results
    
    def calculate_SDII(self, precip_collection: ee.ImageCollection,
                       start_date: str, end_date: str,
                       wet_threshold: float = 1.0) -> ee.ImageCollection:
        """
        Calculate Simple Daily Intensity Index
        
        Formula: SDII = sum(RR on wet days) / count(wet days)
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_annual_sdii(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Sum precipitation on wet days
            wet_precip = annual.map(
                lambda img: img.updateMask(img.gte(wet_threshold))
            ).sum()
            
            # Count wet days
            wet_days = annual.map(lambda img: img.gte(wet_threshold)).sum()
            
            # Calculate SDII
            sdii = wet_precip.divide(wet_days)
            
            return sdii.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_sdii)
        )
        
        return results
    
    def calculate_PRCPTOT(self, precip_collection: ee.ImageCollection,
                          start_date: str, end_date: str,
                          wet_threshold: float = 1.0) -> ee.ImageCollection:
        """
        Calculate annual total precipitation on wet days
        
        Formula: PRCPTOT = sum(RR on days where RR ≥ wet_threshold)
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_annual_prcptot(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Sum precipitation on wet days only
            wet_total = annual.map(
                lambda img: img.updateMask(img.gte(wet_threshold))
            ).sum()
            
            return wet_total.set({
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        
        results = ee.ImageCollection.fromImages(
            years.map(calculate_annual_prcptot)
        )
        
        return results
    
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
                                    scale: int = 5000, max_pixels: int = 1e6) -> pd.DataFrame:
        """
        Extract time series data ensuring ALL data points are processed
        
        Args:
            image_collection: Collection with calculated indices
            scale: Scale in meters for reduction
            max_pixels: Maximum pixels to process
            
        Returns:
            pandas DataFrame with time series
        """
        try:
            # Get collection size 
            collection_size = image_collection.size().getInfo()
            print(f"Processing ALL {collection_size} images in climate index extraction...")
            
            def reduce_image(image):
                """Extract date and average values from an image."""
                # Get the timestamp
                date = ee.Date(image.get('system:time_start'))
                date_string = date.format('YYYY-MM-dd')
                
                # Get the average value 
                reduced = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.geometry,
                    scale=scale,
                    maxPixels=max_pixels,
                    bestEffort=True,
                    tileScale=2
                )
                
                return ee.Feature(None, {'date': date_string, 'value': reduced})
            
            # Map over the entire collection
            features = image_collection.map(reduce_image)
            
            print("Extracting ALL time series data...")
            
            # Get the data from Earth Engine
            feature_collection_data = features.getInfo()
            
            # Process the results
            records = []
            for feature in feature_collection_data['features']:
                props = feature['properties']
                
                # Extract the value (it might be nested in the 'value' dict)
                value_dict = props.get('value', {})
                
                # Get the first non-null numeric value
                actual_value = None
                for key, val in value_dict.items():
                    if val is not None and isinstance(val, (int, float)):
                        actual_value = val
                        break
                
                if actual_value is not None:
                    records.append({
                        'date': props['date'],
                        'value': actual_value
                    })
            
            # Convert to DataFrame
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                print(f"Successfully extracted {len(df)} data points")
                return df
            else:
                print("No valid data points extracted")
                return pd.DataFrame(columns=['date', 'value'])
                
        except Exception as e:
            print(f"Error in optimized extraction: {str(e)}")
            # Fallback to regular extraction
            return self.extract_time_series(image_collection, scale)
    
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
        elif index_name == 'TN10p':
            return self.calculate_TN10p(tmin_collection, start_date, end_date, **kwargs)
        elif index_name == 'WSDI':
            return self.calculate_WSDI(tmax_collection, start_date, end_date, **kwargs)
        elif index_name == 'GSL':
            # For GSL, use tmax as mean temperature if tmean not available
            tmean_collection = tmax_collection  # Simplified approach
            return self.calculate_GSL(tmean_collection, start_date, end_date)
        elif index_name == 'FD':
            return self.calculate_FD(tmin_collection, start_date, end_date)
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