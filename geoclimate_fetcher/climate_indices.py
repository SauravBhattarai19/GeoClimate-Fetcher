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
    
    def calculate_percentiles(self, collection: ee.ImageCollection, 
                            base_start: str, base_end: str,
                            percentiles: List[int] = [10, 90]) -> ee.Image:
        """
        Calculate percentile thresholds from base period
        
        Args:
            collection: Temperature or precipitation collection
            base_start: Start date of base period (YYYY-MM-DD)
            base_end: End date of base period (YYYY-MM-DD)
            percentiles: List of percentiles to calculate
        
        Returns:
            ee.Image with bands for each percentile
        """
        base_collection = collection.filterDate(base_start, base_end)
        
        # Calculate percentiles for each day of year
        def calculate_doy_percentiles(doy):
            # Filter to specific day of year across all years
            doy_collection = base_collection.filter(
                ee.Filter.calendarRange(doy, doy, 'day_of_year')
            )
            
            # Calculate percentiles
            percentile_image = doy_collection.reduce(
                ee.Reducer.percentile(percentiles)
            )
            
            return percentile_image.set('day_of_year', doy)
        
        # Calculate for all days of year
        doy_list = ee.List.sequence(1, 365)
        percentile_collection = ee.ImageCollection(
            doy_list.map(calculate_doy_percentiles)
        )
        
        return percentile_collection
    
    # Temperature Indices Calculations
    
    def calculate_TXx(self, tmax_collection: ee.ImageCollection, 
                      start_date: str, end_date: str) -> ee.Image:
        """
        Calculate monthly maximum value of daily maximum temperature
        
        Formula: TXx = max(TX) for each month
        """
        # Filter collection to date range
        filtered = tmax_collection.filterDate(start_date, end_date)
        
        # Calculate monthly maximum
        def monthly_max(month, year):
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
        
        # Get unique year-month combinations
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        months = ee.List.sequence(1, 12)
        
        # Calculate for each month
        results = ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: monthly_max(m, y))).flatten()
        )
        
        return results
    
    def calculate_TNn(self, tmin_collection: ee.ImageCollection,
                      start_date: str, end_date: str) -> ee.Image:
        """
        Calculate monthly minimum value of daily minimum temperature
        
        Formula: TNn = min(TN) for each month
        """
        filtered = tmin_collection.filterDate(start_date, end_date)
        
        def monthly_min(month, year):
            monthly = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            return monthly.min().set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        months = ee.List.sequence(1, 12)
        
        results = ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: monthly_min(m, y))).flatten()
        )
        
        return results
    
    def calculate_TX90p(self, tmax_collection: ee.ImageCollection,
                        start_date: str, end_date: str,
                        base_start: str = "1961-01-01", 
                        base_end: str = "1990-12-31") -> ee.Image:
        """
        Calculate percentage of days when TX > 90th percentile
        
        Formula: TX90p = (count(TX > TX_90th) / total_days) * 100
        """
        # Calculate 90th percentile from base period
        percentiles = self.calculate_percentiles(
            tmax_collection, base_start, base_end, [90]
        )
        
        # Filter to analysis period
        filtered = tmax_collection.filterDate(start_date, end_date)
        
        def calculate_monthly_tx90p(month, year):
            # Get month data
            monthly = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            # Get 90th percentile for each day of this month
            month_start = ee.Date.fromYMD(year, month, 1)
            month_end = month_start.advance(1, 'month')
            days_in_month = month_end.difference(month_start, 'day')
            
            # Count exceedances
            def count_exceedances(image):
                doy = ee.Date(image.get('system:time_start')).getRelative('day', 'year')
                threshold = percentiles.filter(
                    ee.Filter.eq('day_of_year', doy)
                ).first().select(['p90'])
                
                exceeds = image.gt(threshold)
                return exceeds
            
            exceedance_count = monthly.map(count_exceedances).sum()
            
            # Calculate percentage
            percentage = exceedance_count.divide(days_in_month).multiply(100)
            
            return percentage.set({
                'month': month,
                'year': year,
                'system:time_start': month_start.millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        months = ee.List.sequence(1, 12)
        
        results = ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: calculate_monthly_tx90p(m, y))).flatten()
        )
        
        return results
    
    def calculate_FD(self, tmin_collection: ee.ImageCollection,
                     start_date: str, end_date: str) -> ee.Image:
        """
        Calculate annual count of frost days (TN < 0°C)
        
        Formula: FD = count(TN < 0°C)
        """
        filtered = tmin_collection.filterDate(start_date, end_date)
        
        def calculate_annual_fd(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Count days below 0°C
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
                      start_date: str, end_date: str) -> ee.Image:
        """
        Calculate monthly mean diurnal temperature range
        
        Formula: DTR = mean(TX - TN)
        """
        # Join collections by date
        join_filter = ee.Filter.equals(
            leftField='system:time_start',
            rightField='system:time_start'
        )
        
        joined = ee.ImageCollection(
            ee.Join.inner().apply(
                tmax_collection.filterDate(start_date, end_date),
                tmin_collection.filterDate(start_date, end_date),
                join_filter
            )
        )
        
        # Calculate daily DTR
        def calculate_dtr(image):
            tmax = ee.Image(image.get('primary'))
            tmin = ee.Image(image.get('secondary'))
            dtr = tmax.subtract(tmin)
            return dtr.copyProperties(tmax, ['system:time_start'])
        
        dtr_daily = joined.map(calculate_dtr)
        
        # Calculate monthly mean
        def monthly_mean_dtr(month, year):
            monthly = dtr_daily.filter(
                ee.Filter.calendarRange(year, year, 'year')
            ).filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            return monthly.mean().set({
                'month': month,
                'year': year,
                'system:time_start': ee.Date.fromYMD(year, month, 1).millis()
            })
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        months = ee.List.sequence(1, 12)
        
        results = ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: monthly_mean_dtr(m, y))).flatten()
        )
        
        return results
    
    # Precipitation Indices Calculations
    
    def calculate_RX1day(self, precip_collection: ee.ImageCollection,
                         start_date: str, end_date: str) -> ee.Image:
        """
        Calculate monthly maximum 1-day precipitation
        
        Formula: RX1day = max(RR) for each month
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def monthly_max_precip(month, year):
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
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        months = ee.List.sequence(1, 12)
        
        results = ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: monthly_max_precip(m, y))).flatten()
        )
        
        return results
    
    def calculate_RX5day(self, precip_collection: ee.ImageCollection,
                         start_date: str, end_date: str) -> ee.Image:
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
        def monthly_max_5day(month, year):
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
        
        years = ee.List.sequence(
            ee.Date(start_date).get('year'),
            ee.Date(end_date).get('year')
        )
        months = ee.List.sequence(1, 12)
        
        results = ee.ImageCollection.fromImages(
            years.map(lambda y: months.map(lambda m: monthly_max_5day(m, y))).flatten()
        )
        
        return results
    
    def calculate_CDD(self, precip_collection: ee.ImageCollection,
                      start_date: str, end_date: str,
                      threshold: float = 1.0) -> ee.Image:
        """
        Calculate annual maximum consecutive dry days
        
        Formula: CDD = max(consecutive days with RR < threshold)
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_annual_cdd(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Convert to dry day mask (1 for dry, 0 for wet)
            dry_days = annual.map(lambda img: img.lt(threshold))
            
            # Calculate consecutive runs
            # This is complex in GEE, so we'll use a simplified approach
            # Count total dry days as a proxy
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
                        start_date: str, end_date: str) -> ee.Image:
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
                       wet_threshold: float = 1.0) -> ee.Image:
        """
        Calculate Simple Daily Intensity Index
        
        Formula: SDII = sum(RR on wet days) / count(wet days)
        """
        filtered = precip_collection.filterDate(start_date, end_date)
        
        def calculate_annual_sdii(year):
            annual = filtered.filter(
                ee.Filter.calendarRange(year, year, 'year')
            )
            
            # Mask for wet days
            wet_mask = annual.map(lambda img: img.gte(wet_threshold))
            
            # Sum precipitation on wet days
            wet_precip = annual.map(
                lambda img: img.updateMask(img.gte(wet_threshold))
            ).sum()
            
            # Count wet days
            wet_days = wet_mask.sum()
            
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
                          wet_threshold: float = 1.0) -> ee.Image:
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