"""
Data Manager for Optimal Product Selector
Handles dataset configurations and data download functionality
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
import zipfile
import io
import os
from datetime import datetime
import logging

class DataManager:
    """Manage data operations for product selector"""
    
    def __init__(self):
        """Initialize data manager"""
        self.dataset_configs = self._get_dataset_configs()
    
    def _get_dataset_configs(self) -> Dict:
        """Get dataset configurations
        
        Returns:
            Dictionary with dataset configurations
        """
        return {
            'daymet': {
                'name': 'Daymet V4 Daily Meteorology',
                'variables': {
                    'prcp': {
                        'name': 'Precipitation',
                        'units': 'mm/day',
                        'description': 'Daily total precipitation'
                    },
                    'tmax': {
                        'name': 'Maximum Temperature',
                        'units': '°C',
                        'description': 'Daily maximum temperature'
                    },
                    'tmin': {
                        'name': 'Minimum Temperature',
                        'units': '°C',
                        'description': 'Daily minimum temperature'
                    }
                },
                'spatial_resolution': '1 km',
                'temporal_coverage': '1980-2023',
                'region': 'North America',
                'ee_collection': 'NASA/ORNL/DAYMET_V4'
            },
            'chirps': {
                'name': 'CHIRPS Daily Precipitation',
                'variables': {
                    'prcp': {
                        'name': 'Precipitation',
                        'units': 'mm/day',
                        'description': 'Daily precipitation'
                    }
                },
                'spatial_resolution': '5.5 km',
                'temporal_coverage': '1981-2023',
                'region': 'Global (50°N-50°S)',
                'ee_collection': 'UCSB-CHG/CHIRPS/DAILY'
            },
            'era5': {
                'name': 'ERA5-Land Hourly',
                'variables': {
                    'prcp': {
                        'name': 'Total Precipitation',
                        'units': 'mm/day',
                        'description': 'Daily total precipitation (aggregated)'
                    },
                    'tmax': {
                        'name': 'Maximum Temperature',
                        'units': '°C',
                        'description': 'Daily maximum 2m temperature'
                    },
                    'tmin': {
                        'name': 'Minimum Temperature',
                        'units': '°C',
                        'description': 'Daily minimum 2m temperature'
                    }
                },
                'spatial_resolution': '11 km',
                'temporal_coverage': '1950-2023',
                'region': 'Global',
                'ee_collection': 'ECMWF/ERA5_LAND/DAILY_AGGR'
            },
            'gridmet': {
                'name': 'gridMET',
                'variables': {
                    'prcp': {
                        'name': 'Precipitation',
                        'units': 'mm',
                        'description': 'Daily precipitation accumulation'
                    },
                    'tmax': {
                        'name': 'Maximum Temperature',
                        'units': '°C',
                        'description': 'Daily maximum temperature'
                    },
                    'tmin': {
                        'name': 'Minimum Temperature',
                        'units': '°C',
                        'description': 'Daily minimum temperature'
                    }
                },
                'spatial_resolution': '4 km',
                'temporal_coverage': '1979-2023',
                'region': 'CONUS',
                'ee_collection': 'IDAHO_EPSCOR/GRIDMET'
            },
            'gldas': {
                'name': 'GLDAS-2.1',
                'variables': {
                    'prcp': {
                        'name': 'Precipitation Rate',
                        'units': 'mm/day',
                        'description': 'Precipitation rate (daily aggregated)'
                    },
                    'tmax': {
                        'name': 'Air Temperature',
                        'units': '°C',
                        'description': 'Air temperature at 2m'
                    }
                },
                'spatial_resolution': '25 km',
                'temporal_coverage': '2000-2023',
                'region': 'Global',
                'ee_collection': 'NASA/GLDAS/V021/NOAH/G025/T3H'
            }
        }
    
    def get_available_datasets(self, variable: str = None) -> List[str]:
        """Get list of available datasets
        
        Args:
            variable: Optional variable to filter datasets
            
        Returns:
            List of dataset names
        """
        if variable:
            return [
                dataset_id for dataset_id, config in self.dataset_configs.items()
                if variable in config['variables']
            ]
        return list(self.dataset_configs.keys())
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get dataset information
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset configuration dictionary
        """
        return self.dataset_configs.get(dataset_id, {})
    
    def get_variable_info(self, dataset_id: str, variable: str) -> Dict:
        """Get variable information for a dataset
        
        Args:
            dataset_id: Dataset identifier
            variable: Variable name
            
        Returns:
            Variable configuration dictionary
        """
        dataset_config = self.dataset_configs.get(dataset_id, {})
        variables = dataset_config.get('variables', {})
        return variables.get(variable, {})
    
    def create_analysis_summary_df(self, all_results: Dict, variable: str) -> pd.DataFrame:
        """Create summary DataFrame from analysis results
        
        Args:
            all_results: Dictionary with all analysis results
            variable: Variable name
            
        Returns:
            Summary DataFrame
        """
        try:
            summary_data = []
            
            for station_id, station_results in all_results.items():
                for dataset_name, dataset_results in station_results.items():
                    if 'stats' in dataset_results:
                        stats = dataset_results['stats']
                        row = {
                            'Station_ID': station_id,
                            'Dataset': dataset_name,
                            'Variable': variable,
                            'N_Observations': stats.get('n_observations', 0),
                            'RMSE': stats.get('rmse', np.nan),
                            'MAE': stats.get('mae', np.nan),
                            'R_Squared': stats.get('r2', np.nan),
                            'Correlation': stats.get('correlation', np.nan),
                            'Bias': stats.get('bias', np.nan),
                            'Station_Mean': stats.get('station_mean', np.nan),
                            'Gridded_Mean': stats.get('gridded_mean', np.nan),
                            'Station_StdDev': stats.get('station_std', np.nan),
                            'Gridded_StdDev': stats.get('gridded_std', np.nan)
                        }
                        
                        # Add seasonal stats if available
                        if 'seasonal_stats' in dataset_results:
                            seasonal = dataset_results['seasonal_stats']
                            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                                season_data = seasonal.get(season, {})
                                row.update({
                                    f'{season}_RMSE': season_data.get('rmse', np.nan),
                                    f'{season}_R_Squared': season_data.get('r2', np.nan),
                                    f'{season}_Correlation': season_data.get('correlation', np.nan),
                                    f'{season}_Bias': season_data.get('bias', np.nan)
                                })
                        
                        summary_data.append(row)
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logging.error(f"Error creating summary DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def create_station_summary_df(self, stations_data: Dict, all_results: Dict) -> pd.DataFrame:
        """Create station summary DataFrame
        
        Args:
            stations_data: Dictionary with station information
            all_results: Dictionary with analysis results
            
        Returns:
            Station summary DataFrame
        """
        try:
            station_summary = []
            
            for station_id, station_info in stations_data.items():
                # Find best performing dataset
                best_dataset = 'None'
                best_r2 = -1
                best_rmse = np.inf
                
                if station_id in all_results:
                    for dataset_name, results in all_results[station_id].items():
                        if 'stats' in results:
                            r2 = results['stats'].get('r2', -1)
                            rmse = results['stats'].get('rmse', np.inf)
                            
                            # Use R² as primary metric, RMSE as secondary
                            if r2 > best_r2 or (r2 == best_r2 and rmse < best_rmse):
                                best_r2 = r2
                                best_rmse = rmse
                                best_dataset = dataset_name
                
                row = {
                    'Station_ID': station_id,
                    'Station_Name': station_info.get('name', 'Unknown'),
                    'Latitude': station_info.get('latitude', np.nan),
                    'Longitude': station_info.get('longitude', np.nan),
                    'Elevation': station_info.get('elevation', np.nan),
                    'Best_Dataset': best_dataset,
                    'Best_R_Squared': best_r2 if best_r2 > -1 else np.nan,
                    'Best_RMSE': best_rmse if best_rmse < np.inf else np.nan,
                    'Datasets_Analyzed': len(all_results.get(station_id, {}))
                }
                
                station_summary.append(row)
            
            return pd.DataFrame(station_summary)
            
        except Exception as e:
            logging.error(f"Error creating station summary DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def create_detailed_data_df(self, all_results: Dict, include_time_series: bool = False) -> pd.DataFrame:
        """Create detailed data DataFrame with all observations
        
        Args:
            all_results: Dictionary with analysis results
            include_time_series: Whether to include time series data
            
        Returns:
            Detailed data DataFrame
        """
        try:
            if not include_time_series:
                return pd.DataFrame()
            
            detailed_data = []
            
            for station_id, station_results in all_results.items():
                for dataset_name, dataset_results in station_results.items():
                    if 'merged_data' in dataset_results:
                        merged_data = dataset_results['merged_data']
                        
                        for _, row in merged_data.iterrows():
                            detailed_data.append({
                                'Station_ID': station_id,
                                'Dataset': dataset_name,
                                'Date': row['date'],
                                'Station_Value': row['value_station'],
                                'Gridded_Value': row['value_gridded'],
                                'Difference': row['value_station'] - row['value_gridded'],
                                'Absolute_Difference': abs(row['value_station'] - row['value_gridded'])
                            })
            
            return pd.DataFrame(detailed_data)
            
        except Exception as e:
            logging.error(f"Error creating detailed data DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def prepare_download_data(self, all_results: Dict, stations_data: Dict, variable: str, 
                            include_time_series: bool = False) -> Dict[str, pd.DataFrame]:
        """Prepare data for download
        
        Args:
            all_results: Dictionary with analysis results
            stations_data: Dictionary with station information
            variable: Variable name
            include_time_series: Whether to include time series data
            
        Returns:
            Dictionary with DataFrames for download
        """
        try:
            download_data = {}
            
            # Analysis summary
            summary_df = self.create_analysis_summary_df(all_results, variable)
            if not summary_df.empty:
                download_data['analysis_summary'] = summary_df
            
            # Station summary
            station_summary_df = self.create_station_summary_df(stations_data, all_results)
            if not station_summary_df.empty:
                download_data['station_summary'] = station_summary_df
            
            # Detailed time series data (optional)
            if include_time_series:
                detailed_df = self.create_detailed_data_df(all_results, include_time_series=True)
                if not detailed_df.empty:
                    download_data['detailed_time_series'] = detailed_df
            
            # Dataset information
            dataset_info_data = []
            for dataset_id in self.get_available_datasets():
                config = self.get_dataset_info(dataset_id)
                dataset_info_data.append({
                    'Dataset_ID': dataset_id,
                    'Dataset_Name': config.get('name', 'Unknown'),
                    'Spatial_Resolution': config.get('spatial_resolution', 'Unknown'),
                    'Temporal_Coverage': config.get('temporal_coverage', 'Unknown'),
                    'Region': config.get('region', 'Unknown'),
                    'Variables': ', '.join(config.get('variables', {}).keys())
                })
            
            if dataset_info_data:
                download_data['dataset_information'] = pd.DataFrame(dataset_info_data)
            
            return download_data
            
        except Exception as e:
            logging.error(f"Error preparing download data: {str(e)}")
            return {}
    
    def create_download_zip(self, download_data: Dict[str, pd.DataFrame], 
                           filename_prefix: str = "optimal_product_analysis") -> bytes:
        """Create ZIP file with analysis results
        
        Args:
            download_data: Dictionary with DataFrames to include
            filename_prefix: Prefix for filenames
            
        Returns:
            ZIP file as bytes
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create in-memory ZIP
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                for file_key, df in download_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Create CSV content
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_content = csv_buffer.getvalue()
                        
                        # Add to ZIP
                        filename = f"{filename_prefix}_{file_key}_{timestamp}.csv"
                        zip_file.writestr(filename, csv_content)
                
                # Add metadata file
                metadata = {
                    'Generated': datetime.now().isoformat(),
                    'Tool': 'Optimal Product Selector',
                    'Files_Included': list(download_data.keys()),
                    'Description': 'Statistical analysis comparing meteostat station data with gridded climate datasets'
                }
                
                metadata_content = '\n'.join([f"{key}: {value}" for key, value in metadata.items()])
                zip_file.writestr(f"{filename_prefix}_metadata_{timestamp}.txt", metadata_content)
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
            
        except Exception as e:
            logging.error(f"Error creating download ZIP: {str(e)}")
            return b''
    
    def get_download_filename(self, variable: str, num_stations: int, num_datasets: int) -> str:
        """Generate download filename
        
        Args:
            variable: Variable name
            num_stations: Number of stations analyzed
            num_datasets: Number of datasets compared
            
        Returns:
            Filename string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"optimal_product_analysis_{variable}_{num_stations}stations_{num_datasets}datasets_{timestamp}.zip"
    
    def validate_analysis_results(self, all_results: Dict) -> Tuple[bool, str]:
        """Validate analysis results before download
        
        Args:
            all_results: Dictionary with analysis results
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not all_results:
                return False, "No analysis results available"
            
            # Check if any results have valid statistics
            has_valid_results = False
            
            for station_id, station_results in all_results.items():
                for dataset_name, dataset_results in station_results.items():
                    if 'stats' in dataset_results and dataset_results['stats']:
                        has_valid_results = True
                        break
                
                if has_valid_results:
                    break
            
            if not has_valid_results:
                return False, "No valid statistical results found"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
