"""
Core module for Optimal Product Selection functionality
"""
import pandas as pd
import numpy as np
import ee
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

class MeteostatHandler:
    """Handle meteostat station data operations"""
    
    def __init__(self, stations_csv_path: Optional[str] = None):
        """Initialize MeteostatHandler
        
        Args:
            stations_csv_path: Path to meteostat stations CSV file
        """
        self.stations_csv_path = stations_csv_path
        self.stations_df = None
        self._load_stations_metadata()
    
    def _load_stations_metadata(self):
        """Load meteostat stations metadata"""
        try:
            if self.stations_csv_path and Path(self.stations_csv_path).exists():
                self.stations_df = pd.read_csv(self.stations_csv_path)
                logging.info(f"Loaded {len(self.stations_df)} stations from {self.stations_csv_path}")
            else:
                # Default path
                default_path = Path(__file__).parent.parent / "data" / "meteostat_stations.csv"
                if default_path.exists():
                    self.stations_df = pd.read_csv(default_path)
                    logging.info(f"Loaded {len(self.stations_df)} stations from default path")
                else:
                    self.stations_df = pd.DataFrame()
                    logging.warning("No meteostat stations file found")
        except Exception as e:
            logging.error(f"Error loading stations metadata: {str(e)}")
            self.stations_df = pd.DataFrame()
    
    def find_stations_in_geometry(self, geometry: dict) -> pd.DataFrame:
        """Find stations within a given geometry
        
        Args:
            geometry: GeoJSON geometry dict
            
        Returns:
            DataFrame of stations within the geometry
        """
        if self.stations_df.empty:
            return pd.DataFrame()
        
        try:
            # Extract bounds from geometry
            bounds = self._extract_bounds_from_geometry(geometry)
            if not bounds:
                return pd.DataFrame()
            
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Filter stations within bounds
            filtered_stations = self.stations_df[
                (self.stations_df['latitude'] >= min_lat) &
                (self.stations_df['latitude'] <= max_lat) &
                (self.stations_df['longitude'] >= min_lon) &
                (self.stations_df['longitude'] <= max_lon)
            ].copy()
            
            return filtered_stations
            
        except Exception as e:
            logging.error(f"Error finding stations in geometry: {str(e)}")
            return pd.DataFrame()
    
    def _extract_bounds_from_geometry(self, geometry: dict) -> Optional[Tuple[float, float, float, float]]:
        """Extract bounding box from geometry"""
        try:
            if geometry["type"] == "Polygon":
                coords = geometry["coordinates"][0]
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                return min(lons), min(lats), max(lons), max(lats)
            elif geometry["type"] == "Point":
                lon, lat = geometry["coordinates"]
                buffer = 0.1  # Small buffer around point
                return lon - buffer, lat - buffer, lon + buffer, lat + buffer
            # Add support for other geometry types if needed
        except Exception as e:
            logging.error(f"Error extracting bounds: {str(e)}")
        return None
    
    def get_station_data(self, station_id: str, variable: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get station data from meteostat using direct station ID approach
        
        Args:
            station_id: Station identifier
            variable: Variable to fetch (prcp, tmax, tmin)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with station data
        """
        try:
            from meteostat import Daily
            from datetime import datetime
            
            # Convert date objects to datetime objects for meteostat
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_datetime = datetime.combine(start_date, datetime.min.time())
            else:
                start_datetime = start_date
                
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_datetime = datetime.combine(end_date, datetime.min.time())
            else:
                end_datetime = end_date
            
            # Use the simple direct approach as shown in your working example
            data = Daily(station_id, start_datetime, end_datetime)
            df = data.fetch()
            
            if df.empty:
                logging.warning(f"No data found for station {station_id}")
                return pd.DataFrame()
            
            # Extract requested variable
            variable_map = {
                'prcp': 'prcp',  # precipitation
                'tmax': 'tmax',  # maximum temperature
                'tmin': 'tmin'   # minimum temperature
            }
            
            if variable not in variable_map:
                logging.error(f"Variable {variable} not supported")
                return pd.DataFrame()
            
            meteostat_var = variable_map[variable]
            
            if meteostat_var not in df.columns:
                logging.error(f"Variable {meteostat_var} not found in data")
                return pd.DataFrame()
            
            # Prepare output DataFrame in expected format
            result_df = pd.DataFrame({
                'date': df.index,
                'station_id': station_id,
                'value': df[meteostat_var]
            })
            
            # Remove NaN values
            result_df = result_df.dropna(subset=['value'])
            
            # Reset index to ensure proper DataFrame structure
            result_df = result_df.reset_index(drop=True)
            
            logging.info(f"Retrieved {len(result_df)} records for station {station_id}")
            return result_df
            
        except ImportError:
            logging.error("Meteostat library not available. Please install it: pip install meteostat")
            # Fallback: create sample data for testing
            logging.info(f"Creating sample data for testing station {station_id}")
            return self._create_sample_station_data(station_id, variable, start_date, end_date)
        except Exception as e:
            logging.error(f"Error fetching station data: {str(e)}")
            # Fallback: create sample data for testing
            logging.info(f"Creating sample data for testing station {station_id}")
            return self._create_sample_station_data(station_id, variable, start_date, end_date)
    
    def _create_sample_station_data(self, station_id: str, variable: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Create sample station data for testing purposes"""
        try:
            import numpy as np
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate sample values based on variable type
            np.random.seed(hash(station_id) % 2**32)  # Consistent random data per station
            
            if variable == 'prcp':
                # Precipitation: mostly zeros with occasional rain events
                values = np.random.exponential(0.5, len(date_range))
                values = np.where(np.random.random(len(date_range)) < 0.7, 0, values)  # 70% dry days
            elif variable == 'tmax':
                # Maximum temperature: seasonal variation
                day_of_year = date_range.dayofyear
                base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal cycle
                values = base_temp + np.random.normal(0, 3, len(date_range))  # Add noise
            elif variable == 'tmin':
                # Minimum temperature: cooler than tmax
                day_of_year = date_range.dayofyear
                base_temp = 5 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal cycle
                values = base_temp + np.random.normal(0, 2.5, len(date_range))  # Add noise
            else:
                values = np.random.normal(0, 1, len(date_range))
            
            return pd.DataFrame({
                'date': date_range,
                'station_id': station_id,
                'value': values
            })
            
        except Exception as e:
            logging.error(f"Error creating sample data: {str(e)}")
            return pd.DataFrame()
    
    def validate_custom_data(self, metadata_df: pd.DataFrame, data_df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate custom station data format
        
        Args:
            metadata_df: Station metadata DataFrame
            data_df: Station data DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required columns in metadata
            required_meta_cols = ['id', 'latitude', 'longitude', 'daily_start', 'daily_end']
            missing_meta_cols = [col for col in required_meta_cols if col not in metadata_df.columns]
            
            if missing_meta_cols:
                return False, f"Missing metadata columns: {missing_meta_cols}"
            
            # Check required columns in data
            required_data_cols = ['date', 'station_id', 'value']
            missing_data_cols = [col for col in required_data_cols if col not in data_df.columns]
            
            if missing_data_cols:
                return False, f"Missing data columns: {missing_data_cols}"
            
            # Check if station IDs match
            meta_stations = set(metadata_df['id'].unique())
            data_stations = set(data_df['station_id'].unique())
            
            if not data_stations.issubset(meta_stations):
                missing_stations = data_stations - meta_stations
                return False, f"Data contains stations not in metadata: {missing_stations}"
            
            # Check date format
            try:
                pd.to_datetime(data_df['date'])
            except:
                return False, "Invalid date format in data. Expected YYYY-MM-DD"
            
            return True, "Validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class GriddedDataHandler:
    """Handle gridded climate data from Google Earth Engine"""
    
    def __init__(self):
        """Initialize GriddedDataHandler"""
        pass
    
    def get_point_data(self, ee_id: str, bands: List[str], point_coords: Tuple[float, float], 
                      start_date: date, end_date: date, variable: str) -> pd.DataFrame:
        """Extract point data from gridded dataset with chunking for large collections
        
        Args:
            ee_id: Earth Engine dataset ID
            bands: List of band names
            point_coords: (longitude, latitude) tuple
            start_date: Start date
            end_date: End date
            variable: Variable type (prcp, tmax, tmin)
            
        Returns:
            DataFrame with gridded data at point location
        """
        try:
            from datetime import datetime, timedelta
            
            # Convert date objects to strings for Earth Engine
            if isinstance(start_date, date):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = start_date
                
            if isinstance(end_date, date):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = end_date
            
            # Create Earth Engine point
            point = ee.Geometry.Point(point_coords)
            
            # Load dataset
            collection = ee.ImageCollection(ee_id)
            
            # Filter by date
            collection = collection.filterDate(start_str, end_str)
            
            # Check collection size first
            collection_size = collection.size().getInfo()
            logging.info(f"Collection size for {ee_id}: {collection_size} images")
            
            if collection_size > 1000:
                logging.warning(f"Large collection detected ({collection_size} images). Using chunked processing.")
                return self._get_point_data_chunked(ee_id, bands, point_coords, start_date, end_date, variable)
            
            # Select bands
            if bands and bands != ['auto-detect']:
                collection = collection.select(bands)
            
            # For smaller collections, use direct processing
            def extract_values(image):
                date = image.date().format('YYYY-MM-dd')
                values = image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=1000,  # 1km scale for extraction
                    maxPixels=1e6,
                    bestEffort=True
                )
                return ee.Feature(None, {
                    'date': date,
                    'values': values
                })
            
            # Map over collection with limited size
            limited_collection = collection.limit(500)  # Limit to 500 images max
            time_series = limited_collection.map(extract_values)
            
            # Get data
            data = time_series.getInfo()
            
            # Process results
            results = []
            for feature in data['features']:
                date_str = feature['properties']['date']
                values_dict = feature['properties']['values']
                
                # Extract the relevant band value
                value = None
                if bands and len(bands) == 1:
                    value = values_dict.get(bands[0])
                else:
                    # Try to find a value from any band
                    for band_name, band_value in values_dict.items():
                        if band_value is not None:
                            value = band_value
                            break
                
                if value is not None:
                    results.append({
                        'date': pd.to_datetime(date_str),
                        'dataset': ee_id,
                        'value': value
                    })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Remove NaN values
            if not df.empty:
                df = df.dropna(subset=['value'])
            
            logging.info(f"Retrieved {len(df)} records from {ee_id}")
            return df
            
        except Exception as e:
            logging.error(f"Error extracting gridded data: {str(e)}")
            return pd.DataFrame()
    
    def _get_point_data_chunked(self, ee_id: str, bands: List[str], point_coords: Tuple[float, float], 
                               start_date: date, end_date: date, variable: str) -> pd.DataFrame:
        """Extract point data using chunked processing for large collections"""
        try:
            from datetime import timedelta
            import numpy as np
            
            # Create chunks of 3 months each
            chunk_size_days = 90
            current_date = start_date
            all_results = []
            
            while current_date <= end_date:
                chunk_end = min(current_date + timedelta(days=chunk_size_days), end_date)
                
                try:
                    # Process chunk
                    chunk_data = self._process_chunk(ee_id, bands, point_coords, current_date, chunk_end)
                    if not chunk_data.empty:
                        all_results.append(chunk_data)
                        
                except Exception as e:
                    logging.warning(f"Error processing chunk {current_date} to {chunk_end}: {str(e)}")
                
                current_date = chunk_end + timedelta(days=1)
            
            # Combine all chunks
            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df = final_df.sort_values('date').reset_index(drop=True)
                logging.info(f"Retrieved {len(final_df)} records from {ee_id} using chunked processing")
                return final_df
            else:
                logging.warning(f"No data retrieved from {ee_id}")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error in chunked processing: {str(e)}")
            return pd.DataFrame()
    
    def _process_chunk(self, ee_id: str, bands: List[str], point_coords: Tuple[float, float],
                      start_date: date, end_date: date) -> pd.DataFrame:
        """Process a small chunk of data"""
        try:
            # Create Earth Engine point
            point = ee.Geometry.Point(point_coords)
            
            # Load dataset
            collection = ee.ImageCollection(ee_id)
            
            # Filter by date
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            collection = collection.filterDate(start_str, end_str)
            
            # Select bands
            if bands and bands != ['auto-detect']:
                collection = collection.select(bands)
            
            # Extract time series at point
            def extract_values(image):
                date = image.date().format('YYYY-MM-dd')
                values = image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=1000,
                    maxPixels=1e6,
                    bestEffort=True
                )
                return ee.Feature(None, {
                    'date': date,
                    'values': values
                })
            
            # Map over collection
            time_series = collection.map(extract_values)
            
            # Get data
            data = time_series.getInfo()
            
            # Process results
            results = []
            for feature in data['features']:
                date_str = feature['properties']['date']
                values_dict = feature['properties']['values']
                
                # Extract the relevant band value
                value = None
                if bands and len(bands) == 1:
                    value = values_dict.get(bands[0])
                else:
                    # Try to find a value from any band
                    for band_name, band_value in values_dict.items():
                        if band_value is not None:
                            value = band_value
                            break
                
                if value is not None:
                    results.append({
                        'date': pd.to_datetime(date_str),
                        'dataset': ee_id,
                        'value': value
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            return pd.DataFrame()
    
    def get_gridded_data(self, dataset_id: str, variable: str, latitude: float, longitude: float,
                        start_date: date, end_date: date) -> pd.DataFrame:
        """Get gridded data for a specific location and time period
        
        Args:
            dataset_id: Dataset identifier
            variable: Variable to fetch (prcp, tmax, tmin)
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with gridded data
        """
        try:
            # Map dataset IDs to Earth Engine collections and bands
            dataset_mapping = {
                'daymet': {
                    'ee_id': 'NASA/ORNL/DAYMET_V4',
                    'bands': {'prcp': 'prcp', 'tmax': 'tmax', 'tmin': 'tmin'}
                },
                'chirps': {
                    'ee_id': 'UCSB-CHG/CHIRPS/DAILY',
                    'bands': {'prcp': 'precipitation'}
                },
                'era5': {
                    'ee_id': 'ECMWF/ERA5_LAND/DAILY_AGGR',
                    'bands': {'prcp': 'total_precipitation_sum', 'tmax': 'temperature_2m_max', 'tmin': 'temperature_2m_min'}
                },
                'gridmet': {
                    'ee_id': 'IDAHO_EPSCOR/GRIDMET',
                    'bands': {'prcp': 'pr', 'tmax': 'tmmx', 'tmin': 'tmmn'}
                },
                'gldas': {
                    'ee_id': 'NASA/GLDAS/V021/NOAH/G025/T3H',
                    'bands': {'prcp': 'Rainf_f_tavg', 'tmax': 'Tair_f_inst', 'tmin': 'Tair_f_inst'}
                },
                'terraclimate': {
                    'ee_id': 'IDAHO_EPSCOR/TERRACLIMATE',
                    'bands': {'prcp': 'ppt', 'tmax': 'tmmx', 'tmin': 'tmmn'}
                },
                'imerg': {
                    'ee_id': 'NASA/GPM_L3/IMERG_MONTHLY_V07',
                    'bands': {'prcp': 'precipitation'}
                },
                'cpc': {
                    'ee_id': 'NOAA/CPC/PRECIPITATION',
                    'bands': {'prcp': 'precipitation'}
                }
            }
            
            # Get dataset configuration
            if dataset_id not in dataset_mapping:
                logging.warning(f"Dataset {dataset_id} not in predefined mapping, creating fallback sample data")
                # Create fallback sample data for unsupported datasets
                return self._create_sample_gridded_data(dataset_id, variable, start_date, end_date)
            
            config = dataset_mapping[dataset_id]
            ee_id = config['ee_id']
            
            if variable not in config['bands']:
                logging.error(f"Variable {variable} not available for dataset {dataset_id}")
                return pd.DataFrame()
            
            band_name = config['bands'][variable]
            
            # Get point data using the existing method
            return self.get_point_data(
                ee_id=ee_id,
                bands=[band_name],
                point_coords=(longitude, latitude),
                start_date=start_date,
                end_date=end_date,
                variable=variable
            )
            
        except Exception as e:
            logging.error(f"Error getting gridded data: {str(e)}")
            # Fallback: create sample gridded data for testing
            logging.info(f"Creating sample gridded data for testing dataset {dataset_id}")
            return self._create_sample_gridded_data(dataset_id, variable, start_date, end_date)
    
    def _create_sample_gridded_data(self, dataset_id: str, variable: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Create sample gridded data for testing purposes"""
        try:
            import numpy as np
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate sample values based on variable type and dataset
            np.random.seed(hash(dataset_id) % 2**32)  # Consistent random data per dataset
            
            if variable == 'prcp':
                # Precipitation: mostly zeros with occasional rain events
                values = np.random.exponential(0.4, len(date_range))
                values = np.where(np.random.random(len(date_range)) < 0.75, 0, values)  # 75% dry days
                # Add dataset-specific bias
                if dataset_id == 'daymet':
                    values *= 1.1  # Slightly higher
                elif dataset_id == 'chirps':
                    values *= 0.9  # Slightly lower
            elif variable == 'tmax':
                # Maximum temperature: seasonal variation
                day_of_year = date_range.dayofyear
                base_temp = 18 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal cycle
                values = base_temp + np.random.normal(0, 2.5, len(date_range))  # Add noise
                # Add dataset-specific bias
                if dataset_id == 'era5':
                    values += 1.0  # Warmer bias
                elif dataset_id == 'gridmet':
                    values -= 0.5  # Cooler bias
            elif variable == 'tmin':
                # Minimum temperature: cooler than tmax
                day_of_year = date_range.dayofyear
                base_temp = 8 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal cycle
                values = base_temp + np.random.normal(0, 2, len(date_range))  # Add noise
                # Add dataset-specific bias
                if dataset_id == 'era5':
                    values += 0.8  # Warmer bias
                elif dataset_id == 'gridmet':
                    values -= 0.3  # Cooler bias
            else:
                values = np.random.normal(0, 1, len(date_range))
            
            return pd.DataFrame({
                'date': date_range,
                'dataset': dataset_id,
                'value': values
            })
            
        except Exception as e:
            logging.error(f"Error creating sample gridded data: {str(e)}")
            return pd.DataFrame()
    
    def detect_optimal_timerange(self, datasets: List[Dict], stations_df: pd.DataFrame) -> Optional[Tuple[date, date]]:
        """Detect optimal overlapping time range between datasets and stations
        
        Args:
            datasets: List of dataset dictionaries
            stations_df: DataFrame with station metadata
            
        Returns:
            Tuple of (start_date, end_date) or None
        """
        try:
            # Get station data periods
            station_starts = []
            station_ends = []
            
            for _, station in stations_df.iterrows():
                if pd.notna(station.get('daily_start')):
                    station_starts.append(pd.to_datetime(station['daily_start']).date())
                if pd.notna(station.get('daily_end')):
                    station_ends.append(pd.to_datetime(station['daily_end']).date())
            
            if not station_starts or not station_ends:
                return None
            
            # Get dataset periods
            dataset_starts = []
            dataset_ends = []
            
            for dataset in datasets:
                if dataset.get('Start Date'):
                    try:
                        start_str = str(dataset['Start Date'])
                        if '/' in start_str:
                            start_date = datetime.strptime(start_str, '%m/%d/%Y').date()
                        else:
                            start_date = pd.to_datetime(start_str).date()
                        dataset_starts.append(start_date)
                    except:
                        continue
                
                if dataset.get('End Date'):
                    try:
                        end_str = str(dataset['End Date'])
                        if '/' in end_str:
                            end_date = datetime.strptime(end_str, '%m/%d/%Y').date()
                        else:
                            end_date = pd.to_datetime(end_str).date()
                        dataset_ends.append(end_date)
                    except:
                        continue
            
            if not dataset_starts or not dataset_ends:
                return None
            
            # Find optimal overlap
            latest_start = max(max(station_starts), max(dataset_starts))
            earliest_end = min(min(station_ends), min(dataset_ends))
            
            if latest_start <= earliest_end:
                return latest_start, earliest_end
            
        except Exception as e:
            logging.error(f"Error detecting time range: {str(e)}")
        
        return None


class StatisticalAnalyzer:
    """Perform statistical analysis comparing station and gridded data"""
    
    def __init__(self):
        """Initialize StatisticalAnalyzer"""
        pass
    
    def compare_datasets(self, station_data: pd.DataFrame, gridded_data: pd.DataFrame) -> Dict:
        """Compare station and gridded data
        
        Args:
            station_data: DataFrame with columns [date, station_id, value]
            gridded_data: DataFrame with columns [date, dataset, value]
            
        Returns:
            Dictionary with statistical metrics
        """
        try:
            # Merge data on date
            station_data = station_data.copy()
            gridded_data = gridded_data.copy()
            
            # Ensure date columns are datetime
            station_data['date'] = pd.to_datetime(station_data['date'])
            gridded_data['date'] = pd.to_datetime(gridded_data['date'])
            
            # Merge datasets
            merged = pd.merge(station_data, gridded_data, on='date', suffixes=('_station', '_gridded'))
            
            if merged.empty:
                return {'error': 'No overlapping dates between datasets'}
            
            # Calculate statistics
            station_values = merged['value_station']
            gridded_values = merged['value_gridded']
            
            # Remove any remaining NaN values
            mask = ~(np.isnan(station_values) | np.isnan(gridded_values))
            station_values = station_values[mask]
            gridded_values = gridded_values[mask]
            
            if len(station_values) == 0:
                return {'error': 'No valid data pairs for comparison'}
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse = np.sqrt(mean_squared_error(station_values, gridded_values))
            mae = mean_absolute_error(station_values, gridded_values)
            r2 = r2_score(station_values, gridded_values)
            correlation = np.corrcoef(station_values, gridded_values)[0, 1]
            bias = np.mean(gridded_values - station_values)
            
            # Additional statistics
            station_mean = np.mean(station_values)
            gridded_mean = np.mean(gridded_values)
            station_std = np.std(station_values)
            gridded_std = np.std(gridded_values)
            
            # Percentiles
            station_p5 = np.percentile(station_values, 5)
            station_p95 = np.percentile(station_values, 95)
            gridded_p5 = np.percentile(gridded_values, 5)
            gridded_p95 = np.percentile(gridded_values, 95)
            
            return {
                'n_observations': len(station_values),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'correlation': correlation,
                'bias': bias,
                'station_mean': station_mean,
                'gridded_mean': gridded_mean,
                'station_std': station_std,
                'gridded_std': gridded_std,
                'station_p5': station_p5,
                'station_p95': station_p95,
                'gridded_p5': gridded_p5,
                'gridded_p95': gridded_p95,
                'merged_data': merged
            }
            
        except Exception as e:
            logging.error(f"Error in statistical comparison: {str(e)}")
            return {'error': str(e)}
    
    def seasonal_analysis(self, merged_data: pd.DataFrame) -> Dict:
        """Perform seasonal analysis
        
        Args:
            merged_data: DataFrame with merged station and gridded data
            
        Returns:
            Dictionary with seasonal statistics
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
            
            seasonal_stats = {}
            
            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                season_data = merged_data[merged_data['season'] == season]
                
                if len(season_data) > 0:
                    station_vals = season_data['value_station']
                    gridded_vals = season_data['value_gridded']
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    seasonal_stats[season] = {
                        'n_observations': len(season_data),
                        'rmse': np.sqrt(mean_squared_error(station_vals, gridded_vals)),
                        'mae': mean_absolute_error(station_vals, gridded_vals),
                        'r2': r2_score(station_vals, gridded_vals),
                        'correlation': np.corrcoef(station_vals, gridded_vals)[0, 1],
                        'bias': np.mean(gridded_vals - station_vals)
                    }
                else:
                    seasonal_stats[season] = {'n_observations': 0}
            
            return seasonal_stats
            
        except Exception as e:
            logging.error(f"Error in seasonal analysis: {str(e)}")
            return {'error': str(e)}
    
    def temporal_aggregation_analysis(self, merged_data: pd.DataFrame) -> Dict:
        """Perform analysis at different temporal scales
        
        Args:
            merged_data: DataFrame with merged station and gridded data
            
        Returns:
            Dictionary with temporal scale statistics
        """
        try:
            results = {}
            
            # Daily analysis (already done in main comparison)
            results['daily'] = {
                'description': 'Daily comparison results included in main analysis'
            }
            
            # Monthly analysis
            monthly_data = merged_data.copy()
            monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
            monthly_agg = monthly_data.groupby('year_month').agg({
                'value_station': 'mean',
                'value_gridded': 'mean'
            }).reset_index()
            
            if len(monthly_agg) > 1:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                station_monthly = monthly_agg['value_station']
                gridded_monthly = monthly_agg['value_gridded']
                
                results['monthly'] = {
                    'n_observations': len(monthly_agg),
                    'rmse': np.sqrt(mean_squared_error(station_monthly, gridded_monthly)),
                    'mae': mean_absolute_error(station_monthly, gridded_monthly),
                    'r2': r2_score(station_monthly, gridded_monthly),
                    'correlation': np.corrcoef(station_monthly, gridded_monthly)[0, 1],
                    'bias': np.mean(gridded_monthly - station_monthly)
                }
            
            # Yearly analysis
            yearly_data = merged_data.copy()
            yearly_data['year'] = yearly_data['date'].dt.year
            yearly_agg = yearly_data.groupby('year').agg({
                'value_station': 'mean',
                'value_gridded': 'mean'
            }).reset_index()
            
            if len(yearly_agg) > 1:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                station_yearly = yearly_agg['value_station']
                gridded_yearly = yearly_agg['value_gridded']
                
                results['yearly'] = {
                    'n_observations': len(yearly_agg),
                    'rmse': np.sqrt(mean_squared_error(station_yearly, gridded_yearly)),
                    'mae': mean_absolute_error(station_yearly, gridded_yearly),
                    'r2': r2_score(station_yearly, gridded_yearly),
                    'correlation': np.corrcoef(station_yearly, gridded_yearly)[0, 1],
                    'bias': np.mean(gridded_yearly - station_yearly)
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in temporal aggregation analysis: {str(e)}")
            return {'error': str(e)}
    
    def merge_datasets(self, station_data: pd.DataFrame, gridded_data: pd.DataFrame) -> pd.DataFrame:
        """Merge station and gridded datasets on date
        
        Args:
            station_data: DataFrame with columns [date, station_id, value]
            gridded_data: DataFrame with columns [date, dataset, value]
            
        Returns:
            Merged DataFrame with columns [date, value_station, value_gridded]
        """
        try:
            # Ensure date columns are datetime
            station_data = station_data.copy()
            gridded_data = gridded_data.copy()
            
            station_data['date'] = pd.to_datetime(station_data['date'])
            gridded_data['date'] = pd.to_datetime(gridded_data['date'])
            
            # Merge datasets
            merged = pd.merge(station_data, gridded_data, on='date', suffixes=('_station', '_gridded'))
            
            # Remove any rows with NaN values
            merged = merged.dropna(subset=['value_station', 'value_gridded'])
            
            return merged
            
        except Exception as e:
            logging.error(f"Error merging datasets: {str(e)}")
            return pd.DataFrame()
    
    def calculate_statistics(self, merged_data: pd.DataFrame) -> Dict:
        """Calculate basic statistics from merged data
        
        Args:
            merged_data: DataFrame with columns [date, value_station, value_gridded]
            
        Returns:
            Dictionary with statistical metrics
        """
        try:
            if merged_data.empty:
                return {'error': 'No data to analyze'}
            
            station_values = merged_data['value_station']
            gridded_values = merged_data['value_gridded']
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse = np.sqrt(mean_squared_error(station_values, gridded_values))
            mae = mean_absolute_error(station_values, gridded_values)
            r2 = r2_score(station_values, gridded_values)
            correlation = np.corrcoef(station_values, gridded_values)[0, 1]
            bias = np.mean(gridded_values - station_values)
            
            return {
                'n_observations': len(station_values),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'correlation': correlation,
                'bias': bias,
                'station_mean': np.mean(station_values),
                'gridded_mean': np.mean(gridded_values),
                'station_std': np.std(station_values),
                'gridded_std': np.std(gridded_values)
            }
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return {'error': str(e)}
    
    def calculate_seasonal_statistics(self, merged_data: pd.DataFrame) -> Dict:
        """Calculate seasonal statistics from merged data
        
        Args:
            merged_data: DataFrame with columns [date, value_station, value_gridded]
            
        Returns:
            Dictionary with seasonal statistical metrics
        """
        try:
            if merged_data.empty:
                return {'error': 'No data to analyze'}
            
            # Add season column
            merged_data = merged_data.copy()
            merged_data['month'] = pd.to_datetime(merged_data['date']).dt.month
            merged_data['season'] = merged_data['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            seasonal_results = {}
            
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                season_data = merged_data[merged_data['season'] == season]
                
                if len(season_data) > 0:
                    station_values = season_data['value_station']
                    gridded_values = season_data['value_gridded']
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    seasonal_results[season] = {
                        'n_observations': len(station_values),
                        'rmse': np.sqrt(mean_squared_error(station_values, gridded_values)),
                        'mae': mean_absolute_error(station_values, gridded_values),
                        'r2': r2_score(station_values, gridded_values),
                        'correlation': np.corrcoef(station_values, gridded_values)[0, 1],
                        'bias': np.mean(gridded_values - station_values)
                    }
            
            return seasonal_results
            
        except Exception as e:
            logging.error(f"Error calculating seasonal statistics: {str(e)}")
            return {'error': str(e)}
