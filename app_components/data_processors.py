"""
Data Processors for Format Detection and Handling
Automatically detect and process various data formats from different modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import rasterio.plot
from typing import Dict, List, Optional, Tuple, Union, Any
import zipfile
import tempfile
import os
from pathlib import Path
import io
import json
from datetime import datetime
import re


class DataFormatDetector:
    """Detects and processes various data formats"""

    def __init__(self):
        self.supported_csv_formats = {
            'climate_analytics': {
                'required_columns': ['Date', 'Climate_Index_Value'],
                'optional_columns': ['Analysis_Type', 'Dataset'],
                'date_column': 'Date',
                'value_columns': ['Climate_Index_Value']
            },
            'hydrology': {
                'required_columns': ['Date', 'Precipitation'],
                'optional_columns': ['Location', 'Analysis_Method'],
                'date_column': 'Date',
                'value_columns': ['Precipitation']
            },
            'time_series_generic': {
                'required_columns': ['date', 'value'],
                'optional_columns': [],
                'date_column': 'date',
                'value_columns': ['value']
            }
        }

        self.supported_extensions = ['.csv', '.tiff', '.tif', '.zip']

    def detect_file_format(self, uploaded_file) -> Dict[str, Any]:
        """
        Detect the format and type of uploaded file

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Dictionary with format information
        """
        # Handle different file object types (UploadedFile vs BytesIO)
        if hasattr(uploaded_file, 'size'):
            file_size = uploaded_file.size
        else:
            # For BytesIO objects, get size by seeking to end
            current_pos = uploaded_file.tell()
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(current_pos)  # Return to original position

        file_info = {
            'filename': uploaded_file.name,
            'size': file_size,
            'type': None,
            'format': None,
            'valid': False,
            'error': None,
            'metadata': {}
        }

        try:
            # Get file extension
            file_ext = Path(uploaded_file.name).suffix.lower()

            if file_ext not in self.supported_extensions:
                file_info['error'] = f"Unsupported file type: {file_ext}"
                return file_info

            # Reset file pointer
            uploaded_file.seek(0)

            if file_ext == '.csv':
                file_info.update(self._detect_csv_format(uploaded_file))

            elif file_ext in ['.tiff', '.tif']:
                file_info.update(self._detect_tiff_format(uploaded_file))

            elif file_ext == '.zip':
                file_info.update(self._detect_zip_format(uploaded_file))

            return file_info

        except Exception as e:
            file_info['error'] = f"Error detecting format: {str(e)}"
            return file_info

    def _detect_csv_format(self, uploaded_file) -> Dict[str, Any]:
        """Detect CSV format and structure"""
        format_info = {
            'type': 'csv',
            'valid': False,
            'format': 'unknown',
            'metadata': {}
        }

        try:
            # Try to read CSV
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset for later use

            format_info['metadata'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }

            # Detect specific format
            detected_format = self._identify_csv_format(df)
            format_info['format'] = detected_format
            format_info['valid'] = detected_format != 'unknown'

            # Additional metadata for detected format
            if detected_format in self.supported_csv_formats:
                format_config = self.supported_csv_formats[detected_format]
                format_info['metadata']['date_column'] = format_config['date_column']
                format_info['metadata']['value_columns'] = format_config['value_columns']

                # Validate date column
                date_col = format_config['date_column']
                if date_col in df.columns:
                    try:
                        pd.to_datetime(df[date_col])
                        format_info['metadata']['date_valid'] = True
                    except:
                        format_info['metadata']['date_valid'] = False
                        format_info['error'] = f"Date column '{date_col}' contains invalid dates"

            return format_info

        except Exception as e:
            format_info['error'] = f"Error reading CSV: {str(e)}"
            return format_info

    def _detect_tiff_format(self, uploaded_file) -> Dict[str, Any]:
        """Detect TIFF format and metadata"""
        format_info = {
            'type': 'tiff',
            'valid': False,
            'format': 'geotiff',
            'metadata': {}
        }

        try:
            # Save to temporary file for rasterio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            uploaded_file.seek(0)  # Reset for later use

            # Read with rasterio
            with rasterio.open(tmp_file_path) as src:
                # Get dtype from first band, with fallback handling
                try:
                    dtype_str = str(src.dtypes[0]) if src.count > 0 else 'unknown'
                except (IndexError, AttributeError):
                    dtype_str = 'unknown'

                # Get bounds safely
                try:
                    bounds_dict = {
                        'left': float(src.bounds.left),
                        'bottom': float(src.bounds.bottom),
                        'right': float(src.bounds.right),
                        'top': float(src.bounds.top)
                    }
                except (AttributeError, TypeError):
                    bounds_dict = {'left': 0, 'bottom': 0, 'right': 0, 'top': 0}

                format_info['metadata'] = {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': dtype_str,
                    'crs': str(src.crs) if src.crs else 'Unknown',
                    'bounds': bounds_dict,
                    'transform': list(src.transform) if src.transform else [],
                    'nodata': src.nodata,
                    'band_dtypes': [str(dtype) for dtype in src.dtypes] if src.count > 0 else []
                }

                # Read sample data from first band with error handling
                try:
                    if src.count > 0:
                        sample_data = src.read(1, masked=True)
                        # Calculate statistics with NaN handling
                        valid_data = sample_data[~np.isnan(sample_data)]

                        if len(valid_data) > 0:
                            format_info['metadata']['statistics'] = {
                                'min': float(np.min(valid_data)),
                                'max': float(np.max(valid_data)),
                                'mean': float(np.mean(valid_data)),
                                'std': float(np.std(valid_data)),
                                'valid_pixels': len(valid_data),
                                'total_pixels': sample_data.size,
                                'nodata_pixels': sample_data.size - len(valid_data)
                            }
                        else:
                            format_info['metadata']['statistics'] = {
                                'min': None, 'max': None, 'mean': None, 'std': None,
                                'valid_pixels': 0, 'total_pixels': sample_data.size,
                                'nodata_pixels': sample_data.size
                            }
                    else:
                        format_info['metadata']['statistics'] = {
                            'min': None, 'max': None, 'mean': None, 'std': None,
                            'valid_pixels': 0, 'total_pixels': 0, 'nodata_pixels': 0
                        }
                except Exception as stats_error:
                    format_info['metadata']['statistics'] = {
                        'error': f"Could not calculate statistics: {str(stats_error)}"
                    }

            format_info['valid'] = True

            # Clean up temporary file
            os.unlink(tmp_file_path)

            return format_info

        except Exception as e:
            format_info['error'] = f"Error reading TIFF: {str(e)}"
            return format_info

    def _detect_zip_format(self, uploaded_file) -> Dict[str, Any]:
        """Detect ZIP contents"""
        format_info = {
            'type': 'zip',
            'valid': False,
            'format': 'archive',
            'metadata': {}
        }

        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                file_list = zip_ref.namelist()

                # Categorize files
                csv_files = [f for f in file_list if f.endswith('.csv')]
                tiff_files = [f for f in file_list if f.endswith(('.tif', '.tiff'))]
                other_files = [f for f in file_list if not f.endswith(('.csv', '.tif', '.tiff'))]

                format_info['metadata'] = {
                    'total_files': len(file_list),
                    'csv_files': csv_files,
                    'tiff_files': tiff_files,
                    'other_files': other_files,
                    'file_list': file_list
                }

                format_info['valid'] = len(csv_files) > 0 or len(tiff_files) > 0

                # Determine primary format
                if len(tiff_files) > len(csv_files):
                    format_info['format'] = 'spatial_archive'
                elif len(csv_files) > 0:
                    format_info['format'] = 'data_archive'

            uploaded_file.seek(0)  # Reset for later use
            return format_info

        except Exception as e:
            format_info['error'] = f"Error reading ZIP: {str(e)}"
            return format_info

    def _identify_csv_format(self, df: pd.DataFrame) -> str:
        """Identify specific CSV format based on column structure"""
        columns = [col.lower().strip() for col in df.columns]

        # Check for climate analytics format
        if ('date' in columns or 'time' in columns) and 'climate_index_value' in columns:
            return 'climate_analytics'

        # Check for hydrology format
        if ('date' in columns or 'time' in columns) and ('precipitation' in columns or 'precip' in columns):
            return 'hydrology'

        # Check for generic time series
        date_patterns = ['date', 'time', 'datetime', 'timestamp']
        value_patterns = ['value', 'data', 'measurement', 'reading']

        has_date = any(pattern in columns for pattern in date_patterns)
        has_value = any(pattern in columns for pattern in value_patterns)

        if has_date and has_value:
            return 'time_series_generic'

        # Check if it's a general time series by looking for date-like and numeric columns
        date_cols = []
        numeric_cols = []

        for col in df.columns:
            # Check if column contains date-like values
            try:
                pd.to_datetime(df[col].dropna().iloc[:10])  # Test first 10 values
                date_cols.append(col)
            except:
                pass

            # Check if column is numeric
            if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

        if len(date_cols) > 0 and len(numeric_cols) > 0:
            return 'time_series_inferred'

        return 'unknown'

    def process_csv_data(self, uploaded_file, format_info: Dict) -> pd.DataFrame:
        """Process CSV data based on detected format"""
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset

        detected_format = format_info['format']

        if detected_format in self.supported_csv_formats:
            config = self.supported_csv_formats[detected_format]

            # Standardize date column
            date_col = config['date_column']
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)

        elif detected_format == 'time_series_inferred':
            # Auto-detect date and numeric columns
            date_cols = []
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].dropna().iloc[:10])
                    date_cols.append(col)
                except:
                    pass

            if date_cols:
                primary_date_col = date_cols[0]
                df[primary_date_col] = pd.to_datetime(df[primary_date_col])
                df = df.sort_values(primary_date_col)

        return df

    def extract_zip_contents(self, uploaded_file) -> List[Dict]:
        """Extract and process ZIP file contents"""
        extracted_files = []

        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.csv', '.tif', '.tiff')):
                    try:
                        # Extract to memory
                        file_data = zip_ref.read(file_name)

                        # Create a file-like object
                        file_obj = io.BytesIO(file_data)
                        file_obj.name = file_name

                        # Detect format
                        format_info = self.detect_file_format(file_obj)

                        extracted_files.append({
                            'name': file_name,
                            'data': file_obj,
                            'format': format_info
                        })

                    except Exception as e:
                        # More specific error handling for BytesIO issues
                        error_msg = str(e)
                        if "'_io.BytesIO' object has no attribute 'size'" in error_msg:
                            st.warning(f"⚠️ Processing {file_name}: Fixed BytesIO compatibility issue")
                        else:
                            st.warning(f"Could not process {file_name}: {error_msg}")

        uploaded_file.seek(0)  # Reset
        return extracted_files

    def get_column_suggestions(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Suggest appropriate columns for visualization"""
        suggestions = {
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'text_columns': []
        }

        for col in df.columns:
            # Numeric columns - check this FIRST to avoid numeric values being parsed as dates
            if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
                suggestions['numeric_columns'].append(col)

            # Date columns - only check for dates if not already numeric
            elif self._is_likely_date_column(df[col]):
                suggestions['date_columns'].append(col)

            # Categorical columns (limited unique values)
            elif df[col].nunique() < len(df) * 0.5 and df[col].nunique() < 20:
                suggestions['categorical_columns'].append(col)

            # Text columns
            else:
                suggestions['text_columns'].append(col)

        return suggestions

    def _is_likely_date_column(self, series: pd.Series) -> bool:
        """Check if a series is likely to contain dates, with better logic"""
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(series):
            return False

        # Check column name patterns
        col_name = series.name.lower() if series.name else ''
        date_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day']
        if any(keyword in col_name for keyword in date_keywords):
            # Additional check: try parsing a sample
            try:
                sample = series.dropna().iloc[:5]
                pd.to_datetime(sample)
                return True
            except:
                return False

        # For other columns, be more restrictive about date parsing
        # Only consider it a date if it looks like a date format (not pure numbers)
        try:
            sample = series.dropna().iloc[:5]
            if sample.empty:
                return False

            # Check if values look like date strings (contain separators like -, /, etc.)
            sample_str = str(sample.iloc[0]) if len(sample) > 0 else ''
            has_date_separators = any(sep in sample_str for sep in ['-', '/', ' ', ':'])

            if has_date_separators:
                pd.to_datetime(sample)
                return True
            return False
        except:
            return False

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and provide suggestions"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issues': [],
            'suggestions': [],
            'quality_score': 100
        }

        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.2]

        if len(high_missing) > 0:
            quality_report['issues'].append(f"High missing values in columns: {high_missing.index.tolist()}")
            quality_report['suggestions'].append("Consider imputing or removing columns with >20% missing values")
            quality_report['quality_score'] -= 15

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_report['issues'].append(f"Found {duplicates} duplicate rows")
            quality_report['suggestions'].append("Remove duplicate rows for cleaner analysis")
            quality_report['quality_score'] -= 10

        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_report['issues'].append(f"Constant columns: {constant_cols}")
            quality_report['suggestions'].append("Remove constant columns as they don't provide information")
            quality_report['quality_score'] -= 5

        # Check data ranges for outliers (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)]

            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                quality_report['issues'].append(f"Potential outliers in {col}: {len(outliers)} values")
                quality_report['suggestions'].append(f"Review outliers in {col} column")
                quality_report['quality_score'] -= 5

        return quality_report


# Global instance for easy access
data_detector = DataFormatDetector()