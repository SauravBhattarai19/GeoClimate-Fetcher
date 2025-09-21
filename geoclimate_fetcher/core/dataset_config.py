"""
Dataset Configuration Utilities
Handles loading and processing of datasets.json configuration file
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, date


class DatasetConfig:
    """Handles dataset configuration from JSON file"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with dataset configuration

        Args:
            config_path: Path to datasets.json file. If None, uses default location.
        """
        if config_path is None:
            # Default location
            base_path = Path(__file__).parent.parent / "data" / "datasets.json"
        else:
            base_path = Path(config_path)

        self.config_path = base_path
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset configuration not found at: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def get_datasets_for_analysis(self, analysis_type: str) -> Dict[str, Dict]:
        """
        Get all datasets that support the specified analysis type

        Args:
            analysis_type: 'temperature' or 'precipitation'

        Returns:
            Dict mapping dataset IDs to their configurations
        """
        filtered_datasets = {}

        for dataset_id, dataset_info in self._config['datasets'].items():
            if analysis_type in dataset_info.get('supports_analysis', []):
                filtered_datasets[dataset_id] = dataset_info

        return filtered_datasets

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get configuration for a specific dataset"""
        return self._config['datasets'].get(dataset_id)

    def get_band_info(self, dataset_id: str, band_type: str) -> Optional[Dict]:
        """
        Get band information for a specific dataset and band type

        Args:
            dataset_id: Earth Engine dataset ID
            band_type: 'temperature_max', 'temperature_min', 'precipitation', etc.

        Returns:
            Band configuration dictionary or None if not found
        """
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            return None

        return dataset_info.get('bands', {}).get(band_type)

    def apply_scaling(self, dataset_id: str, band_type: str, value: float) -> float:
        """
        Apply scaling factor and offset to convert to standard units

        Args:
            dataset_id: Earth Engine dataset ID
            band_type: Band type identifier
            value: Original value

        Returns:
            Scaled value in standard units
        """
        band_info = self.get_band_info(dataset_id, band_type)
        if not band_info:
            return value

        scaling_factor = band_info.get('scaling_factor', 1.0)
        offset = band_info.get('offset', 0.0)

        return (value * scaling_factor) + offset

    def get_date_range(self, dataset_id: str) -> Tuple[str, str]:
        """
        Get available date range for a dataset

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            return None, None

        return dataset_info.get('start_date'), dataset_info.get('end_date')

    def validate_date_range(self, dataset_id: str, start_date: date, end_date: date) -> Tuple[bool, str]:
        """
        Validate if requested date range is within dataset availability

        Args:
            dataset_id: Dataset to validate against
            start_date: Requested start date
            end_date: Requested end date

        Returns:
            Tuple of (is_valid, error_message)
        """
        dataset_start, dataset_end = self.get_date_range(dataset_id)

        if not dataset_start or not dataset_end:
            return False, f"Dataset {dataset_id} not found"

        # Convert string dates to date objects
        try:
            ds_start = datetime.strptime(dataset_start, '%Y-%m-%d').date()
            ds_end = datetime.strptime(dataset_end, '%Y-%m-%d').date()
        except ValueError:
            return False, f"Invalid date format in dataset configuration"

        if start_date < ds_start:
            return False, f"Requested start date {start_date} is before dataset start {ds_start}"

        if end_date > ds_end:
            return False, f"Requested end date {end_date} is after dataset end {ds_end}"

        if start_date >= end_date:
            return False, "Start date must be before end date"

        return True, "Valid date range"

    def get_climate_indices(self, category: Optional[str] = None, complexity: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get available climate indices

        Args:
            category: Filter by category ('temperature', 'precipitation') or None for all
            complexity: Filter by complexity level ('simple', 'percentile') or None for all

        Returns:
            Dict mapping index IDs to their configurations
        """
        indices = self._config.get('climate_indices', {})
        filtered_indices = {}

        for index_id, index_info in indices.items():
            # Filter by category if specified
            if category and index_info.get('category') != category:
                continue

            # Filter by complexity if specified
            if complexity and index_info.get('complexity') != complexity:
                continue

            filtered_indices[index_id] = index_info

        return filtered_indices

    def get_recommended_indices(self, dataset_id: str, analysis_type: str) -> List[str]:
        """
        Get recommended climate indices for a dataset and analysis type

        Args:
            dataset_id: Dataset identifier
            analysis_type: 'temperature' or 'precipitation'

        Returns:
            List of recommended index IDs
        """
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            return []

        recommended = dataset_info.get('recommended_indices', {})
        return recommended.get(analysis_type, [])

    def get_required_bands_for_indices(self, dataset_id: str, index_ids: List[str]) -> Dict[str, str]:
        """
        Get Earth Engine band names required for calculating specific indices

        Args:
            dataset_id: Dataset identifier
            index_ids: List of climate index IDs

        Returns:
            Dict mapping band types to actual Earth Engine band names
        """
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            return {}

        indices = self._config.get('climate_indices', {})
        required_band_types = set()

        # Collect all required band types
        for index_id in index_ids:
            index_info = indices.get(index_id, {})
            required_bands = index_info.get('required_bands', [])
            required_band_types.update(required_bands)

        # Map band types to actual Earth Engine band names
        band_mapping = {}
        dataset_bands = dataset_info.get('bands', {})

        for band_type in required_band_types:
            band_info = dataset_bands.get(band_type, {})
            if band_info:
                band_mapping[band_type] = band_info['band_name']

        return band_mapping

    def get_band_scaling_info(self, dataset_id: str, band_type: str) -> Dict[str, float]:
        """
        Get scaling information for unit conversion

        Returns:
            Dict with 'scaling_factor', 'offset', 'unit', 'original_unit'
        """
        band_info = self.get_band_info(dataset_id, band_type)
        if not band_info:
            return {'scaling_factor': 1.0, 'offset': 0.0, 'unit': '', 'original_unit': ''}

        return {
            'scaling_factor': band_info.get('scaling_factor', 1.0),
            'offset': band_info.get('offset', 0.0),
            'unit': band_info.get('unit', ''),
            'original_unit': band_info.get('original_unit', ''),
            'band_name': band_info.get('band_name', ''),
            'description': band_info.get('description', '')
        }

    def get_dataset_summary_df(self, analysis_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary of datasets as a pandas DataFrame for display

        Args:
            analysis_type: Filter by analysis type or None for all

        Returns:
            DataFrame with dataset information
        """
        if analysis_type:
            datasets = self.get_datasets_for_analysis(analysis_type)
        else:
            datasets = self._config['datasets']

        rows = []
        for dataset_id, info in datasets.items():
            rows.append({
                'Dataset ID': dataset_id,
                'Name': info.get('name', ''),
                'Provider': info.get('provider', ''),
                'Start Date': info.get('start_date', ''),
                'End Date': info.get('end_date', ''),
                'Resolution': info.get('temporal_resolution', ''),
                'Pixel Size (m)': info.get('pixel_size_m', ''),
                'Coverage': info.get('geographic_coverage', 'Global'),
                'Supports': ', '.join(info.get('supports_analysis', []))
            })

        return pd.DataFrame(rows)


# Global instance for easy access
_dataset_config_instance = None

def get_dataset_config() -> DatasetConfig:
    """Get singleton instance of DatasetConfig"""
    global _dataset_config_instance
    if _dataset_config_instance is None:
        _dataset_config_instance = DatasetConfig()
    return _dataset_config_instance


# Convenience functions for backward compatibility
def load_climate_datasets(analysis_type: str) -> Dict[str, Dict]:
    """Load datasets for specific analysis type"""
    return get_dataset_config().get_datasets_for_analysis(analysis_type)


def get_band_mapping(dataset_id: str, index_ids: List[str]) -> Dict[str, str]:
    """Get band mapping for dataset and indices"""
    return get_dataset_config().get_required_bands_for_indices(dataset_id, index_ids)