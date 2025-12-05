"""
Module for handling dataset metadata from CSV catalogs.

Memory Optimization Notes:
- This module is designed to be cached using @st.cache_resource
- DataFrames are optimized with appropriate dtypes to reduce memory footprint
- The singleton pattern ensures only one instance exists in memory
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import glob
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

# Singleton instance for caching
_catalog_instance: Optional['MetadataCatalog'] = None


def get_metadata_catalog(data_dir: Optional[str] = None) -> 'MetadataCatalog':
    """
    Get a cached singleton instance of MetadataCatalog.

    This function ensures only one instance of MetadataCatalog exists,
    reducing memory usage when multiple modules access the catalog.

    Args:
        data_dir: Directory containing the CSV metadata files.

    Returns:
        MetadataCatalog: Singleton instance
    """
    global _catalog_instance

    if _catalog_instance is None:
        logger.info("Creating new MetadataCatalog singleton instance")
        _catalog_instance = MetadataCatalog(data_dir)

    return _catalog_instance


def clear_metadata_cache():
    """Clear the cached MetadataCatalog instance."""
    global _catalog_instance
    _catalog_instance = None
    logger.info("MetadataCatalog cache cleared")


class MetadataCatalog:
    """
    Class to manage and search dataset metadata from CSV files.

    Memory Optimization:
    - Use get_metadata_catalog() to get a cached singleton instance
    - DataFrames use optimized dtypes (category for low-cardinality strings)
    - Lazy loading of band expansions
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the metadata catalog.

        Args:
            data_dir: Directory containing the CSV metadata files.
                     If None, defaults to 'data/' relative to the package.
        """
        if data_dir is None:
            # Default to 'data/' in the package directory
            package_dir = Path(__file__).resolve().parent.parent
            data_dir = package_dir / 'data'
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self._catalog: Dict[str, pd.DataFrame] = {}
        self._all_datasets: Optional[pd.DataFrame] = None
        self._band_cache: Dict[str, List[str]] = {}  # Cache for expanded band names
        self._load_catalogs()

    def _load_catalogs(self) -> None:
        """Load all CSV catalogs from the data directory with optimized dtypes."""
        csv_files = list(self.data_dir.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        # Skip large station files - they're handled separately
        csv_files = [f for f in csv_files if 'meteostat_stations' not in f.name.lower()]

        for csv_file in csv_files:
            category = csv_file.stem
            try:
                # Load with optimized dtypes
                df = pd.read_csv(csv_file)

                # Optimize string columns to category for low-cardinality columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Convert to category if less than 50% unique values
                        if df[col].nunique() / len(df) < 0.5:
                            df[col] = df[col].astype('category')

                self._catalog[category] = df
                logger.debug(f"Loaded catalog: {category} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {str(e)}")

        # Combine all datasets into a single DataFrame
        if self._catalog:
            self._all_datasets = pd.concat(self._catalog.values(), ignore_index=True)
            logger.info(f"MetadataCatalog loaded: {len(self._all_datasets)} total datasets")
        
    @property
    def categories(self) -> List[str]:
        """
        Get available dataset categories.
        
        Returns:
            List of category names
        """
        return list(self._catalog.keys())
    
    @property
    def all_datasets(self) -> pd.DataFrame:
        """
        Get all datasets combined.
        
        Returns:
            DataFrame containing all datasets
        """
        if self._all_datasets is None:
            raise ValueError("No datasets loaded")
        return self._all_datasets
        
    def get_dataset_by_name(self, name: str) -> Optional[pd.Series]:
        """
        Get a dataset by its exact name.
        
        Args:
            name: The exact name of the dataset
            
        Returns:
            DataFrame row for the dataset or None if not found
        """
        if self._all_datasets is None:
            return None
            
        result = self._all_datasets[self._all_datasets['Dataset Name'] == name]
        return result.iloc[0] if not result.empty else None
        
    def get_datasets_by_category(self, category: str) -> pd.DataFrame:
        """
        Get all datasets in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            DataFrame with datasets in the category
        """
        if category not in self._catalog:
            raise ValueError(f"Category '{category}' not found")
            
        return self._catalog[category]
        
    def search_datasets(self, query: str, threshold: int = 70, 
                       limit: int = 10) -> pd.DataFrame:
        """
        Search for datasets matching the query using fuzzy matching.
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0-100)
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with matching datasets
        """
        if self._all_datasets is None or self._all_datasets.empty:
            return pd.DataFrame()
            
        # Search in dataset names and descriptions
        name_matches = process.extract(
            query, 
            self._all_datasets['Dataset Name'].tolist(),
            scorer=fuzz.partial_ratio,
            limit=limit
        )
        
        desc_matches = process.extract(
            query, 
            self._all_datasets['Description'].tolist(),
            scorer=fuzz.partial_ratio,
            limit=limit
        )
        
        # Get indices of matches - safely handling the tuple unpacking
        name_indices = []
        for match in name_matches:
            if match[1] >= threshold:  # Check if score is above threshold
                name = match[0]  # Get the name 
                # Find corresponding index in dataframe
                indices = self._all_datasets.index[self._all_datasets['Dataset Name'] == name].tolist()
                if indices:  # If found, add the first one
                    name_indices.append(indices[0])
        
        desc_indices = []
        for match in desc_matches:
            if match[1] >= threshold:  # Check if score is above threshold
                desc = match[0]  # Get the description
                indices = self._all_datasets.index[self._all_datasets['Description'] == desc].tolist()
                if indices:  # If found, add the first one
                    desc_indices.append(indices[0])
        
        # Combine unique indices
        all_indices = list(set(name_indices + desc_indices))
        
        # Return empty DataFrame if no matches
        if not all_indices:
            return pd.DataFrame(columns=self._all_datasets.columns)
            
        return self._all_datasets.loc[all_indices].reset_index(drop=True)
        
    def get_bands_for_dataset(self, dataset_name: str) -> List[str]:
        """
        Get available bands for a dataset.

        Uses internal caching to avoid repeated band name expansion.

        Args:
            dataset_name: The name of the dataset

        Returns:
            List of band names
        """
        # Check cache first
        if dataset_name in self._band_cache:
            return self._band_cache[dataset_name]

        dataset = self.get_dataset_by_name(dataset_name)

        if dataset is None:
            return []

        # Parse the band names string from the CSV
        bands_str = dataset.get('Band Names', '')
        if not isinstance(bands_str, str):
            return []

        # Handle different band name notation patterns
        results = []
        
        # If we have "band1, ..., bandN" pattern, try to expand it
        if '…' in bands_str or '...' in bands_str:
            # First split by comma and process each part
            parts = [p.strip() for p in bands_str.split(',')]
            
            for part in parts:
                if '…' in part or '...' in part:
                    # Skip parts that just contain ellipsis
                    if part.strip() in ['…', '...']:
                        continue
                    
                    # Try to expand patterns like "SPEI_01_month, …, SPEI_12_month"
                    try:
                        # Get prefix and suffix (e.g., "SPEI_" and "_month")
                        if '…' in part:
                            prefix, suffix = part.split('…')
                        else:
                            prefix, suffix = part.split('...')
                            
                        # Try to find any numeric pattern
                        import re
                        prefix_match = re.search(r'(\D+)(\d+)$', prefix)
                        if prefix_match:
                            true_prefix = prefix_match.group(1)  # e.g., "SPEI_"
                            start_num = int(prefix_match.group(2))  # e.g., "01" becomes 1
                            
                            # Find ending number in another part in the list
                            for other_part in parts:
                                if other_part.startswith(true_prefix) and not ('…' in other_part or '...' in other_part):
                                    end_match = re.search(r'(\D+)(\d+)', other_part)
                                    if end_match and end_match.group(1) == true_prefix:
                                        end_num = int(end_match.group(2))
                                        # Generate all bands in range
                                        for i in range(start_num, end_num + 1):
                                            band_name = f"{true_prefix}{i:02d}{suffix}"
                                            results.append(band_name)
                    except Exception:
                        # If expansion fails, just add as-is
                        results.append(part)
                else:
                    # Regular band name
                    results.append(part)
        else:
            # Standard comma-separated list
            results = [band.strip() for band in bands_str.split(',')]

        # Remove any empty strings or duplicates
        results = [band for band in results if band and band not in ['…', '...']]
        final_results = list(dict.fromkeys(results))  # Remove duplicates while preserving order

        # Cache the results
        self._band_cache[dataset_name] = final_results

        return final_results
        
    def get_date_range(self, dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the date range for a dataset.
        
        Args:
            dataset_name: The name of the dataset
            
        Returns:
            Tuple of (start_date, end_date)
        """
        dataset = self.get_dataset_by_name(dataset_name)
        
        if dataset is None:
            return None, None
            
        start_date = dataset.get('Start Date')
        end_date = dataset.get('End Date')
        
        return start_date, end_date
        
    def get_ee_id(self, dataset_name: str) -> Optional[str]:
        """
        Get the Earth Engine ID for a dataset.
        
        Args:
            dataset_name: The name of the dataset
            
        Returns:
            Earth Engine ID string or None if not found
        """
        dataset = self.get_dataset_by_name(dataset_name)
        
        if dataset is None:
            return None
            
        return dataset.get('Earth Engine ID')
        
    def get_snippet_type(self, dataset_name: str) -> Optional[str]:
        """
        Get the snippet type (Image/ImageCollection) for a dataset.
        
        Args:
            dataset_name: The name of the dataset
            
        Returns:
            Snippet type string or None if not found
        """
        dataset = self.get_dataset_by_name(dataset_name)
        
        if dataset is None:
            return None
            
        return dataset.get('Snippet Type')