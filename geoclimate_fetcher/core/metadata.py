"""
Module for handling dataset metadata from STAC API and CSV catalogs.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import glob
from rapidfuzz import fuzz, process
import logging

# Import STAC modules
try:
    from .stac_client import STACClient
    from .stac_cache import STACCache
    from .dataset_models import DatasetMetadata, BandMetadata
    STAC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"STAC modules not available: {e}")
    STAC_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetadataCatalog:
    """Class to manage and search dataset metadata from STAC API or CSV files."""

    def __init__(self, use_stac: Optional[bool] = None, data_dir: Optional[str] = None):
        """
        Initialize the metadata catalog.

        Args:
            use_stac: If True, use STAC API. If False, use CSV. If None, auto-detect
                     (use STAC if available, fallback to CSV)
            data_dir: Directory containing the CSV metadata files (for CSV fallback).
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
        self._datasets_cache: List[DatasetMetadata] = []
        self._stac_client: Optional[STACClient] = None

        # Determine whether to use STAC
        if use_stac is None:
            # Auto-detect: prefer STAC if available
            use_stac = STAC_AVAILABLE and os.getenv('ENABLE_STAC_API', 'true').lower() == 'true'

        self.use_stac = use_stac and STAC_AVAILABLE

        # Initialize
        if self.use_stac:
            try:
                logger.info("Initializing MetadataCatalog with STAC API")
                self._init_stac()
            except Exception as e:
                logger.warning(f"STAC initialization failed: {e}")
                # Fallback to CSV if enabled
                if os.getenv('FALLBACK_TO_CSV', 'true').lower() == 'true':
                    logger.info("Falling back to CSV catalog")
                    self.use_stac = False
                    self._load_csv_catalogs()
                else:
                    raise
        else:
            logger.info("Initializing MetadataCatalog with CSV files")
            self._load_csv_catalogs()

    def _init_stac(self):
        """Initialize STAC client and load datasets."""
        self._stac_client = STACClient()
        # Load datasets from STAC (cached if available)
        self._datasets_cache = self._stac_client.fetch_all_datasets()

        # Convert to pandas DataFrame for compatibility with existing code
        self._all_datasets = pd.DataFrame([
            ds.to_dataframe_row() for ds in self._datasets_cache
        ])

        logger.info(f"Loaded {len(self._datasets_cache)} datasets from STAC API")

    def _load_csv_catalogs(self):
        """Load all CSV catalogs from the data directory (fallback method)."""
        self._load_catalogs()
        
    def _load_catalogs(self) -> None:
        """Load all CSV catalogs from the data directory."""
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
        for csv_file in csv_files:
            category = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                self._catalog[category] = df
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
                
        # Combine all datasets into a single DataFrame
        if self._catalog:
            self._all_datasets = pd.concat(self._catalog.values(), ignore_index=True)
        
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
        
        Args:
            dataset_name: The name of the dataset
            
        Returns:
            List of band names
        """
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
        return list(dict.fromkeys(results))  # Remove duplicates while preserving order
        
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

    # ========== NEW STAC-SPECIFIC METHODS ==========

    def get_band_metadata(self, dataset_name: str, band_name: str) -> Optional['BandMetadata']:
        """
        Get detailed band metadata (STAC only).

        Args:
            dataset_name: The name of the dataset
            band_name: The name of the band

        Returns:
            BandMetadata object or None if not found or using CSV
        """
        if not self.use_stac:
            logger.debug("get_band_metadata only available with STAC API")
            return None

        dataset = next((ds for ds in self._datasets_cache if ds.name == dataset_name), None)
        if dataset:
            return dataset.get_band(band_name)
        return None

    def get_dataset_metadata(self, dataset_name: str) -> Optional['DatasetMetadata']:
        """
        Get full dataset metadata object (STAC only).

        Args:
            dataset_name: The name of the dataset

        Returns:
            DatasetMetadata object or None if not found or using CSV
        """
        if not self.use_stac:
            logger.debug("get_dataset_metadata only available with STAC API")
            return None

        return next((ds for ds in self._datasets_cache if ds.name == dataset_name), None)

    def get_dataset_by_ee_id(self, ee_id: str) -> Optional[Dict]:
        """
        Get dataset metadata by Earth Engine ID.
        Works with both STAC and CSV backends.

        Args:
            ee_id: Earth Engine asset ID (e.g., 'ECMWF/ERA5/DAILY')

        Returns:
            Dictionary with dataset metadata or None if not found
        """
        if self.use_stac and self._datasets_cache:
            # Search in STAC cache (list of DatasetMetadata objects converted to dicts)
            for ds in self._datasets_cache:
                if isinstance(ds, dict) and ds.get('id') == ee_id:
                    return ds
                elif hasattr(ds, 'id') and ds.id == ee_id:
                    # Convert DatasetMetadata object to dict
                    return ds.to_dict() if hasattr(ds, 'to_dict') else ds.__dict__

        # Fallback to DataFrame search (works for both STAC and CSV)
        if self._all_datasets is not None and not self._all_datasets.empty:
            matches = self._all_datasets[self._all_datasets['Earth Engine ID'] == ee_id]
            if not matches.empty:
                return matches.iloc[0].to_dict()

        return None

    def get_datasets_by_provider(self, provider: str) -> pd.DataFrame:
        """
        Filter datasets by provider.

        Args:
            provider: Provider name

        Returns:
            DataFrame with datasets from the provider
        """
        if self._all_datasets is None or self._all_datasets.empty:
            return pd.DataFrame()

        return self._all_datasets[self._all_datasets['Provider'] == provider].reset_index(drop=True)

    def get_all_providers(self) -> List[str]:
        """
        Get list of all unique providers.

        Returns:
            List of provider names
        """
        if self._all_datasets is None or self._all_datasets.empty:
            return []

        providers = self._all_datasets['Provider'].unique().tolist()
        return sorted([p for p in providers if p and str(p) != 'nan'])

    def get_all_categories(self) -> List[str]:
        """
        Get list of all unique categories.

        Returns:
            List of category names
        """
        if self._all_datasets is None or self._all_datasets.empty:
            return []

        # For STAC, use the Category field
        if 'Category' in self._all_datasets.columns:
            categories = self._all_datasets['Category'].unique().tolist()
            return sorted([c for c in categories if c and str(c) != 'nan'])

        return []

    def get_all_temporal_resolutions(self) -> List[str]:
        """
        Get list of all unique temporal resolutions.

        Returns:
            List of temporal resolution values
        """
        if self._all_datasets is None or self._all_datasets.empty:
            return []

        if 'Temporal Resolution' in self._all_datasets.columns:
            resolutions = self._all_datasets['Temporal Resolution'].unique().tolist()
            return sorted([r for r in resolutions if r and str(r) != 'nan'])

        return []

    def get_statistics(self) -> Dict:
        """
        Get catalog statistics.

        Returns:
            Dictionary with statistics
        """
        if self.use_stac and self._stac_client:
            return self._stac_client.get_statistics()

        # CSV fallback
        stats = {
            'total_datasets': len(self._all_datasets) if self._all_datasets is not None else 0,
            'total_providers': len(self.get_all_providers()),
            'total_categories': len(self.get_all_categories()),
            'cache_fresh': False,
            'source': 'CSV'
        }
        return stats

    def refresh_cache(self, progress_callback: Optional[callable] = None):
        """
        Refresh STAC cache by fetching latest data from API.

        Args:
            progress_callback: Function(current, total, message) for progress updates
        """
        if not self.use_stac:
            logger.warning("refresh_cache only available with STAC API")
            return

        if self._stac_client:
            # Clear cache and refetch
            self._stac_client.cache.clear_cache()
            self._datasets_cache = self._stac_client.fetch_all_datasets(
                progress_callback=progress_callback
            )

            # Update DataFrame
            self._all_datasets = pd.DataFrame([
                ds.to_dataframe_row() for ds in self._datasets_cache
            ])

            logger.info(f"Cache refreshed: {len(self._datasets_cache)} datasets")

    def is_using_stac(self) -> bool:
        """
        Check if catalog is using STAC API.

        Returns:
            True if using STAC, False if using CSV
        """
        return self.use_stac