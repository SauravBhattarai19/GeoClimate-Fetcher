"""
Module for handling dataset metadata from CSV catalogs.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import glob
from rapidfuzz import fuzz, process

class MetadataCatalog:
    """Class to manage and search dataset metadata from CSV files."""
    
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
            
        # Band names may be comma-separated
        return [band.strip() for band in bands_str.split(',')]
        
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