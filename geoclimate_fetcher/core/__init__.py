"""
Core functionality for GeoClimate-Fetcher.

Memory Optimization:
- Use get_metadata_catalog() for cached singleton access to MetadataCatalog
- Use ee_cache module functions for cached Earth Engine operations
"""

from .gee_auth import GEEAuth, authenticate
from .metadata import MetadataCatalog, get_metadata_catalog, clear_metadata_cache
from .geometry import GeometryHandler
from .exporter import GEEExporter
from .fetchers import ImageCollectionFetcher, StaticRasterFetcher

from .map_widget import UnifiedMapWidget, GeometrySelectionWidget
from .dataset_config import get_dataset_config, DatasetConfig

# Earth Engine caching utilities
from .ee_cache import (
    get_geometry_area_cached,
    get_collection_size_cached,
    get_band_names_cached,
    clear_ee_cache
)

__all__ = [
    'GEEAuth',
    'authenticate',
    'MetadataCatalog',
    'get_metadata_catalog',
    'clear_metadata_cache',
    'GeometryHandler',
    'GEEExporter',
    'ImageCollectionFetcher',
    'StaticRasterFetcher',
    'UnifiedMapWidget',
    'GeometrySelectionWidget',
    'get_dataset_config',
    'DatasetConfig',
    # EE Cache utilities
    'get_geometry_area_cached',
    'get_collection_size_cached',
    'get_band_names_cached',
    'clear_ee_cache'
]