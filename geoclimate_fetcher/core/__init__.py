"""
Core functionality for GeoClimate-Fetcher.
"""

from .gee_auth import GEEAuth, authenticate
from .metadata import MetadataCatalog
from .geometry import GeometryHandler
from .exporter import GEEExporter
from .fetchers import ImageCollectionFetcher, StaticRasterFetcher

from .map_widget import UnifiedMapWidget, GeometrySelectionWidget
from .dataset_config import get_dataset_config, DatasetConfig

__all__ = [
    'GEEAuth',
    'authenticate',
    'MetadataCatalog',
    'GeometryHandler',
    'GEEExporter',
    'ImageCollectionFetcher',
    'StaticRasterFetcher',
    'UnifiedMapWidget',
    'GeometrySelectionWidget',
    'get_dataset_config',
    'DatasetConfig'
]