"""
Core functionality for GeoClimate-Fetcher.
"""

from .gee_auth import GEEAuth, authenticate
from .metadata import MetadataCatalog
from .geometry import GeometryHandler
from .exporter import GEEExporter
from .fetchers import ImageCollectionFetcher, StaticRasterFetcher

__all__ = [
    'GEEAuth', 
    'authenticate',
    'MetadataCatalog',
    'GeometryHandler',
    'GEEExporter',
    'ImageCollectionFetcher',
    'StaticRasterFetcher'
]