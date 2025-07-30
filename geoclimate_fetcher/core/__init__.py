"""
Core functionality for GeoClimate-Fetcher.
"""

from .gee_auth import GEEAuth, authenticate
from .metadata import MetadataCatalog
from .geometry import GeometryHandler
from .exporter import GEEExporter
from .fetchers import ImageCollectionFetcher, StaticRasterFetcher

# Import temporal disaggregation classes if available
try:
    from .temporal_disaggregation import TemporalDisaggregationHandler, BiasCorrection, OptimalSelection
    temporal_disaggregation_available = True
except ImportError:
    temporal_disaggregation_available = False

__all__ = [
    'GEEAuth', 
    'authenticate',
    'MetadataCatalog',
    'GeometryHandler',
    'GEEExporter',
    'ImageCollectionFetcher',
    'StaticRasterFetcher'
]

if temporal_disaggregation_available:
    __all__.extend([
        'TemporalDisaggregationHandler',
        'BiasCorrection', 
        'OptimalSelection'
    ])