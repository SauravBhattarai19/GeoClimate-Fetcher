"""
Earth Engine Caching Utilities

This module provides caching for expensive Earth Engine operations like:
- geometry.area().getInfo()
- collection.size().getInfo()
- image.bandNames().getInfo()

Memory Optimization Notes:
- Uses Streamlit's caching decorators when available
- Falls back to simple LRU cache for non-Streamlit contexts
- Caches are automatically cleared after TTL expires
"""

import hashlib
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL for non-Streamlit contexts
_cache: Dict[str, Tuple[Any, datetime]] = {}
_cache_ttl_seconds = 600  # 10 minutes default TTL


def _get_cache_key(*args) -> str:
    """Generate a cache key from arguments."""
    key_str = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_from_cache(key: str) -> Tuple[bool, Any]:
    """Get value from cache if not expired."""
    if key in _cache:
        value, timestamp = _cache[key]
        if datetime.now() - timestamp < timedelta(seconds=_cache_ttl_seconds):
            return True, value
        else:
            del _cache[key]
    return False, None


def _set_cache(key: str, value: Any) -> None:
    """Set value in cache with current timestamp."""
    _cache[key] = (value, datetime.now())


def clear_ee_cache() -> int:
    """Clear all cached Earth Engine values."""
    global _cache
    count = len(_cache)
    _cache = {}
    logger.info(f"Cleared {count} cached EE values")
    return count


def get_geometry_area_cached(geometry, use_streamlit: bool = True) -> float:
    """
    Get geometry area in square kilometers with caching.

    This caches the result of geometry.area().divide(1000000).getInfo()
    to avoid repeated expensive server calls.

    Args:
        geometry: Earth Engine Geometry object
        use_streamlit: Whether to try using Streamlit cache

    Returns:
        Area in square kilometers
    """
    import ee

    # Get a hashable representation of the geometry
    try:
        geo_json = geometry.getInfo()
        cache_key = _get_cache_key('area', geo_json)
    except Exception as e:
        logger.warning(f"Could not get geometry info for caching: {e}")
        # Fall back to direct computation
        return geometry.area().divide(1000000).getInfo()

    # Check cache
    found, value = _get_from_cache(cache_key)
    if found:
        logger.debug(f"Cache hit for geometry area")
        return value

    # Compute and cache
    try:
        area_km2 = geometry.area().divide(1000000).getInfo()
        _set_cache(cache_key, area_km2)
        logger.debug(f"Cached geometry area: {area_km2:.2f} km²")
        return area_km2
    except Exception as e:
        logger.error(f"Error computing geometry area: {e}")
        return 0.0


def get_collection_size_cached(ee_id: str, start_date: str, end_date: str,
                               bounds_geometry=None) -> int:
    """
    Get ImageCollection size with caching.

    This caches the result of collection.size().getInfo()
    to avoid repeated expensive server calls.

    Args:
        ee_id: Earth Engine collection ID
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        bounds_geometry: Optional geometry for spatial filtering

    Returns:
        Number of images in the collection
    """
    import ee

    # Create cache key
    bounds_key = None
    if bounds_geometry:
        try:
            bounds_key = json.dumps(bounds_geometry.getInfo())
        except:
            bounds_key = str(bounds_geometry)

    cache_key = _get_cache_key('collection_size', ee_id, start_date, end_date, bounds_key)

    # Check cache
    found, value = _get_from_cache(cache_key)
    if found:
        logger.debug(f"Cache hit for collection size: {ee_id}")
        return value

    # Compute and cache
    try:
        collection = ee.ImageCollection(ee_id).filterDate(start_date, end_date)

        if bounds_geometry:
            collection = collection.filterBounds(bounds_geometry)

        size = collection.size().getInfo()
        _set_cache(cache_key, size)
        logger.debug(f"Cached collection size for {ee_id}: {size}")
        return size
    except Exception as e:
        logger.error(f"Error getting collection size for {ee_id}: {e}")
        return 0


def get_band_names_cached(ee_id: str) -> List[str]:
    """
    Get band names for an ImageCollection with caching.

    This caches the result of collection.first().bandNames().getInfo()
    to avoid repeated expensive server calls.

    Args:
        ee_id: Earth Engine collection ID

    Returns:
        List of band names
    """
    import ee

    cache_key = _get_cache_key('band_names', ee_id)

    # Check cache
    found, value = _get_from_cache(cache_key)
    if found:
        logger.debug(f"Cache hit for band names: {ee_id}")
        return value

    # Compute and cache
    try:
        collection = ee.ImageCollection(ee_id)
        first_image = collection.first()
        bands = first_image.bandNames().getInfo()
        _set_cache(cache_key, bands)
        logger.debug(f"Cached band names for {ee_id}: {len(bands)} bands")
        return bands
    except Exception as e:
        logger.error(f"Error getting band names for {ee_id}: {e}")
        return []


def get_image_date_cached(image) -> Optional[str]:
    """
    Get image date with caching.

    Args:
        image: Earth Engine Image object

    Returns:
        Date string (YYYY-MM-DD) or None
    """
    import ee

    try:
        # Get a unique identifier for the image
        image_id = image.id().getInfo()
        cache_key = _get_cache_key('image_date', image_id)

        # Check cache
        found, value = _get_from_cache(cache_key)
        if found:
            return value

        # Compute and cache
        date_str = image.date().format('YYYY-MM-dd').getInfo()
        _set_cache(cache_key, date_str)
        return date_str
    except Exception as e:
        logger.error(f"Error getting image date: {e}")
        return None


def get_geometry_bounds_cached(geometry) -> Optional[Dict]:
    """
    Get geometry bounds with caching.

    Args:
        geometry: Earth Engine Geometry object

    Returns:
        Bounds dictionary with coordinates
    """
    import ee

    try:
        geo_json = geometry.getInfo()
        cache_key = _get_cache_key('bounds', geo_json)

        # Check cache
        found, value = _get_from_cache(cache_key)
        if found:
            return value

        # Compute and cache
        bounds = geometry.bounds().getInfo()
        _set_cache(cache_key, bounds)
        return bounds
    except Exception as e:
        logger.error(f"Error getting geometry bounds: {e}")
        return None


def get_collection_date_range_cached(ee_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the date range of an ImageCollection with caching.

    Args:
        ee_id: Earth Engine collection ID

    Returns:
        Tuple of (start_date, end_date) strings
    """
    import ee

    cache_key = _get_cache_key('date_range', ee_id)

    # Check cache
    found, value = _get_from_cache(cache_key)
    if found:
        return value

    try:
        collection = ee.ImageCollection(ee_id)

        # Get first and last dates
        first_date = collection.first().date().format('YYYY-MM-dd').getInfo()

        # Sort descending and get first (most recent)
        last_date = collection.sort('system:time_start', False).first().date().format('YYYY-MM-dd').getInfo()

        result = (first_date, last_date)
        _set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error getting date range for {ee_id}: {e}")
        return (None, None)


# Streamlit-compatible caching (when available)
def try_streamlit_cache():
    """
    Try to use Streamlit caching if available.
    Returns True if Streamlit caching is available.
    """
    try:
        import streamlit as st
        return hasattr(st, 'cache_data')
    except ImportError:
        return False


# Create Streamlit-cached versions if available
try:
    import streamlit as st

    @st.cache_data(ttl=600)  # 10 minutes
    def st_get_geometry_area(geometry_json: str) -> float:
        """Streamlit-cached version of geometry area calculation."""
        import ee
        import json
        geometry = ee.Geometry(json.loads(geometry_json))
        return geometry.area().divide(1000000).getInfo()

    @st.cache_data(ttl=300)  # 5 minutes
    def st_get_collection_size(ee_id: str, start_date: str, end_date: str,
                               bounds_json: Optional[str] = None) -> int:
        """Streamlit-cached version of collection size query."""
        import ee
        import json
        collection = ee.ImageCollection(ee_id).filterDate(start_date, end_date)
        if bounds_json:
            bounds = ee.Geometry(json.loads(bounds_json))
            collection = collection.filterBounds(bounds)
        return collection.size().getInfo()

    @st.cache_data(ttl=3600)  # 1 hour
    def st_get_band_names(ee_id: str) -> List[str]:
        """Streamlit-cached version of band names query."""
        import ee
        collection = ee.ImageCollection(ee_id)
        return collection.first().bandNames().getInfo()

    STREAMLIT_CACHE_AVAILABLE = True
    logger.info("Streamlit caching available for EE operations")

except ImportError:
    STREAMLIT_CACHE_AVAILABLE = False
    logger.info("Streamlit not available, using basic caching")
