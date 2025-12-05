"""
Memory Optimization Utilities for GeoClimate Intelligence Platform

This module provides centralized memory management functions including:
- Streamlit caching decorators for expensive operations
- Session state cleanup utilities
- Garbage collection triggers
- DataFrame optimization helpers
- Temporary file cleanup

Memory Optimization Strategy:
1. Use @st.cache_resource for singleton objects (MetadataCatalog, connections)
2. Use @st.cache_data for computed data with TTL (Earth Engine results)
3. Clean up session state on module transitions
4. Optimize DataFrame dtypes to reduce memory footprint
5. Trigger garbage collection after large operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import gc
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from datetime import datetime, timedelta
from functools import wraps
import logging

# Configure logging for memory operations
logger = logging.getLogger(__name__)

# =============================================================================
# MEMORY CONSTANTS AND CONFIGURATION
# =============================================================================

# Session state keys that should persist across module transitions
PERSISTENT_STATE_KEYS = {
    'auth_complete', 'project_id', 'app_mode',
    'cookie_gee_auth_token', 'cookie_gee_project_id'
}

# Session state keys for each module (for targeted cleanup)
MODULE_STATE_KEYS = {
    'data_explorer': {
        'geometry_complete', 'geometry_selected', 'dataset_selected',
        'bands_selected', 'dates_selected', 'current_dataset', 'selected_bands',
        'start_date', 'end_date', 'geometry_handler', 'download_path',
        'drawn_features', 'download_complete', 'download_results',
        'post_download_results', 'selected_file_info'
    },
    'climate_analytics': {
        'climate_step', 'climate_analysis_type', 'climate_geometry_complete',
        'climate_dataset_selected', 'climate_selected_dataset', 'climate_date_range_set',
        'climate_indices_selected', 'climate_selected_indices', 'climate_start_date',
        'climate_end_date', 'climate_geometry_handler', 'climate_analysis_complete',
        'climate_spatial_results', 'climate_results', 'climate_export_configured',
        'climate_calculator'
    },
    'hydrology': {
        'hydro_geometry_complete', 'hydro_dataset_selected', 'hydro_dates_selected',
        'hydro_current_dataset', 'hydro_precipitation_data', 'hydro_analysis_results',
        'hydro_geometry', 'hydro_area_km2', 'hydro_dataset_info', 'hydro_start_date',
        'hydro_end_date', 'hydro_run_analysis', 'hydro_show_sample', 'hydro_analyzer',
        'hydro_analysis_ready'
    },
    'product_selector': {
        'ps_step', 'ps_geometry_complete', 'ps_stations_selected',
        'ps_datasets_selected', 'ps_comparison_results', 'ps_geometry',
        'ps_stations_df', 'ps_selected_stations', 'ps_gridded_data'
    },
    'data_visualizer': {
        'direct_visualization_data', 'visualizer_data', 'uploaded_file_data',
        'viz_processed_data', 'viz_current_file'
    },
    'multi_geometry_export': {
        'mg_geometries', 'mg_dataset_selected', 'mg_export_results',
        'mg_geometry_data'
    }
}

# Large data keys that should be cleaned up aggressively
LARGE_DATA_KEYS = {
    'hydro_precipitation_data', 'climate_spatial_results', 'climate_results',
    'direct_visualization_data', 'visualizer_data', 'post_download_results',
    'download_results', 'ps_comparison_results', 'ps_gridded_data',
    'mg_export_results', 'hydro_analyzer'
}

# Cache TTL values (in seconds)
CACHE_TTL = {
    'metadata': 3600,       # 1 hour for dataset metadata
    'stations': 3600,       # 1 hour for station metadata
    'ee_geometry': 600,     # 10 minutes for geometry calculations
    'ee_collection': 300,   # 5 minutes for collection info
    'processed_data': 1800  # 30 minutes for processed results
}


# =============================================================================
# SESSION STATE CLEANUP FUNCTIONS
# =============================================================================

def cleanup_module_state(module_name: str, preserve_keys: Optional[Set[str]] = None) -> int:
    """
    Clean up session state for a specific module.

    Args:
        module_name: Name of the module to clean up
        preserve_keys: Optional set of keys to preserve

    Returns:
        Number of keys cleaned up
    """
    if module_name not in MODULE_STATE_KEYS:
        logger.warning(f"Unknown module: {module_name}")
        return 0

    preserve = preserve_keys or set()
    preserve.update(PERSISTENT_STATE_KEYS)

    cleaned = 0
    keys_to_clean = MODULE_STATE_KEYS[module_name] - preserve

    for key in keys_to_clean:
        if key in st.session_state:
            # Delete the key
            del st.session_state[key]
            cleaned += 1
            logger.debug(f"Cleaned session state key: {key}")

    # Trigger garbage collection after cleanup
    gc.collect()

    logger.info(f"Cleaned {cleaned} session state keys for module: {module_name}")
    return cleaned


def cleanup_large_data_keys() -> int:
    """
    Clean up all large data keys from session state.

    Returns:
        Number of keys cleaned up
    """
    cleaned = 0
    for key in LARGE_DATA_KEYS:
        if key in st.session_state:
            del st.session_state[key]
            cleaned += 1
            logger.debug(f"Cleaned large data key: {key}")

    # Force garbage collection
    gc.collect()

    logger.info(f"Cleaned {cleaned} large data keys")
    return cleaned


def cleanup_other_modules(current_module: str) -> int:
    """
    Clean up session state for all modules except the current one.
    This is useful when switching between modules to free memory.

    Args:
        current_module: The module that should NOT be cleaned up

    Returns:
        Total number of keys cleaned up
    """
    total_cleaned = 0

    for module_name in MODULE_STATE_KEYS:
        if module_name != current_module:
            total_cleaned += cleanup_module_state(module_name)

    return total_cleaned


def get_session_state_memory_estimate() -> Dict[str, Any]:
    """
    Estimate memory usage of session state.

    Returns:
        Dictionary with memory usage estimates
    """
    import sys

    estimates = {}
    total_size = 0

    for key in st.session_state.keys():
        try:
            value = st.session_state[key]
            # Estimate size based on type
            if isinstance(value, pd.DataFrame):
                size = value.memory_usage(deep=True).sum()
            elif isinstance(value, (list, dict)):
                size = sys.getsizeof(value)
                # Add size of contents for lists/dicts
                if isinstance(value, list):
                    for item in value[:100]:  # Sample first 100 items
                        size += sys.getsizeof(item)
                elif isinstance(value, dict):
                    for k, v in list(value.items())[:100]:
                        size += sys.getsizeof(k) + sys.getsizeof(v)
            elif isinstance(value, np.ndarray):
                size = value.nbytes
            else:
                size = sys.getsizeof(value)

            estimates[key] = size
            total_size += size
        except Exception as e:
            estimates[key] = f"Error: {str(e)}"

    return {
        'keys': estimates,
        'total_bytes': total_size,
        'total_mb': total_size / (1024 * 1024)
    }


# =============================================================================
# DATAFRAME OPTIMIZATION FUNCTIONS
# =============================================================================

def optimize_dataframe_dtypes(df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to appropriate dtypes.

    Args:
        df: DataFrame to optimize
        copy: Whether to create a copy (False modifies in-place when possible)

    Returns:
        Optimized DataFrame
    """
    if df.empty:
        return df

    result = df.copy() if copy else df

    for col in result.columns:
        col_type = result[col].dtype

        # Optimize integers
        if col_type == 'int64':
            c_min = result[col].min()
            c_max = result[col].max()

            if c_min >= -128 and c_max <= 127:
                result[col] = result[col].astype('int8')
            elif c_min >= -32768 and c_max <= 32767:
                result[col] = result[col].astype('int16')
            elif c_min >= -2147483648 and c_max <= 2147483647:
                result[col] = result[col].astype('int32')

        # Optimize floats
        elif col_type == 'float64':
            c_min = result[col].min()
            c_max = result[col].max()

            # Check if float32 is sufficient (considering precision loss)
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                result[col] = result[col].astype('float32')

        # Convert object columns with few unique values to category
        elif col_type == 'object':
            num_unique = result[col].nunique()
            num_total = len(result[col])

            # Convert to category if ratio of unique to total is less than 50%
            if num_unique / num_total < 0.5:
                result[col] = result[col].astype('category')

    return result


def reduce_dataframe_memory(df: pd.DataFrame,
                            columns_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reduce DataFrame memory by keeping only necessary columns and optimizing dtypes.

    Args:
        df: DataFrame to reduce
        columns_to_keep: List of columns to keep (None keeps all)

    Returns:
        Reduced DataFrame
    """
    if df.empty:
        return df

    # Select columns
    if columns_to_keep:
        available_cols = [c for c in columns_to_keep if c in df.columns]
        result = df[available_cols]
    else:
        result = df

    # Optimize dtypes
    result = optimize_dataframe_dtypes(result, copy=True)

    return result


# =============================================================================
# GARBAGE COLLECTION AND CLEANUP
# =============================================================================

def force_garbage_collection() -> int:
    """
    Force garbage collection and return number of objects collected.

    Returns:
        Number of unreachable objects collected
    """
    # Collect garbage in all generations
    collected = gc.collect()
    logger.debug(f"Garbage collection freed {collected} objects")
    return collected


def cleanup_temp_files(max_age_hours: int = 1) -> int:
    """
    Clean up temporary files older than specified age.

    Args:
        max_age_hours: Maximum age in hours before cleanup

    Returns:
        Number of files cleaned up
    """
    temp_dir = tempfile.gettempdir()
    cleaned = 0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

    # Look for GeoClimate-related temp files
    patterns = ['*.tif', '*.csv', '*.nc', '*.zip', 'geotiff_*']

    for pattern in patterns:
        for file_path in Path(temp_dir).glob(pattern):
            try:
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_time:
                        file_path.unlink()
                        cleaned += 1
                elif file_path.is_dir():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_time:
                        shutil.rmtree(file_path, ignore_errors=True)
                        cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean temp file {file_path}: {e}")

    logger.info(f"Cleaned {cleaned} temporary files")
    return cleaned


# =============================================================================
# CACHING DECORATORS AND UTILITIES
# =============================================================================

def clear_all_caches():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("All Streamlit caches cleared")


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about cache usage.
    Note: Limited info available from Streamlit's cache API.

    Returns:
        Dictionary with cache information
    """
    return {
        'cache_data_active': True,
        'cache_resource_active': True,
        'note': 'Use clear_all_caches() to reset caches if memory is high'
    }


# =============================================================================
# CACHED DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_resource(ttl=CACHE_TTL['metadata'])
def get_cached_metadata_catalog():
    """
    Get a cached instance of MetadataCatalog.
    This is a singleton that persists across reruns and users.

    Returns:
        MetadataCatalog instance
    """
    from geoclimate_fetcher.core import MetadataCatalog
    logger.info("Creating new MetadataCatalog instance (cached)")
    return MetadataCatalog()


@st.cache_data(ttl=CACHE_TTL['stations'])
def get_cached_stations_data(bounds: Optional[tuple] = None) -> pd.DataFrame:
    """
    Get cached Meteostat stations data, optionally filtered by bounds.

    Args:
        bounds: Optional tuple of (min_lon, min_lat, max_lon, max_lat) for filtering

    Returns:
        DataFrame with stations data
    """
    from pathlib import Path

    # Load stations file
    stations_path = Path(__file__).parent.parent / "geoclimate_fetcher" / "data" / "meteostat_stations.csv"

    if not stations_path.exists():
        logger.warning("Meteostat stations file not found")
        return pd.DataFrame()

    logger.info("Loading Meteostat stations (cached)")

    # Read with optimized dtypes
    df = pd.read_csv(
        stations_path,
        dtype={
            'id': 'string',
            'name': 'string',
            'country': 'category',
            'region': 'category',
            'latitude': 'float32',
            'longitude': 'float32',
            'elevation': 'float32'
        }
    )

    # Filter by bounds if provided
    if bounds:
        min_lon, min_lat, max_lon, max_lat = bounds
        df = df[
            (df['latitude'] >= min_lat) &
            (df['latitude'] <= max_lat) &
            (df['longitude'] >= min_lon) &
            (df['longitude'] <= max_lon)
        ]

    return df


@st.cache_data(ttl=CACHE_TTL['metadata'])
def get_cached_datasets_json() -> Dict:
    """
    Get cached datasets.json configuration.

    Returns:
        Dictionary with dataset configurations
    """
    import json
    from pathlib import Path

    datasets_path = Path(__file__).parent.parent / "geoclimate_fetcher" / "data" / "datasets.json"

    if not datasets_path.exists():
        logger.warning("datasets.json not found")
        return {}

    logger.info("Loading datasets.json (cached)")

    with open(datasets_path, 'r') as f:
        return json.load(f)


@st.cache_data(ttl=CACHE_TTL['ee_geometry'])
def get_cached_geometry_area(geometry_json: str) -> float:
    """
    Cache Earth Engine geometry area calculations.

    Args:
        geometry_json: JSON string representation of the geometry

    Returns:
        Area in square kilometers
    """
    import ee
    import json

    try:
        geo_dict = json.loads(geometry_json)
        geometry = ee.Geometry(geo_dict)
        area_km2 = geometry.area().divide(1000000).getInfo()
        return area_km2
    except Exception as e:
        logger.error(f"Error calculating geometry area: {e}")
        return 0.0


@st.cache_data(ttl=CACHE_TTL['ee_collection'])
def get_cached_collection_size(ee_id: str, start_date: str, end_date: str,
                               bounds_json: Optional[str] = None) -> int:
    """
    Cache Earth Engine collection size queries.

    Args:
        ee_id: Earth Engine collection ID
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        bounds_json: Optional JSON string of bounds geometry

    Returns:
        Number of images in the collection
    """
    import ee
    import json

    try:
        collection = ee.ImageCollection(ee_id).filterDate(start_date, end_date)

        if bounds_json:
            bounds = ee.Geometry(json.loads(bounds_json))
            collection = collection.filterBounds(bounds)

        size = collection.size().getInfo()
        return size
    except Exception as e:
        logger.error(f"Error getting collection size: {e}")
        return 0


@st.cache_data(ttl=CACHE_TTL['ee_collection'])
def get_cached_band_names(ee_id: str) -> List[str]:
    """
    Cache Earth Engine band names query.

    Args:
        ee_id: Earth Engine collection ID

    Returns:
        List of band names
    """
    import ee

    try:
        collection = ee.ImageCollection(ee_id)
        first_image = collection.first()
        bands = first_image.bandNames().getInfo()
        return bands
    except Exception as e:
        logger.error(f"Error getting band names: {e}")
        return []


# =============================================================================
# MODULE TRANSITION HANDLER
# =============================================================================

def handle_module_transition(new_module: str, force_cleanup: bool = False):
    """
    Handle transition between modules with memory cleanup.

    Args:
        new_module: The module being transitioned to
        force_cleanup: Whether to force cleanup of current module too
    """
    current_module = st.session_state.get('app_mode')

    # Only clean up if actually switching modules
    if current_module and current_module != new_module:
        logger.info(f"Module transition: {current_module} -> {new_module}")

        # Clean up the previous module's state
        cleanup_module_state(current_module)

        # Clean up temp files
        cleanup_temp_files(max_age_hours=1)

        # Force garbage collection
        force_garbage_collection()

    if force_cleanup:
        cleanup_other_modules(new_module)


# =============================================================================
# MEMORY MONITORING (for debugging)
# =============================================================================

def log_memory_status():
    """Log current memory status for debugging."""
    import sys

    # Get session state estimate
    state_info = get_session_state_memory_estimate()

    logger.info(f"Session state memory: {state_info['total_mb']:.2f} MB")
    logger.info(f"Number of session state keys: {len(state_info['keys'])}")

    # Log large keys
    for key, size in sorted(state_info['keys'].items(),
                           key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                           reverse=True)[:10]:
        if isinstance(size, (int, float)) and size > 1024 * 1024:  # > 1MB
            logger.info(f"  Large key: {key} = {size / (1024*1024):.2f} MB")


def create_memory_report() -> str:
    """
    Create a detailed memory report for debugging.

    Returns:
        Formatted string with memory report
    """
    report = []
    report.append("=" * 50)
    report.append("GeoClimate Memory Report")
    report.append("=" * 50)

    # Session state
    state_info = get_session_state_memory_estimate()
    report.append(f"\nSession State: {state_info['total_mb']:.2f} MB")
    report.append(f"Number of keys: {len(state_info['keys'])}")

    # Top 10 largest keys
    report.append("\nTop 10 largest session state keys:")
    sorted_keys = sorted(
        [(k, v) for k, v in state_info['keys'].items() if isinstance(v, (int, float))],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for key, size in sorted_keys:
        report.append(f"  {key}: {size / (1024*1024):.2f} MB")

    # Python garbage collector stats
    report.append("\nGarbage Collector Stats:")
    for i, gen in enumerate(gc.get_stats()):
        report.append(f"  Generation {i}: {gen}")

    report.append("=" * 50)

    return "\n".join(report)
