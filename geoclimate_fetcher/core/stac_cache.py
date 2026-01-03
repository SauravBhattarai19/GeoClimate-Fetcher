"""
Multi-layer caching system for STAC catalog data.

Implements a two-tier caching strategy:
1. In-memory cache: Instant access during session
2. Disk cache: Persistent storage with TTL (default 24 hours)
"""

import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
from .dataset_models import DatasetMetadata, BandMetadata


logger = logging.getLogger(__name__)


class STACCache:
    """Multi-layer caching for STAC data with TTL support."""

    # Cache directory and settings
    CACHE_DIR = Path.home() / ".geoclimate_fetcher" / "stac_cache"
    CACHE_VERSION = "v1"
    CACHE_TTL = 86400  # 24 hours in seconds

    def __init__(self, cache_dir: Optional[Path] = None, ttl: int = CACHE_TTL):
        """
        Initialize cache manager.

        Args:
            cache_dir: Custom cache directory (default: ~/.geoclimate_fetcher/stac_cache)
            ttl: Time-to-live in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.ttl = ttl
        self._memory_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"STAC cache initialized: {self.cache_dir} (TTL: {ttl}s)")

    def _get_cache_path(self, key: str, use_json: bool = True) -> Path:
        """
        Get path for cache file.

        Args:
            key: Cache key
            use_json: If True, use .json extension, otherwise .pkl

        Returns:
            Path to cache file
        """
        ext = "json" if use_json else "pkl"
        safe_key = key.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}_{self.CACHE_VERSION}.{ext}"

    def is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cache file is still valid (within TTL).

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False

        # Check modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        return age.total_seconds() < self.ttl

    def get_root_catalog(self) -> Optional[Dict]:
        """
        Get cached root catalog.

        Returns:
            Root catalog dict or None if not cached/expired
        """
        key = "root_catalog"

        # Check memory cache
        with self._cache_lock:
            if key in self._memory_cache:
                logger.debug("Root catalog found in memory cache")
                return self._memory_cache[key]

        # Check disk cache
        cache_path = self._get_cache_path(key)
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Store in memory cache
                    with self._cache_lock:
                        self._memory_cache[key] = data
                    logger.debug("Root catalog loaded from disk cache")
                    return data
            except Exception as e:
                logger.warning(f"Error loading root catalog from cache: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def set_root_catalog(self, data: Dict):
        """
        Cache root catalog.

        Args:
            data: Root catalog dictionary
        """
        key = "root_catalog"

        # Store in memory
        with self._cache_lock:
            self._memory_cache[key] = data

        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Root catalog cached")
        except Exception as e:
            logger.warning(f"Error caching root catalog: {e}")

    def get_provider_catalog(self, provider: str) -> Optional[Dict]:
        """
        Get cached provider catalog.

        Args:
            provider: Provider name

        Returns:
            Provider catalog dict or None if not cached/expired
        """
        key = f"provider_{provider}"

        # Check memory cache
        with self._cache_lock:
            if key in self._memory_cache:
                logger.debug(f"Provider catalog {provider} found in memory cache")
                return self._memory_cache[key]

        # Check disk cache
        cache_path = self._get_cache_path(key)
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Store in memory cache
                    with self._cache_lock:
                        self._memory_cache[key] = data
                    logger.debug(f"Provider catalog {provider} loaded from disk cache")
                    return data
            except Exception as e:
                logger.warning(f"Error loading provider catalog {provider} from cache: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def set_provider_catalog(self, provider: str, data: Dict):
        """
        Cache provider catalog.

        Args:
            provider: Provider name
            data: Provider catalog dictionary
        """
        key = f"provider_{provider}"

        # Store in memory
        with self._cache_lock:
            self._memory_cache[key] = data

        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Provider catalog {provider} cached")
        except Exception as e:
            logger.warning(f"Error caching provider catalog {provider}: {e}")

    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get cached dataset.

        Args:
            dataset_id: Dataset ID (e.g., "ECMWF/ERA5_LAND/DAILY_AGGR")

        Returns:
            DatasetMetadata or None if not cached/expired
        """
        key = f"dataset_{dataset_id}"

        # Check memory cache
        with self._cache_lock:
            if key in self._memory_cache:
                logger.debug(f"Dataset {dataset_id} found in memory cache")
                return self._memory_cache[key]

        # Check disk cache (using pickle for DatasetMetadata objects)
        cache_path = self._get_cache_path(key, use_json=False)
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # Store in memory cache
                    with self._cache_lock:
                        self._memory_cache[key] = data
                    logger.debug(f"Dataset {dataset_id} loaded from disk cache")
                    return data
            except Exception as e:
                logger.warning(f"Error loading dataset {dataset_id} from cache: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def set_dataset(self, dataset_id: str, data: DatasetMetadata):
        """
        Cache dataset.

        Args:
            dataset_id: Dataset ID
            data: DatasetMetadata object
        """
        key = f"dataset_{dataset_id}"

        # Store in memory
        with self._cache_lock:
            self._memory_cache[key] = data

        # Store on disk (using pickle)
        cache_path = self._get_cache_path(key, use_json=False)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Dataset {dataset_id} cached")
        except Exception as e:
            logger.warning(f"Error caching dataset {dataset_id}: {e}")

    def get_all_datasets(self) -> Optional[List[DatasetMetadata]]:
        """
        Get cached full dataset list.

        Returns:
            List of DatasetMetadata or None if not cached/expired
        """
        key = "all_datasets"

        # Check memory cache
        with self._cache_lock:
            if key in self._memory_cache:
                logger.debug("All datasets found in memory cache")
                return self._memory_cache[key]

        # Check disk cache
        cache_path = self._get_cache_path(key, use_json=False)
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # Store in memory cache
                    with self._cache_lock:
                        self._memory_cache[key] = data
                    logger.info(f"Loaded {len(data)} datasets from disk cache")
                    return data
            except Exception as e:
                logger.warning(f"Error loading all datasets from cache: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def set_all_datasets(self, datasets: List[DatasetMetadata]):
        """
        Cache full dataset list.

        Args:
            datasets: List of DatasetMetadata objects
        """
        key = "all_datasets"

        # Store in memory
        with self._cache_lock:
            self._memory_cache[key] = datasets

        # Store on disk
        cache_path = self._get_cache_path(key, use_json=False)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(datasets, f)
            logger.info(f"Cached {len(datasets)} datasets to disk")
        except Exception as e:
            logger.warning(f"Error caching all datasets: {e}")

    def clear_cache(self, memory_only: bool = False):
        """
        Clear all cached data.

        Args:
            memory_only: If True, only clear memory cache (keep disk cache)
        """
        # Clear memory cache
        with self._cache_lock:
            self._memory_cache.clear()
        logger.info("Memory cache cleared")

        # Clear disk cache if requested
        if not memory_only:
            try:
                for cache_file in self.cache_dir.glob(f"*_{self.CACHE_VERSION}.*"):
                    cache_file.unlink()
                logger.info("Disk cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing disk cache: {e}")

    def clear_expired(self):
        """Remove expired cache files from disk."""
        try:
            removed_count = 0
            for cache_file in self.cache_dir.glob(f"*_{self.CACHE_VERSION}.*"):
                if not self.is_cache_valid(cache_file):
                    cache_file.unlink()
                    removed_count += 1
            if removed_count > 0:
                logger.info(f"Removed {removed_count} expired cache files")
        except Exception as e:
            logger.warning(f"Error clearing expired cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        stats = {
            'memory_cache_size': len(self._memory_cache),
            'disk_cache_files': 0,
            'total_disk_size_mb': 0.0,
            'cache_dir': str(self.cache_dir),
            'ttl_seconds': self.ttl,
            'valid_files': 0,
            'expired_files': 0
        }

        try:
            total_size = 0
            valid_count = 0
            expired_count = 0

            for cache_file in self.cache_dir.glob(f"*_{self.CACHE_VERSION}.*"):
                total_size += cache_file.stat().st_size
                if self.is_cache_valid(cache_file):
                    valid_count += 1
                else:
                    expired_count += 1

            stats['disk_cache_files'] = valid_count + expired_count
            stats['total_disk_size_mb'] = round(total_size / (1024 * 1024), 2)
            stats['valid_files'] = valid_count
            stats['expired_files'] = expired_count

        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")

        return stats

    def is_fresh(self) -> bool:
        """
        Check if the all_datasets cache is fresh.

        Returns:
            True if cache exists and is valid
        """
        cache_path = self._get_cache_path("all_datasets", use_json=False)
        return self.is_cache_valid(cache_path)
