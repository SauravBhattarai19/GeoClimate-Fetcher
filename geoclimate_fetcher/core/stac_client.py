"""
Client for fetching and parsing Google Earth Engine STAC catalog.

This module provides the STACClient class for accessing the Earth Engine
STAC API at storage.googleapis.com/earthengine-stac.
"""

import requests
import logging
import time
import pickle
import gzip
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from .dataset_models import DatasetMetadata, BandMetadata
from .stac_cache import STACCache


logger = logging.getLogger(__name__)


class STACClient:
    """Client for fetching and parsing Google Earth Engine STAC catalog."""

    BASE_URL = "https://storage.googleapis.com/earthengine-stac/catalog"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, cache_manager: Optional[STACCache] = None):
        """
        Initialize STAC client.

        Args:
            cache_manager: STACCache instance (creates new one if None)
        """
        self.cache = cache_manager or STACCache()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeoClimate-Fetcher/1.0'
        })

        logger.info("STAC client initialized")

    def _load_snapshot(self) -> Optional[Dict]:
        """
        Load pre-built STAC snapshot from repository.

        Returns:
            Snapshot dict or None if not found
        """
        # Try pickle first (fastest)
        snapshot_path = Path(__file__).parent.parent / 'data' / 'stac_snapshot.pkl'

        if snapshot_path.exists():
            try:
                logger.info("Loading pre-built STAC snapshot...")
                with open(snapshot_path, 'rb') as f:
                    snapshot = pickle.load(f)

                # Validate snapshot structure
                if not isinstance(snapshot, dict) or 'datasets' not in snapshot:
                    logger.warning("Invalid snapshot structure, ignoring")
                    return None

                logger.info(f"✅ Loaded snapshot with {snapshot.get('dataset_count', 0)} datasets")
                logger.info(f"   Generated: {snapshot.get('generated_at', 'unknown')}")

                # Show age warning if old
                self._check_snapshot_age(snapshot.get('generated_at'))

                return snapshot

            except Exception as e:
                logger.warning(f"Error loading snapshot: {e}")
                return None

        # Try compressed JSON fallback
        json_gz_path = Path(__file__).parent.parent / 'data' / 'stac_snapshot.json.gz'
        if json_gz_path.exists():
            try:
                logger.info("Loading compressed JSON snapshot...")
                with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f:
                    snapshot = json.load(f)

                logger.info(f"✅ Loaded snapshot with {snapshot.get('dataset_count', 0)} datasets")
                return snapshot

            except Exception as e:
                logger.warning(f"Error loading JSON snapshot: {e}")
                return None

        logger.info("No pre-built snapshot found")
        return None

    def _check_snapshot_age(self, generated_at_str: Optional[str]):
        """Check snapshot age and warn if stale."""
        if not generated_at_str:
            return

        try:
            generated_at = datetime.fromisoformat(generated_at_str)
            age = datetime.now() - generated_at
            age_days = age.days

            if age_days > 60:
                logger.warning(f"⚠️  Snapshot is {age_days} days old (consider refreshing)")
            elif age_days > 30:
                logger.info(f"💡 Snapshot is {age_days} days old")
            else:
                logger.debug(f"Snapshot age: {age_days} days")

        except Exception as e:
            logger.debug(f"Could not parse snapshot age: {e}")

    def _parse_snapshot_datasets(self, snapshot: Dict) -> List[DatasetMetadata]:
        """
        Parse snapshot data into DatasetMetadata objects.

        Args:
            snapshot: Snapshot dictionary

        Returns:
            List of DatasetMetadata objects
        """
        datasets = []

        for ds_data in snapshot.get('datasets', []):
            try:
                # Parse dates
                start_date = None
                end_date = None

                if ds_data.get('start_date'):
                    try:
                        start_date = datetime.fromisoformat(ds_data['start_date'])
                    except:
                        pass

                if ds_data.get('end_date'):
                    try:
                        end_date = datetime.fromisoformat(ds_data['end_date'])
                    except:
                        pass

                # Parse bands
                bands = []
                for band_data in ds_data.get('bands', []):
                    band = BandMetadata(
                        name=band_data.get('name', ''),
                        description=band_data.get('description', ''),
                        units=band_data.get('units'),
                        scale=band_data.get('scale'),
                        offset=band_data.get('offset'),
                        wavelength=band_data.get('wavelength'),
                        center_wavelength=band_data.get('center_wavelength'),
                        gsd=band_data.get('gsd'),
                        data_type=band_data.get('data_type'),
                        bitmask=band_data.get('bitmask'),
                        classes=band_data.get('classes'),
                        minimum=band_data.get('minimum'),
                        maximum=band_data.get('maximum')
                    )
                    bands.append(band)

                # Create dataset
                dataset = DatasetMetadata(
                    id=ds_data.get('id', ''),
                    name=ds_data.get('name', ''),
                    description=ds_data.get('description', ''),
                    provider=ds_data.get('provider', 'Unknown'),
                    all_providers=ds_data.get('all_providers', []),
                    snippet_type=ds_data.get('snippet_type', 'ImageCollection'),
                    start_date=start_date,
                    end_date=end_date,
                    temporal_resolution=ds_data.get('temporal_resolution', 'Unknown'),
                    pixel_size=ds_data.get('pixel_size'),
                    bands=bands,
                    keywords=ds_data.get('keywords', []),
                    license=ds_data.get('license', 'Unknown'),
                    stac_version=ds_data.get('stac_version', '1.0.0'),
                    category=ds_data.get('category', 'Other'),
                    spatial_extent=ds_data.get('spatial_extent')
                )
                datasets.append(dataset)

            except Exception as e:
                logger.warning(f"Error parsing dataset {ds_data.get('name', 'unknown')}: {e}")
                continue

        return datasets

    def get_snapshot_age_days(self) -> Optional[int]:
        """
        Get age of current snapshot in days.

        Returns:
            Age in days or None if no snapshot
        """
        snapshot = self._load_snapshot()
        if not snapshot:
            return None

        generated_at_str = snapshot.get('generated_at')
        if not generated_at_str:
            return None

        try:
            generated_at = datetime.fromisoformat(generated_at_str)
            age = datetime.now() - generated_at
            return age.days
        except:
            return None

    def _fetch_with_retry(self, url: str) -> Dict:
        """
        Fetch URL with exponential backoff retry.

        Args:
            url: URL to fetch

        Returns:
            Parsed JSON response

        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()

            except requests.Timeout:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    raise

            except requests.RequestException as e:
                logger.warning(f"Error fetching {url}: {e} (attempt {attempt + 1})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    raise

        raise requests.RequestException(f"Failed to fetch {url} after {self.MAX_RETRIES} attempts")

    def fetch_root_catalog(self) -> Dict:
        """
        Fetch the root STAC catalog.

        Returns:
            Root catalog dictionary

        Raises:
            requests.RequestException: If fetch fails
        """
        # Check cache first
        cached = self.cache.get_root_catalog()
        if cached:
            logger.info("Using cached root catalog")
            return cached

        # Fetch from API
        logger.info("Fetching root catalog from STAC API")
        url = f"{self.BASE_URL}/catalog.json"
        data = self._fetch_with_retry(url)

        # Cache result
        self.cache.set_root_catalog(data)

        logger.info(f"Root catalog fetched successfully (STAC v{data.get('stac_version', 'unknown')})")
        return data

    def fetch_provider_catalog(self, provider_name: str) -> Dict:
        """
        Fetch a specific provider's catalog.

        Args:
            provider_name: Provider directory name (e.g., "ECMWF", "NASA")

        Returns:
            Provider catalog dictionary

        Raises:
            requests.RequestException: If fetch fails
        """
        # Check cache first
        cached = self.cache.get_provider_catalog(provider_name)
        if cached:
            logger.debug(f"Using cached catalog for provider {provider_name}")
            return cached

        # Fetch from API
        logger.debug(f"Fetching catalog for provider {provider_name}")
        url = f"{self.BASE_URL}/{provider_name}/catalog.json"
        data = self._fetch_with_retry(url)

        # Cache result
        self.cache.set_provider_catalog(provider_name, data)

        return data

    def fetch_dataset(self, dataset_id: str) -> DatasetMetadata:
        """
        Fetch a specific dataset's metadata.

        Args:
            dataset_id: Dataset ID (e.g., "ECMWF/ERA5_LAND/DAILY_AGGR")

        Returns:
            DatasetMetadata object

        Raises:
            requests.RequestException: If fetch fails
        """
        # Check cache first
        cached = self.cache.get_dataset(dataset_id)
        if cached:
            logger.debug(f"Using cached metadata for dataset {dataset_id}")
            return cached

        # Convert dataset ID to STAC URL path
        # e.g., "ECMWF/ERA5_LAND/DAILY_AGGR" -> "ECMWF/ECMWF_ERA5_LAND_DAILY_AGGR.json"
        stac_filename = dataset_id.replace('/', '_') + '.json'
        provider = dataset_id.split('/')[0]
        url = f"{self.BASE_URL}/{provider}/{stac_filename}"

        # Fetch from API
        logger.debug(f"Fetching metadata for dataset {dataset_id}")
        collection = self._fetch_with_retry(url)

        # Parse into DatasetMetadata
        dataset = DatasetMetadata.from_stac_collection(collection)

        # Cache result
        self.cache.set_dataset(dataset_id, dataset)

        return dataset

    def get_provider_names(self) -> List[str]:
        """
        Get list of all provider names from root catalog.

        Returns:
            List of provider names
        """
        root = self.fetch_root_catalog()
        providers = []

        for link in root.get('links', []):
            if link.get('rel') == 'child':
                # Extract provider name from title or href
                provider_name = link.get('title', '')
                if provider_name:
                    providers.append(provider_name)

        logger.info(f"Found {len(providers)} providers in catalog")
        return sorted(providers)

    def fetch_all_datasets(
        self,
        providers: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_datasets: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[DatasetMetadata]:
        """
        Fetch all datasets from pre-built snapshot, cache, or STAC API.

        Priority order (unless force_refresh=True):
        1. Pre-built snapshot (instant, ~0.5s)
        2. Cache (fast, if fresh)
        3. STAC API (slow, ~2-3 min)

        Args:
            providers: Filter by specific providers (None = all providers)
            progress_callback: Function(current, total, message) for progress updates
            max_datasets: Maximum number of datasets to fetch (for testing)
            force_refresh: Skip snapshot/cache and fetch from API

        Returns:
            List of DatasetMetadata objects

        Raises:
            requests.RequestException: If critical fetch fails
        """
        if not force_refresh:
            # Priority 1: Check for pre-built snapshot (FASTEST - instant load!)
            snapshot = self._load_snapshot()
            if snapshot:
                datasets = self._parse_snapshot_datasets(snapshot)

                if progress_callback:
                    progress_callback(1, 1, f"✅ Loaded {len(datasets)} datasets from snapshot")

                # Apply provider filter if specified
                if providers:
                    datasets = [ds for ds in datasets if ds.provider in providers]
                    logger.info(f"Filtered to {len(datasets)} datasets from providers: {providers}")

                return datasets

            # Priority 2: Check cache
            cached = self.cache.get_all_datasets()
            if cached:
                logger.info(f"Using cached dataset list ({len(cached)} datasets)")
                if progress_callback:
                    progress_callback(1, 1, f"Loaded {len(cached)} datasets from cache")

                # Apply provider filter if specified
                if providers:
                    cached = [ds for ds in cached if ds.provider in providers]
                    logger.info(f"Filtered to {len(cached)} datasets from providers: {providers}")

                return cached

        # Priority 3: Fetch from API (SLOWEST - 2-3 minutes)
        logger.info("Fetching all datasets from STAC API (this may take a few minutes)")

        # Get root catalog
        root = self.fetch_root_catalog()

        # Extract provider links
        provider_links = [link for link in root.get('links', []) if link.get('rel') == 'child']

        # Filter providers if specified
        if providers:
            provider_links = [
                link for link in provider_links
                if link.get('title') in providers
            ]

        total_providers = len(provider_links)
        all_datasets = []
        dataset_count = 0

        logger.info(f"Processing {total_providers} providers")

        # Fetch datasets from each provider
        for i, provider_link in enumerate(provider_links):
            provider_title = provider_link.get('title', 'Unknown')

            if progress_callback:
                progress_callback(
                    i + 1,
                    total_providers,
                    f"Loading {provider_title}... ({len(all_datasets)} datasets so far)"
                )

            try:
                # Get provider catalog
                provider_href = provider_link.get('href', '')
                # Extract provider directory from href
                # e.g., "https://.../catalog/ECMWF/catalog.json" -> "ECMWF"
                provider_dir = provider_href.split('/')[-2] if '/' in provider_href else provider_title

                provider_catalog = self.fetch_provider_catalog(provider_dir)

                # Extract dataset links from provider catalog
                dataset_links = [
                    link for link in provider_catalog.get('links', [])
                    if link.get('rel') == 'child'
                ]

                logger.info(f"Provider {provider_title}: {len(dataset_links)} datasets")

                # Fetch each dataset
                for dataset_link in dataset_links:
                    if max_datasets and dataset_count >= max_datasets:
                        logger.info(f"Reached max_datasets limit ({max_datasets})")
                        break

                    dataset_href = dataset_link.get('href', '')
                    if not dataset_href:
                        continue

                    try:
                        # Fetch dataset collection
                        collection = self._fetch_with_retry(dataset_href)

                        # Parse into DatasetMetadata
                        dataset = DatasetMetadata.from_stac_collection(collection)
                        all_datasets.append(dataset)

                        # Cache individual dataset
                        self.cache.set_dataset(dataset.id, dataset)

                        dataset_count += 1

                    except Exception as e:
                        logger.warning(f"Error fetching dataset {dataset_href}: {e}")
                        continue

                if max_datasets and dataset_count >= max_datasets:
                    break

            except Exception as e:
                logger.warning(f"Error processing provider {provider_title}: {e}")
                continue

        logger.info(f"Successfully fetched {len(all_datasets)} datasets from STAC API")

        # Cache the full list
        self.cache.set_all_datasets(all_datasets)

        if progress_callback:
            progress_callback(
                total_providers,
                total_providers,
                f"Completed! Loaded {len(all_datasets)} datasets"
            )

        return all_datasets

    def search_datasets(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        temporal_resolution: Optional[str] = None
    ) -> List[DatasetMetadata]:
        """
        Search datasets with filters.

        Args:
            query: Search query (matches title, description, keywords)
            providers: Filter by providers
            categories: Filter by categories
            temporal_resolution: Filter by temporal resolution

        Returns:
            List of matching DatasetMetadata objects
        """
        # Get all datasets (from cache or fetch)
        all_datasets = self.fetch_all_datasets(providers=providers)

        query_lower = query.lower() if query else ""
        results = []

        for dataset in all_datasets:
            # Apply category filter
            if categories and dataset.category not in categories:
                continue

            # Apply temporal resolution filter
            if temporal_resolution and dataset.temporal_resolution != temporal_resolution:
                continue

            # Apply query filter
            if query_lower:
                searchable_text = ' '.join([
                    dataset.name,
                    dataset.description,
                    dataset.provider,
                    ' '.join(dataset.keywords),
                    ' '.join([band.name for band in dataset.bands])
                ]).lower()

                if query_lower in searchable_text:
                    results.append(dataset)
            else:
                results.append(dataset)

        logger.info(f"Search for '{query}' found {len(results)} datasets")
        return results

    def get_statistics(self) -> Dict[str, any]:
        """
        Get catalog statistics.

        Returns:
            Dictionary with statistics
        """
        all_datasets = self.fetch_all_datasets()

        providers = set()
        categories = set()
        temporal_resolutions = set()
        total_bands = 0

        for dataset in all_datasets:
            providers.add(dataset.provider)
            categories.add(dataset.category)
            temporal_resolutions.add(dataset.temporal_resolution)
            total_bands += len(dataset.bands)

        cache_stats = self.cache.get_cache_stats()

        return {
            'total_datasets': len(all_datasets),
            'total_providers': len(providers),
            'total_categories': len(categories),
            'total_bands': total_bands,
            'temporal_resolutions': sorted(temporal_resolutions),
            'cache_fresh': self.cache.is_fresh(),
            'cache_stats': cache_stats
        }
