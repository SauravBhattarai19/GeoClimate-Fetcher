# STAC API Integration - Technical Documentation

## Overview

GeoClimate-Fetcher now integrates with Google Earth Engine's STAC (SpatioTemporal Asset Catalog) API, providing access to **900+ datasets** with rich metadata including band descriptions, units, wavelengths, and more.

## What Changed

### Before (CSV-Based)
- ❌ 33 datasets manually maintained in `Datasets.csv`
- ❌ Band names as simple comma-separated strings
- ❌ No band descriptions, units, or wavelengths
- ❌ Manual updates required
- ❌ Limited metadata

### After (STAC API)
- ✅ **900+ datasets** from Google's Earth Engine catalog
- ✅ Rich band metadata (descriptions, units, wavelengths, resolution)
- ✅ **Real-time updates** from Google
- ✅ Smart caching (24-hour TTL)
- ✅ **Automatic fallback** to CSV if STAC unavailable
- ✅ **100% backward compatible**

## Architecture

### Core Components

```
geoclimate_fetcher/core/
├── dataset_models.py      # Data models for STAC metadata
├── stac_cache.py          # Multi-layer caching (memory + disk)
├── stac_client.py         # STAC API client
└── metadata.py            # Modified to support STAC + CSV fallback
```

### Data Flow

```
STAC API (primary)
    ↓
STACClient.fetch_all_datasets()
    ↓
STACCache (24h TTL)
    ├─ Memory cache (session)
    └─ Disk cache (~/.geoclimate_fetcher/stac_cache)
    ↓
MetadataCatalog
    ↓
geodata_explorer.py (UI)
```

## Features

### 1. Smart Caching

**First load**: Fetches all 900+ datasets from STAC API (~2-3 minutes)
**Subsequent loads**: Instant from cache

Cache location: `~/.geoclimate_fetcher/stac_cache/`
Cache TTL: 24 hours (configurable)

### 2. Rich Band Metadata

Example for Sentinel-2 band:
```python
Band: B4 (Red)
├─ Description: "Red"
├─ Wavelength: 664.5nm (S2A) / 665nm (S2B)
├─ Center Wavelength: 0.6645 µm
├─ Resolution (GSD): 10m
└─ Scale: 0.0001
```

### 3. Graceful Fallback

```
Try STAC API
    ↓ (fails)
Try CSV catalog
    ↓ (fails)
Use hardcoded fallback datasets
```

### 4. Backward Compatibility

All existing code continues to work:
```python
from geoclimate_fetcher.core.metadata import MetadataCatalog

# Works exactly as before
catalog = MetadataCatalog()
datasets = catalog.all_datasets
bands = catalog.get_bands_for_dataset("ERA5-Land Daily")
```

## Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
# Enable/disable STAC API (default: true)
ENABLE_STAC_API=true

# Cache duration in seconds (default: 86400 = 24 hours)
STAC_CACHE_TTL=86400

# Fallback to CSV if STAC fails (default: true)
FALLBACK_TO_CSV=true

# Cache directory (default: ~/.geoclimate_fetcher/stac_cache)
STAC_CACHE_DIR=~/.geoclimate_fetcher/stac_cache
```

### Force CSV Mode

```python
# Use CSV instead of STAC
catalog = MetadataCatalog(use_stac=False)
```

### Force STAC Mode

```python
# Use STAC only (no CSV fallback)
import os
os.environ['FALLBACK_TO_CSV'] = 'false'
catalog = MetadataCatalog(use_stac=True)
```

## Usage Examples

### Basic Usage

```python
from geoclimate_fetcher.core.metadata import MetadataCatalog

# Initialize (auto-detects STAC or CSV)
catalog = MetadataCatalog()

# Check what's being used
if catalog.is_using_stac():
    print("Using STAC API 🚀")
else:
    print("Using CSV fallback")

# Get all datasets
datasets_df = catalog.all_datasets
print(f"Loaded {len(datasets_df)} datasets")

# Search datasets
results = catalog.search_datasets("precipitation", threshold=70)

# Get bands for a dataset
bands = catalog.get_bands_for_dataset("ERA5-Land Daily")
```

### Advanced Usage with STAC

```python
from geoclimate_fetcher.core.metadata import MetadataCatalog

catalog = MetadataCatalog(use_stac=True)

# Get rich band metadata
band_meta = catalog.get_band_metadata("Sentinel-2 SR", "B4")
if band_meta:
    print(f"Band: {band_meta.name}")
    print(f"Description: {band_meta.description}")
    print(f"Units: {band_meta.units}")
    print(f"Wavelength: {band_meta.wavelength}")
    print(f"Resolution: {band_meta.gsd}m")

# Get full dataset metadata
dataset_meta = catalog.get_dataset_metadata("ERA5-Land Daily")
if dataset_meta:
    print(f"Provider: {dataset_meta.provider}")
    print(f"Temporal Resolution: {dataset_meta.temporal_resolution}")
    print(f"Date Range: {dataset_meta.start_date} to {dataset_meta.end_date}")
    print(f"Bands: {len(dataset_meta.bands)}")

# Filter by provider
nasa_datasets = catalog.get_datasets_by_provider("NASA")

# Get statistics
stats = catalog.get_statistics()
print(f"Total Datasets: {stats['total_datasets']}")
print(f"Providers: {stats['total_providers']}")
print(f"Total Bands: {stats['total_bands']}")
```

### Refresh Cache

```python
# Clear cache and fetch latest data
def progress_callback(current, total, message):
    print(f"[{current}/{total}] {message}")

catalog.refresh_cache(progress_callback=progress_callback)
```

## UI Features

### Enhanced Dataset Selection

- **900+ datasets** instead of 33
- Search across all metadata (title, description, keywords, bands)
- Filter by provider, category, temporal resolution
- Shows dataset count: "📂 Loaded 947 datasets from STAC API ✨"

### Rich Band Selection

Toggle "Show detailed band information" to see:
- Band descriptions
- Units (K, m, W/m², etc.)
- Wavelengths (for optical sensors)
- Spatial resolution
- Scale factors

Example display:
```
☑ B4
  **Red** | Units: reflectance | λ: 664.5nm (S2A) / 665nm (S2B) | Resolution: 10m
```

## STAC Catalog Structure

### Root Catalog
URL: `https://storage.googleapis.com/earthengine-stac/catalog/catalog.json`

Contains links to 150+ provider catalogs:
- NASA
- USGS
- ECMWF
- ESA/Copernicus
- JAXA
- And many more...

### Provider Catalog
URL: `https://storage.googleapis.com/earthengine-stac/catalog/{PROVIDER}/catalog.json`

Contains links to all datasets from that provider.

### Dataset Collection
URL: `https://storage.googleapis.com/earthengine-stac/catalog/{PROVIDER}/{DATASET_ID}.json`

Full STAC collection with:
- Dataset metadata
- Band information
- Temporal/spatial extent
- License information
- Provider details

## Performance

### First Load (No Cache)
- Time: 2-3 minutes
- Network requests: ~150 (one per provider)
- Data fetched: ~900 datasets
- Cache created: Yes

### Subsequent Loads (Cached)
- Time: < 1 second
- Network requests: 0
- Data source: Disk cache
- Cache fresh for: 24 hours

### Memory Usage
- In-memory cache: ~50-100 MB (active session)
- Disk cache: ~20-30 MB (persistent)

## Troubleshooting

### Issue: STAC API is slow on first load

**Solution**: This is expected. The first load fetches 900+ datasets and caches them. Subsequent loads are instant.

**Progress**: A progress indicator shows loading status.

### Issue: STAC API fails to connect

**Symptoms**: "⚠️ Error loading catalog" message

**Solutions**:
1. Check internet connection
2. Check if `storage.googleapis.com` is accessible
3. Application automatically falls back to CSV if `FALLBACK_TO_CSV=true`
4. Check firewall settings

### Issue: Cache is stale

**Solution**: The cache refreshes automatically after 24 hours. To force refresh:

```python
from geoclimate_fetcher.core.stac_cache import STACCache

cache = STACCache()
cache.clear_cache()  # Clears all cached data
```

Or delete cache directory:
```bash
rm -rf ~/.geoclimate_fetcher/stac_cache/
```

### Issue: Want to disable STAC

**Solution**: Set environment variable:
```bash
export ENABLE_STAC_API=false
```

Or in code:
```python
catalog = MetadataCatalog(use_stac=False)
```

## Cache Management

### View Cache Statistics

```python
from geoclimate_fetcher.core.metadata import MetadataCatalog

catalog = MetadataCatalog(use_stac=True)
stats = catalog.get_statistics()

print(f"Cache Fresh: {stats['cache_fresh']}")
print(f"Cache Stats: {stats['cache_stats']}")
```

### Clear Expired Cache

```python
from geoclimate_fetcher.core.stac_cache import STACCache

cache = STACCache()
cache.clear_expired()  # Remove only expired files
```

### Clear All Cache

```python
cache.clear_cache()  # Remove all cache files
```

## Data Models

### DatasetMetadata

```python
@dataclass
class DatasetMetadata:
    id: str                          # Earth Engine ID
    name: str                        # Human-readable title
    description: str
    provider: str
    snippet_type: str                # "Image" or "ImageCollection"
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    temporal_resolution: str         # "Daily", "Monthly", etc.
    pixel_size: Optional[float]      # meters
    bands: List[BandMetadata]
    keywords: List[str]
    license: str
    category: str                    # Derived category
```

### BandMetadata

```python
@dataclass
class BandMetadata:
    name: str
    description: str
    units: Optional[str]
    scale: Optional[float]
    offset: Optional[float]
    wavelength: Optional[str]
    center_wavelength: Optional[float]
    gsd: Optional[float]             # Ground sample distance
    data_type: Optional[str]
```

## Migration Guide

### For Users

**No action required!** The application automatically uses STAC API and falls back to CSV if needed.

### For Developers

**Existing code continues to work:**
```python
# This still works!
from geoclimate_fetcher.core.metadata import MetadataCatalog

catalog = MetadataCatalog()
datasets = catalog.all_datasets
```

**New STAC-specific features:**
```python
# Get rich metadata (STAC only)
band_meta = catalog.get_band_metadata(dataset_name, band_name)
dataset_meta = catalog.get_dataset_metadata(dataset_name)

# New filtering methods
providers = catalog.get_all_providers()
categories = catalog.get_all_categories()
nasa_datasets = catalog.get_datasets_by_provider("NASA")
```

## API Reference

See the source code for complete API documentation:
- `geoclimate_fetcher/core/dataset_models.py` - Data models
- `geoclimate_fetcher/core/stac_cache.py` - Caching system
- `geoclimate_fetcher/core/stac_client.py` - STAC API client
- `geoclimate_fetcher/core/metadata.py` - MetadataCatalog class

## Future Enhancements

Potential improvements for future versions:

1. **Async fetching**: Use asyncio for parallel dataset fetching
2. **Incremental updates**: Only fetch new/changed datasets
3. **Dataset recommendations**: Suggest datasets based on user's AOI
4. **Advanced filters**: Date range, spatial extent, license type
5. **Dataset comparison**: Compare multiple datasets side-by-side
6. **Visualization presets**: Load optimal vis params from STAC

## Support

For issues or questions:
1. Check this documentation
2. Review error messages (they include fallback info)
3. Open an issue on GitHub
4. Check the cache stats and logs

## License

This STAC integration is part of GeoClimate-Fetcher and follows the same license terms.

---

**Last Updated**: 2026-01-03
**STAC Version**: 1.0.0
**Datasets**: 900+
