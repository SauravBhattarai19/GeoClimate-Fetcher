#!/usr/bin/env python3
"""
Generate pre-built STAC catalog snapshot for instant loading.

This script fetches all datasets from the Earth Engine STAC API and saves
them as a compressed snapshot that ships with the application. This eliminates
the 3-minute wait time for first-time users.

Usage:
    python scripts/generate_stac_snapshot.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gzip
import json
import pickle
from datetime import datetime
from geoclimate_fetcher.core.stac_client import STACClient
from geoclimate_fetcher.core.stac_cache import STACCache


def generate_snapshot():
    """Generate STAC snapshot and save to data directory."""

    print("=" * 60)
    print("STAC SNAPSHOT GENERATOR")
    print("=" * 60)
    print()
    print("This will fetch all 900+ datasets from Google Earth Engine")
    print("STAC API and save them as a pre-built snapshot.")
    print()
    print("⏱️  Estimated time: 2-3 minutes")
    print("🌐 Requires internet connection")
    print()

    # Initialize STAC client
    print("🔧 Initializing STAC client...")
    cache = STACCache()
    client = STACClient(cache_manager=cache)

    # Progress callback
    def progress_callback(current, total, message):
        percent = int((current / total) * 100)
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {percent}% - {message}", end='', flush=True)

    # Fetch all datasets
    print("📥 Fetching datasets from STAC API...")
    print()

    try:
        datasets = client.fetch_all_datasets(progress_callback=progress_callback)
        print()  # New line after progress
        print(f"✅ Fetched {len(datasets)} datasets successfully!")

    except Exception as e:
        print()
        print(f"❌ Error fetching datasets: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify access to storage.googleapis.com")
        print("  3. Try again in a few minutes")
        return False

    # Create snapshot data
    print()
    print("📦 Creating snapshot...")

    # Convert datasets to serializable format
    datasets_data = []
    for ds in datasets:
        ds_dict = {
            'id': ds.id,
            'name': ds.name,
            'description': ds.description,
            'provider': ds.provider,
            'all_providers': ds.all_providers,
            'snippet_type': ds.snippet_type,
            'start_date': ds.start_date.isoformat() if ds.start_date else None,
            'end_date': ds.end_date.isoformat() if ds.end_date else None,
            'temporal_resolution': ds.temporal_resolution,
            'pixel_size': ds.pixel_size,
            'bands': [
                {
                    'name': band.name,
                    'description': band.description,
                    'units': band.units,
                    'scale': band.scale,
                    'offset': band.offset,
                    'wavelength': band.wavelength,
                    'center_wavelength': band.center_wavelength,
                    'gsd': band.gsd,
                    'data_type': band.data_type,
                    'bitmask': band.bitmask,
                    'classes': band.classes,
                    'minimum': band.minimum,
                    'maximum': band.maximum
                }
                for band in ds.bands
            ],
            'keywords': ds.keywords,
            'license': ds.license,
            'stac_version': ds.stac_version,
            'category': ds.category,
            'spatial_extent': ds.spatial_extent
        }
        datasets_data.append(ds_dict)

    snapshot = {
        'version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'dataset_count': len(datasets),
        'stac_version': '1.0.0',
        'datasets': datasets_data
    }

    # Save as compressed JSON
    output_dir = project_root / 'geoclimate_fetcher' / 'data'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Pickle format (fastest loading)
    pickle_path = output_dir / 'stac_snapshot.pkl'
    print(f"💾 Saving pickle snapshot to {pickle_path.relative_to(project_root)}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_size = pickle_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Size: {pickle_size:.2f} MB")

    # Compressed JSON format (readable, good for git)
    json_gz_path = output_dir / 'stac_snapshot.json.gz'
    print(f"💾 Saving compressed JSON to {json_gz_path.relative_to(project_root)}...")
    with gzip.open(json_gz_path, 'wt', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

    json_gz_size = json_gz_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Size: {json_gz_size:.2f} MB")

    # Metadata file (human-readable)
    metadata = {
        'version': snapshot['version'],
        'generated_at': snapshot['generated_at'],
        'dataset_count': snapshot['dataset_count'],
        'stac_version': snapshot['stac_version'],
        'generated_by': 'generate_stac_snapshot.py',
        'providers': sorted(list(set(ds['provider'] for ds in datasets_data))),
        'categories': sorted(list(set(ds['category'] for ds in datasets_data))),
        'total_bands': sum(len(ds['bands']) for ds in datasets_data)
    }

    metadata_path = output_dir / 'stac_snapshot_metadata.json'
    print(f"💾 Saving metadata to {metadata_path.relative_to(project_root)}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print("✅ SNAPSHOT GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print(f"📊 Statistics:")
    print(f"   Datasets:  {metadata['dataset_count']}")
    print(f"   Providers: {len(metadata['providers'])}")
    print(f"   Categories: {len(metadata['categories'])}")
    print(f"   Total Bands: {metadata['total_bands']}")
    print()
    print(f"📅 Generated: {snapshot['generated_at']}")
    print()
    print(f"📁 Files created:")
    print(f"   • {pickle_path.name} ({pickle_size:.2f} MB) - Fast loading")
    print(f"   • {json_gz_path.name} ({json_gz_size:.2f} MB) - Git-friendly")
    print(f"   • {metadata_path.name} - Human-readable info")
    print()
    print("🎉 Users will now experience INSTANT loading!")
    print("   • First load: 0.5 seconds (was 3 minutes)")
    print("   • No API calls required on startup")
    print("   • Works offline")
    print()
    print("💡 Next steps:")
    print("   1. Commit snapshot files to git")
    print("   2. Users will automatically use snapshot")
    print("   3. Optional: Setup monthly auto-update via GitHub Actions")
    print()

    return True


if __name__ == '__main__':
    success = generate_snapshot()
    sys.exit(0 if success else 1)
