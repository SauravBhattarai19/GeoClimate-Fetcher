"""
Test script to verify STAC integration with climate datasets
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from geoclimate_fetcher.core.dataset_config import DatasetConfig
from geoclimate_fetcher.core.metadata import MetadataCatalog
from datetime import datetime

def test_stac_integration():
    """Test that datasets are properly enriched from STAC"""

    print("=" * 80)
    print("STAC INTEGRATION TEST FOR CLIMATE DATASETS")
    print("=" * 80)

    # Test 1: Verify MetadataCatalog.get_dataset_by_ee_id() works
    print("\n✅ Test 1: MetadataCatalog.get_dataset_by_ee_id()")
    catalog = MetadataCatalog(use_stac=True)

    target_ids = [
        "ECMWF/ERA5/DAILY",
        "NASA/ORNL/DAYMET_V4",
        "ECMWF/ERA5_LAND/DAILY_AGGR",
        "NASA/GPM_L3/IMERG_V07"
    ]

    for ee_id in target_ids:
        stac_data = catalog.get_dataset_by_ee_id(ee_id)
        if stac_data:
            print(f"  ✅ {ee_id}")
            print(f"     Name: {stac_data.get('name', 'N/A')[:60]}...")
        else:
            print(f"  ❌ {ee_id} NOT FOUND")
            return False

    # Test 2: Verify DatasetConfig enrichment
    print("\n✅ Test 2: DatasetConfig STAC Enrichment")
    config = DatasetConfig(use_stac=True)

    all_valid = True
    for analysis_type in ['temperature', 'precipitation']:
        print(f"\n  {analysis_type.upper()} Datasets:")
        datasets = config.get_datasets_for_analysis(analysis_type)

        for dataset_id, dataset_info in datasets.items():
            print(f"\n    📊 {dataset_id}")

            # Check required fields from STAC
            required_fields = ['name', 'provider', 'start_date', 'end_date', 'temporal_resolution']

            for field in required_fields:
                value = dataset_info.get(field)
                if value:
                    if field in ['start_date', 'end_date']:
                        print(f"      ✅ {field}: {value}")
                    else:
                        print(f"      ✅ {field}: {str(value)[:50]}...")
                else:
                    print(f"      ❌ {field}: MISSING!")
                    all_valid = False

            # Check climate-specific fields
            bands = dataset_info.get('bands', {})
            if bands:
                print(f"      ✅ bands: {len(bands)} climate bands configured")
            else:
                print(f"      ❌ bands: MISSING!")
                all_valid = False

    # Test 3: Verify date validation works (no "present" errors)
    print("\n✅ Test 3: Date Validation (No 'present' errors)")

    from datetime import date
    test_date_start = date(2020, 1, 1)
    test_date_end = date(2020, 12, 31)

    for dataset_id in target_ids:
        valid, message = config.validate_date_range(dataset_id, test_date_start, test_date_end)
        if valid:
            print(f"  ✅ {dataset_id}: {message}")
        else:
            if "Invalid date format" in message:
                print(f"  ❌ {dataset_id}: {message}")
                all_valid = False
            else:
                # Date range might be outside dataset bounds, that's OK
                print(f"  ⚠️  {dataset_id}: {message} (expected for some datasets)")

    # Final result
    print("\n" + "=" * 80)
    if all_valid:
        print("🎉 ALL TESTS PASSED! STAC Integration successful!")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED! Check errors above.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = test_stac_integration()
    sys.exit(0 if success else 1)
