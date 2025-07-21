"""
Basic demonstration of GeoClimate-Fetcher.

This example shows how to:
1. Authenticate with Google Earth Engine
2. Search for datasets
3. Create an area of interest
4. Download ERA5 precipitation data
"""

import os
import ee
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from geoclimate_fetcher.core import (
    authenticate,
    MetadataCatalog,
    GeometryHandler,
    GEEExporter,
    ImageCollectionFetcher
)

def main():
    """Run the basic demonstration."""
    print("GeoClimate-Fetcher Basic Demo")
    print("============================")
    
    # Step 1: Authenticate with Google Earth Engine
    print("\n1. Authenticating with Google Earth Engine")
    
    # You need to provide your GEE Project ID
    project_id = input("Enter your GEE Project ID: ")
    
    auth = authenticate(project_id)
    if not auth.is_initialized():
        print("Authentication failed. Exiting.")
        return
        
    print("Authentication successful!")
    
    # Step 2: Load the metadata catalog
    print("\n2. Loading dataset catalog")
    catalog = MetadataCatalog()
    
    print(f"Loaded {len(catalog.all_datasets)} datasets across {len(catalog.categories)} categories")
    print(f"Available categories: {', '.join(catalog.categories)}")
    
    # Step 3: Search for precipitation datasets
    print("\n3. Searching for precipitation datasets")
    precip_datasets = catalog.get_datasets_by_category("precipitation")
    
    print(f"Found {len(precip_datasets)} precipitation datasets:")
    for i, (_, row) in enumerate(precip_datasets.iterrows(), 1):
        print(f"  {i}. {row['Dataset Name']} ({row['Earth Engine ID']})")
        
    # Let's use ERA5 for this demo
    dataset_name = "ERA5 Daily Aggregates"
    dataset = catalog.get_dataset_by_name(dataset_name)
    
    if dataset is None:
        print(f"Dataset '{dataset_name}' not found. Trying to find another ERA5 dataset...")
        # Try to find another ERA5 dataset
        era5_datasets = precip_datasets[precip_datasets['Dataset Name'].str.contains('ERA5')]
        if not era5_datasets.empty:
            dataset = era5_datasets.iloc[0]
            dataset_name = dataset['Dataset Name']
            print(f"Using '{dataset_name}' instead.")
        else:
            print("No ERA5 dataset found. Exiting.")
            return
            
    ee_id = dataset['Earth Engine ID']
    print(f"Selected dataset: {dataset_name} ({ee_id})")
    
    # Get available bands
    bands = catalog.get_bands_for_dataset(dataset_name)
    print(f"Available bands: {', '.join(bands)}")
    
    # For ERA5, let's use total precipitation
    selected_bands = ['total_precipitation']
    if 'total_precipitation' not in bands:
        # Try to find a precipitation band
        precip_bands = [b for b in bands if 'precip' in b.lower()]
        if precip_bands:
            selected_bands = [precip_bands[0]]
        else:
            # Just use the first band
            selected_bands = [bands[0]]
            
    print(f"Selected bands: {', '.join(selected_bands)}")
    
    # Step 4: Create an area of interest
    print("\n4. Creating an area of interest")
    geometry_handler = GeometryHandler()
    
    # Create a simple bounding box for San Francisco Bay Area
    sf_bbox = {
        "type": "Polygon",
        "coordinates": [[
            [-122.6, 37.2],
            [-122.6, 37.9],
            [-121.8, 37.9],
            [-121.8, 37.2],
            [-122.6, 37.2]
        ]]
    }
    
    geometry = geometry_handler.set_geometry_from_geojson(sf_bbox, "sf_bay_area")
    
    # Calculate the area
    area = geometry_handler.get_geometry_area()
    print(f"Area of interest: San Francisco Bay Area ({area:.2f} km²)")
    
    # Step 5: Set date range (last 10 days)
    print("\n5. Setting date range")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=10)
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Step 6: Download data
    print("\n6. Downloading data")
    
    # Create the fetcher
    fetcher = ImageCollectionFetcher(ee_id, selected_bands, geometry)
    fetcher.filter_dates(start_date, end_date)
    
    # Get time series average
    print("Calculating spatial averages...")
    data = fetcher.get_time_series_average()
    
    if data.empty:
        print("No data found for the specified parameters. Exiting.")
        return
        
    print(f"Retrieved {len(data)} records")
    print("\nSample data:")
    print(data.head())
    
    # Save to CSV
    output_dir = Path("downloads")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "era5_precipitation_sf_bay_area.csv"
    
    # Create exporter
    exporter = GEEExporter()
    exporter.export_time_series_to_csv(data, output_file)
    
    print(f"\nData saved to {output_file}")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()