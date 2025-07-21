"""
Test script for band selection function
"""
import pandas as pd
from pathlib import Path

def get_bands_for_dataset(dataset_name):
    """Get bands for a dataset directly from the CSV files"""
    # Look in the data directory for CSV files
    data_dir = Path('data')
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return []
    
    print(f"Looking for dataset '{dataset_name}' in CSV files...")
    
    # Try to find the dataset in any CSV file
    for csv_file in data_dir.glob('*.csv'):
        try:
            print(f"Reading {csv_file}...")
            df = pd.read_csv(csv_file)
            if 'Dataset Name' not in df.columns or 'Band Names' not in df.columns:
                print(f"  Missing required columns in {csv_file}")
                continue
                
            # Find the dataset
            dataset_row = df[df['Dataset Name'] == dataset_name]
            if not dataset_row.empty:
                print(f"  Found dataset in {csv_file}")
                bands_str = dataset_row.iloc[0].get('Band Names', '')
                if isinstance(bands_str, str) and bands_str:
                    bands = [band.strip() for band in bands_str.split(',')]
                    print(f"  Bands: {bands}")
                    return bands
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
    
    # If not found, try the Datasets.csv file specifically
    datasets_file = data_dir / 'Datasets.csv'
    if datasets_file.exists():
        try:
            print(f"Reading {datasets_file}...")
            df = pd.read_csv(datasets_file)
            dataset_row = df[df['Dataset Name'] == dataset_name]
            if not dataset_row.empty:
                print(f"  Found dataset in Datasets.csv")
                bands_str = dataset_row.iloc[0].get('Band Names', '')
                if isinstance(bands_str, str) and bands_str:
                    bands = [band.strip() for band in bands_str.split(',')]
                    print(f"  Bands: {bands}")
                    return bands
        except Exception as e:
            print(f"  Error reading Datasets.csv: {e}")
    
    print("No bands found for this dataset")
    return []

def main():
    """Test the function with some dataset names"""
    test_datasets = [
        "Daymet V4 Daily Meteorology (NA)",
        "MODIS Terra Land Surface Temperature (Daily 1km)",
        "ERA5 Daily Aggregates",
        "CHIRPS Daily Precipitation (Version 2.0 Final)"
    ]
    
    for dataset in test_datasets:
        print("\n" + "="*50)
        print(f"Testing dataset: {dataset}")
        bands = get_bands_for_dataset(dataset)
        print(f"Result: {len(bands)} bands found - {bands}")

if __name__ == "__main__":
    main() 