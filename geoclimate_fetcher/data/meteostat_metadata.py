#!/usr/bin/env python3
"""
Simple Meteostat stations downloader - just gets all stations in a CSV file.

Requirements:
pip install meteostat
"""

from meteostat import Stations

def download_stations_csv():
    """Download all Meteostat weather stations to CSV."""
    print("Downloading Meteostat weather stations...")
    
    # Get all stations
    stations = Stations()
    stations_df = stations.fetch()
    
    # Save to CSV
    stations_df.to_csv("meteostat_stations.csv")
    
    print(f"✓ Downloaded {len(stations_df)} stations to meteostat_stations.csv")
    print(f"Columns: {list(stations_df.columns)}")

if __name__ == "__main__":
    download_stations_csv()