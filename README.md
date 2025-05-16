# GeoClimate-Fetcher

An open-source Python package for downloading Google Earth Engine (GEE) climate data for user-defined study areas.

## Prerequisites

- **Python 3.8 to 3.12** (doesn't work on 3.13) - [Download Python](https://www.python.org/downloads/)
- **Google Earth Engine Account** - [Sign up for GEE](https://earthengine.google.com/signup/)
- **Visual Studio Code (Recommended)** - [Download VS Code](https://code.visualstudio.com/download)
- **Git** - [Download Git](https://git-scm.com/downloads)

## Installation

### Quick Installation

1. Clone the repository:
   ```
   git clone https://github.com/Saurav-JSU/GeoClimate-Fetcher.git
   cd GeoClimate-Fetcher
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv fresh_env
   fresh_env\Scripts\activate  # Windows
   source fresh_env/bin/activate  # macOS/Linux
   ```

3. Upgrade pip:
   ```
   python -m pip install --upgrade pip
   ```

4. Install the package in development mode:
   ```
   pip install -e .
   ```

5. If you encounter the following error:
   ```
   ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: '...\site-packages\jupyterlab-4.4.2.dist-info\RECORDtr0pbr8o.tmp' -> '...\site-packages\jupyterlab-4.4.2.dist-info\RECORD'
   Check the permissions.
   ```
   Simply run the install command again:
   ```
   pip install -e .
   ```

## Running the Interactive GUI

After installation, run the interactive Jupyter notebook:

```
python geoclimate_fetcher\notebooks\interactive_gui.ipynb
```

This will launch the interactive GUI in a browser window.

### Alternative Installation Methods

#### Using venv (Windows)

1. Clone the repository:
   ```
   git clone https://github.com/Saurav-JSU/GeoClimate-Fetcher.git
   cd GeoClimate-Fetcher
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv fresh_env
   fresh_env\Scripts\activate
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

#### Using venv (macOS/Linux)

1. Clone the repository:
   ```
   git clone https://github.com/Saurav-JSU/GeoClimate-Fetcher.git
   cd GeoClimate-Fetcher
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv fresh_env
   source fresh_env/bin/activate
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

#### Using Conda

1. Clone the repository:
   ```
   git clone https://github.com/Saurav-JSU/GeoClimate-Fetcher.git
   cd GeoClimate-Fetcher
   ```

2. Create and activate a conda environment:
   ```
   conda env create -f geoclimate_fetcher/environment.yml
   conda activate geoclimate-fetcher
   ```

## Using GeoClimate-Fetcher

The GeoClimate-Fetcher workflow includes the following steps:

1. **Authentication**: Connect to Google Earth Engine with your credentials
2. **Area Selection**: Define your study area by drawing on a map, uploading a GeoJSON file, or entering coordinates
3. **Dataset Selection**: Choose from over 50 climate datasets
4. **Band Selection**: Select specific bands from the dataset
5. **Time Range Selection**: For time-series data, define the period of interest
6. **Download Configuration**: Choose file format, resolution, and output location
   - Files smaller than 50MB will be downloaded directly to your computer
   - Larger files will be automatically exported to Google Drive

## Key Features

- **Authentication Gate**: Secure access to Google Earth Engine with project ID authentication
- **Interactive Area of Interest**: Draw polygons directly on a map or upload shapefiles/GeoJSON
- **Extensive Dataset Catalog**: Search through 54+ climate datasets across multiple categories
- **Band Selection**: Choose specific bands from multi-band datasets
- **Temporal Filtering**: Select time ranges for time-series data
- **Multiple Export Options**: Download directly to local disk or export to Google Drive
- **Extraction Modes**: Average time-series or gridded raster outputs

## Available Datasets

GeoClimate-Fetcher includes catalogs for 54 datasets across 6 categories:

- **Precipitation** (11 datasets): ERA5, GLDAS, and others
- **Temperature** (10 datasets): MODIS LST, ERA5, and others
- **Soil Moisture** (8 datasets): SMAP, GLDAS, and others
- **Evapotranspiration** (17 datasets): TerraClimate, MODIS, and others
- **NDVI** (5 datasets): MODIS, Sentinel-2, and others
- **DEM** (3 datasets): SRTM, NASADEM, ALOS

## Example Code

```python
from geoclimate_fetcher.core import (
    authenticate, MetadataCatalog, GeometryHandler, 
    GEEExporter, ImageCollectionFetcher
)

# Authenticate with GEE
auth = authenticate("your-project-id")

# Create a geometry for San Francisco Bay Area
geometry_handler = GeometryHandler()
sf_bbox = {
    "type": "Polygon",
    "coordinates": [[
        [-122.6, 37.2], [-122.6, 37.9], 
        [-121.8, 37.9], [-121.8, 37.2], 
        [-122.6, 37.2]
    ]]
}
geometry = geometry_handler.set_geometry_from_geojson(sf_bbox, "sf_bay_area")

# Find ERA5 precipitation dataset
catalog = MetadataCatalog()
dataset = catalog.get_dataset_by_name("ERA5 Daily Aggregates")
ee_id = dataset['Earth Engine ID']

# Download time series for January 2021
fetcher = ImageCollectionFetcher(ee_id, ["total_precipitation"], geometry)
fetcher.filter_dates("2021-01-01", "2021-01-31")
data = fetcher.get_time_series_average()

# Save to CSV
exporter = GEEExporter()
exporter.export_time_series_to_csv(data, "precipitation_data.csv")
```

## Troubleshooting

- **Authentication Issues**: Make sure you have a valid Google Earth Engine account and the correct project ID.
- **Installation Problems**: If you encounter issues with dependencies, try installing them manually:
  ```
  pip install earthengine-api geemap ipyleaflet ipywidgets pandas xarray rapidfuzz tqdm rich geopandas rasterio rio-cogeo streamlit jupyterlab
  ```
- **ModuleNotFoundError**: If you encounter a module not found error, make sure your virtual environment is activated and the package is installed in development mode.
- **Permission Error During Installation**: If you get an access denied error during installation, try running the install command again as mentioned in the installation steps.

## License

MIT

## Contact

- **Email**: saurav.bhattarai.1999@gmail.com
- **Lab Website**: [JSU Water Resources Lab](https://bit.ly/jsu_water)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
