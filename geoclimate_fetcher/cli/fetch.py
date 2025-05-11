"""
Command-line interface for GeoClimate-Fetcher.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import ee
from typing import Dict, List, Optional, Union, Any

from geoclimate_fetcher.core import (
    authenticate, 
    MetadataCatalog, 
    GeometryHandler,
    GEEExporter, 
    ImageCollectionFetcher,
    StaticRasterFetcher
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("geoclimate_fetcher")

def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="GeoClimate-Fetcher: Download climate data from Google Earth Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Authentication
    auth_group = parser.add_argument_group("Authentication")
    auth_group.add_argument(
        "--project-id", 
        required=True,
        help="Google Earth Engine project ID"
    )
    auth_group.add_argument(
        "--service-account", 
        help="Service account email (optional)"
    )
    auth_group.add_argument(
        "--key-file", 
        help="Path to service account key file (optional)"
    )
    
    # Area of Interest
    aoi_group = parser.add_argument_group("Area of Interest")
    aoi_group.add_argument(
        "--geojson", 
        help="Path to GeoJSON file with area of interest"
    )
    aoi_group.add_argument(
        "--shapefile", 
        help="Path to Shapefile with area of interest"
    )
    aoi_group.add_argument(
        "--bbox", 
        nargs=4, 
        type=float, 
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Bounding box coordinates (xmin, ymin, xmax, ymax)"
    )
    
    # Dataset
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--dataset-name", 
        help="Name of the dataset to download"
    )
    dataset_group.add_argument(
        "--dataset-id", 
        help="Earth Engine ID of the dataset to download"
    )
    dataset_group.add_argument(
        "--list-datasets", 
        action="store_true",
        help="List available datasets and exit"
    )
    dataset_group.add_argument(
        "--search", 
        help="Search for datasets matching the query"
    )
    dataset_group.add_argument(
        "--category", 
        choices=["precipitation", "temperature", "soil_moisture", 
                "evapotranspiration", "ndvi", "dem"],
        help="Filter datasets by category"
    )
    
    # Bands
    band_group = parser.add_argument_group("Bands")
    band_group.add_argument(
        "--bands", 
        nargs="+", 
        help="Space-separated list of bands to download"
    )
    band_group.add_argument(
        "--list-bands", 
        action="store_true",
        help="List available bands for the specified dataset and exit"
    )
    
    # Time range
    time_group = parser.add_argument_group("Time Range")
    time_group.add_argument(
        "--start-date", 
        help="Start date (YYYY-MM-DD)"
    )
    time_group.add_argument(
        "--end-date", 
        help="End date (YYYY-MM-DD)"
    )
    
    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--extraction-mode", 
        choices=["average", "gridded"], 
        default="average",
        help="Extraction mode for the data"
    )
    export_group.add_argument(
        "--export-mode", 
        choices=["local", "drive"], 
        default="local",
        help="Export mode (local or Google Drive)"
    )
    export_group.add_argument(
        "--output-format", 
        choices=["csv", "netcdf", "geotiff", "cog"], 
        default="csv",
        help="Output file format"
    )
    export_group.add_argument(
        "--scale", 
        type=float, 
        default=1000.0,
        help="Pixel scale in meters"
    )
    export_group.add_argument(
        "--output-dir", 
        default="downloads",
        help="Output directory for local downloads"
    )
    export_group.add_argument(
        "--drive-folder", 
        default="GeoClimateFetcher",
        help="Google Drive folder for Drive exports"
    )
    export_group.add_argument(
        "--filename", 
        help="Output filename (without extension)"
    )
    
    # Misc
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser

def list_datasets(catalog: MetadataCatalog, category: Optional[str] = None, 
                search_query: Optional[str] = None) -> None:
    """
    List available datasets.
    
    Args:
        catalog: Metadata catalog
        category: Optional category to filter by
        search_query: Optional search query
    """
    try:
        if category and category != "all":
            df = catalog.get_datasets_by_category(category)
        elif search_query:
            df = catalog.search_datasets(search_query)
        else:
            df = catalog.all_datasets
            
        # Limit columns for display
        display_df = df[["Dataset Name", "Earth Engine ID", "Provider", 
                        "Start Date", "End Date", "Pixel Size (m)"]]
        
        # Print table
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(show_header=True, header_style="bold")
        
        for col in display_df.columns:
            table.add_column(col)
            
        for _, row in display_df.iterrows():
            table.add_row(*[str(val) for val in row])
            
        console.print(table)
        print(f"\nTotal datasets: {len(df)}")
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        sys.exit(1)

def list_bands(catalog: MetadataCatalog, dataset_name: str) -> None:
    """
    List available bands for a dataset.
    
    Args:
        catalog: Metadata catalog
        dataset_name: Name of the dataset
    """
    try:
        bands = catalog.get_bands_for_dataset(dataset_name)
        
        if not bands:
            print(f"No bands found for dataset: {dataset_name}")
            return
            
        print(f"Available bands for {dataset_name}:")
        for band in bands:
            print(f"  - {band}")
            
    except Exception as e:
        logger.error(f"Error listing bands: {str(e)}")
        sys.exit(1)

def create_geometry(geometry_handler: GeometryHandler, args: argparse.Namespace) -> ee.Geometry:
    """
    Create a geometry from the provided arguments.
    
    Args:
        geometry_handler: Geometry handler
        args: Command-line arguments
        
    Returns:
        Earth Engine geometry
    """
    try:
        if args.geojson:
            # Load GeoJSON file
            with open(args.geojson, 'r') as f:
                geojson_dict = json.load(f)
                
            return geometry_handler.set_geometry_from_geojson(geojson_dict)
            
        elif args.shapefile:
            # Load Shapefile
            return geometry_handler.set_geometry_from_file(args.shapefile)
            
        elif args.bbox:
            # Create bbox geometry
            xmin, ymin, xmax, ymax = args.bbox
            bbox_geojson = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax],
                        [xmin, ymin]
                    ]
                ]
            }
            return geometry_handler.set_geometry_from_geojson(bbox_geojson, "bbox")
            
        else:
            logger.error("No area of interest specified. Use --geojson, --shapefile, or --bbox.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error creating geometry: {str(e)}")
        sys.exit(1)

def main() -> None:
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    logger.debug("Starting GeoClimate-Fetcher CLI")
    
    # Initialize metadata catalog
    try:
        catalog = MetadataCatalog()
        logger.debug(f"Loaded {len(catalog.all_datasets)} datasets")
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        sys.exit(1)
        
    # Handle list/search actions that don't require authentication
    if args.list_datasets:
        list_datasets(catalog, args.category, args.search)
        return
        
    # Authenticate with Earth Engine
    try:
        logger.info(f"Authenticating with project ID: {args.project_id}")
        auth = authenticate(args.project_id, args.service_account, args.key_file)
        
        if not auth.is_initialized():
            logger.error("Authentication failed")
            sys.exit(1)
            
        logger.info("Authentication successful")
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        sys.exit(1)
        
    # Handle operations that require authentication but not geometry
    if args.list_bands:
        if not args.dataset_name:
            logger.error("Dataset name required with --list-bands")
            sys.exit(1)
            
        list_bands(catalog, args.dataset_name)
        return
        
    # Create the geometry handler and set up the AOI
    geometry_handler = GeometryHandler()
    
    # Create geometry from args
    geometry = create_geometry(geometry_handler, args)
    
    # Validate the geometry
    valid, error = geometry_handler.validate_geometry()
    if not valid:
        logger.error(f"Invalid geometry: {error}")
        sys.exit(1)
        
    area = geometry_handler.get_geometry_area()
    logger.info(f"Area of interest: {area:.2f} km²")
    
    # Resolve dataset
    dataset_id = args.dataset_id
    dataset_name = args.dataset_name
    
    if not dataset_id and not dataset_name:
        logger.error("Either --dataset-name or --dataset-id is required")
        sys.exit(1)
        
    # If only name is provided, get the ID from the catalog
    if dataset_name and not dataset_id:
        dataset_id = catalog.get_ee_id(dataset_name)
        if not dataset_id:
            logger.error(f"Dataset name not found: {dataset_name}")
            sys.exit(1)
            
    # If only ID is provided, try to find the name
    if dataset_id and not dataset_name:
        # Find the dataset with this ID
        if catalog.all_datasets is not None:
            matching = catalog.all_datasets[
                catalog.all_datasets['Earth Engine ID'] == dataset_id
            ]
            if not matching.empty:
                dataset_name = matching.iloc[0]['Dataset Name']
                
    # Resolve bands
    bands = args.bands
    if not bands:
        if dataset_name:
            bands = catalog.get_bands_for_dataset(dataset_name)
            if not bands:
                logger.error("No bands found for dataset")
                sys.exit(1)
                
            logger.info(f"Using all available bands: {', '.join(bands)}")
        else:
            logger.error("Band selection required with --bands")
            sys.exit(1)
            
    # Determine snippet type
    snippet_type = None
    if dataset_name:
        snippet_type = catalog.get_snippet_type(dataset_name)
        
    if not snippet_type:
        # Try to determine by checking if it's an ImageCollection
        try:
            # Try as Image first
            ee.Image(dataset_id).getInfo()
            snippet_type = "Image"
        except:
            try:
                # Try as ImageCollection
                ee.ImageCollection(dataset_id).getInfo()
                snippet_type = "ImageCollection"
            except:
                logger.error(f"Unable to determine if {dataset_id} is an Image or ImageCollection")
                sys.exit(1)
                
    logger.info(f"Dataset type: {snippet_type}")
    
    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD.")
            sys.exit(1)
            
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD.")
            sys.exit(1)
            
    # Create exporter
    exporter = GEEExporter()
    
    # Set up the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    filename = args.filename
    if not filename:
        if dataset_name:
            sanitized_name = dataset_name.replace(' ', '_').lower()
        else:
            sanitized_name = dataset_id.split('/')[-1].lower()
            
        geometry_name = geometry_handler.current_geometry_name or 'custom_aoi'
        sanitized_geometry = geometry_name.replace(' ', '_').lower()
        
        filename = f"{sanitized_name}_{sanitized_geometry}"
        
    # Process based on snippet type
    try:
        if snippet_type == "ImageCollection":
            # Ensure dates are provided
            if not start_date or not end_date:
                logger.error("Start and end dates are required for ImageCollection datasets")
                sys.exit(1)
                
            logger.info(f"Fetching {dataset_id} from {start_date} to {end_date}")
            
            # Create fetcher
            fetcher = ImageCollectionFetcher(dataset_id, bands, geometry)
            fetcher.filter_dates(start_date, end_date)
            
            # Extract data
            if args.extraction_mode == 'average':
                # Check if the dataset is potentially large
                logger.info("Checking dataset size...")
                # Use chunked processing for potentially large datasets
                logger.info("Using chunked processing for time series to handle large datasets")
                data = fetcher.get_time_series_average_chunked()
                
                # Export
                if args.export_mode == 'local':
                    # Local export
                    if args.output_format == 'csv':
                        file_path = output_dir / f"{filename}.csv"
                        logger.info(f"Saving to {file_path}")
                        exporter.export_time_series_to_csv(data, file_path)
                    else:
                        # Error handling code remains the same
                        pass
                else:  # Google Drive
                    # Export to Drive with chunking
                    logger.info(f"Exporting to Google Drive folder '{args.drive_folder}' using chunked processing")
                    task_ids = exporter.export_time_series_to_drive_chunked(
                        data, filename, args.drive_folder
                    )
                    logger.info(f"Started {len(task_ids)} export tasks to Google Drive")

            else:  # gridded
                # Get gridded data
                logger.info("Fetching gridded data")
                data = fetcher.get_gridded_data(scale=args.scale)
                
                # Export
                if args.export_mode == 'local':
                    # Local export
                    if args.output_format == 'netcdf':
                        file_path = output_dir / f"{filename}.nc"
                        logger.info(f"Saving to {file_path}")
                        exporter.export_gridded_data_to_netcdf(data, file_path)
                    else:
                        logger.error(f"Unsupported output format for gridded time series: {args.output_format}")
                        logger.error("Use netcdf format for gridded time series data")
                        sys.exit(1)
                        
                else:  # Google Drive
                    logger.error("Export to Drive for gridded time series not fully implemented")
                    logger.error("Please use local export with NetCDF format")
                    sys.exit(1)
                    
        else:  # Image
            logger.info(f"Fetching {dataset_id}")
            
            # Create fetcher
            fetcher = StaticRasterFetcher(dataset_id, bands, geometry)
            
            # Extract data
            if args.extraction_mode == 'average':
                # Get mean values
                logger.info("Calculating spatial averages")
                data = fetcher.get_mean_values()
                
                # Convert to DataFrame
                import pandas as pd
                df = pd.DataFrame([data])
                
                # Export
                if args.export_mode == 'local':
                    # Local export
                    if args.output_format == 'csv':
                        file_path = output_dir / f"{filename}.csv"
                        logger.info(f"Saving to {file_path}")
                        exporter.export_time_series_to_csv(df, file_path)
                    else:
                        logger.error(f"Unsupported output format for spatial averages: {args.output_format}")
                        logger.error("Use csv format for spatial average data")
                        sys.exit(1)
                        
                else:  # Google Drive
                    # Create a feature collection from the data
                    fc = ee.FeatureCollection([ee.Feature(None, data)])
                    
                    # Export to Drive
                    logger.info(f"Exporting to Google Drive folder '{args.drive_folder}'")
                    
                    exporter.export_table_to_drive(
                        fc, filename, args.drive_folder, wait=True
                    )
                    
            else:  # gridded
                # Create the image with only selected bands
                image = ee.Image(dataset_id).select(bands)
                
                # Export
                if args.export_mode == 'local':
                    # Local export
                    if args.output_format == 'geotiff':
                        file_path = output_dir / f"{filename}.tif"
                        logger.info(f"Saving to {file_path}")
                        exporter.export_image_to_local(image, file_path, geometry, args.scale)
                        
                    elif args.output_format == 'cog':
                        file_path = output_dir / f"{filename}.tif"
                        logger.info(f"Saving to {file_path} as Cloud Optimized GeoTIFF")
                        exporter.export_to_cloud_optimized_geotiff(image, file_path, geometry, args.scale)
                        
                    else:  # netcdf
                        # Get pixel values
                        logger.info("Fetching pixel values")
                        pixel_data = fetcher.get_pixel_values(scale=args.scale)
                        
                        # Convert to xarray
                        import xarray as xr
                        import numpy as np
                        
                        # Get arrays and dimensions
                        first_band = next(iter(pixel_data.values()))
                        height, width = first_band.shape
                        
                        # Create dummy coordinates
                        coords = {
                            'y': np.arange(height),
                            'x': np.arange(width)
                        }
                        
                        data_vars = {
                            band: (['y', 'x'], array)
                            for band, array in pixel_data.items()
                        }
                        
                        ds = xr.Dataset(data_vars=data_vars, coords=coords)
                        
                        # Save to NetCDF
                        file_path = output_dir / f"{filename}.nc"
                        logger.info(f"Saving to {file_path}")
                        exporter.export_gridded_data_to_netcdf(ds, file_path)
                        
                else:  # Google Drive
                    logger.info(f"Exporting to Google Drive folder '{args.drive_folder}'")
                    
                    exporter.export_image_to_drive(
                        image, filename, args.drive_folder, geometry, args.scale, wait=True
                    )
                    
        logger.info("Export complete!")
        
    except Exception as e:
        logger.error(f"Error during export: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()