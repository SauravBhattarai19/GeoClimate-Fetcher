"""
Utility functions shared across the application
"""
import streamlit as st
import pandas as pd
import ee
import geemap
import tempfile
import os
from datetime import datetime
from pathlib import Path


def go_back_to_step(step_name):
    """Navigate back to a specific step in the application"""
    if step_name == "home":
        st.session_state.app_mode = None
        st.rerun()
    elif step_name == "geometry":
        # Go back to step 1: Reset all selections
        # Reset both variable names (geometry_selected and geometry_complete) for compatibility
        st.session_state.geometry_selected = False
        st.session_state.geometry_complete = False
        st.session_state.dataset_selected = False
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False

        # Also reset geometry handler state if it exists
        if 'geometry_handler' in st.session_state:
            try:
                st.session_state.geometry_handler._current_geometry = None
                st.session_state.geometry_handler._current_geometry_name = None
            except:
                pass

        # Clear any download state
        if 'download_complete' in st.session_state:
            st.session_state.download_complete = False
        if 'download_results' in st.session_state:
            st.session_state.download_results = None

        st.rerun()
    elif step_name == "dataset":
        # Go back to step 2: Keep geometry, reset dataset and onwards
        st.session_state.dataset_selected = False
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False
        st.rerun()
    elif step_name == "bands":
        # Go back to step 3: Keep geometry and dataset, reset bands and onwards
        st.session_state.bands_selected = False
        st.session_state.dates_selected = False
        st.rerun()
    elif step_name == "dates":
        # Go back to step 4: Keep geometry, dataset and bands, reset dates
        st.session_state.dates_selected = False
        st.rerun()
    else:
        # Handle other navigation steps if needed
        st.session_state.app_mode = step_name
        st.rerun()


def _detect_temporal_resolution(ee_id):
    """
    Detect temporal resolution based on dataset ID
    Returns: 'daily', 'hourly', '30min', etc.
    """
    ee_id_lower = ee_id.lower()

    # Known hourly datasets
    if any(x in ee_id_lower for x in ['gpm_l3/imerg', 'era5', 'gfs']):
        return 'hourly'

    # Known 30-minute datasets
    if any(x in ee_id_lower for x in ['goes', 'abi']):
        return '30min'

    # Most climate datasets are daily
    return 'daily'


def get_bands_for_dataset(dataset_name):
    """
    Get bands for a dataset using STAC API or CSV fallback.

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of band names
    """
    import os
    import pandas as pd
    from pathlib import Path

    # Try STAC-powered MetadataCatalog first
    try:
        from geoclimate_fetcher.core.metadata import MetadataCatalog
        catalog = MetadataCatalog(use_stac=True)
        bands = catalog.get_bands_for_dataset(dataset_name)
        if bands:
            return bands
    except Exception as e:
        print(f"STAC lookup failed, falling back to CSV: {e}")

    # Fallback to direct CSV reading
    data_dir = Path('geoclimate_fetcher') / 'data'
    if not data_dir.exists():
        return []

    # Try to find the dataset in any CSV file
    for csv_file in data_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'Dataset Name' not in df.columns or 'Band Names' not in df.columns:
                continue

            # Find the dataset
            dataset_row = df[df['Dataset Name'] == dataset_name]
            if not dataset_row.empty:
                bands_str = dataset_row.iloc[0].get('Band Names', '')
                if isinstance(bands_str, str) and bands_str:
                    return [band.strip() for band in bands_str.split(',')]
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # If not found, try the Datasets.csv file specifically
    datasets_file = data_dir / 'Datasets.csv'
    if datasets_file.exists():
        try:
            df = pd.read_csv(datasets_file)
            dataset_row = df[df['Dataset Name'] == dataset_name]
            if not dataset_row.empty:
                bands_str = dataset_row.iloc[0].get('Band Names', '')
                if isinstance(bands_str, str) and bands_str:
                    return [band.strip() for band in bands_str.split(',')]
        except Exception as e:
            print(f"Error reading Datasets.csv: {e}")

    return []


def download_ee_data_simple(dataset, bands, geometry, start_date, end_date, export_format='GeoTIFF', scale=30):
    """
    Simplified download function using proven geemap approach.
    Based on the working implementation from the backup app.

    Args:
        dataset: Dataset dictionary with ee_id and snippet_type
        bands: List of band names to download
        geometry: Earth Engine geometry
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        export_format: Output format ('GeoTIFF', 'CSV')
        scale: Pixel resolution in meters

    Returns:
        dict: {'success': bool, 'file_path': str, 'file_data': bytes, 'message': str}
    """
    try:
        ee_id = dataset.get('ee_id')
        snippet_type = dataset.get('snippet_type', 'Image')

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            temp_path = temp_file.name

        # Process based on dataset type
        if snippet_type == 'ImageCollection':
            return _download_image_collection_simple(
                ee_id, bands, geometry, start_date, end_date, temp_path, export_format, scale
            )
        else:
            return _download_image_simple(
                ee_id, bands, geometry, temp_path, export_format, scale
            )

    except Exception as e:
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"Download failed: {str(e)}"
        }


def _download_image_simple(ee_id, bands, geometry, temp_path, export_format, scale):
    """Download a single Earth Engine Image using geemap"""
    try:
        # Load the image
        image = ee.Image(ee_id)

        # Select bands if specified
        if bands:
            image = image.select(bands)

        # Clip to geometry
        image = image.clip(geometry)

        if export_format == 'CSV':
            # For CSV, get zonal statistics
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=2e9
            )

            # Convert to DataFrame
            stats_dict = stats.getInfo()
            if stats_dict:
                df = pd.DataFrame([stats_dict])
                df.insert(0, 'geometry_area_km2', geometry.area().divide(1000000).getInfo())

                # Save to temporary CSV
                csv_path = temp_path.replace('.tif', '.csv')
                df.to_csv(csv_path, index=False)

                # Read back as bytes
                with open(csv_path, 'rb') as f:
                    file_data = f.read()

                os.unlink(csv_path)  # Clean up

                return {
                    'success': True,
                    'file_path': csv_path,
                    'file_data': file_data,
                    'message': f"CSV generated with {len(stats_dict)} bands"
                }
            else:
                return {
                    'success': False,
                    'file_path': None,
                    'file_data': None,
                    'message': "No data available for the selected region"
                }

        else:  # GeoTIFF
            # Use geemap's reliable export function
            geemap.ee_export_image(
                image,
                filename=temp_path,
                scale=scale,
                region=geometry,
                file_per_band=False
            )

            # Check if file was created
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # Read file as bytes
                with open(temp_path, 'rb') as f:
                    file_data = f.read()

                file_size_mb = len(file_data) / (1024 * 1024)

                return {
                    'success': True,
                    'file_path': temp_path,
                    'file_data': file_data,
                    'message': f"GeoTIFF exported successfully ({file_size_mb:.1f} MB)"
                }
            else:
                return {
                    'success': False,
                    'file_path': None,
                    'file_data': None,
                    'message': "Export failed - no file generated"
                }

    except Exception as e:
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"Image processing error: {str(e)}"
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


def _download_image_collection_simple(ee_id, bands, geometry, start_date, end_date, temp_path, export_format, scale):
    """Download Earth Engine ImageCollection using simplified approach"""
    try:
        # Load the collection
        collection = ee.ImageCollection(ee_id)

        # Apply date filter if dates are provided
        if start_date and end_date:
            try:
                # Convert string dates to proper format for Earth Engine
                start_ee = ee.Date(str(start_date))
                end_ee = ee.Date(str(end_date))
                collection = collection.filterDate(start_ee, end_ee)
            except Exception as date_error:
                return {
                    'success': False,
                    'file_path': None,
                    'file_data': None,
                    'message': f"Date filter error: {str(date_error)}. Check date format (YYYY-MM-DD)"
                }

        # Filter by bounds
        collection = collection.filterBounds(geometry)

        # Select bands if specified
        if bands:
            collection = collection.select(bands)

        # Check collection size
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            return {
                'success': False,
                'file_path': None,
                'file_data': None,
                'message': "No images found for the specified date range and region"
            }

        if export_format == 'CSV':
            # Use the new chunked processing system
            try:
                from geoclimate_fetcher.core.download_utils import process_image_collection_chunked

                # Detect temporal resolution (basic heuristic)
                temporal_resolution = _detect_temporal_resolution(ee_id)

                return process_image_collection_chunked(
                    collection=collection,
                    bands=bands,
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    export_format='CSV',
                    scale=scale,
                    temporal_resolution=temporal_resolution
                )

            except Exception as csv_error:
                return {
                    'success': False,
                    'file_path': None,
                    'file_data': None,
                    'message': f"CSV processing failed: {str(csv_error)}"
                }

        elif export_format == 'GeoTIFF':
            # Use the new chunked processing system for GeoTIFF
            try:
                from geoclimate_fetcher.core.download_utils import process_image_collection_chunked

                # Detect temporal resolution (basic heuristic)
                temporal_resolution = _detect_temporal_resolution(ee_id)

                return process_image_collection_chunked(
                    collection=collection,
                    bands=bands,
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    export_format='GeoTIFF',
                    scale=scale,
                    temporal_resolution=temporal_resolution
                )
            except Exception as tiff_error:
                return {
                    'success': False,
                    'file_path': None,
                    'file_data': None,
                    'message': f"GeoTIFF processing failed: {str(tiff_error)}"
                }

        elif export_format == 'NetCDF':
            # For NetCDF, create proper multi-dimensional structure with time axis
            try:
                return _export_netcdf_with_time(collection, temp_path, geometry, scale, collection_size, ee_id)
            except Exception as nc_error:
                return {
                    'success': False,
                    'file_path': None,
                    'file_data': None,
                    'message': f"NetCDF processing failed: {str(nc_error)}"
                }

        else:
            return {
                'success': False,
                'file_path': None,
                'file_data': None,
                'message': f"Unsupported export format: {export_format}"
            }

    except Exception as e:
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"ImageCollection processing error: {str(e)}"
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

def _export_individual_geotiffs(collection, temp_path, geometry, scale, collection_size):
    """Export individual GeoTIFF files for small collections with time info"""
    import zipfile
    from pathlib import Path

    # Create temporary directory for individual files
    temp_dir = Path(temp_path).parent / 'geotiff_collection'
    temp_dir.mkdir(exist_ok=True)

    try:
        # Convert collection to list and export each image
        images_list = collection.toList(collection_size)
        exported_files = []

        # Create progress placeholder for larger collections
        progress_placeholder = st.empty() if collection_size > 5 else None

        for i in range(collection_size):
            try:
                # Show progress for larger collections - replace, don't add
                if progress_placeholder and i % 5 == 0:
                    progress_placeholder.info(f"Exporting GeoTIFF {i+1}/{collection_size}...")

                image = ee.Image(images_list.get(i))

                # Get image date for filename
                date_str = image.date().format('YYYY_MM_dd').getInfo()
                time_str = image.date().format('HH_mm_ss').getInfo()

                # Create filename with time info
                filename = f"image_{date_str}_{time_str}.tif"
                file_path = temp_dir / filename

                # Clip to geometry
                clipped_image = image.clip(geometry)

                # Export using geemap
                geemap.ee_export_image(
                    clipped_image,
                    filename=str(file_path),
                    scale=scale,
                    region=geometry,
                    file_per_band=False
                )

                if file_path.exists() and file_path.stat().st_size > 0:
                    exported_files.append(file_path)
                else:
                    st.warning(f"Failed to export image {i+1}: {filename}")

            except Exception as img_error:
                st.warning(f"Error exporting image {i+1}: {str(img_error)}")
                continue

        if exported_files:
            # Create ZIP file containing all GeoTIFFs
            zip_path = temp_path.replace('.tif', '_collection.zip')

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in exported_files:
                    zipf.write(file_path, file_path.name)

            # Read ZIP as bytes
            with open(zip_path, 'rb') as f:
                file_data = f.read()

            # Cleanup
            for file_path in exported_files:
                file_path.unlink()
            temp_dir.rmdir()
            os.unlink(zip_path)

            file_size_mb = len(file_data) / (1024 * 1024)

            return {
                'success': True,
                'file_path': zip_path,
                'file_data': file_data,
                'message': f"ZIP archive with {len(exported_files)} individual daily GeoTIFF files (each with {len(bands) if bands else 'all'} bands) ({file_size_mb:.1f} MB)"
            }
        else:
            return {
                'success': False,
                'file_path': None,
                'file_data': None,
                'message': "No GeoTIFF files could be exported"
            }

    except Exception as e:
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"Individual GeoTIFF export failed: {str(e)}"
        }
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def _export_netcdf_with_time(collection, temp_path, geometry, scale, collection_size, ee_id):
    """Export NetCDF with proper multi-dimensional structure and time axis"""
    try:
        # For now, return an informative message that NetCDF is not yet fully implemented
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"NetCDF export is not yet implemented. Please use CSV for time series data or GeoTIFF for spatial data. Collection has {collection_size} images."
        }
    except Exception as e:
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"NetCDF processing failed: {str(e)}"
        }


def _export_composite_geotiff(collection, temp_path, geometry, scale, collection_size, start_date, end_date):
    """Export temporal composite GeoTIFF with time info in metadata"""
    try:
        # Create median composite for large collections
        composite = collection.median()

        # Clip to geometry
        composite = composite.clip(geometry)

        # Export using geemap with metadata
        geemap.ee_export_image(
            composite,
            filename=temp_path,
            scale=scale,
            region=geometry,
            file_per_band=False
        )

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            # Read file as bytes
            with open(temp_path, 'rb') as f:
                file_data = f.read()

            file_size_mb = len(file_data) / (1024 * 1024)

            return {
                'success': True,
                'file_path': temp_path,
                'file_data': file_data,
                'message': f"Temporal composite GeoTIFF from {collection_size} images ({start_date} to {end_date}) ({file_size_mb:.1f} MB)"
            }
        else:
            return {
                'success': False,
                'file_path': None,
                'file_data': None,
                'message': "Composite GeoTIFF export failed"
            }

    except Exception as e:
        return {
            'success': False,
            'file_path': None,
            'file_data': None,
            'message': f"Composite GeoTIFF creation failed: {str(e)}"
        }
