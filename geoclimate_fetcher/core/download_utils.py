"""
Shared download utilities for Earth Engine data processing
Can be used across multiple modules (geodata_explorer, hydrology_analyzer, etc.)
"""
import ee
import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import geemap


def calculate_optimal_chunk_size(temporal_resolution, total_days):
    """
    Calculate optimal chunk size based on temporal resolution and total time period

    Args:
        temporal_resolution: 'daily', 'hourly', '30min', etc.
        total_days: Total number of days in the time period

    Returns:
        tuple: (chunk_size_days, max_images_per_chunk)
    """
    # Earth Engine has limits around 5000 elements per collection operation
    MAX_ELEMENTS = 5000

    if temporal_resolution == 'daily':
        # Daily: 365 images per year, can handle ~13 years
        images_per_day = 1
        max_days_per_chunk = min(MAX_ELEMENTS // images_per_day, 3650)  # Max 10 years
        chunk_days = min(total_days, max_days_per_chunk)

    elif temporal_resolution == 'hourly':
        # Hourly: 8760 images per year, need smaller chunks
        images_per_day = 24
        max_days_per_chunk = min(MAX_ELEMENTS // images_per_day, 180)  # ~6 months
        chunk_days = min(total_days, max_days_per_chunk)

    elif temporal_resolution == '30min':
        # 30-minute: 17520 images per year, need even smaller chunks
        images_per_day = 48
        max_days_per_chunk = min(MAX_ELEMENTS // images_per_day, 90)  # ~3 months
        chunk_days = min(total_days, max_days_per_chunk)

    elif temporal_resolution == 'yearly':
        # Yearly: Climate indices already aggregated to 1 image per year
        # Don't chunk - process the entire period as one
        images_per_day = 1/365  # 1 image per year
        chunk_days = total_days  # Process entire period

    elif temporal_resolution == 'monthly':
        # Monthly: Climate indices already aggregated to 1 image per month
        # Don't chunk - process the entire period as one
        images_per_day = 1/30  # 1 image per month
        chunk_days = total_days  # Process entire period

    else:
        # Default: assume daily
        images_per_day = 1
        max_days_per_chunk = min(MAX_ELEMENTS // images_per_day, 1825)  # 5 years
        chunk_days = min(total_days, max_days_per_chunk)

    max_images = chunk_days * images_per_day

    return chunk_days, max_images


def create_date_chunks(start_date, end_date, chunk_days):
    """
    Create date chunks for processing large time periods

    Args:
        start_date: Start date string (YYYY-MM-DD) or datetime.date object
        end_date: End date string (YYYY-MM-DD) or datetime.date object
        chunk_days: Number of days per chunk

    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    from datetime import datetime, timedelta, date

    chunks = []

    # Handle both string and datetime.date inputs
    if isinstance(start_date, date):
        current_start = datetime.combine(start_date, datetime.min.time())
    elif isinstance(start_date, str):
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        current_start = start_date  # Assume it's already a datetime

    if isinstance(end_date, date):
        end_dt = datetime.combine(end_date, datetime.min.time())
    elif isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = end_date  # Assume it's already a datetime

    while current_start <= end_dt:
        chunk_end = min(current_start + timedelta(days=chunk_days - 1), end_dt)
        chunks.append((
            current_start.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        ))
        current_start = chunk_end + timedelta(days=1)

    return chunks


def process_image_collection_chunked(collection, bands, geometry, start_date, end_date,
                                   export_format, scale, temporal_resolution='daily'):
    """
    Process large image collections using chunking mechanism

    Args:
        collection: Earth Engine ImageCollection
        bands: List of band names
        geometry: Earth Engine geometry
        start_date: Start date string
        end_date: End date string
        export_format: 'CSV', 'GeoTIFF', 'NetCDF'
        scale: Pixel resolution in meters
        temporal_resolution: 'daily', 'hourly', '30min'

    Returns:
        Dict with success status and results
    """
    try:
        # Calculate total time period - handle both string and date object inputs
        from datetime import datetime, date

        if isinstance(start_date, date):
            start_dt = datetime.combine(start_date, datetime.min.time())
        elif isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date

        if isinstance(end_date, date):
            end_dt = datetime.combine(end_date, datetime.min.time())
        elif isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date

        total_days = (end_dt - start_dt).days + 1

        # Get optimal chunk size
        chunk_days, max_images_per_chunk = calculate_optimal_chunk_size(temporal_resolution, total_days)

        # If small enough, process normally
        if total_days <= chunk_days:
            # Convert dates to strings for consistency
            start_str = start_dt.strftime('%Y-%m-%d')
            end_str = end_dt.strftime('%Y-%m-%d')
            return _process_single_chunk(collection, bands, geometry, start_str, end_str,
                                       export_format, scale, temporal_resolution)

        # Large collection - process in chunks
        st.info(f"Large time period detected ({total_days} days). Processing in chunks of {chunk_days} days...")

        # Convert dates to strings for chunking
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        date_chunks = create_date_chunks(start_str, end_str, chunk_days)

        if export_format == 'CSV':
            return _process_csv_chunks(collection, bands, geometry, date_chunks, scale, temporal_resolution)
        elif export_format == 'GeoTIFF':
            return _process_geotiff_chunks(collection, bands, geometry, date_chunks, scale, temporal_resolution)
        else:
            return {
                'success': False,
                'message': f"Chunked processing not yet implemented for {export_format}"
            }

    except Exception as e:
        return {
            'success': False,
            'message': f"Chunked processing failed: {str(e)}"
        }


def _process_single_chunk(collection, bands, geometry, start_date, end_date,
                         export_format, scale, temporal_resolution):
    """Process a single chunk (normal processing)"""
    # Filter collection
    filtered_collection = collection.filterDate(start_date, end_date).filterBounds(geometry)

    if bands:
        filtered_collection = filtered_collection.select(bands)

    collection_size = filtered_collection.size().getInfo()

    if collection_size == 0:
        return {
            'success': False,
            'message': "No images found for the specified date range and region"
        }

    if export_format == 'CSV':
        return _export_csv_single_chunk(filtered_collection, geometry, scale, collection_size)
    elif export_format == 'GeoTIFF':
        return _export_geotiff_single_chunk(filtered_collection, geometry, scale, collection_size)

    return {'success': False, 'message': f"Unsupported format: {export_format}"}


def _process_csv_chunks(collection, bands, geometry, date_chunks, scale, temporal_resolution):
    """Process CSV export using optimized chunked fetcher"""
    all_dfs = []
    chunk_progress = st.empty()

    try:
        # Get collection EE ID for creating new fetchers
        ee_id = collection.getInfo()['id'] if hasattr(collection, 'getInfo') else None
        if not ee_id:
            # Fallback to slow method if we can't get EE ID
            return _process_csv_chunks_legacy(collection, bands, geometry, date_chunks, scale, temporal_resolution)

        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            chunk_progress.info(f"Processing optimized chunk {chunk_idx + 1}/{len(date_chunks)}: {chunk_start} to {chunk_end}")

            try:
                # Create optimized fetcher for this chunk
                from geoclimate_fetcher.core.fetchers.collection import ImageCollectionFetcher
                chunk_fetcher = ImageCollectionFetcher(ee_id, bands, geometry)
                chunk_fetcher = chunk_fetcher.filter_dates(chunk_start, chunk_end)

                # Use optimized method
                chunk_df = chunk_fetcher.get_time_series_average(
                    export_format='CSV',
                    user_scale=scale,
                    dataset_native_scale=None  # Will be optimized automatically
                )

                if chunk_df is not None and not chunk_df.empty:
                    all_dfs.append(chunk_df)
                    chunk_progress.info(f"Chunk {chunk_idx + 1}: Retrieved {len(chunk_df)} optimized records")
                else:
                    chunk_progress.info(f"Chunk {chunk_idx + 1}: No data found")

            except Exception as chunk_error:
                st.warning(f"Error in optimized chunk {chunk_idx + 1}: {str(chunk_error)}")
                # Fallback to legacy method for this chunk
                chunk_collection = collection.filterDate(chunk_start, chunk_end).filterBounds(geometry)
                if bands:
                    chunk_collection = chunk_collection.select(bands)
                chunk_size = chunk_collection.size().getInfo()
                if chunk_size > 0:
                    chunk_rows = _extract_csv_data_from_collection(chunk_collection, geometry, scale, chunk_size)
                    if chunk_rows:
                        chunk_df = pd.DataFrame(chunk_rows)
                        all_dfs.append(chunk_df)

        if all_dfs:
            # Combine all chunks into single DataFrame
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Sort by date column (handle different date column names)
            date_cols = [col for col in combined_df.columns if 'date' in col.lower()]
            if date_cols:
                combined_df = combined_df.sort_values(date_cols[0])

            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                csv_path = temp_file.name
                combined_df.to_csv(csv_path, index=False)

            # Read as bytes
            with open(csv_path, 'rb') as f:
                file_data = f.read()

            os.unlink(csv_path)  # Cleanup

            chunk_progress.success(f"✅ Completed! Processed {len(combined_df)} temporal records from {len(date_chunks)} optimized chunks")

            return {
                'success': True,
                'file_path': csv_path,
                'file_data': file_data,
                'message': f"Time series CSV with {len(combined_df)} temporal records (processed in {len(date_chunks)} optimized chunks)"
            }
        else:
            return {
                'success': False,
                'message': "No data could be extracted from any chunks"
            }

    except Exception as e:
        # Complete fallback to legacy method
        st.warning(f"Optimized chunking failed ({str(e)}), falling back to legacy method...")
        return _process_csv_chunks_legacy(collection, bands, geometry, date_chunks, scale, temporal_resolution)
    finally:
        if 'chunk_progress' in locals():
            chunk_progress.empty()


def _process_csv_chunks_legacy(collection, bands, geometry, date_chunks, scale, temporal_resolution):
    """Legacy CSV chunking method using slow image-by-image processing"""
    all_rows = []
    chunk_progress = st.empty()

    try:
        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            chunk_progress.info(f"Processing legacy chunk {chunk_idx + 1}/{len(date_chunks)}: {chunk_start} to {chunk_end}")

            # Filter collection for this chunk
            chunk_collection = collection.filterDate(chunk_start, chunk_end).filterBounds(geometry)

            if bands:
                chunk_collection = chunk_collection.select(bands)

            chunk_size = chunk_collection.size().getInfo()

            if chunk_size > 0:
                # Process this chunk using legacy method
                chunk_rows = _extract_csv_data_from_collection(chunk_collection, geometry, scale, chunk_size)
                all_rows.extend(chunk_rows)

        if all_rows:
            # Combine all chunks into single CSV
            df = pd.DataFrame(all_rows)
            df = df.sort_values('datetime') if 'datetime' in df.columns else df

            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                csv_path = temp_file.name
                df.to_csv(csv_path, index=False)

            # Read as bytes
            with open(csv_path, 'rb') as f:
                file_data = f.read()

            os.unlink(csv_path)  # Cleanup

            chunk_progress.success(f"✅ Completed! Processed {len(all_rows)} temporal records from {len(date_chunks)} legacy chunks")

            return {
                'success': True,
                'file_path': csv_path,
                'file_data': file_data,
                'message': f"Time series CSV with {len(all_rows)} temporal records (processed in {len(date_chunks)} legacy chunks)"
            }
        else:
            return {
                'success': False,
                'message': "No data could be extracted from any chunks"
            }

    except Exception as e:
        return {
            'success': False,
            'message': f"Legacy chunked CSV processing failed: {str(e)}"
        }
    finally:
        if 'chunk_progress' in locals():
            chunk_progress.empty()


def _extract_csv_data_from_collection(collection, geometry, scale, collection_size):
    """Extract CSV data from a single collection"""
    rows = []
    images_list = collection.toList(collection_size)

    progress_placeholder = st.empty()

    try:
        for i in range(collection_size):
            try:
                # Progress update
                if i % 10 == 0:
                    progress_placeholder.info(f"Extracting data from image {i+1}/{collection_size}...")

                image = ee.Image(images_list.get(i))

                # Get date components
                date_obj = image.date()
                datetime_str = date_obj.format('YYYY-MM-dd HH:mm:ss').getInfo()

                # Get statistics
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=scale,
                    maxPixels=1e9
                ).getInfo()

                # Create row
                row = {
                    'datetime': datetime_str,
                    **stats
                }

                rows.append(row)

            except Exception as img_error:
                st.warning(f"Error processing image {i+1}: {str(img_error)}")
                continue

        return rows

    finally:
        progress_placeholder.empty()


def _process_geotiff_chunks(collection, bands, geometry, date_chunks, scale, temporal_resolution):
    """Process GeoTIFF export using chunking - creates separate ZIP files per chunk"""
    chunk_files = []
    chunk_progress = st.empty()

    try:
        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            chunk_progress.info(f"Processing GeoTIFF chunk {chunk_idx + 1}/{len(date_chunks)}: {chunk_start} to {chunk_end}")

            # Filter collection for this chunk
            chunk_collection = collection.filterDate(chunk_start, chunk_end).filterBounds(geometry)

            if bands:
                chunk_collection = chunk_collection.select(bands)

            chunk_size = chunk_collection.size().getInfo()

            if chunk_size > 0:
                # Create temporary file for this chunk
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                    chunk_temp_path = temp_file.name

                # Process this chunk
                chunk_result = _export_geotiff_single_chunk(chunk_collection, geometry, scale, chunk_size, chunk_temp_path)

                if chunk_result['success']:
                    chunk_files.append({
                        'name': f"chunk_{chunk_idx + 1}_{chunk_start}_to_{chunk_end}.zip",
                        'data': chunk_result['file_data'],
                        'size': len(chunk_result['file_data'])
                    })

        if chunk_files:
            # If only one chunk, return it directly
            if len(chunk_files) == 1:
                chunk_progress.success(f"✅ Completed! Single chunk with GeoTIFF files")
                return {
                    'success': True,
                    'file_path': chunk_files[0]['name'],
                    'file_data': chunk_files[0]['data'],
                    'message': f"GeoTIFF collection (single chunk)"
                }

            # Multiple chunks - create master ZIP
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as master_zip:
                master_zip_path = master_zip.name

                with zipfile.ZipFile(master_zip_path, 'w', zipfile.ZIP_DEFLATED) as master_zipf:
                    for chunk_file in chunk_files:
                        master_zipf.writestr(chunk_file['name'], chunk_file['data'])

            # Read master ZIP as bytes
            with open(master_zip_path, 'rb') as f:
                master_data = f.read()

            os.unlink(master_zip_path)  # Cleanup

            total_size_mb = sum(cf['size'] for cf in chunk_files) / (1024 * 1024)

            chunk_progress.success(f"✅ Completed! Processed {len(chunk_files)} chunks")

            return {
                'success': True,
                'file_path': 'geotiff_chunks.zip',
                'file_data': master_data,
                'message': f"Master ZIP with {len(chunk_files)} chunk files ({total_size_mb:.1f} MB total)"
            }
        else:
            return {
                'success': False,
                'message': "No GeoTIFF chunks could be created"
            }

    except Exception as e:
        return {
            'success': False,
            'message': f"Chunked GeoTIFF processing failed: {str(e)}"
        }
    finally:
        if 'chunk_progress' in locals():
            chunk_progress.empty()


def _export_csv_single_chunk(collection, geometry, scale, collection_size):
    """Export CSV for a single chunk"""
    rows = _extract_csv_data_from_collection(collection, geometry, scale, collection_size)

    if rows:
        df = pd.DataFrame(rows)
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            csv_path = temp_file.name
            df.to_csv(csv_path, index=False)

        with open(csv_path, 'rb') as f:
            file_data = f.read()

        os.unlink(csv_path)

        return {
            'success': True,
            'file_path': csv_path,
            'file_data': file_data,
            'message': f"Time series CSV with {len(rows)} temporal records"
        }
    else:
        return {
            'success': False,
            'message': "No temporal data could be extracted"
        }


def _export_geotiff_single_chunk(collection, geometry, scale, collection_size, temp_path=None):
    """Export GeoTIFF for a single chunk"""
    if temp_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            temp_path = temp_file.name

    return export_individual_geotiffs_standalone(collection, temp_path, geometry, scale, collection_size)


def export_individual_geotiffs_standalone(collection, temp_path, geometry, scale, collection_size):
    """
    Standalone version of individual GeoTIFF export to avoid circular imports
    """
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

                # Export using geemap with explicit EPSG:4326 projection
                geemap.ee_export_image(
                    clipped_image,
                    filename=str(file_path),
                    scale=scale,
                    region=geometry,
                    file_per_band=False,
                    crs='EPSG:4326'  # Explicitly set coordinate reference system
                )

                if file_path.exists() and file_path.stat().st_size > 0:
                    exported_files.append(file_path)
                else:
                    st.warning(f"Failed to export image {i+1}: {filename}")

            except Exception as img_error:
                st.warning(f"Error exporting image {i+1}: {str(img_error)}")
                continue

        if progress_placeholder:
            progress_placeholder.empty()

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
                'message': f"ZIP archive with {len(exported_files)} individual daily GeoTIFF files ({file_size_mb:.1f} MB)"
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