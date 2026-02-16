"""
Climate Analysis Runner
Integrates ClimateIndicesCalculator with download_utils for robust server-side computation
"""

import ee
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import tempfile
import os
import zipfile
from pathlib import Path

from .dataset_config import get_dataset_config
from ..climate_indices import ClimateIndicesCalculator
from .download_utils import process_image_collection_chunked
from app_utils import download_ee_data_simple


def run_climate_analysis_with_chunking(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run climate analysis with flexible export options (Google Drive or local)

    Args:
        config: Analysis configuration dictionary
                Required keys: analysis_type, dataset_id, selected_indices,
                              start_date, end_date, geometry
                Optional keys: export_method ('drive'/'local'/'preview'),
                              spatial_scale (meters), temporal_resolution,
                              temporal_only (bool) - if True, skip spatial processing

    Returns:
        Results dictionary with success status and output data
    """
    try:
        # Extract configuration
        analysis_type = config['analysis_type']
        dataset_id = config['dataset_id']
        selected_indices = config['selected_indices']
        start_date = config['start_date']
        end_date = config['end_date']
        geometry = config['geometry']

        # Extract export configuration
        export_method = config.get('export_method', 'local')  # 'drive', 'local', 'preview'
        spatial_scale = config.get('spatial_scale', 1000)  # meters
        temporal_resolution = config.get('temporal_resolution', 'yearly')  # yearly/monthly
        temporal_only = config.get('temporal_only', False)  # Skip spatial processing if True

        dataset_config = get_dataset_config()

        st.info(f"🔧 Initializing analysis for {len(selected_indices)} indices over {start_date} to {end_date}")

        # Initialize calculator with dataset configuration
        calculator = ClimateIndicesCalculator(geometry, dataset_id)

        # Get dataset information
        dataset_info = dataset_config.get_dataset_info(dataset_id)
        if not dataset_info:
            return {'success': False, 'error': f'Dataset {dataset_id} not found in configuration'}

        # Get required bands for selected indices
        band_mapping = dataset_config.get_required_bands_for_indices(dataset_id, selected_indices)

        st.info(f"📊 Loading Earth Engine collections for {dataset_info['name']}")

        # Load Earth Engine collections with error handling
        collections = {}
        base_collection = ee.ImageCollection(dataset_id)

        # Get the first image to check available bands
        try:
            first_image = base_collection.first()
            available_bands = first_image.bandNames().getInfo()
            st.info(f"📋 Available bands in {dataset_id}: {available_bands}")
        except Exception as e:
            st.warning(f"⚠️ Could not determine available bands: {str(e)}")
            available_bands = []

        for band_type, ee_band_name in band_mapping.items():
            try:
                if available_bands and ee_band_name not in available_bands:
                    st.error(f"❌ Band '{ee_band_name}' not found in dataset. Available: {available_bands}")
                    return {'success': False, 'error': f'Band {ee_band_name} not found in {dataset_id}'}

                # Filter date range first, then select band
                collection = base_collection.filterDate(start_date, end_date).select([ee_band_name])

                # Verify the collection has data
                collection_size = collection.size().getInfo()
                if collection_size == 0:
                    st.warning(f"⚠️ No data found for band '{ee_band_name}' in the specified date range")

                collections[band_type] = collection
                st.info(f"✅ Loaded {band_type} band '{ee_band_name}': {collection_size} images")

            except Exception as e:
                st.error(f"❌ Failed to load band '{ee_band_name}' for {band_type}: {str(e)}")
                return {'success': False, 'error': f'Failed to load band {ee_band_name}: {str(e)}'}

        # Calculate each index
        results = {}
        all_time_series = {}
        all_spatial_data = {}
        all_index_collections = {}  # Store ImageCollections for later spatial processing

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate adaptive scale based on area size to prevent memory issues
        try:
            area_km2 = ee.Geometry(geometry).area().divide(1e6).getInfo()
            if area_km2 > 500000:  # Very large (>500,000 km²)
                adaptive_scale = 20000  # 20km resolution
                st.warning(f"⚠️ Very large study area ({area_km2:,.0f} km²). Using coarse resolution ({adaptive_scale}m) to prevent memory issues.")
            elif area_km2 > 100000:  # Large (>100,000 km²)
                adaptive_scale = 10000  # 10km resolution
                st.info(f"ℹ️ Large study area ({area_km2:,.0f} km²). Using adaptive resolution ({adaptive_scale}m).")
            elif area_km2 > 10000:  # Medium (>10,000 km²)
                adaptive_scale = 5000   # 5km resolution
            else:  # Small (<10,000 km²)
                adaptive_scale = 1000   # 1km resolution
        except:
            adaptive_scale = 5000  # Default fallback
            area_km2 = 0

        if temporal_only:
            st.info(f"📈 Extracting temporal data for {len(selected_indices)} indices...")

        # Use expander for detailed logs
        with st.expander("📋 Detailed Processing Log", expanded=False):
            for i, index_name in enumerate(selected_indices):
                status_text.text(f"Processing {index_name} ({i + 1}/{len(selected_indices)})...")
                progress = (i + 1) / len(selected_indices)
                progress_bar.progress(progress)

                try:
                    # Calculate the climate index
                    st.write(f"🔄 Computing {index_name}...")

                    # Get threshold parameters for this index if available
                    threshold_kwargs = {}
                    if 'threshold_params' in config and index_name in config['threshold_params']:
                        threshold_kwargs = config['threshold_params'][index_name]
                        st.write(f"  🎛️ Custom parameters: {threshold_kwargs}")

                    # Get percentile parameters for this index if available
                    percentile_kwargs = {}
                    if 'percentile_params' in config and index_name in config['percentile_params']:
                        percentile_config = config['percentile_params'][index_name]
                        percentile_kwargs = {
                            'percentile': percentile_config['percentile'],
                            'base_start': percentile_config['base_start'],
                            'base_end': percentile_config['base_end']
                        }
                        st.write(f"  📊 Percentile: {percentile_config['percentile']}th, base {percentile_config['base_start'][:4]}-{percentile_config['base_end'][:4]}")

                    # Combine all parameters
                    all_kwargs = {**threshold_kwargs, **percentile_kwargs}

                    index_result = calculator.calculate_simple_index(
                        index_name, collections, start_date, end_date,
                        temporal_resolution=temporal_resolution,
                        **all_kwargs
                    )

                    # Store the ImageCollection for later spatial processing
                    all_index_collections[index_name] = index_result

                    # Extract time series data with ADAPTIVE SCALE and retry logic
                    st.write(f"  📈 Extracting time series (scale={adaptive_scale}m)...")

                    extraction_success = False
                    current_scale = adaptive_scale
                    max_retries = 3

                    for retry in range(max_retries):
                        try:
                            time_series_df = calculator.extract_time_series_optimized(
                                index_result, scale=current_scale, max_pixels=1e6
                            )
                            extraction_success = True
                            break
                        except Exception as extraction_error:
                            error_msg = str(extraction_error).lower()
                            if "memory" in error_msg or "limit exceeded" in error_msg:
                                # Increase scale to reduce memory usage
                                current_scale = int(current_scale * 2)
                                if retry < max_retries - 1:
                                    st.write(f"  ⚠️ Memory limit hit. Retrying with coarser scale ({current_scale}m)...")
                                else:
                                    raise extraction_error
                            else:
                                raise extraction_error

                    if extraction_success and not time_series_df.empty:
                        all_time_series[index_name] = time_series_df
                        st.write(f"  ✅ Extracted {len(time_series_df)} data points")
                    else:
                        st.write(f"  ⚠️ No data points extracted")

                    # SPATIAL PROCESSING (skip if temporal_only mode)
                    if temporal_only:
                        # Skip spatial processing, mark as pending for on-demand export
                        spatial_result = {
                            'success': True,
                            'export_method': 'pending',
                            'message': f'{index_name} spatial export pending (user-triggered)',
                            'pending': True,
                            'collection_available': True
                        }
                        st.write(f"  ⏳ Spatial export ready (on-demand)")
                    else:
                        # Generate spatial data based on export_method
                        collection_size = index_result.size().getInfo()
                        st.write(f"  🗺️ Processing spatial data ({collection_size} images, {export_method} mode)")

                        # Branch based on export_method - critical for proper export handling
                        if export_method == 'drive':
                            # Google Drive batch export - handles large files via Earth Engine tasks
                            st.write(f"    ☁️ Submitting to Google Drive...")
                            spatial_result = _export_to_google_drive(
                                index_result, geometry, index_name, spatial_scale
                            )

                        elif export_method == 'preview':
                            # Preview mode - single sample image
                            st.write(f"    📱 Generating preview sample...")
                            spatial_result = _generate_preview_sample(
                                index_result, geometry, index_name, spatial_scale
                            )

                        elif export_method == 'auto':
                            # Smart Auto mode: estimate size, try local first, fallback to drive
                            st.write(f"    🤖 Smart Auto: estimating optimal method...")

                            # Estimate file size based on area and resolution
                            try:
                                pixels_per_image = area_km2 / ((spatial_scale / 1000) ** 2)
                                estimated_size_per_image_mb = (pixels_per_image * 4) / (1024 * 1024)
                                total_estimated_mb = estimated_size_per_image_mb * collection_size

                                st.write(f"    📊 Est. size: {total_estimated_mb:.1f} MB ({collection_size} × {estimated_size_per_image_mb:.1f} MB)")

                                # Decision threshold: < 50MB for local, otherwise Drive
                                if total_estimated_mb < 50 and estimated_size_per_image_mb < 100:
                                    st.write(f"    ✅ Size suitable for local download...")
                                    spatial_result = process_image_collection_chunked(
                                        collection=index_result,
                                        bands=None,
                                        geometry=geometry,
                                        start_date=start_date,
                                        end_date=end_date,
                                        export_format='GeoTIFF',
                                        scale=spatial_scale,
                                        temporal_resolution=temporal_resolution
                                    )

                                    # Check if local export succeeded
                                    if not spatial_result.get('success') or not spatial_result.get('file_data'):
                                        st.write(f"    ⚠️ Local failed, falling back to Google Drive...")
                                        spatial_result = _export_to_google_drive(
                                            index_result, geometry, index_name, spatial_scale
                                        )
                                else:
                                    st.write(f"    📤 Size too large for local, using Google Drive...")
                                    spatial_result = _export_to_google_drive(
                                        index_result, geometry, index_name, spatial_scale
                                    )
                            except Exception as size_err:
                                st.write(f"    ⚠️ Size estimation failed, using Google Drive...")
                                spatial_result = _export_to_google_drive(
                                    index_result, geometry, index_name, spatial_scale
                                )

                        else:  # 'local' or default
                            # Direct local download
                            st.write(f"    💻 Direct local download...")
                            spatial_result = process_image_collection_chunked(
                                collection=index_result,
                                bands=None,
                                geometry=geometry,
                                start_date=start_date,
                                end_date=end_date,
                                export_format='GeoTIFF',
                                scale=spatial_scale,
                                temporal_resolution=temporal_resolution
                            )

                        # Standardize result structure
                        if spatial_result.get('success'):
                            if spatial_result.get('file_data') and 'export_method' not in spatial_result:
                                spatial_result['export_method'] = 'local'
                                spatial_result['actual_size_mb'] = len(spatial_result['file_data']) / (1024 * 1024)
                                spatial_result['filename'] = spatial_result.get('filename', f'{index_name}_spatial.zip')
                            elif spatial_result.get('drive_folder') and 'export_method' not in spatial_result:
                                spatial_result['export_method'] = 'drive'
                                spatial_result['estimated_size_mb'] = spatial_result.get('estimated_size_mb', collection_size * 5)

                            if 'export_method' not in spatial_result:
                                spatial_result['export_method'] = export_method

                            st.write(f"  ✅ Spatial: {spatial_result.get('export_method', 'unknown')} method")
                        else:
                            st.write(f"  ⚠️ Spatial failed: {spatial_result.get('message', spatial_result.get('error', 'Unknown'))}")

                            # Create fallback CSV result
                            if not time_series_df.empty:
                                csv_data = time_series_df.to_csv(index=False).encode('utf-8')
                                spatial_result = {
                                    'success': True,
                                    'export_method': 'local',
                                    'file_data': csv_data,
                                    'filename': f'{index_name}_timeseries.csv',
                                    'actual_size_mb': len(csv_data) / (1024 * 1024),
                                    'message': f'Time series CSV for {index_name} (spatial failed)',
                                    'fallback': True
                                }
                                st.write(f"  📊 Fallback: time series CSV provided")
                            else:
                                spatial_result = {
                                    'success': False,
                                    'export_method': 'failed',
                                    'error': f'Both spatial and temporal failed for {index_name}'
                                }

                    if spatial_result['success']:
                        all_spatial_data[index_name] = spatial_result

                    results[index_name] = {
                        'time_series': time_series_df,
                        'spatial_data': spatial_result,
                        'success': True
                    }

                except Exception as index_error:
                    error_msg = str(index_error)

                    # Provide specific guidance for percentile-based indices
                    if index_name in ['TX90p', 'TX10p', 'TN90p', 'TN10p', 'R95p', 'R99p', 'R75p']:
                        # Get the actual base period that was used
                        actual_base_period = "1980-2000"  # default
                        if 'percentile_params' in config and index_name in config['percentile_params']:
                            percentile_config = config['percentile_params'][index_name]
                            start_year = percentile_config['base_start'][:4]
                            end_year = percentile_config['base_end'][:4]
                            actual_base_period = f"{start_year}-{end_year}"

                        if "base period" in error_msg.lower():
                            st.write(f"  ❌ Base period data issue ({actual_base_period})")
                            detailed_error = f"Percentile-based index requires {actual_base_period} baseline data: {error_msg}"
                        else:
                            st.write(f"  ❌ Percentile calculation failed: {error_msg}")
                            detailed_error = f"Percentile calculation error: {error_msg}"
                    else:
                        st.write(f"  ❌ Error: {error_msg}")
                        detailed_error = error_msg

                    results[index_name] = {
                        'success': False,
                        'error': detailed_error
                    }
                    continue

        progress_bar.progress(1.0)

        # Calculate success statistics
        total_indices = len(selected_indices)
        successful_indices = sum(1 for r in results.values() if r['success'])
        failed_indices = total_indices - successful_indices

        # Provide concise completion message (consolidated)
        if failed_indices == 0:
            status_text.text(f"✅ All {total_indices} indices processed successfully!")
        elif successful_indices == 0:
            status_text.text(f"❌ All {total_indices} indices failed!")
            st.error(f"All climate indices failed. Check the 'Detailed Processing Log' expander above.")
        else:
            status_text.text(f"⚠️ {successful_indices}/{total_indices} indices succeeded")
            st.warning(f"{successful_indices}/{total_indices} indices succeeded. {failed_indices} failed - check logs above.")

        # Generate combined outputs
        combined_results = _generate_combined_outputs(
            all_time_series, all_spatial_data, config
        )

        # Validate result structure for smart download compatibility
        validated_results = _validate_smart_download_structure(results, export_method)

        # Overall success is true if at least one index succeeded
        overall_success = successful_indices > 0

        result_dict = {
            'success': overall_success,
            'individual_results': validated_results,
            'time_series_data': all_time_series,
            'time_series_csv': combined_results.get('time_series_csv'),
            'spatial_data_zip': combined_results.get('spatial_data_zip'),
            'analysis_report': combined_results.get('analysis_report'),
            'image_collections': all_index_collections,  # Store ImageCollections for geemap visualization
            'geometry': geometry,  # Store geometry for map centering
            'summary': {
                'total_indices': total_indices,
                'successful_indices': successful_indices,
                'failed_indices': failed_indices,
                'time_period': f"{start_date} to {end_date}",
                'dataset': dataset_info['name']
            },
            'temporal_only': temporal_only
        }

        # If temporal_only mode, store config for later spatial processing
        if temporal_only:
            result_dict['spatial_export_config'] = {
                'dataset_id': dataset_id,
                'geometry': geometry,
                'start_date': start_date,
                'end_date': end_date,
                'spatial_scale': spatial_scale,
                'temporal_resolution': temporal_resolution,
                'selected_indices': selected_indices,
                'export_method': export_method,
                'threshold_params': config.get('threshold_params', {}),
                'percentile_params': config.get('percentile_params', {})
            }
            # Concise final message
            if successful_indices > 0:
                st.info(f"📊 Temporal data ready for {successful_indices} indices. Spatial export available below.")

        return result_dict

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': 'Check the configuration and try again with a smaller time period if necessary'
        }


def _generate_spatial_data_chunked(index_collection: ee.ImageCollection,
                                 geometry: ee.Geometry,
                                 index_name: str,
                                 start_date: str,
                                 end_date: str) -> Dict[str, Any]:
    """
    Generate spatial data (yearly GeoTIFF files) using chunked processing

    Args:
        index_collection: Calculated climate index collection
        geometry: Area of interest
        index_name: Name of the climate index
        start_date: Analysis start date
        end_date: Analysis end date

    Returns:
        Dictionary with spatial data results
    """
    try:
        # Check collection size
        collection_size = index_collection.size().getInfo()

        if collection_size == 0:
            return {'success': False, 'error': 'Empty collection for spatial data'}

        st.info(f"🗺️ Generating {collection_size} spatial files for {index_name}...")

        # For large collections, use the chunked processing from download_utils
        if collection_size > 50:  # Use chunking for large collections
            spatial_result = process_image_collection_chunked(
                collection=index_collection,
                bands=None,  # Already selected
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                export_format='GeoTIFF',
                scale=1000,  # Use 1km resolution for climate indices
                temporal_resolution='monthly'  # Most indices are monthly/annual
            )
        else:
            # Use direct processing for smaller collections
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                temp_path = temp_file.name

            spatial_result = _export_individual_geotiffs_simple(
                index_collection, temp_path, geometry, 1000, collection_size, index_name
            )

        return spatial_result

    except Exception as e:
        return {
            'success': False,
            'error': f'Spatial data generation failed: {str(e)}'
        }

# NOTE: The custom spatial export functions below (_export_individual_geotiffs_simple,
# _export_to_google_drive, _generate_preview_sample, _generate_spatial_data_local)
# are DEPRECATED and replaced by process_image_collection_chunked for consistency with GeoData Explorer

def _export_individual_geotiffs_simple(collection: ee.ImageCollection,
                                     temp_path: str,
                                     geometry: ee.Geometry,
                                     scale: int,
                                     collection_size: int,
                                     index_name: str) -> Dict[str, Any]:
    """
    Export individual GeoTIFF files for smaller collections
    """
    import geemap

    temp_dir = Path(temp_path).parent / f'{index_name}_geotiffs'
    temp_dir.mkdir(exist_ok=True)

    try:
        images_list = collection.toList(collection_size)
        exported_files = []

        for i in range(collection_size):
            try:
                image = ee.Image(images_list.get(i))

                # Get date information
                date_info = image.get('system:time_start')
                year = image.get('year')
                month = image.get('month')

                # Create filename
                if year and month:
                    filename = f"{index_name}_{year.getInfo()}_{month.getInfo():02d}.tif"
                else:
                    filename = f"{index_name}_{i:03d}.tif"

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

            except Exception as img_error:
                st.warning(f"Failed to export image {i+1}: {str(img_error)}")
                continue

        if exported_files:
            # Create ZIP file
            zip_path = temp_path.replace('.tif', f'_{index_name}_spatial.zip')

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

            return {
                'success': True,
                'export_method': 'local',
                'file_path': zip_path,
                'file_data': file_data,
                'filename': f"{index_name}_spatial.zip",
                'actual_size_mb': len(file_data) / (1024 * 1024),
                'message': f"Generated {len(exported_files)} spatial files for {index_name}"
            }
        else:
            return {
                'success': False,
                'error': f"No spatial files could be exported for {index_name}"
            }

    except Exception as e:
        return {
            'success': False,
            'error': f"Spatial export failed: {str(e)}"
        }
    finally:
        # Cleanup
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def _generate_combined_outputs(all_time_series: Dict[str, pd.DataFrame],
                             all_spatial_data: Dict[str, Dict],
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate combined output files from analysis results

    Args:
        all_time_series: Time series data for each index
        all_spatial_data: Spatial data for each index
        config: Analysis configuration

    Returns:
        Dictionary with combined output data
    """
    outputs = {}

    try:
        # Generate combined time series CSV
        if all_time_series:
            combined_df = _create_combined_time_series_csv(all_time_series, config)
            outputs['time_series_csv'] = combined_df.to_csv(index=False).encode('utf-8')

        # Generate combined spatial ZIP
        if all_spatial_data:
            spatial_zip_data = _create_combined_spatial_zip(all_spatial_data, config)
            outputs['spatial_data_zip'] = spatial_zip_data

        # Generate analysis report
        outputs['analysis_report'] = _create_analysis_report(
            all_time_series, all_spatial_data, config
        ).encode('utf-8')

    except Exception as e:
        st.error(f"Error generating combined outputs: {str(e)}")

    return outputs


def _create_combined_time_series_csv(all_time_series: Dict[str, pd.DataFrame],
                                   config: Dict[str, Any]) -> pd.DataFrame:
    """Create combined time series CSV with all indices"""
    combined_rows = []

    for index_name, df in all_time_series.items():
        for _, row in df.iterrows():
            combined_rows.append({
                'Date': row.get('date', row.get('datetime', '')),
                'Climate_Index': index_name,
                'Value': row.get('value', row.get('Climate_Index_Value', 0)),
                'Analysis_Type': config['analysis_type'],
                'Dataset': config['dataset_id'],
                'Start_Date': config['start_date'],
                'End_Date': config['end_date'],
                'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    return pd.DataFrame(combined_rows)


def _create_combined_spatial_zip(all_spatial_data: Dict[str, Dict],
                               config: Dict[str, Any]) -> bytes:
    """Create combined ZIP file with all spatial data"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
        temp_zip_path = temp_zip.name

    try:
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as master_zip:
            for index_name, spatial_result in all_spatial_data.items():
                if spatial_result['success'] and 'file_data' in spatial_result:
                    # Add each index's spatial data as a separate entry
                    zip_entry_name = f"{index_name}_spatial_data.zip"
                    master_zip.writestr(zip_entry_name, spatial_result['file_data'])

        # Read master ZIP as bytes
        with open(temp_zip_path, 'rb') as f:
            zip_data = f.read()

        os.unlink(temp_zip_path)
        return zip_data

    except Exception as e:
        st.error(f"Error creating combined spatial ZIP: {str(e)}")
        return b''


def _create_analysis_report(all_time_series: Dict[str, pd.DataFrame],
                          all_spatial_data: Dict[str, Dict],
                          config: Dict[str, Any]) -> str:
    """Create comprehensive analysis report"""

    dataset_config = get_dataset_config()
    dataset_info = dataset_config.get_dataset_info(config['dataset_id'])

    report = f"""Climate Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS CONFIGURATION:
=======================
Analysis Type: {config['analysis_type'].title()}
Dataset: {dataset_info.get('name', 'Unknown')} ({config['dataset_id']})
Provider: {dataset_info.get('provider', 'Unknown')}
Time Period: {config['start_date']} to {config['end_date']}
Selected Indices: {', '.join(config['selected_indices'])}

PROCESSING RESULTS:
==================
Total Indices Requested: {len(config['selected_indices'])}
Successfully Processed: {len(all_time_series)}

"""

    # Add results for each index
    for index_name in config['selected_indices']:
        report += f"\n{index_name.upper()}:\n"
        report += "-" * (len(index_name) + 1) + "\n"

        if index_name in all_time_series:
            df = all_time_series[index_name]
            if not df.empty:
                values = df['value'].dropna()
                report += f"  Time Series Records: {len(df)}\n"
                report += f"  Mean Value: {values.mean():.4f}\n"
                report += f"  Standard Deviation: {values.std():.4f}\n"
                report += f"  Minimum Value: {values.min():.4f}\n"
                report += f"  Maximum Value: {values.max():.4f}\n"
            else:
                report += "  No time series data available\n"
        else:
            report += "  Processing failed\n"

        if index_name in all_spatial_data:
            spatial_result = all_spatial_data[index_name]
            if spatial_result['success']:
                report += f"  Spatial Files: Generated successfully\n"
                report += f"  Details: {spatial_result.get('message', 'N/A')}\n"
            else:
                report += f"  Spatial Files: Failed - {spatial_result.get('error', 'Unknown error')}\n"

    report += f"""

DATA INTERPRETATION:
==================
This climate analysis provides both temporal and spatial climate index data.

TIME SERIES DATA:
- Contains area-averaged values for each climate index
- Suitable for trend analysis and statistical modeling
- Can be used for correlation studies and time series analysis

SPATIAL DATA:
- Individual GeoTIFF files for spatial analysis
- Suitable for GIS applications and spatial modeling
- Can be used for mapping and regional analysis

RECOMMENDED APPLICATIONS:
- Climate change impact assessment
- Extreme event analysis
- Agricultural and ecological studies
- Water resources planning
- Climate model validation

For technical questions about the methodology, refer to the ETCCDI standards documentation.

Report generated by GeoClimate Intelligence Platform
"""

    return report


# =============================================================================
# Google Drive Export Functions
# =============================================================================

def _export_to_google_drive(index_collection: ee.ImageCollection,
                          geometry: ee.Geometry,
                          index_name: str,
                          scale: int = 1000) -> Dict[str, Any]:
    """
    Export climate index results to Google Drive using Earth Engine batch export

    Args:
        index_collection: Calculated climate index collection
        geometry: Area of interest
        index_name: Name of the climate index
        scale: Spatial resolution in meters

    Returns:
        Dictionary with Google Drive export results
    """
    try:
        collection_size = index_collection.size().getInfo()

        if collection_size == 0:
            return {'success': False, 'error': f'Empty collection for {index_name}'}

        # Create unique folder name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        drive_folder = f"ClimateIndices_{index_name}_{timestamp}"

        st.info(f"🚀 Starting Google Drive export for {index_name} ({collection_size} files)")
        st.info(f"📁 Google Drive folder: {drive_folder}")
        st.info(f"⚙️ Resolution: {scale}m")

        # Convert collection to list for batch export
        images_list = index_collection.toList(collection_size)
        tasks = []
        failed_tasks = 0

        progress_placeholder = st.empty()

        for i in range(collection_size):
            try:
                progress_placeholder.info(f"Submitting export task {i+1}/{collection_size}...")

                image = ee.Image(images_list.get(i))

                # Create meaningful filename from image properties
                date_info = image.date()
                year = date_info.get('year')
                month = date_info.get('month')

                # Try to get year/month from image properties, fallback to index
                try:
                    year_val = year.getInfo() if year else None
                    month_val = month.getInfo() if month else None

                    if year_val and month_val:
                        filename = f"{index_name}_{year_val}_{month_val:02d}"
                    else:
                        # Fallback to date string
                        date_str = date_info.format('YYYY_MM').getInfo()
                        filename = f"{index_name}_{date_str}"
                except:
                    # Ultimate fallback to index
                    filename = f"{index_name}_{i+1:03d}"

                # Clip image to geometry
                clipped_image = image.clip(geometry)

                # Submit export task to Google Drive
                task = ee.batch.Export.image.toDrive(
                    image=clipped_image,
                    description=filename,
                    folder=drive_folder,
                    fileNamePrefix=filename,
                    scale=scale,
                    region=geometry,
                    maxPixels=1e9,
                    fileFormat='GeoTIFF'
                )

                # Start the task
                task.start()

                task_info = {
                    'task_id': task.id,
                    'filename': f"{filename}.tif",
                    'status': 'SUBMITTED',
                    'description': f"{index_name} - Image {i+1}"
                }
                tasks.append(task_info)

            except Exception as task_error:
                st.warning(f"Failed to submit export task {i+1}: {str(task_error)}")
                failed_tasks += 1
                continue

        progress_placeholder.empty()

        if tasks:
            success_count = len(tasks)
            drive_url = f"https://drive.google.com/drive/folders/{drive_folder}"

            st.success(f"✅ {index_name}: Submitted {success_count} export tasks to Google Drive")
            if failed_tasks > 0:
                st.warning(f"⚠️ {failed_tasks} tasks failed to submit")

            return {
                'success': True,
                'export_method': 'drive',
                'drive_folder': drive_folder,
                'drive_url': drive_url,
                'tasks': tasks,
                'total_tasks': success_count,
                'failed_tasks': failed_tasks,
                'task_id': tasks[0]['task_id'] if tasks else None,
                'estimated_size_mb': success_count * 10,  # Rough estimate
                'scale': scale,
                'message': f"Submitted {success_count} export tasks to Google Drive folder: {drive_folder}"
            }
        else:
            return {
                'success': False,
                'error': f'All export tasks failed for {index_name}'
            }

    except Exception as e:
        return {
            'success': False,
            'error': f'Google Drive export failed for {index_name}: {str(e)}'
        }


def _generate_preview_sample(index_collection: ee.ImageCollection,
                           geometry: ee.Geometry,
                           index_name: str,
                           scale: int = 1000) -> Dict[str, Any]:
    """
    Generate a single sample GeoTIFF file for preview

    Args:
        index_collection: Calculated climate index collection
        geometry: Area of interest
        index_name: Name of the climate index
        scale: Spatial resolution in meters

    Returns:
        Dictionary with preview sample results
    """
    try:
        collection_size = index_collection.size().getInfo()

        if collection_size == 0:
            return {'success': False, 'error': f'Empty collection for {index_name}'}

        st.info(f"📷 Generating preview sample for {index_name}")

        # Get the first image as sample
        sample_image = ee.Image(index_collection.first()).clip(geometry)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            temp_path = temp_file.name

        # Use geemap for export
        import geemap

        # Get sample date for filename
        try:
            date_str = sample_image.date().format('YYYY_MM_dd').getInfo()
            sample_filename = f"{index_name}_sample_{date_str}.tif"
        except:
            sample_filename = f"{index_name}_sample.tif"

        # Export sample using geemap
        geemap.ee_export_image(
            sample_image,
            filename=temp_path,
            scale=scale,
            region=geometry,
            file_per_band=False
        )

        # Read sample file as bytes
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            with open(temp_path, 'rb') as f:
                sample_data = f.read()

            # Cleanup
            os.unlink(temp_path)

            file_size_mb = len(sample_data) / (1024 * 1024)

            return {
                'success': True,
                'export_method': 'preview',  # Preview mode - single sample
                'file_data': sample_data,
                'filename': sample_filename,
                'total_images': collection_size,
                'scale': scale,
                'file_size_mb': file_size_mb,
                'actual_size_mb': file_size_mb,  # For compatibility
                'message': f"Generated preview sample ({file_size_mb:.2f} MB) - {collection_size} total images available"
            }
        else:
            return {
                'success': False,
                'error': f'Failed to generate preview sample for {index_name}'
            }

    except Exception as e:
        return {
            'success': False,
            'error': f'Preview generation failed for {index_name}: {str(e)}'
        }


def _generate_spatial_data_local(index_collection: ee.ImageCollection,
                               geometry: ee.Geometry,
                               index_name: str,
                               scale: int = 1000) -> Dict[str, Any]:
    """
    Generate spatial data locally (for small datasets only)

    This is the original implementation but with size limits
    """
    try:
        collection_size = index_collection.size().getInfo()

        if collection_size == 0:
            return {'success': False, 'error': f'Empty collection for {index_name}'}

        st.info(f"💻 Generating local files for {index_name} ({collection_size} images)")
        st.info(f"⚙️ Resolution: {scale}m")

        # Use the existing local export function but with custom scale
        # Create platform-appropriate temporary path
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            temp_path = temp_file.name

        return _export_individual_geotiffs_simple(
            index_collection,
            temp_path,
            geometry,
            scale,
            collection_size,
            index_name
        )

    except Exception as e:
        return {
            'success': False,
            'error': f'Local export failed for {index_name}: {str(e)}'
        }


def get_recommended_scale(dataset_id: str, default: int = 1000) -> Dict[str, Any]:
    """
    Get recommended spatial scales based on dataset characteristics

    Args:
        dataset_id: Earth Engine dataset ID
        default: Default scale if no specific recommendation

    Returns:
        Dictionary with scale recommendations and time estimates
    """
    scale_recommendations = {
        'ECMWF/ERA5/DAILY': {
            'native': 27830,  # ~28km native resolution
            'recommended': 25000,  # 25km recommended
            'options': {
                'Ultra High (1km)': {'scale': 1000, 'time_factor': 100, 'size_factor': 625},
                'High (5km)': {'scale': 5000, 'time_factor': 25, 'size_factor': 25},
                'Medium (10km)': {'scale': 10000, 'time_factor': 7, 'size_factor': 6.25},
                'Low (25km - Native)': {'scale': 25000, 'time_factor': 1, 'size_factor': 1},
                'Very Low (50km)': {'scale': 50000, 'time_factor': 0.3, 'size_factor': 0.25}
            }
        },
        'NASA/ORNL/DAYMET_V4': {
            'native': 1000,  # 1km native resolution
            'recommended': 1000,  # Keep native 1km
            'options': {
                'Ultra High (1km - Native)': {'scale': 1000, 'time_factor': 1, 'size_factor': 1},
                'High (2km)': {'scale': 2000, 'time_factor': 0.4, 'size_factor': 0.25},
                'Medium (5km)': {'scale': 5000, 'time_factor': 0.1, 'size_factor': 0.04},
                'Low (10km)': {'scale': 10000, 'time_factor': 0.05, 'size_factor': 0.01},
                'Very Low (25km)': {'scale': 25000, 'time_factor': 0.02, 'size_factor': 0.0016}
            }
        }
    }

    if dataset_id in scale_recommendations:
        return scale_recommendations[dataset_id]
    else:
        # Default recommendations for unknown datasets
        return {
            'native': default,
            'recommended': default,
            'options': {
                'High (1km)': {'scale': 1000, 'time_factor': 10, 'size_factor': 100},
                'Medium (5km)': {'scale': 5000, 'time_factor': 2, 'size_factor': 4},
                'Low (10km)': {'scale': 10000, 'time_factor': 1, 'size_factor': 1},
                'Very Low (25km)': {'scale': 25000, 'time_factor': 0.3, 'size_factor': 0.16}
            }
        }


def estimate_export_time(collection_size: int, scale: int, base_time_minutes: int = 2) -> str:
    """
    Estimate export time based on collection size and scale

    Args:
        collection_size: Number of images in collection
        scale: Spatial resolution in meters
        base_time_minutes: Base time per image at 10km resolution

    Returns:
        Human-readable time estimate string
    """
    # Scale factor - higher resolution = longer time
    scale_factor = (10000 / scale) ** 2  # Quadratic relationship with resolution

    # Total time in minutes
    total_minutes = collection_size * base_time_minutes * scale_factor

    if total_minutes < 60:
        return f"~{int(total_minutes)} minutes"
    elif total_minutes < 1440:  # Less than 1 day
        hours = total_minutes / 60
        return f"~{hours:.1f} hours"
    else:
        days = total_minutes / 1440
        return f"~{days:.1f} days"


def _validate_smart_download_structure(results: Dict[str, Any], export_method: str) -> Dict[str, Any]:
    """
    Validate and fix result structure for smart download compatibility

    Args:
        results: Raw analysis results
        export_method: The export method used ('local', 'drive', 'preview', 'auto', 'pending')

    Returns:
        Validated results with proper structure
    """
    validated_results = {}

    for index_name, index_result in results.items():
        if not index_result.get('success', False):
            # Keep failed results as-is
            validated_results[index_name] = index_result
            continue

        # Ensure spatial_data has required structure
        if 'spatial_data' in index_result:
            spatial_data = index_result['spatial_data']

            # Ensure export_method is present and valid
            if 'export_method' not in spatial_data:
                # Infer export method from data characteristics
                if spatial_data.get('file_data'):
                    spatial_data['export_method'] = 'local'
                elif spatial_data.get('drive_folder') or spatial_data.get('task_id'):
                    spatial_data['export_method'] = 'drive'
                elif spatial_data.get('pending'):
                    spatial_data['export_method'] = 'pending'
                else:
                    # No data available - mark as failed
                    spatial_data['export_method'] = 'failed'
                    spatial_data['error'] = 'No export data generated'

            # Validate export_method is recognized
            valid_methods = ['local', 'drive', 'preview', 'pending', 'failed']
            if spatial_data.get('export_method') not in valid_methods:
                # Force to valid method based on available data
                if spatial_data.get('file_data'):
                    spatial_data['export_method'] = 'local'
                elif spatial_data.get('drive_folder'):
                    spatial_data['export_method'] = 'drive'
                elif spatial_data.get('pending'):
                    spatial_data['export_method'] = 'pending'
                else:
                    spatial_data['export_method'] = 'failed'

            # Ensure size information is present for local/preview
            if spatial_data.get('export_method') in ['local', 'preview'] and 'actual_size_mb' not in spatial_data:
                if spatial_data.get('file_data'):
                    spatial_data['actual_size_mb'] = len(spatial_data['file_data']) / (1024 * 1024)
                else:
                    spatial_data['actual_size_mb'] = 0

            # Ensure filename is present for local exports
            if spatial_data.get('export_method') == 'local' and 'filename' not in spatial_data:
                spatial_data['filename'] = f"{index_name}_spatial.zip"

            # Ensure preview has required fields
            if spatial_data.get('export_method') == 'preview':
                if 'file_size_mb' not in spatial_data:
                    spatial_data['file_size_mb'] = spatial_data.get('actual_size_mb', 0)
                if 'total_images' not in spatial_data:
                    spatial_data['total_images'] = 1

            # Ensure drive export has required fields
            if spatial_data.get('export_method') == 'drive':
                if 'drive_url' not in spatial_data:
                    spatial_data['drive_url'] = 'https://drive.google.com/drive/'
                if 'total_tasks' not in spatial_data:
                    spatial_data['total_tasks'] = 0

        validated_results[index_name] = index_result

        # Log validation results (compact)
        if 'spatial_data' in index_result:
            method = index_result['spatial_data'].get('export_method', 'missing')
            if method == 'failed':
                st.warning(f"⚠️ {index_name}: export={method}")
            elif method == 'pending':
                st.info(f"⏳ {index_name}: export={method}")
            else:
                st.info(f"✅ {index_name}: export={method}")

    return validated_results