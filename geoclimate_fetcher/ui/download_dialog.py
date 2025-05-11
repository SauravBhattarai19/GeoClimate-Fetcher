"""
Download dialog widget for configuring and initiating downloads.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, Dict, Any, List, Union, Tuple
import pandas as pd
from pathlib import Path
import ee
from datetime import datetime, date

from geoclimate_fetcher.core.exporter import GEEExporter
from geoclimate_fetcher.core.metadata import MetadataCatalog
from geoclimate_fetcher.core.geometry import GeometryHandler

# Import our GeometryStateManager
try:
    # Try to find it in the module first
    from geoclimate_fetcher.state_manager import GeometryStateManager
except ImportError:
    # Define it inline if not found
    class GeometryStateManager:
        """Singleton class to store global geometry state across widgets"""
        
        _instance = None
        
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super(GeometryStateManager, cls).__new__(cls)
                cls._instance.geometry_handler = GeometryHandler()
                cls._instance.geometry_set = False
                cls._instance.debug_mode = True
            return cls._instance
        
        def set_geometry(self, geo_json=None, geometry=None, name="user_drawn_geometry"):
            """Set the geometry from GeoJSON or direct EE geometry"""
            if self.debug_mode:
                print(f"GeometryStateManager: Setting geometry with name '{name}'")
            
            if geo_json is not None:
                self.geometry_handler.set_geometry_from_geojson(geo_json, name)
                self.geometry_set = True
                
                if self.debug_mode:
                    try:
                        area = self.geometry_handler.get_geometry_area()
                        print(f"GeometryStateManager: Set geometry from GeoJSON, area = {area:.2f} km²")
                    except Exception as e:
                        print(f"GeometryStateManager: Error calculating area: {str(e)}")
            
            elif geometry is not None:
                self.geometry_handler._current_geometry = geometry
                self.geometry_handler._current_geometry_name = name
                self.geometry_set = True
                
                if self.debug_mode:
                    try:
                        area = self.geometry_handler.get_geometry_area()
                        print(f"GeometryStateManager: Set direct geometry, area = {area:.2f} km²")
                    except Exception as e:
                        print(f"GeometryStateManager: Error calculating area: {str(e)}")
        
        def get_geometry_handler(self):
            """Get the geometry handler with current geometry"""
            if self.debug_mode:
                print(f"GeometryStateManager: Getting geometry handler, geometry set = {self.geometry_set}")
                if self.geometry_set:
                    try:
                        area = self.geometry_handler.get_geometry_area()
                        print(f"GeometryStateManager: Current geometry area = {area:.2f} km²")
                    except Exception as e:
                        print(f"GeometryStateManager: Error calculating area: {str(e)}")
            
            return self.geometry_handler
        
        def has_geometry(self):
            """Check if geometry is set"""
            return self.geometry_set

# Create a global state manager
global_state_manager = GeometryStateManager()

class DownloadDialogWidget:
    """Widget for configuring and initiating downloads from Earth Engine."""
    
    def __init__(self, metadata_catalog: MetadataCatalog, geometry_handler: GeometryHandler, 
                on_download_complete: Optional[Callable] = None):
        """
        Initialize the download dialog widget.
        
        Args:
            metadata_catalog: MetadataCatalog instance
            geometry_handler: GeometryHandler instance
            on_download_complete: Callback function to execute after download completion
        """
        self.metadata_catalog = metadata_catalog
        
        # First use the provided geometry_handler
        self.geometry_handler = geometry_handler
        
        # Then check if we have a geometry in the provided handler
        has_geometry = False
        try:
            has_geometry = self.geometry_handler.current_geometry is not None
            if has_geometry:
                print(f"DownloadDialogWidget: Using provided geometry handler with area {self.geometry_handler.get_geometry_area():.2f} km²")
        except:
            pass
            
        # If no geometry in provided handler, use the global state
        if not has_geometry:
            print("DownloadDialogWidget: No geometry in provided handler, using global state manager")
            self.geometry_handler = global_state_manager.get_geometry_handler()
        
        self.on_download_complete = on_download_complete
        self.exporter = GEEExporter()
        
        # Print debug info about geometry
        print(f"Initializing DownloadDialogWidget with geometry_handler")
        if self.geometry_handler.current_geometry is not None:
            print(f"Geometry is set: {self.geometry_handler.current_geometry_name}")
            try:
                area = self.geometry_handler.get_geometry_area()
                print(f"Area: {area:.2f} km²")
            except Exception as e:
                print(f"Error getting area: {str(e)}")
        else:
            print("No geometry is set in the geometry_handler")
            # Try to get geometry from global state as a backup
            if global_state_manager.has_geometry():
                self.geometry_handler = global_state_manager.get_geometry_handler()
                print("Using geometry from global state manager instead")
        
        # Download parameters
        self.dataset_name = None
        self.ee_id = None
        self.bands = []
        self.start_date = None
        self.end_date = None
        self.download_path = None
        
        # Create UI components
        self.title = widgets.HTML("<h3>Download Configuration</h3>")
        
        self.extraction_mode_dropdown = widgets.Dropdown(
            options=[
                ('Average time-series', 'average'),
                ('Gridded rasters', 'gridded')
            ],
            description='Extraction Mode:',
            style={'description_width': 'initial'}
        )
        
        self.export_mode_dropdown = widgets.Dropdown(
            options=[
                ('Direct to local disk', 'local'),
                ('Export to Google Drive', 'drive')
            ],
            description='Export Mode:',
            style={'description_width': 'initial'}
        )
        
        self.output_format_dropdown = widgets.Dropdown(
            options=[
                ('CSV (for time-series)', 'csv'),
                ('NetCDF (for gridded data)', 'netcdf'),
                ('GeoTIFF (for static rasters)', 'geotiff'),
                ('Cloud Optimized GeoTIFF (COG)', 'cog')
            ],
            description='Output Format:',
            style={'description_width': 'initial'}
        )
        
        self.drive_folder_input = widgets.Text(
            value='GeoClimateFetcher',
            description='Drive Folder:',
            placeholder='Enter Google Drive folder name',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.local_path_input = widgets.Text(
            value='downloads',
            description='Local Path:',
            placeholder='Enter local directory path',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.filename_input = widgets.Text(
            value='',
            description='Filename:',
            placeholder='Enter filename (without extension)',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.scale_input = widgets.FloatText(
            value=1000.0,
            description='Scale (m):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        self.download_button = widgets.Button(
            description='Start Download',
            button_style='success',
            icon='download'
        )
        
        self.progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#2196f3'}
        )
        
        self.output = widgets.Output()
        
        # Bind events
        self.download_button.on_click(self._on_download_button_click)
        self.export_mode_dropdown.observe(self._on_export_mode_change, names='value')
        self.extraction_mode_dropdown.observe(self._on_extraction_mode_change, names='value')
        
        # Layout
        drive_folder_box = widgets.VBox([self.drive_folder_input])
        self.drive_folder_accordion = widgets.Accordion(children=[drive_folder_box])
        self.drive_folder_accordion.set_title(0, 'Google Drive Settings')
        self.drive_folder_accordion.selected_index = None  # Initially collapsed
        
        local_path_box = widgets.VBox([self.local_path_input])
        self.local_path_accordion = widgets.Accordion(children=[local_path_box])
        self.local_path_accordion.set_title(0, 'Local Path Settings')
        self.local_path_accordion.selected_index = 0  # Initially expanded
        
        # Main widget
        self.widget = widgets.VBox([
            self.title,
            self.extraction_mode_dropdown,
            self.export_mode_dropdown,
            self.output_format_dropdown,
            self.drive_folder_accordion,
            self.local_path_accordion,
            self.filename_input,
            self.scale_input,
            self.download_button,
            self.progress,
            self.output
        ])
        
        # Initialize UI state
        self._on_export_mode_change({'new': self.export_mode_dropdown.value})
        self._on_extraction_mode_change({'new': self.extraction_mode_dropdown.value})
        
    def display(self):
        """Display the download dialog widget."""
        display(self.widget)
        
    def set_parameters(self, dataset_name: str, ee_id: str, bands: List[str], 
                     start_date: Optional[date] = None, end_date: Optional[date] = None):
        """
        Set the download parameters.
        
        Args:
            dataset_name: Dataset name
            ee_id: Earth Engine ID
            bands: List of bands to download
            start_date: Optional start date for time series
            end_date: Optional end date for time series
        """
        self.dataset_name = dataset_name
        self.ee_id = ee_id
        self.bands = bands
        self.start_date = start_date
        self.end_date = end_date
        
        # Update filename suggestion
        sanitized_name = dataset_name.replace(' ', '_').lower()
        geometry_name = self.geometry_handler.current_geometry_name or 'custom_aoi'
        sanitized_geometry = geometry_name.replace(' ', '_').lower()
        
        self.filename_input.value = f"{sanitized_name}_{sanitized_geometry}"
        
        with self.output:
            clear_output()
            print(f"Ready to download {dataset_name}")
            print(f"Selected bands: {', '.join(bands)}")
            if start_date and end_date:
                print(f"Date range: {start_date} to {end_date}")
                
    def _on_export_mode_change(self, change):
        """Handle export mode change."""
        if change['new'] == 'drive':
            self.drive_folder_accordion.selected_index = 0  # Expand
            self.local_path_accordion.selected_index = None  # Collapse
        else:
            self.drive_folder_accordion.selected_index = None  # Collapse
            self.local_path_accordion.selected_index = 0  # Expand
            
    def _on_extraction_mode_change(self, change):
        """Handle extraction mode change."""
        extraction_mode = change['new']
        
        # Update output format options based on extraction mode
        if extraction_mode == 'average':
            self.output_format_dropdown.options = [('CSV (for time-series)', 'csv')]
        else:  # gridded
            self.output_format_dropdown.options = [
                ('NetCDF (for gridded data)', 'netcdf'),
                ('GeoTIFF (for static rasters)', 'geotiff'),
                ('Cloud Optimized GeoTIFF (COG)', 'cog')
            ]
            
    def _on_download_button_click(self, button):
        """Handle download button click."""
        with self.output:
            clear_output()
            
            # Additional geometry check with detailed info
            print("Checking geometry before download...")
            
            # Try using the global state first as a last resort
            if self.geometry_handler is None or self.geometry_handler.current_geometry is None:
                print("No geometry in local handler, checking global state...")
                if global_state_manager.has_geometry():
                    print("Found geometry in global state, using it")
                    self.geometry_handler = global_state_manager.get_geometry_handler()
            
            # Normal check for geometry
            if self.geometry_handler is None:
                print("Error: geometry_handler is None")
                return
                
            # More detailed check for geometry
            if self.geometry_handler.current_geometry is None:
                print("Error: No area of interest selected. Please select an area first.")
                
                # Let's try one more approach - check if there's a user_roi on the map
                try:
                    from geemap.geemap import Map
                    maps = [obj for obj in globals().values() if isinstance(obj, Map)]
                    if maps:
                        map_obj = maps[0]
                        if hasattr(map_obj, 'user_roi') and map_obj.user_roi is not None:
                            print("Found geometry in map.user_roi, applying it now...")
                            self.geometry_handler.set_geometry_from_geojson(map_obj.user_roi, "recovered_aoi")
                            # Also update global state
                            global_state_manager.set_geometry(geo_json=map_obj.user_roi, name="recovered_aoi")
                except Exception as e:
                    print(f"Attempted recovery failed: {str(e)}")
                    
                # Check again after recovery attempt
                if self.geometry_handler.current_geometry is None:
                    return
                
            # Validate parameters
            if not self.dataset_name or not self.ee_id or not self.bands:
                print("Error: Missing dataset parameters. Please select a dataset and bands first.")
                return
                
            extraction_mode = self.extraction_mode_dropdown.value
            export_mode = self.export_mode_dropdown.value
            output_format = self.output_format_dropdown.value
            filename = self.filename_input.value.strip()
            
            if not filename:
                print("Error: Please provide a filename.")
                return
                
            # Set up extraction parameters
            scale = self.scale_input.value
            
            # Get snippet type (Image/ImageCollection)
            snippet_type = self.metadata_catalog.get_snippet_type(self.dataset_name)
            
            # Create the appropriate fetcher based on snippet type
            try:
                geometry = self.geometry_handler.current_geometry
                
                # Verify we have a geometry
                if geometry is None:
                    print("Error: Still no geometry available for download. Cannot proceed.")
                    return
                    
                print(f"Using geometry with area: {self.geometry_handler.get_geometry_area():.2f} km²")
                
                # Initialize progress
                self.progress.value = 10

                if snippet_type == 'ImageCollection':
                    from geoclimate_fetcher.core.fetchers import ImageCollectionFetcher
                    
                    # Ensure dates are provided for ImageCollection
                    if not self.start_date or not self.end_date:
                        print("Error: Start and end dates are required for time series data.")
                        self.progress.value = 0
                        return
                        
                    # Create fetcher
                    fetcher = ImageCollectionFetcher(self.ee_id, self.bands, geometry)
                    fetcher.filter_dates(self.start_date, self.end_date)
                    
                    # Extract data
                    if extraction_mode == 'average':
                        # Use chunked processing to handle large datasets
                        print("Fetching time series averages using chunked processing...")
                        data = fetcher.get_time_series_average_chunked()
                        
                        # Update progress
                        self.progress.value = 50
                        
                        # Export based on selected mode
                        if export_mode == 'local':
                            # Local export
                            local_path = Path(self.local_path_input.value)
                            local_path.mkdir(parents=True, exist_ok=True)
                            file_path = local_path / f"{filename}.csv"
                            
                            print(f"Saving to {file_path}...")
                            self.exporter.export_time_series_to_csv(data, file_path)
                            
                            print(f"Download complete: {file_path}")
                            
                        else:  # Google Drive
                            # Export to Drive with chunking
                            folder = self.drive_folder_input.value
                            print(f"Exporting to Google Drive folder '{folder}' using chunked processing...")
                            
                            task_ids = self.exporter.export_time_series_to_drive_chunked(
                                data, filename, folder
                            )
                            
                            print(f"Started {len(task_ids)} export tasks to Google Drive folder: {folder}")
                            
                    else:  # gridded
                        # Get gridded data
                        print("Fetching gridded data...")
                        data = fetcher.get_gridded_data(scale=scale)
                        
                        # Update progress
                        self.progress.value = 50
                        
                        # Export based on selected mode and format
                        if export_mode == 'local':
                            # Local export
                            local_path = Path(self.local_path_input.value)
                            local_path.mkdir(parents=True, exist_ok=True)
                            
                            if output_format == 'netcdf':
                                file_path = local_path / f"{filename}.nc"
                                print(f"Saving to {file_path}...")
                                self.exporter.export_gridded_data_to_netcdf(data, file_path)
                                print(f"Download complete: {file_path}")
                                
                            elif output_format in ['geotiff', 'cog']:
                                # For GeoTIFF, export each time step as a separate file
                                print(f"Saving time series as individual GeoTIFF files to {local_path}/{filename}_YYYYMMDD.tif...")
                                
                                # Create subdirectory with the filename
                                tiff_dir = local_path / filename
                                self.exporter.export_time_series_to_geotiff(data, tiff_dir)
                                
                                print(f"Download complete: {tiff_dir}")
                                
                            else:
                                print("Error: Unsupported output format. Please use NetCDF or GeoTIFF.")
                                self.progress.value = 0
                                return
                                
                        else:  # Google Drive
                            folder = self.drive_folder_input.value
                            
                            # For time series in Drive, we need to export each image separately
                            if output_format in ['geotiff', 'cog']:
                                print(f"Exporting time series to Google Drive folder '{folder}'...")
                                
                                # Get times from the dataset
                                times = pd.to_datetime(data.time.values)
                                
                                # For each date, export an image to Drive
                                success_count = 0
                                total_dates = len(times)
                                
                                with tqdm(total=total_dates, desc="Exporting dates to Drive") as pbar:
                                    for i, time_value in enumerate(times):
                                        date_str = time_value.strftime('%Y%m%d')
                                        date_filename = f"{filename}_{date_str}"
                                        
                                        try:
                                            # Create an image for this date with all bands
                                            img_date = time_value.strftime('%Y-%m-%d')
                                            next_day = (time_value + timedelta(days=1)).strftime('%Y-%m-%d')
                                            
                                            # Filter collection to this date
                                            date_img = self.collection.filter(ee.Filter.date(img_date, next_day)).first()
                                            
                                            if date_img is not None:
                                                # Export to Drive
                                                self.exporter.export_image_to_drive(
                                                    date_img.select(self.bands), 
                                                    date_filename, 
                                                    folder, 
                                                    geometry, 
                                                    scale, 
                                                    wait=False  # Don't wait for each - just start the tasks
                                                )
                                                success_count += 1
                                        except Exception as e:
                                            print(f"Error exporting date {date_str}: {str(e)}")
                                        
                                        # Update progress bar
                                        pbar.update(1)
                                
                                print(f"Submitted {success_count} out of {total_dates} dates for export to Google Drive")
                                print("Exports will continue in the background. Check your Drive folder when complete.")
                                
                            else:  # Export NetCDF to Drive
                                print("Error: Direct NetCDF export to Google Drive is not supported.")
                                print("Please use GeoTIFF format for Google Drive exports or save NetCDF locally.")
                                self.progress.value = 0
                                return
                            
                else:  # Image
                    from geoclimate_fetcher.core.fetchers import StaticRasterFetcher
                    
                    # Create fetcher
                    fetcher = StaticRasterFetcher(self.ee_id, self.bands, geometry)
                    
                    # Extract data
                    if extraction_mode == 'average':
                        # Get mean values
                        print("Fetching spatial averages...")
                        data = fetcher.get_mean_values()
                        
                        # Convert to DataFrame
                        df = pd.DataFrame([data])
                        
                        # Update progress
                        self.progress.value = 50
                        
                        # Export based on selected mode
                        if export_mode == 'local':
                            # Local export
                            local_path = Path(self.local_path_input.value)
                            local_path.mkdir(parents=True, exist_ok=True)
                            file_path = local_path / f"{filename}.csv"
                            
                            print(f"Saving to {file_path}...")
                            self.exporter.export_time_series_to_csv(df, file_path)
                            
                            print(f"Download complete: {file_path}")
                            
                        else:  # Google Drive
                            # Create a feature collection from the data
                            fc = ee.FeatureCollection([ee.Feature(None, data)])
                            
                            # Export to Drive
                            folder = self.drive_folder_input.value
                            print(f"Exporting to Google Drive folder '{folder}'...")
                            
                            self.exporter.export_table_to_drive(
                                fc, filename, folder, wait=True
                            )
                            
                            print(f"Export to Google Drive complete. Check your Drive folder: {folder}")
                            
                    else:  # gridded
                        # Create the image with only selected bands
                        image = ee.Image(self.ee_id).select(self.bands)
                        
                        # Update progress
                        self.progress.value = 30
                        
                        # Export based on selected mode and format
                        if export_mode == 'local':
                            # Local export
                            local_path = Path(self.local_path_input.value)
                            local_path.mkdir(parents=True, exist_ok=True)
                            
                            if output_format == 'geotiff':
                                file_path = local_path / f"{filename}.tif"
                                print(f"Saving to {file_path}...")
                                self.exporter.export_image_to_local(image, file_path, geometry, scale)
                                
                            elif output_format == 'cog':
                                file_path = local_path / f"{filename}.tif"
                                print(f"Saving to {file_path} as Cloud Optimized GeoTIFF...")
                                self.exporter.export_to_cloud_optimized_geotiff(image, file_path, geometry, scale)
                                
                            else:  # netcdf
                                # Get pixel values
                                print("Fetching pixel values...")
                                pixel_data = fetcher.get_pixel_values(scale=scale)
                                
                                # Convert to xarray
                                # This is a simplified version and doesn't include proper geolocation
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
                                file_path = local_path / f"{filename}.nc"
                                print(f"Saving to {file_path}...")
                                self.exporter.export_gridded_data_to_netcdf(ds, file_path)
                                
                            print(f"Download complete: {file_path}")
                            
                        else:  # Google Drive
                            folder = self.drive_folder_input.value
                            print(f"Exporting to Google Drive folder '{folder}'...")
                            
                            self.exporter.export_image_to_drive(
                                image, filename, folder, geometry, scale, wait=True
                            )
                            
                            print(f"Export to Google Drive complete. Check your Drive folder: {folder}")
                            
                # Complete
                self.progress.value = 100
                
                if self.on_download_complete:
                    self.on_download_complete()
                    
            except Exception as e:
                print(f"Error during download: {str(e)}")
                import traceback
                traceback.print_exc()
                self.progress.value = 0