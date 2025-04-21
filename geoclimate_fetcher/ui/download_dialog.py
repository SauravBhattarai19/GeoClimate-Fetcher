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
        self.geometry_handler = geometry_handler
        self.on_download_complete = on_download_complete
        self.exporter = GEEExporter()
        
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
            value='downloads/',
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
            
            # Validate parameters
            if not self.dataset_name or not self.ee_id or not self.bands:
                print("Error: Missing dataset parameters. Please select a dataset and bands first.")
                return
                
            if not self.geometry_handler.current_geometry:
                print("Error: No area of interest selected. Please select an area first.")
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
                        # Get time series average
                        print("Fetching time series averages...")
                        data = fetcher.get_time_series_average()
                        
                        # Update progress
                        self.progress.value = 50
                        
                        # Export based on selected mode
                        if export_mode == 'local':
                            # Local export
                            local_path = Path(self.local_path_input.value)
                            file_path = local_path / f"{filename}.csv"
                            
                            print(f"Saving to {file_path}...")
                            self.exporter.export_time_series_to_csv(data, file_path)
                            
                            print(f"Download complete: {file_path}")
                            
                        else:  # Google Drive
                            # Create a feature collection from the DataFrame
                            features = []
                            for _, row in data.iterrows():
                                properties = {col: row[col] for col in data.columns}
                                features.append(ee.Feature(None, properties))
                                
                            fc = ee.FeatureCollection(features)
                            
                            # Export to Drive
                            folder = self.drive_folder_input.value
                            print(f"Exporting to Google Drive folder '{folder}'...")
                            
                            self.exporter.export_table_to_drive(
                                fc, filename, folder, wait=True
                            )
                            
                            print(f"Export to Google Drive complete. Check your Drive folder: {folder}")
                            
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
                            
                            if output_format == 'netcdf':
                                file_path = local_path / f"{filename}.nc"
                                print(f"Saving to {file_path}...")
                                self.exporter.export_gridded_data_to_netcdf(data, file_path)
                                
                            else:  # geotiff or cog
                                # Convert xarray to ee.Image for export
                                print("Error: Direct GeoTIFF export for gridded time series not implemented.")
                                print("Please use NetCDF format or export to Google Drive.")
                                self.progress.value = 0
                                return
                                
                            print(f"Download complete: {file_path}")
                            
                        else:  # Google Drive
                            folder = self.drive_folder_input.value
                            print(f"Exporting to Google Drive folder '{folder}'...")
                            
                            # This would require converting the xarray to an EE Image
                            # This is complex and not fully implemented here
                            print("Error: Export to Drive for gridded time series not implemented.")
                            print("Please use local export with NetCDF format.")
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
                self.progress.value = 0