"""
Map widget for selecting Area of Interest.
"""
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import geemap.foliumap as geemap
import ee
from typing import Optional, Callable, Dict, Any, List, Union
import json
from pathlib import Path

from geoclimate_fetcher.core.geometry import GeometryHandler

class MapWidget:
    """Widget for selecting Area of Interest on a map."""
    
    def __init__(self, on_geometry_selected: Optional[Callable] = None):
        """
        Initialize the map widget.
        
        Args:
            on_geometry_selected: Callback function to execute after geometry selection
        """
        self.geometry_handler = GeometryHandler()
        self.on_geometry_selected = on_geometry_selected
        self.current_draw_items = None
        self.map = None
        
        # Create UI components
        self.title = widgets.HTML("<h3>Area of Interest Selection</h3>")
        
        self.draw_button = widgets.Button(
            description='Draw Polygon',
            button_style='info',
            icon='pencil'
        )
        
        self.clear_button = widgets.Button(
            description='Clear',
            button_style='warning',
            icon='trash'
        )
        
        self.upload_button = widgets.FileUpload(
            accept='.shp,.geojson,.json',
            multiple=False,
            description='Upload Shapefile/GeoJSON',
            style={'description_width': 'initial'}
        )
        
        self.save_button = widgets.Button(
            description='Save Selection',
            button_style='success',
            icon='check'
        )
        
        self.status_label = widgets.Label(value='No area selected')
        
        self.output = widgets.Output()
        
        # Bind button clicks
        self.draw_button.on_click(self._on_draw_button_click)
        self.clear_button.on_click(self._on_clear_button_click)
        self.save_button.on_click(self._on_save_button_click)
        self.upload_button.observe(self._on_file_upload, names='value')
        
        # Create the map
        self.map_widget = widgets.Output()
        with self.map_widget:
            self.map = geemap.Map()
            self.map.add_draw_control(export=False)
            display(self.map)
        
        # Control widgets
        self.controls = widgets.HBox([
            self.draw_button,
            self.clear_button,
            self.upload_button,
            self.save_button
        ])
        
        # Main widget
        self.widget = widgets.VBox([
            self.title,
            widgets.HTML("<p>Select an area of interest by drawing a polygon or uploading a file:</p>"),
            self.controls,
            self.status_label,
            self.map_widget,
            self.output
        ])
        
    def display(self):
        """Display the map widget."""
        display(self.widget)
        
    def _on_draw_button_click(self, button):
        """Enable drawing mode on the map."""
        with self.output:
            clear_output()
            print("Draw a polygon on the map. Click points to create vertices, then click the first point to close.")
            
            # Clear any existing geometries
            if self.current_draw_items:
                self.map.draw_control.clear()
                self.current_draw_items = None
                
            # Activate drawing
            self.map.draw_control.activate_mode('polygon')
            
    def _on_clear_button_click(self, button):
        """Clear all drawn geometries."""
        with self.output:
            clear_output()
            
            # Clear the map
            if hasattr(self.map, 'draw_control'):
                self.map.draw_control.clear()
                
            self.current_draw_items = None
            self.status_label.value = 'No area selected'
            
    def _on_save_button_click(self, button):
        """Save the current geometry selection."""
        with self.output:
            clear_output()
            
            if self.current_draw_items:
                try:
                    # Create EE geometry
                    if isinstance(self.current_draw_items, dict):  # GeoJSON dict
                        self.geometry_handler.set_geometry_from_geojson(self.current_draw_items)
                    else:  # Drawn geometry from draw control
                        geojson = self.current_draw_items
                        self.geometry_handler.set_geometry_from_drawn(geojson)
                        
                    # Calculate area
                    area = self.geometry_handler.get_geometry_area()
                    
                    # Update status
                    self.status_label.value = f'Selected area: {area:.2f} km²'
                    
                    # Validate the geometry
                    valid, error = self.geometry_handler.validate_geometry()
                    
                    if valid:
                        print(f"Area selected: {area:.2f} km²")
                        
                        if self.on_geometry_selected:
                            self.on_geometry_selected(self.geometry_handler)
                    else:
                        print(f"Invalid geometry: {error}")
                        
                except Exception as e:
                    print(f"Error saving geometry: {str(e)}")
            else:
                print("No area selected. Please draw a polygon or upload a file first.")
                
    def _on_file_upload(self, change):
        """Handle file upload."""
        with self.output:
            clear_output()
            
            if not change.new:
                return
                
            try:
                # Get the first file
                file_info = next(iter(change.new.values()))
                
                # Check file type
                filename = file_info['name']
                content = file_info['content']
                
                if filename.endswith('.geojson') or filename.endswith('.json'):
                    # Parse GeoJSON
                    geojson_dict = json.loads(content.tobytes().decode('utf-8'))
                    
                    # Store the GeoJSON
                    self.current_draw_items = geojson_dict
                    
                    # Add to map
                    self.map.draw_control.clear()
                    self.map.add_geojson(geojson_dict, layer_name='uploaded_geometry')
                    
                    print(f"GeoJSON file '{filename}' loaded successfully.")
                    
                elif filename.endswith('.shp'):
                    # For Shapefiles, we need to save them temporarily
                    temp_dir = Path('temp_shp')
                    temp_dir.mkdir(exist_ok=True)
                    
                    temp_file = temp_dir / filename
                    with open(temp_file, 'wb') as f:
                        f.write(content)
                        
                    print(f"Shapefile upload not fully supported. Please upload a GeoJSON file instead.")
                    
                    # We'd need the accompanying .dbf, .shx, etc. files to read a shapefile properly
                    # This is just a placeholder for now
                    
                else:
                    print(f"Unsupported file type: {filename}")
                    return
                    
            except Exception as e:
                print(f"Error processing uploaded file: {str(e)}")
                
    def get_geometry_handler(self) -> GeometryHandler:
        """
        Get the current geometry handler.
        
        Returns:
            GeometryHandler instance
        """
        return self.geometry_handler
        
    def has_geometry(self) -> bool:
        """
        Check if a geometry has been selected.
        
        Returns:
            True if a geometry is selected, False otherwise
        """
        return self.geometry_handler.current_geometry is not None