"""
Map widget for selecting Area of Interest.
"""
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import ee
from typing import Optional, Callable, Dict, Any, List, Union
import json
from pathlib import Path

# Try different geemap import approaches
try:
    import geemap
    print("Using geemap from standard import")
except ImportError:
    try:
        import geemap.foliumap as geemap
        print("Using geemap.foliumap")
    except ImportError:
        print("ERROR: geemap not found. Please install with: pip install geemap")

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
        self.draw_control = None
        
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
            try:
                # Create map without reinitializing EE
                self.map = geemap.Map(ee_initialize=False)
                
                # Define a universal on_draw handler 
                def universal_handler(target, action, geo_json):
                    """Handles drawing events from any version of draw control"""
                    print(f"Drawing detected! Action: {action}")
                    self.current_draw_items = geo_json
                
                # Try different methods to add drawing controls
                try:
                    # Try to access existing draw control first
                    if hasattr(self.map, 'draw_control'):
                        self.draw_control = self.map.draw_control
                        print("Using existing draw control")
                    elif hasattr(self.map, '_draw_control'):
                        self.draw_control = self.map._draw_control
                        print("Using existing _draw_control")
                    else:
                        # Method 1: Using the direct parameter-less approach
                        self.map.add_draw_control()
                        # Try to get the draw control
                        if hasattr(self.map, 'draw_control'):
                            self.draw_control = self.map.draw_control
                        elif hasattr(self.map, '_draw_control'):
                            self.draw_control = self.map._draw_control
                        print("Added draw control with no parameters")
                except Exception as e1:
                    print(f"Method 1 failed: {str(e1)}")
                    try:
                        # Method 2: Try with ipyleaflet DrawControl directly
                        try:
                            from ipyleaflet import DrawControl
                            draw_control = DrawControl(
                                polygon={"shapeOptions": {"color": "#0000FF"}},
                                rectangle={"shapeOptions": {"color": "#0000FF"}},
                                circle={"shapeOptions": {"color": "#0000FF"}},
                                circlemarker={},
                                polyline={}
                            )
                            
                            # Set up callback to track drawn items
                            draw_control.on_draw(universal_handler)
                            self.map.add_control(draw_control)
                            self.draw_control = draw_control
                            print("Added ipyleaflet DrawControl directly")
                        except Exception as e2a:
                            print(f"Failed to add ipyleaflet DrawControl: {str(e2a)}")
                            # Try again with another approach
                            self.map.add_draw_control(export=False)
                            if hasattr(self.map, 'draw_control'):
                                self.draw_control = self.map.draw_control
                            elif hasattr(self.map, '_draw_control'):
                                self.draw_control = self.map._draw_control
                            print("Added draw control with export=False")
                    except Exception as e2:
                        print(f"Method 2 failed: {str(e2)}")
                        print("Could not add drawing controls. Map may not have drawing functionality.")
                
                # Now add event handlers to ALL possible draw controls
                try:
                    # Try to add handler to the draw control we found
                    if self.draw_control is not None:
                        if hasattr(self.draw_control, 'on_draw'):
                            self.draw_control.on_draw(universal_handler)
                            print("Added on_draw handler to draw_control")
                    
                    # Try to add handler to the map's draw control methods
                    if hasattr(self.map, 'on_draw'):
                        self.map.on_draw(lambda feature: self._on_map_draw(feature))
                        print("Added on_draw handler to map")
                    
                    # Other common methods
                    for method_name in ['on_draw_created', 'on_draw_change']:
                        if hasattr(self.map, method_name):
                            getattr(self.map, method_name)(lambda feature: self._on_map_draw(feature))
                            print(f"Added {method_name} handler to map")
                except Exception as e:
                    print(f"Error setting up draw handlers: {str(e)}")
                
                # Print available attributes for debugging
                map_attrs = [attr for attr in dir(self.map) if not attr.startswith('_') and 'draw' in attr.lower()]
                print(f"Map drawing-related attributes: {map_attrs}")
                
                display(self.map)
                
            except Exception as e:
                print(f"Error initializing map: {str(e)}")
                print("You may need to install the correct version of geemap.")
                print("Try: pip install geemap --upgrade")
        
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
    
    def _on_map_draw(self, feature):
        """Unified handler for map drawing events"""
        print(f"Map draw event detected! Feature: {type(feature)}")
        self.current_draw_items = feature
        
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
                self._clear_drawings()
                self.current_draw_items = None
            
            # Prompt user to use drawing tools in the map interface
            print("Please use the drawing tools in the map interface to draw a polygon.")
            print("Look for the polygon icon in the toolbar on the left side of the map.")
            
    def _clear_drawings(self):
        """Clear all drawings based on available API."""
        try:
            # Try multiple methods to clear drawings
            if self.draw_control is not None:
                try:
                    if hasattr(self.draw_control, 'clear'):
                        self.draw_control.clear()
                        print("Cleared drawings using draw_control.clear()")
                        return
                except Exception as e:
                    print(f"Error clearing with draw_control: {str(e)}")
            
            # Try map's drawing controls
            for obj_name in ['_draw_control', 'draw_control']:
                if hasattr(self.map, obj_name):
                    ctrl = getattr(self.map, obj_name)
                    try:
                        if hasattr(ctrl, 'clear'):
                            ctrl.clear()
                            print(f"Cleared drawings using map.{obj_name}.clear()")
                            return
                    except:
                        pass
            
            # Try alternative clear methods
            for method_name in ['clear_drawings', 'clear_controls', 'clear_layers']:
                if hasattr(self.map, method_name):
                    try:
                        getattr(self.map, method_name)()
                        print(f"Cleared drawings using map.{method_name}()")
                        return
                    except:
                        pass
                    
            print("Could not clear drawings using any known method")
            
        except Exception as e:
            print(f"Warning: Could not clear drawings: {str(e)}")
            print("Please use the trash icon in the drawing tools to clear drawings manually.")
            
    def _on_clear_button_click(self, button):
        """Clear all drawn geometries."""
        with self.output:
            clear_output()
            
            # Clear the map
            self._clear_drawings()
            self.current_draw_items = None
            self.status_label.value = 'No area selected'
            
    def _on_save_button_click(self, button):
        """Save the current geometry selection."""
        with self.output:
            clear_output()
            
            # Debug information
            print("Trying to save the drawn geometry...")
            if self.current_draw_items:
                print(f"current_draw_items type: {type(self.current_draw_items)}")
            else:
                print("current_draw_items is None, trying alternative methods...")
            
            # Try multiple ways to get drawn geometries
            if not self.current_draw_items:
                # Check map's user_roi/user_rois properties first 
                for attr in ['user_roi', 'user_rois', '_user_roi', '_user_rois']:
                    if hasattr(self.map, attr):
                        try:
                            value = getattr(self.map, attr)
                            if value:
                                print(f"Found geometry in map.{attr}")
                                self.current_draw_items = value
                                break
                        except Exception as e:
                            print(f"Error getting {attr}: {str(e)}")
                
                # If still not found, check other properties
                if not self.current_draw_items:
                    for attr in ['_last_draw', 'last_draw', 'last_active_drawing', 'layers']:
                        if hasattr(self.map, attr):
                            try:
                                value = getattr(self.map, attr)
                                if value:
                                    print(f"Found potential geometry in map.{attr}")
                                    self.current_draw_items = value
                                    break
                            except Exception as e:
                                print(f"Error getting {attr}: {str(e)}")
            
            # For geemap versions that use last_active_drawing
            if hasattr(self.map, 'draw_last_feature'):
                try:
                    feature = self.map.draw_last_feature()
                    if feature:
                        print("Got geometry from map.draw_last_feature()")
                        self.current_draw_items = feature
                except Exception as e:
                    print(f"Error calling draw_last_feature: {str(e)}")
            
            # Last resort - check if we have the draw control's features
            if self.draw_control and not self.current_draw_items:
                try:
                    if hasattr(self.draw_control, 'data'):
                        data = self.draw_control.data
                        if data and len(data.get('features', [])) > 0:
                            print("Got geometry from draw_control.data")
                            self.current_draw_items = data
                except Exception as e:
                    print(f"Error getting draw_control.data: {str(e)}")
            
            # Add manual approach for last resort
            if not self.current_draw_items:
                try:
                    # Create a simple polygon for testing if nothing else works
                    manual_input = input("Do you want to enter coordinates manually? (y/n): ")
                    if manual_input.lower() == 'y':
                        print("Enter coordinates as: lon1,lat1 lon2,lat2 lon3,lat3...")
                        coords_str = input("Coordinates: ")
                        coords = []
                        for pair in coords_str.split():
                            lon, lat = map(float, pair.split(','))
                            coords.append([lon, lat])
                        # Close the polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        
                        manual_geojson = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [coords]
                            }
                        }
                        self.current_draw_items = manual_geojson
                        print("Created manual geometry")
                except Exception as e:
                    print(f"Error creating manual geometry: {str(e)}")
            
            if self.current_draw_items:
                try:
                    # Debug the geometry
                    print(f"Found geometry of type: {type(self.current_draw_items)}")
                    
                    # Create EE geometry based on the type
                    if isinstance(self.current_draw_items, dict):
                        # GeoJSON dict - most common case
                        print("Processing GeoJSON dictionary")
                        
                        # Handle FeatureCollection format
                        if 'type' in self.current_draw_items and self.current_draw_items['type'] == 'FeatureCollection':
                            if len(self.current_draw_items['features']) > 0:
                                # Get the first feature from collection
                                feature = self.current_draw_items['features'][0]
                                if 'geometry' in feature:
                                    geometry = feature['geometry']
                                    print(f"Using first geometry from FeatureCollection: {geometry['type']}")
                                    self.geometry_handler.set_geometry_from_geojson(geometry)
                                else:
                                    self.geometry_handler.set_geometry_from_geojson(feature)
                            else:
                                print("Error: FeatureCollection is empty")
                        # Handle Feature format 
                        elif 'type' in self.current_draw_items and self.current_draw_items['type'] == 'Feature':
                            if 'geometry' in self.current_draw_items:
                                geometry = self.current_draw_items['geometry']
                                print(f"Using geometry from Feature: {geometry['type']}")
                                self.geometry_handler.set_geometry_from_geojson(geometry)
                            else:
                                self.geometry_handler.set_geometry_from_geojson(self.current_draw_items)
                        # Handle direct geometry format
                        else:
                            print("Using direct geometry format")
                            self.geometry_handler.set_geometry_from_geojson(self.current_draw_items)
                    
                    elif isinstance(self.current_draw_items, ee.Geometry):
                        # Already an EE Geometry
                        print("Using existing EE Geometry")
                        self.geometry_handler._current_geometry = self.current_draw_items
                        self.geometry_handler._current_geometry_name = "drawn_aoi"
                    
                    elif isinstance(self.current_draw_items, ee.FeatureCollection):
                        # EE FeatureCollection
                        print("Using EE FeatureCollection")
                        # Get first feature geometry
                        first_feature = ee.Feature(self.current_draw_items.first())
                        geometry = first_feature.geometry()
                        self.geometry_handler._current_geometry = geometry
                        self.geometry_handler._current_geometry_name = "drawn_aoi"
                    
                    elif hasattr(self.current_draw_items, '__geo_interface__'):
                        # Object with __geo_interface__ (like shapely)
                        geo_interface = self.current_draw_items.__geo_interface__
                        print(f"Using __geo_interface__: {geo_interface['type']}")
                        self.geometry_handler.set_geometry_from_geojson(geo_interface)
                    
                    else:
                        # Try to convert other types
                        print(f"Converting unknown type to geometry: {type(self.current_draw_items)}")
                        try:
                            # Last resort: try to convert to ee.Geometry
                            geometry = ee.Geometry(self.current_draw_items)
                            self.geometry_handler._current_geometry = geometry
                            self.geometry_handler._current_geometry_name = "drawn_aoi"
                        except:
                            # Try to convert to a GeoJSON
                            import json
                            try:
                                # If it's a string, try parsing it as JSON
                                if isinstance(self.current_draw_items, str):
                                    geo_json = json.loads(self.current_draw_items)
                                    self.geometry_handler.set_geometry_from_geojson(geo_json)
                                else:
                                    print(f"Could not convert type {type(self.current_draw_items)} to geometry")
                            except:
                                print(f"Could not parse item as GeoJSON: {self.current_draw_items}")
                        
                    # Calculate area if geometry is set
                    if self.geometry_handler.current_geometry:
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
                    else:
                        print("Error: Failed to set a valid geometry")
                        
                except Exception as e:
                    print(f"Error saving geometry: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            else:
                print("No area selected. Please draw a polygon or upload a file first.")
                print("Draw a polygon using the drawing tools in the map toolbar.")
                print("Look for the polygon icon in the toolbar on the left side of the map.")
                
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
                    
                    # Add to map - using available API
                    try:
                        # Clear existing drawings
                        self._clear_drawings()
                            
                        # Add the GeoJSON - try different methods
                        try:
                            self.map.add_geojson(geojson_dict, layer_name='uploaded_geometry')
                        except:
                            try:
                                # Alternative method for some versions
                                self.map.add_layer(geojson_dict, {}, 'uploaded_geometry')
                            except:
                                print("Warning: Could not visualize the uploaded GeoJSON on the map.")
                                print("But the geometry has been loaded and can be saved.")
                    except:
                        print("Warning: Could not visualize the uploaded GeoJSON on the map.")
                        print("But the geometry has been loaded and can be saved.")
                    
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