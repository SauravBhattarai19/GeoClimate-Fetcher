import streamlit as st
import folium
from folium.plugins import Draw
import json
from pathlib import Path
import sys
import ee

# Add geoclimate_fetcher to path
project_root = Path(__file__).parent.parent
geoclimate_path = project_root / "geoclimate_fetcher"
if str(geoclimate_path) not in sys.path:
    sys.path.insert(0, str(geoclimate_path))

from geoclimate_fetcher.core import GeometryHandler
from streamlit_folium import st_folium

class GeometryComponent:
    """Universal component for selecting area of interest across all modules"""
    
    def __init__(self, session_prefix="", geometry_complete_key=None, title="📍 Select Area of Interest"):
        """
        Initialize the geometry component with configurable session state keys.
        
        Args:
            session_prefix (str): Prefix for session state keys (e.g., "climate_" for climate module)
            geometry_complete_key (str): Custom key for geometry completion status
            title (str): Custom title for the component
        """
        self.session_prefix = session_prefix
        self.title = title
        
        # Set up session state keys
        self.handler_key = f"{session_prefix}geometry_handler"
        self.complete_key = geometry_complete_key or f"{session_prefix}geometry_complete"
        
        # Initialize geometry handler if not exists
        if self.handler_key not in st.session_state:
            st.session_state[self.handler_key] = GeometryHandler()
    
    @property
    def geometry_handler(self):
        """Get the geometry handler for this component"""
        return st.session_state[self.handler_key]
    
    @property
    def is_complete(self):
        """Check if geometry selection is complete"""
        return st.session_state.get(self.complete_key, False)
    
    def set_complete(self, value=True):
        """Set geometry completion status"""
        st.session_state[self.complete_key] = value
    
    def create_map(self, center_lat=39.8283, center_lon=-98.5795, zoom=4):
        """Create a folium map with drawing tools"""
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap',
            attr='Map data © OpenStreetMap contributors'
        )
          # Add additional tile layers
        folium.TileLayer(
            tiles='Stamen Terrain',
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
            name='Terrain'
        ).add_to(m)
        folium.TileLayer(
            tiles='CartoDB positron',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='Light'
        ).add_to(m)
        
        # Add drawing tools
        draw = Draw(
            export=True,
            position='topright',
            draw_options={
                'polyline': False,
                'rectangle': {
                    'shapeOptions': {
                        'color': '#ff7f0e',
                        'fillColor': '#ff7f0e',
                        'fillOpacity': 0.3
                    }
                },
                'polygon': {
                    'shapeOptions': {
                        'color': '#1f77b4',
                        'fillColor': '#1f77b4',
                        'fillOpacity': 0.3
                    }
                },
                'circle': False,
                'marker': False,
                'circlemarker': False
            },
            edit_options={
                'edit': True,
                'remove': True
            }
        )
        draw.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_geometry_from_coordinates(self, min_lon, min_lat, max_lon, max_lat):
        """Create Earth Engine geometry from coordinates"""
        try:
            geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
            self.geometry_handler._current_geometry = geometry
            self.geometry_handler._current_geometry_name = "coordinates_aoi"
            return True, "Geometry created successfully from coordinates"
        except Exception as e:
            return False, f"Error creating geometry: {str(e)}"
    
    def create_geometry_from_geojson(self, geojson_data, simplify_tolerance=500):
        """
        Create Earth Engine geometry from GeoJSON with union and simplification.

        Handles multiple features by:
        1. Unioning all features into a single geometry
        2. Simplifying the boundary to reduce vertices

        Args:
            geojson_data: GeoJSON data (dict or string)
            simplify_tolerance: Simplification tolerance in meters (default 500m)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Handle different GeoJSON formats
            if isinstance(geojson_data, str):
                geojson_dict = json.loads(geojson_data)
            else:
                geojson_dict = geojson_data

            # Extract and process geometry based on type
            if geojson_dict.get("type") == "FeatureCollection" and "features" in geojson_dict:
                features = geojson_dict["features"]

                if len(features) == 0:
                    return False, "FeatureCollection is empty"

                elif len(features) == 1:
                    # Single feature - process normally
                    geometry_dict = features[0]["geometry"]
                    geometry = ee.Geometry(geometry_dict)

                else:
                    # Multiple features - union and simplify
                    st.info(f"ℹ️ Processing {len(features)} features: unioning and simplifying boundaries...")

                    # Convert all features to EE features
                    ee_features = []
                    for feature in features:
                        if "geometry" in feature:
                            geom = ee.Geometry(feature["geometry"])
                            ee_features.append(ee.Feature(geom))

                    # Create FeatureCollection
                    feature_collection = ee.FeatureCollection(ee_features)

                    # Union all features (dissolve internal boundaries)
                    unified = feature_collection.union(maxError=1)

                    # Get the unified geometry
                    geometry = unified.geometry()

                    # Simplify the outer boundary
                    geometry = geometry.simplify(maxError=simplify_tolerance)

                    st.success(f"✅ Unified {len(features)} features into single geometry (simplified with {simplify_tolerance}m tolerance)")

            elif geojson_dict.get("type") == "Feature" and "geometry" in geojson_dict:
                # Single Feature
                geometry_dict = geojson_dict["geometry"]
                geometry = ee.Geometry(geometry_dict)

            else:
                # Direct geometry
                geometry = ee.Geometry(geojson_dict)

            # Store the geometry
            self.geometry_handler._current_geometry = geometry
            self.geometry_handler._current_geometry_name = "uploaded_aoi"
            return True, "Geometry created successfully from GeoJSON"

        except Exception as e:
            return False, f"Error processing GeoJSON: {str(e)}"
    
    def render(self, continue_button_text="Continue to Next Step", show_continue_button=True):
        """Render the geometry selection component"""
        st.markdown(f"## {self.title}")
        
        # Check if geometry already selected
        if self.is_complete:
            st.success("✅ Area of interest already selected!")
            
            # Show current geometry info
            try:
                handler = self.geometry_handler
                if handler.current_geometry:
                    area = handler.get_geometry_area()
                    name = handler.current_geometry_name
                    st.info(f"Current area: {name} ({area:.2f} km²)")
            except Exception:
                pass
            
            if show_continue_button:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(continue_button_text, type="primary", key=f"{self.session_prefix}continue"):
                        return True
                with col2:
                    if st.button("Select Different Area", key=f"{self.session_prefix}different"):
                        self.set_complete(False)
                        st.rerun()
            else:
                if st.button("Select Different Area", key=f"{self.session_prefix}different"):
                    self.set_complete(False)
                    st.rerun()
            return False
        
        st.markdown("""
        Choose your area of interest using one of the methods below.
        You can draw on the map, upload a GeoJSON file, or enter coordinates manually.
        """)
        
        # Method selection
        method = st.radio(
            "Select method:",
            ["🗺️ Draw on Map", "📁 Upload GeoJSON", "📐 Enter Coordinates"],
            horizontal=True,
            key=f"{self.session_prefix}method_selection"
        )
        
        if method == "🗺️ Draw on Map":
            return self.render_map_method()
        elif method == "📁 Upload GeoJSON":
            return self.render_upload_method()
        elif method == "📐 Enter Coordinates":
            return self.render_coordinates_method()
        
        return False
    
    def render_map_method(self):
        """Render the interactive map method"""
        st.markdown("### 🗺️ Interactive Map")
        st.info("Use the drawing tools in the top-right corner to draw a rectangle or polygon on the map.")
        
        # Create the map
        m = self.create_map()
        
        # Display the map and capture interactions
        map_data = st_folium(
            m,
            key=f"{self.session_prefix}aoi_map",
            width=700,
            height=500,
            returned_objects=["all_drawings", "last_object_clicked"]
        )
        
        # Process drawn features
        if map_data['all_drawings'] and len(map_data['all_drawings']) > 0:
            st.success(f"✅ Found {len(map_data['all_drawings'])} drawn feature(s)")
            
            # Use the most recent drawing
            latest_drawing = map_data['all_drawings'][-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(latest_drawing, expanded=False)
            
            with col2:
                if st.button("Use This Area", type="primary", key=f"{self.session_prefix}use_area"):
                    success, message = self.create_geometry_from_geojson(latest_drawing['geometry'])
                    if success:
                        st.success(f"✅ {message}")
                        self.set_complete(True)
                        return True
                    else:
                        st.error(f"❌ {message}")
        else:
            st.info("👆 Draw a rectangle or polygon on the map above to select your area of interest.")
        
        return False
    
    def render_upload_method(self):
        """Render the file upload method"""
        st.markdown("### 📁 Upload GeoJSON File")
        
        uploaded_file = st.file_uploader(
            "Choose a GeoJSON file",
            type=["geojson", "json"],
            help="Upload a GeoJSON file containing your area of interest",
            key=f"{self.session_prefix}file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                geojson_data = json.loads(uploaded_file.getvalue().decode())
                
                # Show preview
                st.success("✅ File loaded successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**File Contents:**")
                    st.json(geojson_data, expanded=False)
                
                with col2:
                    st.markdown("**Actions:**")
                    if st.button("Use This File", type="primary", key=f"{self.session_prefix}use_file"):
                        success, message = self.create_geometry_from_geojson(geojson_data)
                        if success:
                            st.success(f"✅ {message}")
                            self.set_complete(True)
                            return True
                        else:
                            st.error(f"❌ {message}")
                            
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        return False
    
    def render_coordinates_method(self):
        """Render the manual coordinates entry method"""
        st.markdown("### 📐 Enter Bounding Box Coordinates")
        
        # Coordinate input form
        with st.form(f"{self.session_prefix}coordinates_form"):
            st.markdown("Enter the bounding box coordinates (in decimal degrees):")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Southwest Corner:**")
                min_lon = st.number_input(
                    "Minimum Longitude",
                    value=-95.0,
                    min_value=-180.0,
                    max_value=180.0,
                    format="%.6f",
                    help="Western boundary"
                )
                min_lat = st.number_input(
                    "Minimum Latitude",
                    value=30.0,
                    min_value=-90.0,
                    max_value=90.0,
                    format="%.6f",
                    help="Southern boundary"
                )
            
            with col2:
                st.markdown("**Northeast Corner:**")
                max_lon = st.number_input(
                    "Maximum Longitude",
                    value=-94.0,
                    min_value=-180.0,
                    max_value=180.0,
                    format="%.6f",
                    help="Eastern boundary"
                )
                max_lat = st.number_input(
                    "Maximum Latitude",
                    value=31.0,
                    min_value=-90.0,
                    max_value=90.0,
                    format="%.6f",
                    help="Northern boundary"
                )
            
            # Validation
            valid = True
            if min_lon >= max_lon:
                st.error("❌ Minimum longitude must be less than maximum longitude")
                valid = False
            if min_lat >= max_lat:
                st.error("❌ Minimum latitude must be less than maximum latitude")
                valid = False
            
            submitted = st.form_submit_button("Create Bounding Box", type="primary", disabled=not valid)
            
            if submitted and valid:
                success, message = self.create_geometry_from_coordinates(min_lon, min_lat, max_lon, max_lat)
                if success:
                    st.success(f"✅ {message}")
                    self.set_complete(True)
                    return True
                else:
                    st.error(f"❌ {message}")
        
        # Show preview map
        if st.checkbox("Show preview map", key=f"{self.session_prefix}show_preview"):
            preview_map = folium.Map(
                location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
                zoom_start=6
            )
            
            # Add bounding box
            bbox_coords = [
                [min_lat, min_lon],
                [min_lat, max_lon],
                [max_lat, max_lon],
                [max_lat, min_lon],
                [min_lat, min_lon]
            ]
            
            folium.Polygon(
                locations=bbox_coords,
                color="red",
                fill_color="red",
                fill_opacity=0.2,
                popup="Selected Area"
            ).add_to(preview_map)
            
            # Fit bounds
            preview_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
            
            st_folium(preview_map, width=700, height=300, key=f"{self.session_prefix}preview_map")
        
        return False 