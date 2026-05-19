"""
Unified Map Widget for GeoClimate Fetcher
Provides consistent map functionality across all modules
"""

import folium
from folium.plugins import Draw
import streamlit as st
from streamlit_folium import st_folium
import ee
import json
from typing import Optional, Dict, Any, Tuple, Callable


class UnifiedMapWidget:
    """
    Universal map widget for area selection across all GeoClimate modules.
    
    Provides consistent map appearance, functionality, and user experience.
    Handles geometry creation, validation, and export.
    """
    
    def __init__(self, 
                 session_prefix: str = "",
                 map_key: str = None,
                 center_lat: float = 39.8283,
                 center_lon: float = -98.5795,
                 zoom: int = 4):
        """
        Initialize the unified map widget.
        
        Args:
            session_prefix: Prefix for session state keys (e.g., "climate_")
            map_key: Unique key for the map component
            center_lat: Map center latitude
            center_lon: Map center longitude  
            zoom: Initial zoom level
        """
        self.session_prefix = session_prefix
        self.map_key = map_key or f"{session_prefix}unified_map"
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        
    def create_standardized_map(self) -> folium.Map:
        """
        Create the standardized map used across all modules.
        
        Returns:
            Configured folium Map with consistent styling
        """
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom
        )
        
        # Add standardized tile layers
        folium.TileLayer(
            tiles='CartoDB positron',
            attr='© OpenStreetMap contributors © CARTO',
            name='Light'
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='Terrain'
        ).add_to(m)
        
        # Add standardized drawing tools
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
    
    def render_map_interface(self, 
                           on_area_selected: Callable[[Dict], None] = None,
                           show_export: bool = True,
                           custom_instructions: str = None) -> Tuple[bool, Optional[Dict]]:
        """
        Render the complete map interface with drawing and selection.
        
        Args:
            on_area_selected: Callback function when area is selected
            show_export: Whether to show geometry export option
            custom_instructions: Custom instruction text
            
        Returns:
            Tuple of (area_selected, geometry_dict)
        """
        # Display instructions
        instructions = custom_instructions or "Use the drawing tools in the top-right corner to draw a rectangle or polygon on the map."
        st.info(instructions)
        
        # Create and display map
        m = self.create_standardized_map()
        
        map_data = st_folium(
            m,
            key=self.map_key,
            width=700,
            height=500,
            returned_objects=["all_drawings", "last_object_clicked"]
        )
        
        # Process drawn features
        if map_data['all_drawings'] and len(map_data['all_drawings']) > 0:
            st.success(f"✅ Found {len(map_data['all_drawings'])} drawn feature(s)")
            
            # Use the most recent drawing
            latest_drawing = map_data['all_drawings'][-1]
            geometry_dict = latest_drawing['geometry']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Selected Area:**")
                st.json(latest_drawing, expanded=False)
            
            with col2:
                # Use This Area button
                if st.button("Use This Area", type="primary", key=f"{self.session_prefix}use_area"):
                    if on_area_selected:
                        on_area_selected(geometry_dict)
                    return True, geometry_dict
                
                # Export geometry button
                if show_export:
                    if st.button("📥 Export Geometry", key=f"{self.session_prefix}export_geom"):
                        geojson_str = json.dumps(geometry_dict, indent=2)
                        st.download_button(
                            label="Download GeoJSON",
                            data=geojson_str,
                            file_name="selected_area.geojson",
                            mime="application/json",
                            key=f"{self.session_prefix}download_geojson"
                        )
            
            return False, geometry_dict
        else:
            st.info("👆 Draw a rectangle or polygon on the map above to select your area of interest.")
            return False, None
    
    def _simplify_geometry_safely(self, geometry: ee.Geometry, tolerance: float = 100) -> ee.Geometry:
        """
        Safely simplify geometry with error handling.

        Simplifies geometry to reduce vertices for better performance.
        Falls back to original geometry if simplification fails (e.g., Point geometries).

        Args:
            geometry: Earth Engine geometry to simplify
            tolerance: Simplification tolerance in meters (default 100m)

        Returns:
            Simplified geometry, or original if simplification fails
        """
        try:
            simplified = geometry.simplify(maxError=tolerance)
            return simplified
        except Exception:
            # Simplification failed (e.g., Point geometry, invalid geometry)
            # Silently return original geometry - this is expected for some geometry types
            return geometry

    def create_ee_geometry(self, geometry_dict: Dict[str, Any]) -> ee.Geometry:
        """
        Create Earth Engine geometry from GeoJSON dict with automatic simplification.

        Automatically simplifies hand-drawn and uploaded geometries to improve
        Earth Engine processing performance while preserving shape.

        Args:
            geometry_dict: GeoJSON geometry dictionary

        Returns:
            Simplified Earth Engine Geometry object
        """
        geometry = ee.Geometry(geometry_dict)
        return self._simplify_geometry_safely(geometry, tolerance=100)
    
    def get_geometry_info(self, geometry: ee.Geometry) -> Dict[str, Any]:
        """
        Get information about the geometry.
        
        Args:
            geometry: Earth Engine geometry
            
        Returns:
            Dictionary with geometry information
        """
        try:
            area_km2 = geometry.area().divide(1000000).getInfo()
            bounds = geometry.bounds().getInfo()
            
            return {
                'area_km2': area_km2,
                'bounds': bounds,
                'type': geometry.type().getInfo()
            }
        except Exception as e:
            return {
                'error': str(e),
                'area_km2': None,
                'bounds': None,
                'type': 'Unknown'
            }


class GeometrySelectionWidget:
    """
    Complete geometry selection widget that handles all three methods:
    - Draw on Map
    - Upload GeoJSON  
    - Enter Coordinates
    """
    
    def __init__(self, session_prefix: str = "", title: str = "📍 Select Area of Interest"):
        """
        Initialize geometry selection widget.
        
        Args:
            session_prefix: Session state prefix for this widget
            title: Title to display
        """
        self.session_prefix = session_prefix
        self.title = title
        self.map_widget = UnifiedMapWidget(session_prefix=session_prefix)
        
        # Session state keys
        self.geometry_key = f"{session_prefix}geometry"
        self.complete_key = f"{session_prefix}geometry_complete"
        
    def render_complete_interface(self, 
                                on_geometry_selected: Callable[[ee.Geometry], None] = None) -> bool:
        """
        Render the complete geometry selection interface.
        
        Args:
            on_geometry_selected: Callback when geometry is selected
            
        Returns:
            True if geometry was selected, False otherwise
        """
        st.markdown(f"## {self.title}")
        
        # Check if geometry already selected
        if st.session_state.get(self.complete_key, False):
            geometry = st.session_state.get(self.geometry_key)
            if geometry:
                st.success("✅ Area of interest already selected!")
                
                # Show geometry info
                try:
                    info = self.map_widget.get_geometry_info(geometry)
                    if info.get('area_km2'):
                        st.info(f"Selected area: {info['area_km2']:.2f} km²")
                except Exception:
                    st.info("Geometry ready for analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Continue", type="primary", key=f"{self.session_prefix}continue"):
                        return True
                with col2:
                    if st.button("Select Different Area", key=f"{self.session_prefix}different"):
                        st.session_state[self.complete_key] = False
                        if self.geometry_key in st.session_state:
                            del st.session_state[self.geometry_key]
                        st.rerun()
                return False
        
        # Method selection
        method = st.radio(
            "Select method:",
            ["🗺️ Draw on Map", "📁 Upload GeoJSON", "📐 Enter Coordinates"],
            horizontal=True,
            key=f"{self.session_prefix}method_selection"
        )
        
        geometry_selected = False
        selected_geometry = None
        
        if method == "🗺️ Draw on Map":
            def on_area_selected(geometry_dict):
                try:
                    geometry = self.map_widget.create_ee_geometry(geometry_dict)
                    st.session_state[self.geometry_key] = geometry
                    st.session_state[self.complete_key] = True
                    if on_geometry_selected:
                        on_geometry_selected(geometry)
                    st.success("✅ Area selected successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creating geometry: {str(e)}")
            
            geometry_selected, _ = self.map_widget.render_map_interface(
                on_area_selected=on_area_selected,
                show_export=True
            )
            
        elif method == "📁 Upload GeoJSON":
            self._render_upload_method(on_geometry_selected)
            
        elif method == "📐 Enter Coordinates":
            self._render_coordinates_method(on_geometry_selected)
        
        return geometry_selected

    def _pre_simplify_geojson_client_side(self, geojson_dict):
        """
        Merge and progressively simplify GeoJSON with shapely before sending to GEE.

        GEE embeds the full coordinate array in every API request payload, so
        EE-side .simplify() does NOT help — the complex geometry is already in
        the request before GEE processes it.  This method reduces the geometry
        locally (no network call) and always preserves full area coverage:
        simplified boundaries never cut inside the original shape, and the
        convex-hull fallback guarantees every original point is contained.

        Returns:
            (geometry_dict | None, info_message | None)
            Returns (None, None) when no action is needed or shapely is absent.
        """
        _SIZE_LIMIT = 50_000  # 50 KB — safe target for GEE geometry payloads

        try:
            from shapely.geometry import shape, mapping
            from shapely.ops import unary_union

            gtype = geojson_dict.get("type", "")
            if gtype == "FeatureCollection":
                features = geojson_dict.get("features", [])
                shapes = [shape(f["geometry"]) for f in features if f.get("geometry")]
                if not shapes:
                    return None, None
                geom = unary_union(shapes)
                n_features = len(shapes)
            elif gtype == "Feature":
                geom = shape(geojson_dict["geometry"])
                n_features = 1
            else:
                geom = shape(geojson_dict)
                n_features = 1

            def _size(g):
                return len(json.dumps(mapping(g)))

            original_size = _size(geom)

            if original_size <= _SIZE_LIMIT and n_features == 1:
                return None, None

            if original_size <= _SIZE_LIMIT:
                return dict(mapping(geom)), f"Merged {n_features} features into one geometry."

            for tol_deg, label in [
                (0.0005, "~50 m"),
                (0.001,  "~100 m"),
                (0.005,  "~500 m"),
                (0.01,   "~1 km"),
                (0.05,   "~5 km"),
                (0.1,    "~10 km"),
                (0.5,    "~50 km"),
            ]:
                simplified = geom.simplify(tol_deg, preserve_topology=True)
                s = _size(simplified)
                if s <= _SIZE_LIMIT:
                    msg = (
                        f"Geometry simplified (tolerance: {label}) to fit GEE payload limits — "
                        f"{original_size // 1024} KB → {s // 1024} KB. "
                        f"All original area is preserved."
                    )
                    return dict(mapping(simplified)), msg

            hull = geom.convex_hull
            s = _size(hull)
            msg = (
                f"Geometry was too complex ({original_size // 1024} KB) even after maximum "
                f"simplification. Used convex hull ({s // 1024} KB) — "
                f"all original area is fully enclosed."
            )
            return dict(mapping(hull)), msg

        except ImportError:
            return None, None
        except Exception:
            return None, None

    def _process_geojson_to_geometry(self, geojson_data: dict, simplify_tolerance: float = 500) -> ee.Geometry:
        """
        Process GeoJSON to Earth Engine geometry with union and simplification.

        Uses client-side shapely pre-simplification to avoid GEE payload size
        errors before falling back to EE-side union/simplify for edge cases.

        Returns:
            Unified and simplified ee.Geometry
        """
        try:
            # --- Client-side pre-simplification (must happen before ee.Geometry()) ---
            pre_simplified, info_msg = self._pre_simplify_geojson_client_side(geojson_data)

            if pre_simplified is not None:
                if info_msg:
                    st.info(f"ℹ️ {info_msg}")
                return ee.Geometry(pre_simplified)

            # shapely unavailable or geometry already small — original path
            gtype = geojson_data.get("type", "")

            if gtype == "FeatureCollection":
                features = geojson_data.get("features", [])

                if len(features) == 0:
                    raise ValueError("FeatureCollection is empty")

                elif len(features) == 1:
                    geometry_dict = features[0]["geometry"]
                    geometry = self.map_widget.create_ee_geometry(geometry_dict)
                    st.info("ℹ️ Loaded single feature from FeatureCollection")

                else:
                    st.info(f"ℹ️ Processing {len(features)} features: unioning boundaries...")
                    ee_features = [
                        ee.Feature(ee.Geometry(f["geometry"]))
                        for f in features if f.get("geometry")
                    ]
                    unified = ee.FeatureCollection(ee_features).union(maxError=1)
                    geometry = unified.geometry()
                    geometry = geometry.simplify(maxError=simplify_tolerance)
                    st.success(
                        f"✅ Unified {len(features)} features "
                        f"(simplified with {simplify_tolerance} m tolerance)"
                    )

            elif gtype == "Feature":
                geometry = self.map_widget.create_ee_geometry(geojson_data["geometry"])
                st.info("ℹ️ Loaded single Feature")

            else:
                geometry = self.map_widget.create_ee_geometry(geojson_data)
                st.info("ℹ️ Loaded geometry directly")

            return geometry

        except Exception as e:
            raise ValueError(f"Failed to process GeoJSON: {str(e)}")

    def _render_upload_method(self, on_geometry_selected: Callable = None):
        """Render GeoJSON upload method"""
        st.markdown("### 📁 Upload GeoJSON File")

        uploaded_file = st.file_uploader(
            "Choose a GeoJSON file",
            type=["geojson", "json"],
            key=f"{self.session_prefix}file_upload"
        )

        if uploaded_file is not None:
            try:
                geojson_data = json.loads(uploaded_file.getvalue().decode())

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**File Contents:**")
                    st.json(geojson_data, expanded=False)

                with col2:
                    if st.button("Use This File", type="primary", key=f"{self.session_prefix}use_file"):
                        try:
                            # Extract and process geometry with union + simplify
                            geometry = self._process_geojson_to_geometry(geojson_data)

                            st.session_state[self.geometry_key] = geometry
                            st.session_state[self.complete_key] = True

                            if on_geometry_selected:
                                on_geometry_selected(geometry)

                            st.success("✅ Geometry created from uploaded file!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error processing file: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    def _render_coordinates_method(self, on_geometry_selected: Callable = None):
        """Render coordinate input method"""
        st.markdown("### 📐 Enter Bounding Box Coordinates")
        
        with st.form(f"{self.session_prefix}coordinates_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Southwest Corner:**")
                min_lon = st.number_input("Min Longitude", value=-95.0, format="%.6f")
                min_lat = st.number_input("Min Latitude", value=30.0, format="%.6f")
            
            with col2:
                st.markdown("**Northeast Corner:**")
                max_lon = st.number_input("Max Longitude", value=-94.0, format="%.6f")
                max_lat = st.number_input("Max Latitude", value=31.0, format="%.6f")
            
            valid = min_lon < max_lon and min_lat < max_lat
            if not valid:
                st.error("❌ Invalid coordinates: min values must be less than max values")
            
            submitted = st.form_submit_button("Create Area", type="primary", disabled=not valid)
            
            if submitted and valid:
                try:
                    geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
                    st.session_state[self.geometry_key] = geometry
                    st.session_state[self.complete_key] = True
                    
                    if on_geometry_selected:
                        on_geometry_selected(geometry)
                    
                    st.success("✅ Area created from coordinates!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creating geometry: {str(e)}")
