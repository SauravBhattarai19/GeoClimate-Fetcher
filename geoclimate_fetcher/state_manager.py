"""
State manager to ensure geometry is shared between widgets
"""
import ee
from geoclimate_fetcher.core.geometry import GeometryHandler

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