"""
Geemap Compatibility Module

This module patches xyzservices before importing geemap to prevent 
BoxKeyError issues that occur with certain versions of xyzservices, 
python-box, and geemap.

Usage:
    Instead of: import geemap.foliumap as geemap
    Use: from geemap_compat import geemap
"""

def _patch_xyzservices_for_geemap():
    """Patch xyzservices to work with geemap's Box conversion."""
    try:
        import xyzservices
        from xyzservices import TileProvider
        
        # Check if xyz_to_folium exists and if we need to patch it
        if not hasattr(xyzservices.providers, 'xyz_to_folium'):
            def safe_xyz_to_folium():
                """Convert xyzservices providers to folium-compatible dict with safe keys."""
                result = {}
                
                def process_provider(provider, prefix=""):
                    if isinstance(provider, TileProvider):
                        # Sanitize key: replace dots and hyphens with underscores
                        safe_key = prefix.replace(".", "_").replace("-", "_")
                        if safe_key:
                            result[safe_key] = provider
                    elif hasattr(provider, '__iter__'):
                        for key in provider:
                            try:
                                child = provider[key]
                                new_prefix = f"{prefix}_{key}" if prefix else key
                                process_provider(child, new_prefix)
                            except Exception:
                                pass
                
                for name in dir(xyzservices.providers):
                    if not name.startswith('_'):
                        try:
                            provider = getattr(xyzservices.providers, name)
                            process_provider(provider, name)
                        except Exception:
                            pass
                
                return result
            
            xyzservices.providers.xyz_to_folium = safe_xyz_to_folium
    except ImportError:
        pass  # xyzservices not installed
    except Exception:
        pass  # Silently ignore patching errors


def _patch_geemap_basemaps():
    """Alternative patch: directly modify geemap's basemaps module."""
    try:
        # Import basemaps module before foliumap tries to use it
        from geemap import basemaps as geemap_basemaps
        
        # Patch xyz_to_folium to return a safe dictionary
        original_func = getattr(geemap_basemaps, 'xyz_to_folium', None)
        
        if original_func is not None:
            def safe_xyz_to_folium():
                try:
                    result = original_func()
                    # Sanitize all keys
                    safe_result = {}
                    for key, value in result.items():
                        safe_key = str(key).replace(".", "_").replace("-", "_").replace(" ", "_")
                        safe_result[safe_key] = value
                    return safe_result
                except Exception:
                    return {}
            
            geemap_basemaps.xyz_to_folium = safe_xyz_to_folium
    except ImportError:
        pass
    except Exception:
        pass


# Apply patches before importing geemap
_patch_xyzservices_for_geemap()
_patch_geemap_basemaps()

# Now safely import geemap.foliumap
try:
    import geemap.foliumap as geemap
except Exception as e:
    # If import still fails, try importing base geemap
    try:
        import geemap
        # Create a minimal foliumap-compatible interface if needed
        if not hasattr(geemap, 'Map'):
            import folium
            class Map(folium.Map):
                """Fallback Map class when geemap.foliumap fails to import."""
                def __init__(self, *args, **kwargs):
                    # Set defaults similar to geemap
                    kwargs.setdefault('location', [0, 0])
                    kwargs.setdefault('zoom_start', 2)
                    super().__init__(*args, **kwargs)
                
                def add_ee_layer(self, ee_object, vis_params=None, name="Layer"):
                    """Placeholder for EE layer - requires proper geemap."""
                    pass
            
            geemap.Map = Map
    except Exception:
        raise ImportError(
            f"Failed to import geemap: {e}\n"
            "Please ensure geemap is installed: pip install geemap>=0.35.0"
        )

# Export geemap for use by other modules
__all__ = ['geemap']
