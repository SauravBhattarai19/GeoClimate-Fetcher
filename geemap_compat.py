"""
Geemap Compatibility Module

This module ensures geemap.foliumap is loaded (not the ipyleaflet geemap.Map).

Root cause: geemap/__init__.py does `from .geemap import *` by default, which
exports `basemaps = box.Box(xyz_to_leaflet(), ...)` into the geemap package
namespace, overwriting the `geemap.basemaps` module attribute with a Box object.
When foliumap.py then does `from . import basemaps`, it receives the Box (not the
module) and `basemaps.xyz_to_folium()` fails with BoxKeyError.

Fix: set USE_FOLIUM=1 before any geemap import. This makes geemap/__init__.py
load `from .foliumap import *` instead, so geemap.py never runs and basemaps
stays as the module.

Usage:
    Instead of: import geemap.foliumap as geemap
    Use: from geemap_compat import geemap
"""

import os
import sys

# Must be set before any geemap import to force foliumap backend
os.environ.setdefault("USE_FOLIUM", "1")

# Pre-load geemap.basemaps as a real module into sys.modules so that even if
# something later overwrites the package attribute, the module is already cached.
try:
    import importlib
    _bm_spec = importlib.util.find_spec("geemap.basemaps")
    if _bm_spec is not None and "geemap.basemaps" not in sys.modules:
        _bm = importlib.util.module_from_spec(_bm_spec)
        sys.modules["geemap.basemaps"] = _bm
        _bm_spec.loader.exec_module(_bm)
except Exception:
    pass

# Now import — with USE_FOLIUM=1, __init__.py loads foliumap instead of geemap.py
try:
    import geemap.foliumap as geemap
except Exception as e:
    # Fallback: build a minimal folium-based Map that works in Streamlit
    try:
        import folium
        import types

        geemap = types.ModuleType("geemap_fallback")

        class Map(folium.Map):
            """Minimal folium Map used when geemap.foliumap cannot be imported."""

            def __init__(self, *args, **kwargs):
                kwargs.setdefault("location", [0, 0])
                kwargs.setdefault("zoom_start", 2)
                super().__init__(*args, **kwargs)

            def addLayer(self, ee_object, vis_params=None, name="Layer",
                         shown=True, opacity=1.0):
                pass  # requires full geemap

            add_layer = addLayer

            def setCenter(self, lon, lat, zoom=None):
                self.location = [lat, lon]
                if zoom:
                    self.zoom_start = zoom

            def add_colorbar(self, *args, **kwargs):
                pass

            def add_layer_control(self):
                folium.LayerControl().add_to(self)

            def to_streamlit(self, width=None, height=600, **kwargs):
                import streamlit.components.v1 as components
                self.add_layer_control()
                return components.html(
                    self.get_root().render(), width=width, height=height
                )

        geemap.Map = Map

    except Exception:
        raise ImportError(
            f"Failed to import geemap.foliumap: {e}\n"
            "Please ensure geemap is installed: pip install geemap"
        )

__all__ = ["geemap"]
