### GeoClimate Fetcher

Interactive Streamlit application and Python package to explore, analyze, and download climate datasets from Google Earth Engine (GEE). It provides a guided workflow: authenticate with GEE, define an Area of Interest (AOI), select datasets, preview on an interactive map, and export results in multiple formats.

Built with Streamlit, Google Earth Engine, and geemap.

---

### Features

- **One-time GEE authentication**: Sign in with a Google Cloud Project ID; optional Service Account support.
- **AOI selection**: Draw polygons/rectangles on a folium map, upload geometry, or use predefined boundaries.
- **Dataset catalog**: Browse and filter Earth Engine climate datasets via an integrated metadata catalog.
- **Visualization**: Preview layers on the map; inspect time series.
- **Exports**:
  - GeoTIFF gridded rasters
  - NetCDF gridded datasets
  - CSV time series
  - Chunked Google Drive exports for large tables/time series
- **Session persistence**: Authentication remembered via secure cookies for a smooth return experience.

---

### Requirements

- Python 3.8+
- A Google Earth Engine account (free) — see the [Earth Engine signup page](https://earthengine.google.com/signup/)
- A Google Cloud Project ID with Earth Engine enabled

On Linux, make sure you have system packages commonly required by geospatial stacks (e.g., GDAL dependencies). If you hit installation issues with `rasterio`/`geopandas`, consult their docs for system prerequisites.

---

### Installation

```bash
# Clone the repository
git clone <this-repo-url>
cd <repo-directory>

# (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Optional: install the package in editable mode to use the library APIs
pip install -e .
```

---

### Run the App

Choose one of the following methods:

- Run directly with Streamlit (recommended):

```bash
streamlit run app.py
```

- Or via the bundled launcher module:

```bash
python -m geoclimate_fetcher.run_webapp
```

The app will open in your browser (by default at `http://localhost:8501`).

---

### First-time Authentication

When you open the app, you’ll see a one-time authentication step:

1. Enter your Google Cloud Project ID.
2. Optionally provide a Service Account email and upload a JSON key if you prefer SA-based auth.
3. Submit. If successful, the app stores a secure token so you won’t need to re-authenticate frequently.

Notes:
- You must have a valid Earth Engine account and the project must have Earth Engine enabled.
- If authentication fails, the UI will explain the error so you can adjust credentials.

---

### Typical Workflow

1. **Authenticate** with Google Earth Engine.
2. **Define AOI**:
   - Draw on the interactive map (rectangle or polygon).
   - Upload/import geometry.
3. **Select Dataset(s)** from the catalog and adjust parameters (date range, bands, indices, etc.).
4. **Preview** data on the map and optionally inspect time series.
5. **Export** results:
   - GeoTIFF (imagery)
   - NetCDF (gridded data)
   - CSV (time series)
   - Google Drive (chunked export for large tables)

Exports are handled by the internal `GEEExporter` and may perform chunking for long time series to improve reliability.

---

### Using the Library APIs (optional)

After `pip install -e .`, you can use selected components programmatically:

```python
from geoclimate_fetcher.core import authenticate, MetadataCatalog, GeometryHandler, GEEExporter
import ee

# Authenticate with your GCP project (user auth flow previously done via earthengine)
auth = authenticate(project_id="your-gcp-project-id")

# Build or load an AOI
geometry_handler = GeometryHandler()
ee_geometry = geometry_handler.create_polygon([[lon, lat], [lon, lat], [lon, lat], [lon, lat]])

# Inspect available datasets
catalog = MetadataCatalog()
datasets = catalog.search(text="temperature")

# Export helper
exporter = GEEExporter()
# ... use catalog and exporter to fetch/export as needed ...
```

API surfaces include: `authenticate`, `MetadataCatalog`, `GeometryHandler`, `GEEExporter`, `ImageCollectionFetcher`, `StaticRasterFetcher`, and `DataVisualizer`.

---

### Project Structure (high level)

- `app.py` — main Streamlit UI (GeoClimate Intelligence Platform)
- `main.py` — alternate minimal Streamlit entry using modular components
- `app_components/` — Streamlit UI components (auth, geometry, dataset, download)
- `geoclimate_fetcher/core/` — core logic (auth, metadata, exporter, fetchers)
- `geoclimate_fetcher/run_webapp.py` — simple launcher for `streamlit run app.py`
- `requirements.txt` — runtime dependencies
- `pyproject.toml` — package metadata

---

### Troubleshooting

- **Earth Engine authentication errors**: Ensure your EE account is active, the GCP project is valid, and that you’ve enabled Earth Engine. If using a Service Account, confirm the key JSON and IAM roles.
- **Map not rendering**: Verify `geemap`, `folium`, and `streamlit-folium` installed (they’re in `requirements.txt`).
- **Exports taking too long**: Large AOIs or long time ranges can be slow. Prefer chunked exports or reduce date spans.
- **Port conflicts**: Run with a custom port: `streamlit run app.py --server.port 8502`.

---

### License

MIT License. See the repository’s `LICENSE` if provided.

---

### Links

- Homepage: [GeoClimate Fetcher Repository](https://github.com/Saurav-JSU/GeoClimate-Fetcher)
- Bug Tracker: [Issues](https://github.com/Saurav-JSU/GeoClimate-Fetcher/issues)

Acknowledgements: Google Earth Engine, Streamlit, geemap, folium, pandas, xarray, and the broader open-source geospatial community.

