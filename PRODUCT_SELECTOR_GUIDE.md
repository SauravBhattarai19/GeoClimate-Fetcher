# GeoClimate-Fetcher: Optimal Product Selector Module Analysis

## Overview
The Optimal Product Selector is a sophisticated module within GeoClimate-Fetcher that enables users to compare meteorological datasets against ground station data to identify the best data sources for their specific geographic location and analysis needs.

## Directory Structure

```
GeoClimate-Fetcher/
├── geoclimate_fetcher/
│   ├── core/
│   │   ├── product_selector.py       # Core selection logic (handlers)
│   │   └── dataset_config.py         # Dataset configuration management
│   └── data/
│       ├── datasets.json             # PRIMARY DATASET DEFINITION (key file!)
│       ├── Datasets.csv              # Legacy CSV format dataset list
│       └── meteostat_stations.csv    # Station metadata
├── app_components/
│   ├── product_selector_component.py         # Main UI component (1559 lines)
│   ├── product_selector_visualizer.py        # Results visualization (503 lines)
│   ├── product_selector_data_manager.py      # Data operations (467 lines)
│   └── [data_processors.py]                  # Utility functions
└── interface/
    └── product_selector.py                   # Interface entry point
```

## Key Files and Their Roles

### 1. `/geoclimate_fetcher/data/datasets.json` - PRIMARY DATASET CONFIGURATION
**Purpose:** Central configuration file for all available climate datasets

**Structure:**
```json
{
  "datasets": {
    "ECMWF/ERA5/DAILY": {
      "name": "ERA5 Daily Aggregates",
      "provider": "ECMWF",
      "start_date": "1979-01-02",
      "end_date": "2020-07-09",
      "temporal_resolution": "daily",
      "pixel_size_m": 27830,
      "snippet_type": "ImageCollection",
      "geographic_coverage": "Global",
      "description": "ERA5 daily surface climate variables...",
      
      "bands": {
        "temperature_max": {
          "band_name": "maximum_2m_air_temperature",
          "unit": "°C",
          "original_unit": "K",
          "scaling_factor": 1.0,
          "offset": -273.15,
          "description": "Daily maximum 2m air temperature"
        },
        "precipitation": {
          "band_name": "total_precipitation",
          "unit": "mm/day",
          "original_unit": "m",
          "scaling_factor": 1000.0,
          "offset": 0.0,
          "description": "Daily total precipitation"
        }
      },
      
      "supports_analysis": ["temperature", "precipitation"],
      "recommended_indices": {
        "temperature": ["TXx", "TNn", "DTR", "TX90p", "TN10p", "FD", "SU"],
        "precipitation": ["RX1day", "CDD", "PRCPTOT", "R95p", "R10mm", "SDII"]
      }
    }
  },
  
  "climate_indices": { ... },
  "metadata": { ... }
}
```

### Dataset Entry Properties:

- **name**: Human-readable dataset name
- **provider**: Data provider organization
- **start_date** / **end_date**: Data availability (YYYY-MM-DD format)
- **temporal_resolution**: daily, monthly, weekly, etc.
- **pixel_size_m**: Spatial resolution in meters
- **snippet_type**: ImageCollection or Image
- **geographic_coverage**: Global, Regional, or specific area description
- **description**: What the dataset contains
- **bands**: Available data variables with:
  - `band_name`: Earth Engine band identifier
  - `unit`: Standard output unit (°C, mm, etc.)
  - `original_unit`: Native unit in the dataset
  - `scaling_factor`: Multiplication factor for unit conversion
  - `offset`: Additive offset for unit conversion
  - `description`: What this band represents
- **supports_analysis**: Array of ["temperature"] and/or ["precipitation"]
- **recommended_indices**: Climate indices suited for this dataset

---

## 2. Core Classes and Their Functions

### A. `/geoclimate_fetcher/core/dataset_config.py` - DatasetConfig Class

**Responsibilities:**
- Load and parse datasets.json
- Provide dataset lookup and filtering
- Handle unit conversions
- Validate date ranges
- Support climate index queries

**Key Methods:**
```python
get_datasets_for_analysis(analysis_type)     # Filter by "temperature" or "precipitation"
get_dataset_info(dataset_id)                 # Retrieve full dataset info
get_band_info(dataset_id, band_type)         # Get band configuration
apply_scaling(dataset_id, band_type, value)  # Convert units
get_date_range(dataset_id)                   # Get data availability
validate_date_range(dataset_id, start, end)  # Check date coverage
get_climate_indices(category, complexity)    # Get available indices
get_recommended_indices(dataset_id, analysis_type)  # Suggested analyses
```

### B. `/geoclimate_fetcher/core/product_selector.py` - Core Handlers

#### GriddedDataHandler
**Responsibilities:**
- Fetch data from Google Earth Engine
- Apply unit conversions
- Handle data extraction from datasets

**Key Features:**
- Automatic unit conversion using datasets.json metadata
- Supports JSON-based conversion (primary) and hardcoded fallback
- Large collection chunking (90-day chunks for >1000 images)
- Band mapping and extraction

**Unit Conversion Process:**
1. Try JSON-based conversion first (from datasets.json)
2. Fall back to hardcoded conversions if not found
3. Infer units based on provider patterns

#### MeteostatHandler
**Responsibilities:**
- Load and query meteostat station data
- Extract station data for specific variables
- Support custom station data validation

**Key Features:**
- Loads from CSV (default: geoclimate_fetcher/data/meteostat_stations.csv)
- Finds stations within geographic bounds
- Retrieves historical station data

#### StatisticalAnalyzer
**Responsibilities:**
- Compare station vs. gridded data
- Calculate performance metrics
- Seasonal and temporal analysis

**Key Metrics Calculated:**
- `n_observations`: Count of overlapping data points
- `rmse`: Root Mean Square Error (lower is better)
- `mae`: Mean Absolute Error (lower is better)
- `r2`: Coefficient of determination (higher is better, primary metric)
- `correlation`: Pearson correlation (higher is better)
- `bias`: Mean difference (closer to 0 is better)

---

## 3. UI Components

### `/app_components/product_selector_component.py` - Main Component

**Workflow (6 Steps):**

1. **Geometry Selection** - Select area of interest
   - Draw on map / Upload GeoJSON / Enter coordinates

2. **Station Selection** - Choose measurement points
   - Auto-discover meteostat stations or upload custom data

3. **Variable Selection** - Choose what to analyze
   - Temperature (tmax, tmin) or Precipitation (prcp)

4. **Dataset Selection** - Select gridded datasets to compare
   - Filtered by variable type, from datasets.json

5. **Time Range Selection** - Choose analysis period
   - Auto-suggest optimal overlapping period

6. **Analysis & Results** - View comparisons and recommendations
   - Statistical metrics and visualizations

---

## 4. How "Optimal" Datasets are Determined

### Primary Selection Metric: R² (Coefficient of Determination)

The system uses **R²** as the main metric:

```python
best_dataset = max(station_results.keys(), 
                   key=lambda x: station_results[x]['stats'].get('r2', -1))
```

**R² Scale:**
- 1.0 = perfect fit
- 0.5 = explains 50% of variance
- 0.0 = as good as just using the mean
- <0 = worse than the mean

### Secondary Metrics (for detailed analysis):
1. **RMSE** - Average prediction error (lower is better)
2. **MAE** - Mean absolute difference (lower is better)
3. **Correlation** - Linear relationship strength (higher is better)
4. **Bias** - Systematic over/underestimation (0 is best)

### Display:
- Per-station best dataset (highest R²)
- Multi-metric bar charts comparing all datasets
- Station map colored by best R² values

---

## 5. How to Add a New Dataset

### Step 1: Add Entry to datasets.json

```json
{
  "PROVIDER/DATASET_ID": {
    "name": "Human-Readable Name",
    "provider": "Organization",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "temporal_resolution": "daily",
    "pixel_size_m": 1000,
    "snippet_type": "ImageCollection",
    "geographic_coverage": "Global",
    "description": "Dataset description",
    
    "bands": {
      "temperature_max": {
        "band_name": "ee_band_name",
        "unit": "°C",
        "original_unit": "K",
        "scaling_factor": 1.0,
        "offset": -273.15,
        "description": "Daily maximum temperature"
      },
      "precipitation": {
        "band_name": "precip_band",
        "unit": "mm/day",
        "original_unit": "m",
        "scaling_factor": 1000.0,
        "offset": 0.0,
        "description": "Daily precipitation"
      }
    },
    
    "supports_analysis": ["temperature", "precipitation"],
    "recommended_indices": {
      "temperature": ["TXx", "TNn", "DTR"],
      "precipitation": ["RX1day", "CDD", "PRCPTOT"]
    }
  }
}
```

### Step 2: Configure Unit Conversions

Ensure band entries have proper:
- `original_unit`: What the data actually is
- `unit`: What it should be standardized to
- `scaling_factor`: Multiply value by this
- `offset`: Then add this

Common conversions:
- Temperature: K → °C (scaling=1.0, offset=-273.15)
- Precipitation: m → mm (scaling=1000.0, offset=0.0)
- Precipitation: kg/m²/s → mm/day (scaling=86400, offset=0.0)

### Step 3: Specify Variable Support

Add to `supports_analysis`:
- "temperature" if has temperature_max or temperature_min
- "precipitation" if has precipitation

### Step 4: Add Recommended Indices

List indices from climate_indices section that dataset supports:
- Temperature: TXx, TNn, DTR, TX90p, TN10p, FD, SU
- Precipitation: RX1day, CDD, PRCPTOT, R95p, R10mm, SDII

### Step 5: Test

1. Verify data loads via Google Earth Engine
2. Check UI displays dataset in selection screen
3. Run test analysis with sample station data
4. Verify unit conversions are correct
5. Check statistical comparison works

---

## 6. Dataset Addition Checklist

- [ ] Get Earth Engine Dataset ID (PROVIDER/DATASET/VERSION)
- [ ] Document availability dates (start, end)
- [ ] Identify spatial resolution (meters)
- [ ] Identify temporal resolution (daily, monthly, etc.)
- [ ] List all available bands/variables
- [ ] Identify native units for each band
- [ ] Calculate conversion factors:
  - Temperature to °C
  - Precipitation to mm (or mm/day)
- [ ] Test conversions with sample values
- [ ] Add entry to datasets.json
- [ ] Set supports_analysis
- [ ] Add recommended climate indices
- [ ] Test data retrieval
- [ ] Verify UI display
- [ ] Validate analysis results

---

## 7. Code Examples

### Load Datasets Configuration

```python
from geoclimate_fetcher.core.dataset_config import DatasetConfig

config = DatasetConfig()
temp_datasets = config.get_datasets_for_analysis("temperature")
precip_datasets = config.get_datasets_for_analysis("precipitation")
```

### Get Band Information

```python
band_info = config.get_band_info("NASA/ORNL/DAYMET_V4", "temperature_max")
print(band_info['band_name'])      # 'tmax'
print(band_info['unit'])           # '°C'
print(band_info['original_unit'])  # '°C'
print(band_info['scaling_factor']) # 1.0
```

### Validate Date Range

```python
from datetime import date
is_valid, msg = config.validate_date_range(
    "NASA/ORNL/DAYMET_V4",
    date(2010, 1, 1),
    date(2020, 12, 31)
)
```

### Access Recommended Indices

```python
indices = config.get_recommended_indices("NASA/ORNL/DAYMET_V4", "temperature")
# Returns: ['TXx', 'TNn', 'DTR', 'TX90p', 'TN10p', 'FD', 'SU']
```

---

## 8. Extension Points

### Add New Analysis Type
1. Add to dataset's `supports_analysis` list
2. Add new indices to `climate_indices` in datasets.json
3. Implement in climate_analysis.py

### Add New Ranking Metric
1. Calculate in StatisticalAnalyzer.calculate_statistics()
2. Update ProductSelectorVisualizer.create_comparison_overview()
3. Modify _display_station_results() logic

### Support Custom Gridded Data
1. Upload metadata CSV: [id, latitude, longitude, daily_start, daily_end]
2. Upload data CSV: [date, station_id, value]
3. System validates via MeteostatHandler.validate_custom_data()

---

## 9. Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| datasets.json | ~416 | Central dataset configuration |
| product_selector.py | 1404 | Core data handlers (GriddedDataHandler, MeteostatHandler, StatisticalAnalyzer) |
| dataset_config.py | 302 | Dataset loading and utilities |
| product_selector_component.py | 1559 | Main UI wizard |
| product_selector_visualizer.py | 503 | Result visualizations |
| product_selector_data_manager.py | 467 | Data export/download |

---

## 10. Data Flow

```
User Input (Geography, Station, Variable, Datasets, Time)
        ↓
ProductSelectorComponent wizard steps
        ↓
datasets.json filtered by supports_analysis
        ↓
MeteostatHandler.get_station_data()
GriddedDataHandler.get_gridded_data()
        ↓
GriddedDataHandler.apply_unit_conversion()
  ├→ JSON conversion (datasets.json)
  ├→ Hardcoded fallback
  └→ Provider pattern inference
        ↓
StatisticalAnalyzer:
  - merge_datasets()
  - calculate_statistics() → R², RMSE, MAE, Correlation, Bias
  - calculate_seasonal_statistics()
        ↓
Find best dataset (max R²)
        ↓
ProductSelectorVisualizer:
  - create_comparison_overview()
  - create_station_map()
  - create_scatter_plot()
  - create_time_series_plot()
  - create_seasonal_boxplot()
        ↓
User sees results with recommendations
```

---

## Summary

The Optimal Product Selector is built on:

1. **Centralized Configuration** - datasets.json defines all datasets, bands, and conversions
2. **Modular Architecture** - Separate handlers for data, analysis, UI, visualization
3. **Automatic Unit Conversion** - JSON-first with intelligent fallback
4. **Statistical Comparison** - R² is the primary optimality metric
5. **Guided UI Workflow** - 6-step wizard for analysis
6. **Rich Visualization** - Multi-metric comparison charts and maps

**To add a dataset:** Edit datasets.json with full metadata, band definitions, unit conversions, and supported analyses. Test via the UI.

For questions or issues, review the source files listed above or refer to the actual datasets.json file in `/geoclimate_fetcher/data/`.
