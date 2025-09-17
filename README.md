# 🌍 GeoClimate Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-4285F4.svg)](https://earthengine.google.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive web-based platform for climate data analysis, visualization, and intelligence built on Google Earth Engine. The platform provides advanced tools for downloading, analyzing, and visualizing climate datasets with a focus on user-friendly interfaces and scientific accuracy.

🚀 **[Try the Live Platform](https://geeclimate.streamlit.app)** - No installation required!

![Platform Screenshot](pictures/platform.png)

## 🚀 Key Features

### 🌡️ **Climate Intelligence Hub**
- **20 Climate Indices**: ETCCDI-compliant indices including percentile-based (TX90p, TN10p, R95p) and threshold-based (SU, FD, SDII)
- **Advanced Analytics**: Server-side Earth Engine processing with trend analysis and statistical modeling
- **Smart Dataset Selection**: Automatic recommendations based on analysis type and geographic coverage
- **Compact UI**: Organized tabs and progressive disclosure for efficient index selection
- **Export Flexibility**: Multiple formats (GeoTIFF, CSV) with optimized compression

### 🗺️ **GEE Data Explorer**
- **Interactive Map Interface**: Browse and visualize 40+ Earth Engine datasets
- **Intelligent Band Discovery**: Automatic band detection and metadata extraction
- **Geometry Tools**: Draw, upload, or select from predefined regions
- **Real-time Preview**: Live visualization of selected data with customizable styling
- **Smart Download**: Chunked processing for large datasets with progress tracking

### 🎯 **Optimal Product Selector**
- **Station Discovery**: Find and select meteostat weather stations in your area
- **Dataset Comparison**: Compare multiple gridded climate datasets against station data
- **Statistical Analysis**: Comprehensive correlation and bias analysis
- **Optimal Recommendations**: Identify the best data source for your specific needs
- **Performance Metrics**: RMSE, correlation, bias analysis for data quality assessment

### 📊 **Data Visualization Suite**
- **Interactive Charts**: Time series, correlation, and distribution analysis with Plotly
- **Spatial Mapping**: Advanced cartographic visualization with color legends and data ranges
- **Statistical Analysis**: Automated trend detection and pattern recognition
- **Export Options**: High-quality charts and downloadable data formats
- **Climate-Specific Visualizations**: Tailored charts for climate index analysis

### 💧 **Hydrology Analyzer**
- **Precipitation Analysis**: Comprehensive precipitation data analysis and modeling
- **Return Period Analysis**: Statistical analysis of extreme precipitation events
- **IDF Curves**: Intensity-Duration-Frequency curve generation
- **Drought Indices**: Multiple drought indicator calculations
- **Climate-Hydrology Integration**: Links climate data with hydrological processes

## 📋 **Supported Datasets**

### Climate Datasets
- **ERA5 Daily Aggregates** (1979-2020): Global reanalysis data at 27km resolution
- **Daymet V4** (1980-2023): High-resolution North American meteorology at 1km

### Earth Engine Collections (40+)
- MODIS Land Products (MOD13Q1, MOD11A1, MOD16A2)
- Landsat Collection 2 (4, 5, 7, 8, 9)
- Sentinel-1/2 Collections
- CHIRPS Precipitation Data
- SRTM Elevation Data
- And many more...

### Climate Indices (20 Total)
#### Standard Indices (6)
- **TXx**: Maximum Temperature
- **TNn**: Minimum Temperature
- **DTR**: Diurnal Temperature Range
- **RX1day**: Maximum 1-day Precipitation
- **CDD**: Consecutive Dry Days
- **PRCPTOT**: Annual Total Precipitation

#### Percentile-Based Indices (7)
- **TX90p/TX10p**: Warm/Cool Days (temperature percentiles)
- **TN90p/TN10p**: Warm/Cool Nights (temperature percentiles)
- **R95p/R99p/R75p**: Wet Day Precipitation (precipitation percentiles)

#### Threshold-Based Indices (7)
- **TXn/TNx**: Coldest Day/Warmest Night
- **FD**: Frost Days (< 0°C)
- **SU**: Summer Days (> 25°C)
- **R10mm/R20mm**: Heavy/Very Heavy Rain Days
- **SDII**: Simple Daily Intensity Index

## 🛠️ **Installation**

### Prerequisites
- Python 3.8 or higher
- Google Earth Engine account
- 4GB+ RAM recommended for large datasets

### Quick Start
```bash
# Clone the repository
git clone https://github.com/sauravbhattarai19/geoclimate-platform.git
cd geoclimate-platform

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine
earthengine authenticate

# Run the application
streamlit run app.py
```

### Development Installation
```bash
# Clone and install in development mode
git clone https://github.com/sauravbhattarai19/geoclimate-platform.git
cd geoclimate-platform
pip install -e .
```

## 🎯 **Quick Usage**

### 🌐 Online Platform (Recommended)
1. **Visit**: [geeclimate.streamlit.app](https://geeclimate.streamlit.app)
2. **Authenticate**: One-time Google Earth Engine authentication
3. **Select Module**:
   - 🌡️ **Climate Intelligence Hub**: For climate analysis and index calculations
   - 🗺️ **GEE Data Explorer**: For browsing and downloading Earth Engine datasets
   - 🎯 **Optimal Product Selector**: For comparing and selecting optimal data sources
   - 💧 **Hydrology Analyzer**: For precipitation and hydrological analysis
   - 📊 **Data Visualizer**: For advanced data visualization and analysis

### 💻 Local Development
1. **Clone and Install**:
   ```bash
   git clone https://github.com/sauravbhattarai19/geoclimate-platform.git
   cd geoclimate-platform
   pip install -r requirements.txt
   ```

2. **Launch Locally**:
   ```bash
   streamlit run app.py
   ```

3. **Navigate to**: `http://localhost:8501`

## 📁 **Project Structure**

```
geoclimate-platform/
├── app.py                      # Main Streamlit application
├── app_components/             # Reusable UI components
│   ├── auth_component.py       # Google Earth Engine authentication
│   ├── download_component.py   # Data download interface
│   ├── geometry_component.py   # Geometry selection tools
│   └── visualization_utils.py  # Shared visualization functions
├── interface/                  # Specialized interface modules
│   ├── climate_analytics.py    # Climate Intelligence Hub
│   ├── geodata_explorer.py     # GEE Data Explorer
│   ├── product_selector.py     # Optimal Product Selector
│   ├── hydrology_analyzer.py   # Hydrology analysis tools
│   ├── data_visualizer.py      # Data visualization suite
│   └── router.py               # Interface routing logic
├── geoclimate_fetcher/         # Core package
│   ├── climate_indices.py      # Climate index calculations
│   ├── core/                   # Core functionality
│   │   ├── dataset_config.py   # Dataset configuration management
│   │   ├── download_utils.py   # Download and export utilities
│   │   ├── climate_analysis_runner.py  # Climate analysis engine
│   │   └── gee_auth.py         # Earth Engine authentication
│   └── data/                   # Configuration and reference data
│       ├── datasets.json       # Dataset metadata and configuration
│       └── meteostat_stations.csv  # Weather station metadata
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Package configuration
└── README.md                  # This file
```

## 🔧 **Configuration**

### Earth Engine Authentication
The platform uses Google Earth Engine for data processing. Authentication options:

1. **Interactive Authentication** (Recommended for first-time users)
   ```bash
   earthengine authenticate
   ```

2. **Service Account** (For deployment)
   ```python
   import ee
   ee.Initialize(ee.ServiceAccountCredentials(
       email='your-service-account@project.iam.gserviceaccount.com',
       key_file='/path/to/service-account-key.json'
   ))
   ```

### Dataset Configuration
Customize available datasets by editing `geoclimate_fetcher/data/datasets.json`:

```json
{
  "datasets": {
    "YOUR_DATASET_ID": {
      "name": "Your Dataset Name",
      "provider": "Data Provider",
      "start_date": "1980-01-01",
      "end_date": "2023-12-31",
      "bands": {
        "temperature_max": {
          "band_name": "tmax",
          "unit": "°C",
          "scaling_factor": 1.0,
          "offset": 0.0
        }
      }
    }
  }
}
```

## 📊 **Analysis Capabilities**

### Climate Analysis Workflow
1. **Dataset Selection**: Choose from ERA5, Daymet, or custom datasets
2. **Temporal Configuration**: Set analysis period and temporal resolution
3. **Spatial Configuration**: Define study area using interactive map
4. **Index Selection**: Choose from 20 ETCCDI-compliant climate indices
5. **Export Configuration**: Select output format and resolution
6. **Analysis Execution**: Server-side processing with progress tracking
7. **Results Visualization**: Interactive charts and downloadable outputs

### Advanced Features
- **Percentile Calculations**: 1980-2000 base period for climate normals
- **Trend Analysis**: Mann-Kendall trend tests and Sen's slope estimation
- **Multi-index Analysis**: Process multiple indices simultaneously
- **Smart Chunking**: Automatic data chunking for large spatial extents
- **Export Optimization**: Compressed outputs with metadata preservation

## 🌐 **Deployment**

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
The platform is currently deployed on:
- **Streamlit Cloud**: [geeclimate.streamlit.app](https://geeclimate.streamlit.app) (Live Production)

Additional deployment options:
- **Google Cloud Run**: Containerized deployment
- **AWS EC2/ECS**: Full control deployment
- **Azure Container Instances**: Quick cloud deployment

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

### Development Setup
```bash
# Fork the repository
git clone https://github.com/your-username/geoclimate-platform.git
cd geoclimate-platform

# Create development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

### Contribution Areas
- 🐛 **Bug Fixes**: Report and fix issues
- 🚀 **New Features**: Add climate indices or datasets
- 📚 **Documentation**: Improve guides and examples
- 🎨 **UI/UX**: Enhance user interface and experience
- 🔬 **Scientific Validation**: Verify calculations and methods

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Google Earth Engine**: For providing the computational platform
- **ETCCDI**: For climate index standards and definitions
- **Streamlit**: For the excellent web framework
- **ECMWF**: For ERA5 reanalysis data
- **NASA**: For Earth observation datasets
- **ORNL DAAC**: For Daymet meteorological data

## 📞 **Support & Contact**

- **Documentation**: [Wiki](https://github.com/sauravbhattarai19/geoclimate-platform/wiki)
- **Issues**: [GitHub Issues](https://github.com/sauravbhattarai19/geoclimate-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sauravbhattarai19/geoclimate-platform/discussions)
- **Email**: saurav.bhattarai.1999@gmail.com

## 📈 **Citation**

If you use this platform in your research, please cite:

```bibtex
@software{geoclimate_platform,
  title={GeoClimate Intelligence Platform: A Web-Based Tool for Climate Data Analysis},
  author={Bhattarai, Saurav},
  year={2025},
  url={https://github.com/sauravbhattarai19/geoclimate-platform},
  version={0.1.0}
}
```

## 🔗 **Related Projects**

- [Google Earth Engine](https://earthengine.google.com/) - Cloud-based planetary analysis
- [Climate Data Store](https://cds.climate.copernicus.eu/) - EU climate data portal
- [ETCCDI](http://etccdi.pacificclimate.org/) - Climate change indices
- [Streamlit](https://streamlit.io/) - Python web app framework

---

<div align="center">

**Built with ❤️ for the climate science community**

[📚 Documentation](https://github.com/sauravbhattarai19/geoclimate-platform/wiki) • [🐛 Issues](https://github.com/sauravbhattarai19/geoclimate-platform/issues) • [💬 Discussions](https://github.com/sauravbhattarai19/geoclimate-platform/discussions)

</div>