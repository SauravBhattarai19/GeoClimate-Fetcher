# 🌍 GeoClimate Fetcher

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-4285F4.svg)](https://earthengine.google.com)
[![License](https://img.shields.io/badge/License-Non--Commercial-orange.svg)](LICENSE)

A Streamlit web application for downloading and analyzing climate data from Google Earth Engine. Developed as a research tool for climate data analysis and educational purposes.

**Status**: Published in Environmental Modelling & Software. See [Publication](#-publication) section below.

🚀 **[Try the Live Platform](https://geeclimate.streamlit.app/)** - Ready to use online!

![Platform Screenshot](pictures/platform.png)

## 🎯 What This Platform Does

This is a web-based interface that provides five integrated tools for climate data analysis:

### 🔍 **GeoData Explorer**
- Browse and download from [33 Earth Engine datasets](geoclimate_fetcher/data/Datasets.csv)
- **🆕 Interactive geemap preview** before downloading - visualize data instantly!
- **🆕 Adaptive temporal aggregation** - handles hourly, daily, monthly, yearly data intelligently
- **🆕 Smart layer management** - automatic aggregation/sampling for 100+ images
- Interactive map for area selection with geometry clipping
- Automatic band detection with smart color palette selection
- Export to GeoTIFF, NetCDF, and CSV formats
- Percentile-based visualization (5th-95th) to handle outliers

### 🧠 **Climate Analytics**
- Calculate ETCCDI-compliant climate indices
- **Temperature indices**: TXx, TNn, TXn, TNx, TX90p, TX10p, TN90p, TN10p, SU, FD, DTR, GSL, WSDI, CSDI
- **Precipitation indices**: RX1day, RX5day, CDD, R10mm, R20mm, SDII, R95p, R99p, R75p, PRCPTOT
- **🆕 Interactive geemap visualization** with layer toggles for multi-year/multi-month analysis
- **🆕 Integrated colorbar with units** displayed directly on the map
- Trend analysis using Mann-Kendall test and Sen's slope
- Server-side processing with Google Earth Engine
- Time series plots and statistical summaries

### 💧 **Hydrology Analyzer**
- Precipitation data analysis and visualization
- Statistical analysis for hydrological research
- Educational tool for understanding precipitation patterns

### 🎯 **Product Selector**
- Compare Meteostat weather station data with gridded datasets
- Statistical analysis (correlation, bias, RMSE)
- Identify optimal data sources for specific locations

### 📊 **Data Visualizer**
- Visualize downloaded data from any of the above tools
- Interactive charts with Plotly
- Statistical summaries and pattern detection
- Data export capabilities

## 🚀 Quick Start

### Option 1: Use the Live Platform (Recommended)

1. **Visit the hosted application**: [https://geeclimate.streamlit.app/](https://geeclimate.streamlit.app/)

2. **One-time setup** (required for Google Earth Engine access):

   **Prerequisites:**
   - Google Earth Engine account → [Sign up FREE](https://earthengine.google.com/signup/) *(for study & research)*
   - Python installed → [python.org](https://python.org) *(if not already installed)*

   **Setup steps (do this ONCE):**
   - **Install Python** → [python.org](https://python.org) *(if not already installed)*
   - **Open Terminal/Command Prompt**:
     - **Windows**: Search "cmd" or "Command Prompt"
     - **Mac**: Search "Terminal"
     - **Linux**: Ctrl+Alt+T
   - **Install Earth Engine API** *(in terminal)*:
     ```bash
     pip install earthengine-api
     ```
   - **Authenticate with Google** *(in terminal)*:
     ```bash
     earthengine authenticate
     ```
     - A link will appear in terminal → Click it OR it auto-opens browser
     - **Select the Google account** you used to register for Google Earth Engine
     - After successful login, credentials are saved automatically

3. **Upload credentials**: Upload the generated credentials file to the web platform
   - **Find your credentials file**:
     - **Windows**: `C:\Users\[USERNAME]\.config\earthengine\credentials`
     - **Mac/Linux**: `~/.config/earthengine/credentials`
   - File is named exactly `credentials` (no .json/.txt extension)

4. **Start analyzing**: Access all five analysis tools instantly!

### Option 2: Run Locally

**Prerequisites:**
- Python 3.8 or higher
- Google Earth Engine account ([sign up free](https://earthengine.google.com/signup/))

**Installation:**

1. **Clone the repository**
   ```bash
   git clone https://github.com/SauravBhattarai19/GeoClimate-Fetcher.git
   cd GeoClimate-Fetcher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Earth Engine authentication**
   ```bash
   earthengine authenticate
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your web browser at `http://localhost:8501`.

## 📊 Available Datasets

The platform provides access to 33 Earth Engine datasets including:
- **Climate reanalysis**: ERA5-Land, GLDAS, FLDAS
- **Precipitation**: CHIRPS, GPM IMERG
- **Temperature**: Various reanalysis products
- **Land surface**: SRTM DEM, NLCD land cover
- **And more**: [View complete list](geoclimate_fetcher/data/Datasets.csv)

## 🌡️ Climate Indices Reference

### Temperature Indices
| Index | Name | Description | Unit |
|-------|------|-------------|------|
| TXx | Max Tmax | Monthly maximum of daily maximum temperature | °C |
| TNn | Min Tmin | Monthly minimum of daily minimum temperature | °C |
| TX90p | Warm days | Percentage of days when TX > 90th percentile | % |
| TN10p | Cool nights | Percentage of days when TN < 10th percentile | % |
| SU | Summer days | Annual count of days when TX > 25°C | days |
| FD | Frost days | Annual count of days when TN < 0°C | days |
| DTR | Diurnal temperature range | Monthly mean difference between TX and TN | °C |

### Precipitation Indices
| Index | Name | Description | Unit |
|-------|------|-------------|------|
| RX1day | Max 1-day precipitation | Monthly maximum 1-day precipitation | mm |
| RX5day | Max 5-day precipitation | Monthly maximum consecutive 5-day precipitation | mm |
| R95p | Very wet days | Annual total precipitation when RR > 95th percentile | mm |
| CDD | Consecutive dry days | Maximum number of consecutive days with RR < 1mm | days |
| PRCPTOT | Wet-day precipitation | Annual total precipitation from wet days | mm |

*Complete index definitions follow ETCCDI standards*

## 📁 Project Structure

```
GeoClimate-Fetcher/
├── app.py                          # Main Streamlit application
├── interface/                      # User interface modules
│   ├── geodata_explorer.py         # Dataset explorer and downloader
│   ├── climate_analytics.py        # Climate indices calculator
│   ├── hydrology_analyzer.py       # Precipitation analysis
│   ├── product_selector.py         # Data comparison tool
│   └── data_visualizer.py          # Data visualization
├── geoclimate_fetcher/             # Core functionality
│   ├── climate_indices.py          # Climate indices calculations
│   ├── hydrology_analysis.py       # Hydrology analysis functions
│   ├── visualization.py            # Visualization utilities
│   ├── core/                       # Core modules
│   └── data/                       # Dataset catalogs and configs
├── app_components/                 # UI components and utilities
└── requirements.txt                # Python dependencies
```

## 🔧 Dependencies

### Core Requirements
- `streamlit>=1.25.0` - Web application framework
- `earthengine-api>=0.1.380` - Google Earth Engine Python API
- `pandas` - Data manipulation and analysis
- `plotly>=5.14.0` - Interactive visualizations

### Specialized Features
- `meteostat>=1.6.5` - Weather station data (Product Selector)
- `scikit-learn>=1.3.0` - Statistical analysis (Product Selector)
- `folium>=0.14.0` - Interactive maps
- `streamlit-folium>=0.15.0` - Folium integration for Streamlit
- `geemap` - 🆕 Interactive Earth Engine maps with layer controls

[View complete requirements](requirements.txt)

## 🗺️ New Geemap Preview Features

### Interactive Visualization
Both **GeoData Explorer** and **Climate Analytics** now include interactive map previews:
- **Real-time preview** before downloading data
- **Layer toggle controls** for multi-temporal data (switch between dates/months/years)
- **Automatic geometry clipping** - only shows data within your selected area
- **Smart color palettes** - auto-detects data type (temperature, precipitation, vegetation, etc.)
- **Percentile-based scaling** - uses 5th-95th percentile to handle outliers
- **Integrated colorbar** with units displayed directly on the map

### Adaptive Temporal Aggregation
For datasets with many images, the system automatically:
- **Hourly/Sub-daily** → Aggregates to daily composites if >100 images
- **Daily** → Aggregates to weekly or monthly if >100 images
- **Monthly** → Samples evenly if >100 images (cannot aggregate to finer resolution)
- **Yearly** → Displays all (usually <100)
- **Performance limit**: Maximum 100 layers for optimal map performance

## 📖 Usage Guide

1. **Authentication**: Upload your Google Earth Engine credentials file (generated by `earthengine authenticate`)
2. **Select Tool**: Choose from the five available analysis tools
3. **Define Area**: Use the interactive map to select your area of interest
4. **Configure Analysis**: Set parameters (time period, datasets, indices)
5. **Process**: Let Google Earth Engine handle the computation
6. **Download**: Export results in GeoTIFF or CSV format
7. **Visualize**: Use the Data Visualizer for analysis and plotting

## 🎓 Educational Use

This platform was developed for educational and research purposes. It provides:
- Hands-on experience with Google Earth Engine
- Understanding of climate indices and their applications
- Practical data analysis skills for climate research
- Interactive learning environment for geospatial analysis

## ⚠️ Limitations

- **Internet required**: All processing happens on Google Earth Engine servers
- **GEE quotas apply**: Subject to Google Earth Engine usage limits
- **Beta software**: Under active development, some features may be unstable
- **Limited datasets**: Currently supports 33 pre-configured datasets
- **No local processing**: All computation requires Earth Engine authentication

## 🤝 Contributing

We welcome contributions! This is an academic project and we appreciate:
- Bug reports and feature requests via [Issues](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/issues)
- Code contributions via pull requests
- Documentation improvements
- Educational use cases and feedback

### Development Setup
```bash
# Fork the repository
git clone https://github.com/your-username/GeoClimate-Fetcher.git
cd GeoClimate-Fetcher

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py
```

## 👨‍💻 Authors

**Saurav Bhattarai**<sup>a</sup>
Civil Engineer & Geospatial Developer
📧 [saurav.bhattarai.1999@gmail.com](mailto:saurav.bhattarai.1999@gmail.com)
🌐 [sauravbhattarai19.github.io](https://sauravbhattarai19.github.io)
🔗 [GitHub](https://github.com/sauravbhattarai19) | [LinkedIn](https://www.linkedin.com/in/saurav-bhattarai-7133a3176/)

**Nawa Raj Pradhan**<sup>b</sup>

**Rocky Talchabhadel**<sup>a</sup>

<sup>a</sup> Jackson State University

## 🙏 Acknowledgments

This study was conducted for the Department of War under Project No. PE 622182CX3 and PE 633042CX7, Advanced Technology and Applied Technology, Hydrology Mapping.

The first author gratefully acknowledges support from the Oak Ridge Institute for Science and Education (ORISE) fellowship program under the Department of Defense at the U.S. Army Engineer Research and Development Center (ERDC), USACE.

This research was supported by the Hydrological Impacts Computing, Outreach, and Resiliency Partnership (HICORPS) Project, developed in collaboration with the U.S. Army Engineer Research and Development Center (ERDC), WOOLPERT, and Taylor Engineering. Additional funding was provided by the Microsoft-Atlanta University Center (AUC) Data Science Mini-Grant Program, which advances data science research and education at Historically Black Colleges and Universities (HBCUs) through virtually integrated projects promoting innovative solutions to real-world problems.

We thank Jackson State University for providing computational resources and infrastructure support throughout this research. Special appreciation goes to the JSU Water Lab for their invaluable assistance during training workshops and beta version testing phases, which were critical for platform development and validation.

The authors thank Google Earth Engine for providing free access to planetary-scale computing and datasets that made this comprehensive analysis platform possible, and Streamlit for the open-source framework that enabled interactive platform deployment. We acknowledge the climate science community for developing and maintaining the ETCCDI climate indices standards that ensure scientific rigor and international comparability. Special thanks to the students and educators who provided feedback during the platform's educational component development, contributing to its pedagogical effectiveness.

## 📄 Publication

This platform has been published in **Environmental Modelling & Software**:

**Bhattarai, S., Pradhan, N. R., & Talchabhadel, R.** (2025). GeoClimate intelligence platform: A web-based framework for environmental data analysis. *Environmental Modelling & Software*, 106826.

- **DOI**: [10.1016/j.envsoft.2025.106826](https://doi.org/10.1016/j.envsoft.2025.106826)
- **Article Link**: [https://www.sciencedirect.com/science/article/pii/S1364815225005109](https://www.sciencedirect.com/science/article/pii/S1364815225005109)
- **License**: Open access under Creative Commons license

## 📄 Citation

If you use this platform in your research, please cite:

```bibtex
@article{bhattarai2025geoclimate,
  title={GeoClimate intelligence platform: A web-based framework for environmental data analysis},
  author={Bhattarai, Saurav and Pradhan, Nawa Raj and Talchabhadel, Rocky},
  journal={Environmental Modelling \& Software},
  volume={},
  pages={106826},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.envsoft.2025.106826},
  url={https://www.sciencedirect.com/science/article/pii/S1364815225005109}
}
```

## 📋 License

This project is **not for commercial use**. It is intended for educational and research purposes only. For commercial licensing inquiries, please contact the author.

## 🔗 Resources

- **Documentation**: [Wiki](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/wiki)
- **Issues**: [GitHub Issues](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/discussions)

---

📚 [Documentation](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/wiki) • 🐛 [Issues](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/issues) • 💬 [Discussions](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/discussions)