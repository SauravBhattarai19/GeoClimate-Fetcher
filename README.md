# 🌍 GeoClimate Fetcher

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-4285F4.svg)](https://earthengine.google.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Streamlit web application for downloading and analyzing climate data from Google Earth Engine. Developed as a research tool for climate data analysis and educational purposes.

**Status**: Under development. Manuscript submitted for peer review.

![Platform Screenshot](pictures/platform.png)

## 🎯 What This Platform Does

This is a web-based interface that provides five integrated tools for climate data analysis:

### 🔍 **GeoData Explorer**
- Browse and download from [33 Earth Engine datasets](geoclimate_fetcher/data/Datasets.csv)
- Interactive map for area selection
- Automatic band detection and metadata extraction
- Export to GeoTIFF and CSV formats

### 🧠 **Climate Analytics**
- Calculate ETCCDI-compliant climate indices
- **Temperature indices**: TXx, TNn, TXn, TNx, TX90p, TX10p, TN90p, TN10p, SU, FD, DTR, GSL, WSDI, CSDI
- **Precipitation indices**: RX1day, RX5day, CDD, R10mm, R20mm, SDII, R95p, R99p, R75p, PRCPTOT
- Trend analysis using Mann-Kendall test and Sen's slope
- Server-side processing with Google Earth Engine

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

### Prerequisites
- Python 3.8 or higher
- Google Earth Engine account ([sign up free](https://earthengine.google.com/signup/))

### Installation

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

[View complete requirements](requirements.txt)

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

**Saurav Bhattarai**
Civil Engineer & Geospatial Developer
📧 [saurav.bhattarai.1999@gmail.com](mailto:saurav.bhattarai.1999@gmail.com)
🌐 [sauravbhattarai19.github.io](https://sauravbhattarai19.github.io)
🔗 [GitHub](https://github.com/sauravbhattarai19) | [LinkedIn](https://www.linkedin.com/in/saurav-bhattarai-7133a3176/)

**Supervision:**
- Dr. Rocky Talchabhadel
- Dr. Nawaraj Pradhan

## 🙏 Acknowledgments

- **Google Earth Engine** - For providing the computational platform and data access
- **Streamlit** - For the excellent web application framework
- **ETCCDI** - For climate indices standards and definitions
- **Claude AI** - For development assistance and code optimization
- **Open source community** - For the many Python libraries that make this possible

## 📄 Citation

If you use this platform in your research, please cite:

```bibtex
@software{geoclimate_fetcher,
  title={GeoClimate Fetcher: A Web-Based Tool for Climate Data Analysis},
  author={Bhattarai, Saurav},
  year={2025},
  url={https://github.com/SauravBhattarai19/GeoClimate-Fetcher},
  version={1.0.0}
}
```

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **Documentation**: [Wiki](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/wiki)
- **Issues**: [GitHub Issues](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/discussions)

---

📚 [Documentation](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/wiki) • 🐛 [Issues](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/issues) • 💬 [Discussions](https://github.com/SauravBhattarai19/GeoClimate-Fetcher/discussions)