# GeoClimate Intelligence Platform - Landing Page

Professional landing page for the GeoClimate Intelligence Platform, a Microsoft AUC partnership project developed at Jackson State University.

## 🌍 Live Demo

Visit the platform: [geeclimate.streamlit.app](https://geeclimate.streamlit.app)

## 📋 Features

- **Modern Responsive Design** - Works perfectly on all devices
- **Microsoft AUC Partnership Showcase** - Highlights the collaboration
- **Team Member Profiles** - Real photos and social links
- **Interactive Media Section** - Podcast, testimonials, workshop gallery
- **QR Code Integration** - Quick access to the platform
- **Professional Styling** - Microsoft-inspired color scheme

## 🚀 Quick Start

### Local Development
1. Open a terminal in this directory
2. Run a local server:
   ```bash
   # Python built-in server
   python -m http.server 8000
   ```
3. Open your browser to `http://localhost:8000`

### GitHub Pages Deployment
See [GITHUB_PAGES_DEPLOYMENT.md](GITHUB_PAGES_DEPLOYMENT.md) for detailed instructions.

## 📁 File Structure

```
landing-page/
├── index.html              # Main landing page
├── styles.css              # All styling
├── script.js               # Interactive functionality
├── photos/                 # Team and workshop photos
│   ├── rocky.png          # Dr. Rocky Talchabhadel
│   ├── saurav.jpeg        # Saurav Bhattarai
│   ├── sunil.JPG          # Sunil Bista
│   ├── subash.JPG         # Subash Poudel
│   ├── samuel.jpeg        # Samuel Terret
│   ├── douglas.jpeg       # Douglas Jones
│   ├── workshop1.jpeg     # Workshop photo 1
│   ├── workshop2.jpeg     # Workshop photo 2
│   └── workshop3.jpeg     # Workshop photo 3
├── README.md              # This file
└── GITHUB_PAGES_DEPLOYMENT.md # Deployment guide
```

## 🖼️ Adding Photos

### Team Photos
Place team member photos in the `photos/` folder with exact names:
- `rocky.png` - Dr. Rocky Talchabhadel
- `saurav.jpeg` - Saurav Bhattarai
- `sunil.JPG` - Sunil Bista
- `subash.JPG` - Subash Poudel
- `samuel.jpeg` - Samuel Terret
- `douglas.jpeg` - Douglas Jones

### Workshop Photos
Add workshop photos as:
- `workshop1.jpeg` - Workshop Photo 1
- `workshop2.jpeg` - Workshop Photo 2
- `workshop3.jpeg` - Workshop Photo 3

## 📱 Responsive Design

The landing page is fully responsive and tested on:
- ✅ Desktop computers
- ✅ Tablets
- ✅ Mobile phones
- ✅ Large displays

## 🔧 Technologies Used

- **HTML5** - Semantic structure
- **CSS3** - Modern styling with Grid and Flexbox
- **JavaScript** - Interactive features
- **Font Awesome** - Icons
- **Google Fonts** - Inter typography
- **QRCode.js** - QR code generation

## 🎯 Project Information

**Project**: GeoClimate Intelligence Platform  
**Institution**: Jackson State University  
**Partnership**: Microsoft AUC (Azure University Credits)  
**Lead**: Dr. Rocky Talchabhadel  
**Developer**: Saurav Bhattarai  

## 📞 Contact

- **Platform**: [geeclimate.streamlit.app](https://geeclimate.streamlit.app)
- **Lab Website**: [bit.ly/jsu_water](https://bit.ly/jsu_water)
- **Developer**: [saurav.bhattarai.1999@gmail.com](mailto:saurav.bhattarai.1999@gmail.com)
- **GitHub**: [github.com/sauravbhattarai19](https://github.com/sauravbhattarai19)

## 📄 License

This project is part of the Microsoft AUC partnership program at Jackson State University.

---

Built with ❤️ for advancing climate research and education.

## Overview

The GeoClimate Intelligence Platform is an advanced Earth Engine climate data analysis and visualization platform that provides researchers and students with powerful tools for climate research and education.

## Features

### 🌍 **GeoData Explorer**
- Download and visualize Earth Engine datasets
- 54+ curated climate datasets across 6 categories
- Interactive area selection with drawing tools
- Multiple export formats (GeoTIFF, NetCDF, CSV)
- Direct download or Google Drive export

### 🧠 **Climate Intelligence Hub**
- Calculate 40+ climate indices based on ETCCDI standards
- Temperature and precipitation analysis
- Extreme events detection
- Cloud-based processing using Google Earth Engine

### 🗺️ **Interactive Mapping**
- Draw study areas directly on map
- Upload shapefiles/GeoJSON files
- Real-time data preview
- Time series visualization

### 📊 **Advanced Analytics**
- Time series analysis and trend detection
- Anomaly detection
- Statistical summaries
- Export capabilities

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python, Streamlit
- **Data Processing**: Google Earth Engine
- **Cloud Platform**: Microsoft Azure (via AUC program)
- **Geospatial Libraries**: GeoPandas, Folium, Rasterio
- **Visualization**: Plotly, Matplotlib

## Team

### Research Team
- **Dr. Rocky Talchabhadel** - Principal Investigator
- **Saurav Bhattarai** - Lead Developer
- **Sunil Bista** - Documentation Specialist
- **Subash Poudel** - Training Coordinator
- **Samuel Terret** - Undergraduate Researcher & Beta Tester
- **Douglas Jones** - Undergraduate Researcher & Beta Tester

## Project Impact

### ✅ **Completed Deliverables**
- ✅ Web Application: [geeclimate.streamlit.app](https://geeclimate.streamlit.app)
- ✅ Open Source Codebase: [GitHub Repository](https://github.com/Saurav-JSU/GeoClimate-Fetcher)
- ✅ Comprehensive Documentation
- ✅ Training Programs for Undergraduates
- ✅ Workshop Materials and User Guides

### 📈 **Project Statistics**
- **54+** Climate Datasets Available
- **6** Data Categories (Precipitation, Temperature, Soil Moisture, etc.)
- **40+** Climate Indices Calculations
- **50+** Students Trained
- **5** Workshops Conducted
- **100%** Workshop Completion Rate

## Training & Education

### Undergraduate Training Program
Comprehensive workshops designed to introduce undergraduate students to advanced geospatial climate analysis techniques.

**Workshop Topics:**
- Introduction to Google Earth Engine
- Climate Data Analysis Fundamentals  
- GeoClimate Platform Tutorial
- Climate Indices Calculation
- Geospatial Data Visualization
- Research Project Development

## Access the Platform

### 🚀 **Live Application**
**URL**: [geeclimate.streamlit.app](https://geeclimate.streamlit.app)

### 📱 **QR Code Access**
Scan the QR code on the landing page to quickly access the platform on mobile devices.

### 💻 **Local Installation**
```bash
git clone https://github.com/Saurav-JSU/GeoClimate-Fetcher.git
cd GeoClimate-Fetcher
pip install -e .
streamlit run app.py
```

## File Structure

```
landing-page/
├── index.html              # Main landing page
├── styles.css              # CSS styling with modern design
├── script.js               # JavaScript functionality
├── demo.html               # Quick preview/demo page
├── microsoft-partnership.html  # MS partnership details
├── server.py               # Custom development server
├── simple-server.py        # Fallback simple server
├── start-server.bat        # Windows batch file
├── package.json            # Project metadata
├── DEPLOYMENT.md           # Deployment guide
└── README.md               # This documentation
```

## 🚀 **How to Run Locally:**

### **Local Development Options:**

**Option 1: Custom Server (Recommended)**
```bash
cd landing-page
python server.py
```

**Option 2: Simple Server (If custom server fails)**
```bash
cd landing-page
python simple-server.py
```

**Option 3: Python Built-in Server**
```bash
cd landing-page
python -m http.server 8000
```

**Option 4: Windows Batch File**
```batch
cd landing-page
start-server.bat
```

All options will automatically open the demo page in your browser at http://localhost:8000

### **Troubleshooting:**

If you encounter server errors:
1. ✅ Try `simple-server.py` instead of `server.py`
2. ✅ Use Python's built-in server: `python -m http.server 8000`
3. ✅ Check if port 8000 is already in use
4. ✅ Try a different port: `python -m http.server 8001`
5. ✅ Make sure you're in the `landing-page` directory

### **Quick Access URLs:**
- 🏠 Demo Page: http://localhost:8000/demo.html
- 📄 Main Landing: http://localhost:8000/index.html
- 🤝 MS Partnership: http://localhost:8000/microsoft-partnership.html
- 🚀 Live Platform: https://geeclimate.streamlit.app

## Features of the Landing Page

### 🎨 **Modern Design**
- Responsive design that works on all devices
- Modern gradient backgrounds and animations
- Microsoft-inspired color scheme
- Smooth animations and transitions
- Accessibility-compliant design

### ⚡ **Interactive Elements**
- Smooth scrolling navigation
- Animated counters for statistics
- QR code generator for mobile access
- Hover effects and micro-interactions
- Mobile-friendly navigation menu

### 🔧 **Technical Features**
- Semantic HTML5 structure
- CSS Grid and Flexbox layouts
- Intersection Observer API for animations
- Service Worker ready for offline functionality
- Performance optimized with lazy loading

### 📱 **Responsive Design**
- Mobile-first approach
- Tablet and desktop optimizations
- Touch-friendly interface
- Flexible grid layouts

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Performance

- **Lighthouse Score**: 95+ Performance
- **Load Time**: < 2 seconds
- **Interactive**: < 1 second
- **Accessibility**: WCAG 2.1 AA compliant

## Deployment

The landing page is static and can be deployed to:
- GitHub Pages
- Netlify
- Vercel
- Azure Static Web Apps
- Any static hosting service

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test across different browsers
5. Submit a pull request

## License

This project is developed under the Microsoft AUC partnership and follows open-source principles.

## Contact

### 📧 **Project Lead**
- **Saurav Bhattarai**: saurav.bhattarai.1999@gmail.com
- **GitHub**: [@sauravbhattarai19](https://github.com/sauravbhattarai19)
- **LinkedIn**: [saurav-bhattarai](https://www.linkedin.com/in/saurav-bhattarai-7133a3176/)

### 🏛️ **Institution**
- **Jackson State University Water Resources Lab**
- **Website**: [bit.ly/jsu_water](https://bit.ly/jsu_water)

## Acknowledgments

- **Microsoft AUC Program** for cloud computing resources
- **Google Earth Engine** for geospatial data processing platform
- **Jackson State University** for institutional support
- **Research team members** for their contributions

---

**Built with ❤️ for advancing climate research and education**
