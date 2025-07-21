# 🌍 GeoClimate Fetcher - Streamlit Web Application

A modern web application for fetching and analyzing geospatial climate data from Google Earth Engine. Built with Streamlit for an intuitive user interface.

## ✨ Features

- **🔐 Easy Authentication** - Simple Google Earth Engine setup
- **📍 Interactive Area Selection** - Draw on map, upload GeoJSON, or enter coordinates
- **📊 Comprehensive Dataset Library** - Access CHIRPS, MODIS, GLDAS, and more
- **🎚️ Flexible Band Selection** - Choose exactly the data you need
- **📅 Time Range Filtering** - Precise temporal data selection
- **💾 Multiple Export Formats** - GeoTIFF, NetCDF, and CSV support
- **☁️ Smart Cloud Integration** - Automatic Google Drive export for large files
- **🚀 Progress Tracking** - Real-time download progress and status
- **🐳 Docker Ready** - Containerized for easy deployment

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd GeoClimate-Fetcher-1
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Open your browser:**
   ```
   http://localhost:8501
   ```

### Option 2: Local Installation

1. **Prerequisites:**
   - Python 3.8-3.12
   - GDAL system libraries
   - Git

2. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements-streamlit.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## 🐳 Docker Usage

### Production Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Development Mode

```bash
# Run with live code reloading
docker-compose --profile dev up

# Access development version at http://localhost:8502
```

### Custom Configuration

```bash
# Use custom environment file
docker-compose --env-file .env.custom up

# Override volumes for different data directories
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

## 📋 Application Workflow

### 1. Authentication
- Enter your Google Earth Engine project ID
- Optional: Configure service account for automated deployments
- Credentials are saved for future sessions

### 2. Area of Interest Selection
Choose from three methods:
- **🗺️ Interactive Map**: Draw polygons or rectangles directly on the map
- **📁 File Upload**: Upload GeoJSON files with your areas of interest
- **📐 Coordinates**: Enter bounding box coordinates manually

### 3. Dataset Selection
- Browse hundreds of available datasets
- Search by name or description
- Filter by category (Climate, Vegetation, etc.)
- View detailed dataset information

### 4. Band and Time Configuration
- Select specific bands/variables you need
- Choose date ranges for time series data
- Preview estimated download sizes

### 5. Download and Export
- Configure output format (GeoTIFF, NetCDF, CSV)
- Set resolution and processing options
- Monitor progress in real-time
- Automatic cloud export for large files

## 🔧 Configuration

### Environment Variables

```bash
# Google Cloud credentials (optional)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Custom data directory
DATA_DIR=/custom/data/path

# Output directory
OUTPUT_DIR=/custom/output/path
```

### Volume Mounts for Docker

```yaml
volumes:
  # Persistent downloads
  - ./downloads:/app/downloads
  
  # Custom data directory
  - /your/data:/app/data:ro
  
  # Credentials
  - ~/.geoclimate-fetcher:/root/.geoclimate-fetcher:ro
  
  # Service account key (if using)
  - /path/to/key.json:/app/credentials/key.json:ro
```

## 📊 Supported Datasets

The application includes access to numerous Earth Engine datasets:

- **🌧️ Climate Data**: CHIRPS precipitation, ERA5 reanalysis, Daymet
- **🛰️ Satellite Imagery**: Landsat, Sentinel-1/2, MODIS
- **🌱 Vegetation**: NDVI, EVI, vegetation indices
- **🌍 Land Surface**: Land surface temperature, soil moisture
- **🏔️ Topography**: SRTM, ASTER DEM
- **🌊 Hydrology**: Global surface water, flood mapping

## ⚙️ Advanced Usage

### Custom Dataset Configuration

Add your own datasets by modifying the metadata catalog:

```python
# In geoclimate_fetcher/data/custom_datasets.csv
Dataset Name,Earth Engine ID,Description,Snippet Type
My Dataset,projects/my-project/assets/my-dataset,Custom dataset,Image
```

### Service Account Authentication

For automated deployments:

1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Mount it in the Docker container or set the environment variable

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

### Large-Scale Processing

For processing large areas or long time series:

- Enable chunking for ImageCollections
- Use Google Drive export for files >50MB
- Consider reducing spatial resolution
- Split large requests into smaller time periods

## 🚦 Health Monitoring

The application includes health checks:

```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Docker health status
docker-compose ps
```

## 🔒 Security Considerations

- **Credentials**: Store Google Cloud credentials securely
- **Network**: Use HTTPS in production (configure reverse proxy)
- **Access**: Implement authentication for multi-user deployments
- **Data**: Ensure proper permissions for output directories

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify Earth Engine project access
   - Check service account permissions
   - Ensure project ID is correct

2. **Large File Issues**
   - Files >50MB automatically go to Google Drive
   - Check Google Drive storage space
   - Monitor Earth Engine quotas

3. **GDAL Installation Issues**
   ```bash
   # Install GDAL system libraries
   sudo apt-get install gdal-bin libgdal-dev
   
   # Install Python GDAL with matching version
   pip install GDAL==$(gdal-config --version)
   ```

4. **Memory Issues**
   - Reduce area size or time range
   - Use lower resolution
   - Enable chunking for large collections

### Debug Mode

Enable debug logging:

```bash
streamlit run main.py --logger.level=debug
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run main.py
```

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Cloud Platforms

#### Google Cloud Run
```bash
gcloud run deploy geoclimate-fetcher \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS ECS/Fargate
```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t geoclimate-fetcher .
docker tag geoclimate-fetcher:latest <account>.dkr.ecr.<region>.amazonaws.com/geoclimate-fetcher:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/geoclimate-fetcher:latest
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: geoclimate-fetcher
spec:
  replicas: 1
  selector:
    matchLabels:
      app: geoclimate-fetcher
  template:
    metadata:
      labels:
        app: geoclimate-fetcher
    spec:
      containers:
      - name: app
        image: geoclimate-fetcher:latest
        ports:
        - containerPort: 8501
```

## 📈 Performance Tips

- **Spatial Resolution**: Use appropriate resolution for your needs
- **Time Ranges**: Limit to necessary date ranges
- **Area Size**: Smaller areas process faster
- **Band Selection**: Only select needed bands
- **Caching**: Results are cached where possible
- **Parallel Processing**: Enable chunking for large collections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Your License Here]

## 📞 Support

- **Issues**: Report bugs on GitHub
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join our discussions

---

**Made with ❤️ for the geospatial community** 