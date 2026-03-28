import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import json
import geemap
import datetime
import os
import zipfile
import tempfile
import io
import shutil
import traceback
from typing import List, Dict, Any, Optional
import math
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.merge import merge
import numpy as np
import requests
from dateutil.relativedelta import relativedelta
import pandas as pd
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    # If plotly not installed, we'll skip the visualization
    go = None
    px = None

# Try to import the separate auth module if available
try:
    from Tool.gee_auth import GEEAuth
except ImportError:
    # Define GEEAuth class inline if module not available
    class GEEAuth:
        """Class to handle Google Earth Engine authentication."""
        
        def __init__(self):
            """Initialize the GEE authentication handler."""
            self._initialized = False
        
        def initialize(self, project_id: str, credentials_content: Optional[str] = None) -> bool:
            """
            Initialize the Earth Engine API with the provided credentials.
            
            Args:
                project_id: The Google Cloud project ID
                credentials_content: Optional credentials file content for web apps
                
            Returns:
                bool: True if authentication was successful, False otherwise
            """
            try:
                # Method: Credentials file content (for web apps with file upload)
                if credentials_content:
                    # Handle uploaded credentials content
                    try:
                        # Validate that the JSON parses correctly
                        creds_data = json.loads(credentials_content)

                        # Check if it's a service account key (has private_key)
                        if "private_key" in creds_data and "client_email" in creds_data:
                            # Temporarily write to disk and use ServiceAccountCredentials
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_sa:
                                tmp_sa.write(credentials_content)
                                tmp_sa_path = tmp_sa.name

                            try:
                                credentials = ee.ServiceAccountCredentials(creds_data["client_email"], tmp_sa_path)
                                ee.Initialize(credentials, project=project_id)
                            finally:
                                if os.path.exists(tmp_sa_path):
                                    os.unlink(tmp_sa_path)
                        else:
                            # This is a user OAuth credentials file
                            # Persist it to ~/.config/earthengine/ so subsequent sessions work
                            ee_creds_dir = os.path.expanduser("~/.config/earthengine")
                            os.makedirs(ee_creds_dir, exist_ok=True)

                            ee_creds_path = os.path.join(ee_creds_dir, "credentials")

                            try:
                                with open(ee_creds_path, "w", encoding="utf-8") as f:
                                    f.write(credentials_content)
                            except Exception as write_err:
                                raise Exception(f"Unable to write credentials file: {write_err}")

                            # Initialize EE with the credentials
                            ee.Initialize(project=project_id)

                    except json.JSONDecodeError:
                        raise Exception("Invalid credentials file format. Please upload a valid Earth Engine credentials file.")
                
                # Try default authentication if no credentials provided
                else:
                    try:
                        ee.Initialize(project=project_id)
                    except Exception as e:
                        raise Exception(
                            "Authentication failed. Please provide:\n"
                            "1. Your Google Cloud Project ID\n"
                            "2. Your Earth Engine credentials file\n\n"
                            "To get credentials: Run 'earthengine authenticate' locally, then upload the credentials file from ~/.config/earthengine/credentials"
                        )
                    
                # Test the connection
                if self.test_connection():
                    self._initialized = True
                    return True
                else:
                    raise Exception("Authentication succeeded but connection test failed")
                    
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                self._initialized = False
                return False
                
        def is_initialized(self) -> bool:
            """Check if Earth Engine has been initialized."""
            return self._initialized
        
        @staticmethod
        def test_connection() -> bool:
            """Test the connection to Earth Engine by making a simple API call."""
            try:
                ee.Image("USGS/SRTMGL1_003").getInfo()
                return True
            except Exception:
                return False

def get_seasonal_snow_density(month: int, use_seasonal: bool = True, custom_density_value: float = None, custom_monthly_densities: dict = None) -> float:
    """
    Get snow density based on month and user preferences
    
    Args:
        month: Month number (1-12)
        use_seasonal: If True, use default seasonal density values
        custom_density_value: If provided, use this constant custom density value
        custom_monthly_densities: Dictionary of month->density mappings for custom monthly values
        
    Returns:
        Snow density in kg/m³
    """
    # Priority: custom_density_value > custom_monthly_densities > seasonal
    if custom_density_value is not None:
        return custom_density_value
    
    if custom_monthly_densities is not None:
        return custom_monthly_densities.get(month, 280)  # Default to 280 if month not found
    
    if use_seasonal:
        # Default seasonal density values based on snow metamorphism
        seasonal_densities = {
            1: 180,   # January - fresh/settled snow
            2: 220,   # February - settling snow
            3: 280,   # March - aging snow
            4: 350,   # April - spring conditions
            5: 420,   # May - wet spring snow
            6: 400,   # June - late season snow
            7: 400,   # July - summer snow (rare)
            8: 400,   # August - summer snow (rare)
            9: 350,   # September - early season snow
            10: 280,  # October - new snow
            11: 220,  # November - early winter
            12: 200   # December - winter snow
        }
        return seasonal_densities.get(month, 280)  # Default to 280 if month invalid
    else:
        # Use constant density (original algorithm)
        return 280  # Average density

def calculate_monthly_gamma_inverse_fit(target_density: float, mean_fsc: float, rho_min: float, rho_max: float) -> float:
    """
    Calculate gamma value for a specific month using the inverse-fit method
    
    Args:
        target_density: Literature-based target density for the month (ρ_m^lit)
        mean_fsc: Mean Fractional Snow Cover for the month (FSC_m)
        rho_min: Minimum density value
        rho_max: Maximum density value
        
    Returns:
        Calculated gamma value for the month
    """
    # Safety checks to prevent log(0) or log(negative) errors
    
    # Clamp the ratio term to prevent log errors
    ratio = (target_density - rho_min) / (rho_max - rho_min)
    ratio = max(1e-9, min(0.99999, ratio))
    
    # Clamp 1 - FSC_m to ensure it's in valid range
    one_minus_fsc = max(1e-9, min(0.99999, 1 - mean_fsc))
    
    # Calculate gamma using the inverse-fit formula
    # γ_m = ln((ρ_m^lit - ρ_min) / (ρ_max - ρ_min)) / ln(1 - FSC_m)
    gamma = np.log(ratio) / np.log(one_minus_fsc)
    
    # Additional safety check to ensure gamma is reasonable
    gamma = max(0.1, min(10.0, gamma))  # Clamp between 0.1 and 10
    
    return gamma

def sinusoidal_gamma_function(t: float, alpha: float, beta: float, phi: float) -> float:
    """
    Sinusoidal function for gamma variation throughout the year
    
    Args:
        t: Month index (1-12)
        alpha: Offset parameter
        beta: Amplitude parameter  
        phi: Phase shift parameter
        
    Returns:
        Gamma value for the given month
    """
    return alpha + beta * np.sin(2 * np.pi * (t - phi) / 12)

def fit_sinusoidal_gamma_curve(monthly_gammas: List[float]) -> tuple:
    """
    Fit a sinusoidal curve to 12 monthly gamma values
    
    Args:
        monthly_gammas: List of 12 gamma values (one for each month)
        
    Returns:
        Tuple of fitted parameters (alpha, beta, phi) and the fitted curve function
    """
    months = np.array(range(1, 13))  # Months 1-12
    
    # Initial parameter guesses
    alpha_guess = np.mean(monthly_gammas)  # Mean of the data
    beta_guess = (np.max(monthly_gammas) - np.min(monthly_gammas)) / 2  # Half the range
    phi_guess = 1  # Start with January
    
    try:
        # Perform curve fitting
        popt, pcov = curve_fit(
            sinusoidal_gamma_function, 
            months, 
            monthly_gammas,
            p0=[alpha_guess, beta_guess, phi_guess],
            maxfev=2000
        )
        
        alpha, beta, phi = popt
        
        # Calculate R² for goodness of fit
        fitted_values = [sinusoidal_gamma_function(m, alpha, beta, phi) for m in months]
        ss_res = np.sum((np.array(monthly_gammas) - np.array(fitted_values)) ** 2)
        ss_tot = np.sum((np.array(monthly_gammas) - np.mean(monthly_gammas)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return alpha, beta, phi, r_squared
        
    except Exception as e:
        st.error(f"Error fitting sinusoidal curve: {str(e)}")
        # Return default parameters if fitting fails
        return alpha_guess, 0, 1, 0

def calculate_mean_fsc_for_months(aoi, year_range: tuple = (2018, 2023)) -> Dict[int, float]:
    """
    Calculate mean Fractional Snow Cover for each month using historical data
    
    Args:
        aoi: Area of Interest geometry
        year_range: Tuple of (start_year, end_year) for historical data
        
    Returns:
        Dictionary mapping month (1-12) to mean FSC value
    """
    monthly_fsc = {}
    
    try:
        for month in range(1, 13):
            fsc_values = []
            
            # Collect FSC data for this month across multiple years
            for year in range(year_range[0], year_range[1] + 1):
                try:
                    start_date = datetime.date(year, month, 1)
                    if month == 12:
                        end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
                    else:
                        end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
                    
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    # Get MODIS Snow Cover data
                    snowExtent = ee.ImageCollection('MODIS/061/MOD10A1') \
                        .select("NDSI_Snow_Cover") \
                        .filterDate(start_date_str, end_date_str) \
                        .filterBounds(aoi)
                    
                    if snowExtent.size().getInfo() > 0:
                        meanNDSI = snowExtent.mean().divide(100)
                        fsc = meanNDSI.multiply(1.45).subtract(0.01).clamp(0, 1)
                        
                        # Get mean FSC value for the AOI
                        mean_fsc_value = fsc.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=aoi,
                            scale=500,
                            maxPixels=1e9
                        ).getInfo()
                        
                        if 'NDSI_Snow_Cover' in mean_fsc_value and mean_fsc_value['NDSI_Snow_Cover'] is not None:
                            fsc_values.append(mean_fsc_value['NDSI_Snow_Cover'])
                
                except Exception as e:
                    continue  # Skip this year if there's an error
            
            # Calculate mean FSC for this month
            if fsc_values:
                monthly_fsc[month] = np.mean(fsc_values)
            else:
                # Use a default value if no data available
                monthly_fsc[month] = 0.5  # Default moderate snow cover
                
    except Exception as e:
        st.error(f"Error calculating mean FSC: {str(e)}")
        # Return default values if calculation fails
        for month in range(1, 13):
            monthly_fsc[month] = 0.5
    
    return monthly_fsc

class SnowDepthCalculator:
    def __init__(self, scale: int = 500, use_seasonal_density: bool = True, custom_density_value: float = None, 
                 custom_monthly_densities: dict = None, density_method: str = "original", 
                 rho_min: float = 100.0, rho_max: float = 500.0, gamma_params: dict = None):
        """Initialize the Snow Depth Calculator
        
        Args:
            scale: Pixel resolution in meters (default: 500)
            use_seasonal_density: Whether to use default seasonal density values
            custom_density_value: Constant custom density value in kg/m³
            custom_monthly_densities: Dictionary of month->density mappings for custom monthly values
            density_method: Method for density calculation ("original", "enhanced_method1", "enhanced_method2")
            rho_min: Minimum density value for enhanced methods (kg/m³)
            rho_max: Maximum density value for enhanced methods (kg/m³)
            gamma_params: Parameters for enhanced methods (dict with gamma values or sinusoidal params)
        """
        self.scale = scale
        self.use_seasonal_density = use_seasonal_density
        self.custom_density_value = custom_density_value
        self.custom_monthly_densities = custom_monthly_densities
        self.density_method = density_method
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.gamma_params = gamma_params or {}
        
    def calculate_monthly_snow_depth(self, aoi, year: int, month: int):
        """Calculate snow depth for a specific month and year"""
        # Create date range for the month
        start_date = datetime.date(year, month, 1)
        if month == 12:
            end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # MODIS Snow Cover
        snowExtent = ee.ImageCollection('MODIS/061/MOD10A1') \
            .select("NDSI_Snow_Cover") \
            .filterDate(start_date_str, end_date_str) \
            .filterBounds(aoi)
        
        meanNDSI = snowExtent.mean().divide(100)
        fsc = meanNDSI.multiply(1.45).subtract(0.01).clamp(0, 1)
        
        # GLDAS SWE
        gldasSWE = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .select("SWE_inst") \
            .filterDate(start_date_str, end_date_str) \
            .filterBounds(aoi) \
            .mean()
        
        # Resample GLDAS
        modisProjection = meanNDSI.projection()
        gldasSWE_resampled = gldasSWE \
            .resample("bilinear") \
            .reproject(crs=modisProjection, scale=self.scale)
        
        # Snow density calculation based on selected method
        if self.density_method == "original":
            # Original method - get density for this month
            density_value = get_seasonal_snow_density(month, self.use_seasonal_density, 
                                                    self.custom_density_value, self.custom_monthly_densities)
            
            # Use the original FSC-based density variation with seasonal base
            if self.use_seasonal_density and self.custom_density_value is None and self.custom_monthly_densities is None:
                rho_min = max(100, density_value - 80)  # Minimum density
                rho_max = min(500, density_value + 80)  # Maximum density
                snowDensity = fsc.expression(
                    "rho_min + (rho_max - rho_min) * (1 - FSC)", {
                        "rho_min": rho_min,
                        "rho_max": rho_max,
                        "FSC": fsc
                    })
            else:
                # Use constant density
                snowDensity = ee.Image.constant(density_value)
                
        elif self.density_method in ["enhanced_method1", "enhanced_method2"]:
            # Enhanced methods using ρ = ρ_min + (ρ_max - ρ_min) * (1 - FSC)^γ
            
            # Determine gamma value for this month
            if self.density_method == "enhanced_method1":
                # Method 1: Use discrete gamma value for this month
                gamma_value = self.gamma_params.get(f'gamma_{month}', 1.0)
            else:
                # Method 2: Use sinusoidal function to calculate gamma
                alpha = self.gamma_params.get('alpha', 1.0)
                beta = self.gamma_params.get('beta', 0.0)
                phi = self.gamma_params.get('phi', 1.0)
                gamma_value = sinusoidal_gamma_function(month, alpha, beta, phi)
            
            # Calculate density using the enhanced formula
            # ρ = ρ_min + (ρ_max - ρ_min) * (1 - FSC)^γ
            snowDensity = fsc.expression(
                "rho_min + (rho_max - rho_min) * pow(1 - FSC, gamma)", {
                    "rho_min": self.rho_min,
                    "rho_max": self.rho_max,
                    "FSC": fsc,
                    "gamma": gamma_value
                })
        else:
            # Fallback to constant density
            snowDensity = ee.Image.constant(280)
        
        # Final calculations
        fineSWE = gldasSWE_resampled.multiply(fsc)
        snowDepth = fineSWE.divide(snowDensity.add(1e-6))
        
        return snowDepth.clip(aoi)

def initialize_earth_engine(project_id: str, credentials_content: Optional[str] = None) -> bool:
    """Initialize Google Earth Engine with user-provided credentials"""
    if not project_id:
        st.error("Please provide a Google Cloud Project ID")
        return False
        
    auth = GEEAuth()
    return auth.initialize(project_id, credentials_content)

def generate_month_list(start_date: datetime.date, end_date: datetime.date) -> List[tuple]:
    """Generate list of (year, month) tuples for the date range"""
    months = []
    current_date = start_date.replace(day=1)  # Start from first day of month
    
    while current_date <= end_date:
        months.append((current_date.year, current_date.month))
        # Add one month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return months

def create_chunks(bounds: List[float], max_pixels: int = 50000000) -> List[List[float]]:
    """Create chunks from bounds to avoid memory limits
    
    Args:
        bounds: [min_lon, min_lat, max_lon, max_lat]
        max_pixels: Maximum number of pixels per chunk
        
    Returns:
        List of chunk bounds
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Estimate area in degrees
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    # Estimate number of pixels at 500m resolution
    # Rough approximation: 1 degree ≈ 111 km
    approx_pixels = (lon_range * 111000) * (lat_range * 111000) / (500 * 500)
    
    if approx_pixels <= max_pixels:
        return [bounds]
    
    # Calculate number of chunks needed
    chunks_needed = math.ceil(approx_pixels / max_pixels)
    chunks_per_side = math.ceil(math.sqrt(chunks_needed))
    
    chunks = []
    lon_step = lon_range / chunks_per_side
    lat_step = lat_range / chunks_per_side
    
    for i in range(chunks_per_side):
        for j in range(chunks_per_side):
            chunk_min_lon = min_lon + i * lon_step
            chunk_max_lon = min_lon + (i + 1) * lon_step
            chunk_min_lat = min_lat + j * lat_step
            chunk_max_lat = min_lat + (j + 1) * lat_step
            
            chunks.append([chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat])
    
    return chunks

def download_chunk(calculator: SnowDepthCalculator, chunk_bounds: List[float], 
                  chunk_id: int, year: int, month: int, output_dir: str) -> str:
    """Download a single chunk for a specific month
    
    Args:
        calculator: SnowDepthCalculator instance
        chunk_bounds: [min_lon, min_lat, max_lon, max_lat]
        chunk_id: Unique identifier for the chunk
        year: Year for the calculation
        month: Month for the calculation
        output_dir: Directory to save the chunk
        
    Returns:
        Path to the downloaded file
    """
    # Create geometry from bounds
    geometry = ee.Geometry.Rectangle(chunk_bounds)
    
    # Calculate snow depth for specific month
    snow_depth = calculator.calculate_monthly_snow_depth(geometry, year, month)
    
    # Create download URL
    url = snow_depth.getDownloadURL({
        'scale': calculator.scale,
        'crs': 'EPSG:4326',
        'region': geometry,
        'format': 'GEO_TIFF'
    })
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()
    
    chunk_path = os.path.join(output_dir, f'snow_depth_{year}_{month:02d}_chunk_{chunk_id}.tif')
    with open(chunk_path, 'wb') as f:
        f.write(response.content)
    
    return chunk_path

def merge_chunks(chunk_paths: List[str], output_path: str):
    """Merge multiple GeoTIFF chunks into a single file"""
    if len(chunk_paths) == 1:
        # If only one chunk, just copy it
        import shutil
        shutil.copy2(chunk_paths[0], output_path)
        return
    
    # Open all chunk files
    src_files_to_mosaic = []
    for chunk_path in chunk_paths:
        src = rasterio.open(chunk_path)
        src_files_to_mosaic.append(src)
    
    # Merge the chunks
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Get metadata from the first chunk
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # Write the merged file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all source files
    for src in src_files_to_mosaic:
        src.close()

def process_monthly_data(calculator: SnowDepthCalculator, aoi_geom, 
                        months_list: List[tuple], progress_bar, status_text) -> bytes:
    """Process AOI and generate monthly snow depth data as a zip file"""
    
    # Get bounds from geometry
    if hasattr(aoi_geom, 'bounds'):
        bounds = list(aoi_geom.bounds().getInfo()['coordinates'][0])
        # Convert from polygon coordinates to bounds
        lons = [coord[0] for coord in bounds]
        lats = [coord[1] for coord in bounds]
        bounds = [min(lons), min(lats), max(lons), max(lats)]
    else:
        # Assume it's already in bounds format
        bounds = aoi_geom
    
    # Create chunks
    chunks = create_chunks(bounds)
    total_tasks = len(months_list) * len(chunks)
    current_task = 0
    
    # Create in-memory zip buffer
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Process each month
            for year, month in months_list:
                month_name = datetime.date(year, month, 1).strftime('%Y_%m')
                status_text.text(f"Processing {month_name} ({len(chunks)} chunks)...")
                
                chunk_paths = []
                
                # Process each chunk for this month
                for chunk_id, chunk_bounds in enumerate(chunks):
                    try:
                        current_task += 1
                        progress_bar.progress(current_task / total_tasks)
                        
                        chunk_path = download_chunk(
                            calculator, chunk_bounds, chunk_id, year, month, temp_dir
                        )
                        chunk_paths.append(chunk_path)
                        
                    except Exception as e:
                        st.warning(f"Error processing chunk {chunk_id+1} for {month_name}: {str(e)}")
                        continue
                
                if not chunk_paths:
                    st.warning(f"No chunks processed successfully for {month_name}")
                    continue
                
                # Merge chunks for this month
                status_text.text(f"Merging chunks for {month_name}...")
                monthly_output_path = os.path.join(temp_dir, f'snow_depth_{month_name}.tif')
                merge_chunks(chunk_paths, monthly_output_path)
                
                # Add to zip file
                status_text.text(f"Adding {month_name} to zip...")
                with open(monthly_output_path, 'rb') as f:
                    zip_file.writestr(f'snow_depth_{month_name}.tif', f.read())
                
                # Clean up chunk files
                for chunk_path in chunk_paths:
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
    
    return zip_buffer.getvalue()

def extract_bounds_from_tiff(tiff_file) -> tuple:
    """Extract bounds from uploaded TIFF file in EPSG:4326"""
    try:
        # Create a temporary file for rasterio to read
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            tmp_file.write(tiff_file.read())
            tmp_path = tmp_file.name
        
        # Reset file pointer if needed
        if hasattr(tiff_file, 'seek'):
            tiff_file.seek(0)
        
        with rasterio.open(tmp_path) as src:
            # Get bounds in the source CRS
            bounds = src.bounds
            crs = src.crs
            
            # Check if we need to transform to EPSG:4326
            needs_transform = False
            if crs is not None:
                try:
                    # Try different ways to check if it's already EPSG:4326
                    crs_string = str(crs).upper()
                    if 'EPSG:4326' not in crs_string and '4326' not in crs_string:
                        # Check if it's geographic (lat/lon) but different datum
                        if crs.is_geographic:
                            needs_transform = not (crs.to_epsg() == 4326)
                        else:
                            needs_transform = True
                except Exception as crs_error:
                    # Assume transformation is needed if we can't determine
                    needs_transform = True
            else:
                needs_transform = False
            
            # Transform bounds if needed
            if needs_transform and crs is not None:
                try:
                    from rasterio.warp import transform_bounds
                    bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
                except Exception as transform_error:
                    st.error(f"Error transforming coordinates: {transform_error}")
                    st.info("Using original bounds - please verify coordinate system")
            
            # Return as [min_lon, min_lat, max_lon, max_lat]
            # Handle both BoundingBox objects and tuples
            if hasattr(bounds, 'left'):
                # BoundingBox object
                result = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            else:
                # Tuple from transform_bounds: (left, bottom, right, top)
                result = [bounds[0], bounds[1], bounds[2], bounds[3]]
        
        # Clean up temporary file
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return result
            
    except Exception as e:
        st.error(f"Error reading TIFF file: {str(e)}")
        st.error(f"File type: {type(tiff_file)}")
        if hasattr(tiff_file, 'name'):
            st.error(f"File name: {tiff_file.name}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None

def resample_to_common_grid(reference_data, target_data, reference_transform, target_transform, 
                           reference_crs, target_crs, target_resolution=500):
    """Resample both datasets to a common grid for comparison"""
    
    # Calculate bounds in original coordinate systems
    ref_bounds = rasterio.transform.array_bounds(reference_data.shape[0], reference_data.shape[1], reference_transform)
    tgt_bounds = rasterio.transform.array_bounds(target_data.shape[0], target_data.shape[1], target_transform)
    
    # Transform reference bounds to EPSG:4326 for comparison
    from rasterio.warp import transform_bounds
    
    # Ensure we have a common CRS for comparison (EPSG:4326)
    common_crs = 'EPSG:4326'
    
    # Transform reference bounds to common CRS
    if str(reference_crs).upper() != 'EPSG:4326':
        ref_bounds_common = transform_bounds(reference_crs, common_crs, *ref_bounds)
    else:
        ref_bounds_common = ref_bounds
    
    # Transform target bounds to common CRS if needed
    if str(target_crs).upper() != 'EPSG:4326':
        tgt_bounds_common = transform_bounds(target_crs, common_crs, *tgt_bounds)
    else:
        tgt_bounds_common = tgt_bounds
    
    # Find intersection of bounds in common CRS
    min_x = max(ref_bounds_common[0], tgt_bounds_common[0])
    min_y = max(ref_bounds_common[1], tgt_bounds_common[1]) 
    max_x = min(ref_bounds_common[2], tgt_bounds_common[2])
    max_y = min(ref_bounds_common[3], tgt_bounds_common[3])
    
    # Check if bounds actually intersect
    if min_x >= max_x or min_y >= max_y:
        st.error("❌ No spatial overlap between datasets!")
        return None, None, None
    
    # Calculate common grid dimensions in the common CRS (EPSG:4326)
    lat_center = (min_y + max_y) / 2
    meters_per_degree_lon = 111000 * np.cos(np.radians(lat_center))
    meters_per_degree_lat = 111000
    
    resolution_lon = target_resolution / meters_per_degree_lon
    resolution_lat = target_resolution / meters_per_degree_lat
    
    width = int((max_x - min_x) / resolution_lon)
    height = int((max_y - min_y) / resolution_lat)
    
    # Ensure minimum dimensions
    if width <= 0 or height <= 0:
        width = max(1, width)
        height = max(1, height)
    
    st.info(f"🔄 Resampling to common grid: {width} x {height} pixels")
    
    # Create common transform in EPSG:4326
    common_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Resample reference data to common grid
    ref_resampled = np.empty((height, width), dtype=np.float32)
    reproject(
        reference_data,
        ref_resampled,
        src_transform=reference_transform,
        src_crs=reference_crs,
        dst_transform=common_transform,
        dst_crs=common_crs,
        resampling=Resampling.bilinear
    )
    
    # Resample target data to common grid
    tgt_resampled = np.empty((height, width), dtype=np.float32)
    reproject(
        target_data,
        tgt_resampled,
        src_transform=target_transform,
        src_crs=target_crs,
        dst_transform=common_transform,
        dst_crs=common_crs,
        resampling=Resampling.bilinear
    )
    
    return ref_resampled, tgt_resampled, common_transform

def calculate_comparison_metrics(reference_data, algorithm_data):
    """Calculate comparison metrics between reference and algorithm data"""
    
    # Remove NaN and invalid values
    valid_mask = (~np.isnan(reference_data)) & (~np.isnan(algorithm_data)) & \
                 (reference_data >= 0) & (algorithm_data >= 0)
    
    if not np.any(valid_mask):
        return None
    
    ref_valid = reference_data[valid_mask]
    alg_valid = algorithm_data[valid_mask]
    
    # Calculate metrics
    metrics = {}
    
    # Basic statistics
    metrics['reference_mean'] = np.mean(ref_valid)
    metrics['reference_std'] = np.std(ref_valid)
    metrics['algorithm_mean'] = np.mean(alg_valid)
    metrics['algorithm_std'] = np.std(alg_valid)
    
    # Comparison metrics
    metrics['correlation'], metrics['p_value'] = pearsonr(ref_valid, alg_valid)
    metrics['rmse'] = np.sqrt(mean_squared_error(ref_valid, alg_valid))
    metrics['mae'] = mean_absolute_error(ref_valid, alg_valid)
    metrics['bias'] = np.mean(alg_valid - ref_valid)
    metrics['relative_bias'] = metrics['bias'] / np.mean(ref_valid) * 100 if np.mean(ref_valid) > 0 else 0
    
    # R-squared
    ss_res = np.sum((ref_valid - alg_valid) ** 2)
    ss_tot = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
    metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Valid pixel count
    metrics['valid_pixels'] = len(ref_valid)
    metrics['total_pixels'] = len(reference_data.flatten())
    metrics['valid_percentage'] = (metrics['valid_pixels'] / metrics['total_pixels']) * 100
    
    return metrics

def create_comparison_plots(reference_data, algorithm_data, metrics):
    """Create comprehensive comparison plots with histograms"""
    if go is None:
        return None
    
    # Create subplots: 2x3 layout
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Model Snow Depth',
            'Our Algorithm Snow Depth',
            'Scatter Plot',
            'Model Snow Depth Histogram',
            'Our Algorithm Snow Depth Histogram',
            'Difference Map'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'bar'}, {'type': 'heatmap'}]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )
    
    # Valid data mask
    valid_mask = (~np.isnan(reference_data)) & (~np.isnan(algorithm_data)) & \
                 (reference_data >= 0) & (algorithm_data >= 0)
    
    # Reference data heatmap
    fig.add_trace(
        go.Heatmap(
            z=reference_data,
            colorscale='Blues',
            name='Reference',
            showscale=False,
            hovertemplate='Reference: %{z:.2f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Algorithm data heatmap
    fig.add_trace(
        go.Heatmap(
            z=algorithm_data,
            colorscale='Blues', 
            name='Algorithm',
            showscale=False,
            hovertemplate='Algorithm: %{z:.2f}m<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Scatter plot
    if np.any(valid_mask):
        ref_valid = reference_data[valid_mask]
        alg_valid = algorithm_data[valid_mask]
        
        # Sample data if too many points
        if len(ref_valid) > 10000:
            sample_idx = np.random.choice(len(ref_valid), 10000, replace=False)
            ref_sample = ref_valid[sample_idx]
            alg_sample = alg_valid[sample_idx]
        else:
            ref_sample = ref_valid
            alg_sample = alg_valid
        
        fig.add_trace(
            go.Scatter(
                x=ref_sample,
                y=alg_sample,
                mode='markers',
                marker=dict(size=3, opacity=0.6, color='blue'),
                name='Data Points',
                hovertemplate='Ref: %{x:.2f}m<br>Alg: %{y:.2f}m<extra></extra>'
            ),
            row=1, col=3
        )
        
        # Add 1:1 line
        max_val = max(np.nanmax(ref_sample), np.nanmax(alg_sample))
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='1:1 Line',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Histograms
        ref_hist = np.histogram(ref_valid, bins=50)
        alg_hist = np.histogram(alg_valid, bins=50)
        
        fig.add_trace(
            go.Bar(
                x=ref_hist[1][:-1],
                y=ref_hist[0],
                name='Reference',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=alg_hist[1][:-1],
                y=alg_hist[0],
                name='Algorithm',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Difference map
    difference = algorithm_data - reference_data
    fig.add_trace(
        go.Heatmap(
            z=difference,
            colorscale='RdBu',
            zmid=0,
            name='Difference',
            showscale=True,
            colorbar=dict(title='Difference (m)', x=1.02),
            hovertemplate='Difference: %{z:.2f}m<extra></extra>'
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Snow Depth Comparison Analysis<br><sub>R² = {metrics["r_squared"]:.3f}, RMSE = {metrics["rmse"]:.3f}m, Bias = {metrics["bias"]:.3f}m</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        height=700,
        showlegend=False,
        font=dict(size=11)
    )
    
    # Update subplot titles
    # Remove labels and ticks from heatmap subplots (row 1 col 1&2, row 2 col 3)
    fig.update_xaxes(title_text="", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="", showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="Our Algorithm Snow Depth (m)", row=1, col=3)
    fig.update_yaxes(title_text="Model Snow Depth (m)", row=1, col=3)
    fig.update_xaxes(title_text="Snow Depth (m)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Snow Depth (m)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    fig.update_xaxes(title_text="", showticklabels=False, row=2, col=3)
    fig.update_yaxes(title_text="", showticklabels=False, row=2, col=3)
    
    return fig

def download_algorithm_data(calculator, bounds, year, month):
    """Download algorithm data for comparison"""
    try:
        # Create geometry from bounds
        geometry = ee.Geometry.Rectangle(bounds)
        
        # Calculate snow depth for specific month
        snow_depth = calculator.calculate_monthly_snow_depth(geometry, year, month)
        
        # Create download URL
        url = snow_depth.getDownloadURL({
            'scale': calculator.scale,
            'crs': 'EPSG:4326',
            'region': geometry,
            'format': 'GEO_TIFF'
        })
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Use a different approach for Windows compatibility
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            tmp_path = tmp_file.name
        
        try:
            # Read the data after closing the file
            with rasterio.open(tmp_path) as src:
                algorithm_data = src.read(1)
                algorithm_transform = src.transform
                algorithm_crs = src.crs
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return algorithm_data, algorithm_transform, algorithm_crs
        
    except Exception as e:
        st.error(f"❌ Error downloading algorithm data: {str(e)}")
        return None, None, None

def create_density_settings_ui(prefix: str = "", default_seasonal: bool = True):
    """
    Create UI components for density settings including enhanced methods
    
    Args:
        prefix: Prefix for unique keys
        default_seasonal: Default value for seasonal density
        
    Returns:
        Tuple of (density_method, density_params)
        where density_params contains all necessary parameters for the selected method
    """
    density_method = st.selectbox(
        "Snow Density Calculation Method:",
        ["Original (Seasonal Look-up Table)", 
         "Enhanced Method 1: Inverse-fit γ per Month", 
         "Enhanced Method 2: Sinusoidal γ Fit"],
        index=0,
        key=f"{prefix}_density_method",
        help="Choose the method for calculating snow density"
    )
    
    density_params = {}
    
    if density_method == "Original (Seasonal Look-up Table)":
        # Original method UI
        density_option = st.radio(
            "Snow Density Configuration:",
            ["🗓️ Default Seasonal Values", "🎯 Custom Values"],
            index=0 if default_seasonal else 1,
            key=f"{prefix}_density_option",
            help="Choose how to configure snow density values"
        )
        
        use_seasonal_density = density_option == "🗓️ Default Seasonal Values"
        custom_density_value = None
        custom_monthly_densities = None
        
        if density_option == "🗓️ Default Seasonal Values":
            st.info("Using predefined month-specific density values based on snow metamorphism")
            
            # Show seasonal density table
            with st.expander("📊 View Default Seasonal Density Values"):
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                densities = [180, 220, 280, 350, 420, 400, 400, 400, 350, 280, 220, 200]
                
                density_df = pd.DataFrame({
                    'Month': months,
                    'Density (kg/m³)': densities,
                    'Snow Condition': [
                        'Fresh/settled snow', 'Settling snow', 'Aging snow', 'Spring conditions',
                        'Wet spring snow', 'Late season snow', 'Summer snow (rare)', 'Summer snow (rare)',
                        'Early season snow', 'New snow', 'Early winter', 'Winter snow'
                    ]
                })
                st.dataframe(density_df, width="stretch")
        
        else:  # Custom Values
            custom_type = st.radio(
                "Custom Density Type:",
                ["⚙️ Constant Value", "📅 Monthly Values"],
                key=f"{prefix}_custom_type",
                help="Choose between a single constant value or define values for each month"
            )
            
            if custom_type == "⚙️ Constant Value":
                custom_density_value = st.number_input(
                    "Constant Density (kg/m³):",
                    min_value=50.0,
                    max_value=600.0,
                    value=280.0,
                    step=10.0,
                    key=f"{prefix}_constant_density",
                    help="Enter a constant snow density value for all months"
                )
                st.info(f"Using constant density of {custom_density_value:.0f} kg/m³ for all months")
                
            else:  # Monthly Values
                st.info("Define custom density values for each month (initialized with default seasonal values)")
                
                # Default seasonal densities
                default_densities = [180, 220, 280, 350, 420, 400, 400, 400, 350, 280, 220, 200]
                months = ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
                
                custom_monthly_densities = {}
                
                # Create input fields in a compact layout
                col1, col2 = st.columns(2)
                
                for i, (month, default_value) in enumerate(zip(months, default_densities)):
                    with col1 if i % 2 == 0 else col2:
                        custom_monthly_densities[i + 1] = st.number_input(
                            f"{month[:3]} (kg/m³):",
                            min_value=50.0,
                            max_value=600.0,
                            value=float(default_value),
                            step=10.0,
                            key=f"{prefix}_month_{i+1}_density",
                            help=f"Density for {month}"
                        )
                
                # Show summary of custom values
                with st.expander("📊 Summary of Custom Monthly Densities"):
                    summary_df = pd.DataFrame({
                        'Month': [m[:3] for m in months],
                        'Custom Density (kg/m³)': [custom_monthly_densities[i+1] for i in range(12)],
                        'Default Density (kg/m³)': default_densities,
                        'Difference': [custom_monthly_densities[i+1] - default_densities[i] for i in range(12)]
                    })
                    st.dataframe(summary_df, width="stretch")
        
        density_params = {
            'method': 'original',
            'use_seasonal_density': use_seasonal_density,
            'custom_density_value': custom_density_value,
            'custom_monthly_densities': custom_monthly_densities
        }
    
    elif density_method == "Enhanced Method 1: Inverse-fit γ per Month":
        st.info("🔬 Enhanced Method 1: Calculate discrete γ values for each month using inverse-fit method")
        
        col1, col2 = st.columns(2)
        with col1:
            rho_min = st.number_input(
                "Minimum Density (ρ_min) kg/m³:",
                min_value=50.0,
                max_value=300.0,
                value=100.0,
                step=10.0,
                key=f"{prefix}_rho_min_method1",
                help="Minimum density value for the enhanced formula"
            )
        
        with col2:
            rho_max = st.number_input(
                "Maximum Density (ρ_max) kg/m³:",
                min_value=300.0,
                max_value=700.0,
                value=500.0,
                step=10.0,
                key=f"{prefix}_rho_max_method1",
                help="Maximum density value for the enhanced formula"
            )
        
        st.markdown("### 📊 Calculate Gamma Values for Each Month")
        
        # Initialize session state for gamma calculations
        if f'{prefix}_gamma_values_method1' not in st.session_state:
            st.session_state[f'{prefix}_gamma_values_method1'] = {}
        
        # Button to calculate gamma values
        if st.button("🧮 Calculate Gamma Values", key=f"{prefix}_calc_gamma_method1"):
            # Use default literature-based target densities for each month
            literature_densities = {
                1: 180, 2: 220, 3: 280, 4: 350, 5: 420, 6: 400,
                7: 400, 8: 400, 9: 350, 10: 280, 11: 220, 12: 200
            }
            
            # Use default FSC values (can be calculated from historical data)
            default_fsc_values = {
                1: 0.6, 2: 0.7, 3: 0.65, 4: 0.5, 5: 0.3, 6: 0.1,
                7: 0.05, 8: 0.05, 9: 0.2, 10: 0.4, 11: 0.5, 12: 0.55
            }
            
            gamma_values = {}
            for month in range(1, 13):
                gamma_values[month] = calculate_monthly_gamma_inverse_fit(
                    literature_densities[month],
                    default_fsc_values[month],
                    rho_min,
                    rho_max
                )
            
            st.session_state[f'{prefix}_gamma_values_method1'] = gamma_values
            st.success("✅ Gamma values calculated successfully!")
        
        # Display gamma values if calculated
        if st.session_state[f'{prefix}_gamma_values_method1']:
            gamma_values = st.session_state[f'{prefix}_gamma_values_method1']
            
            # Create gamma values table
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            gamma_df = pd.DataFrame({
                'Month': months,
                'Gamma (γ)': [gamma_values[i+1] for i in range(12)],
                'Literature Density (kg/m³)': [180, 220, 280, 350, 420, 400, 400, 400, 350, 280, 220, 200]
            })
            
            st.dataframe(gamma_df, width="stretch")
            
            # Create gamma parameters for the calculator
            gamma_params = {f'gamma_{month}': gamma_values[month] for month in range(1, 13)}
        else:
            gamma_params = {}
        
        density_params = {
            'method': 'enhanced_method1',
            'rho_min': rho_min,
            'rho_max': rho_max,
            'gamma_params': gamma_params
        }
    
    elif density_method == "Enhanced Method 2: Sinusoidal γ Fit":
        st.info("📈 Enhanced Method 2: Fit sinusoidal curve to γ values for smooth seasonal variation")
        
        col1, col2 = st.columns(2)
        with col1:
            rho_min = st.number_input(
                "Minimum Density (ρ_min) kg/m³:",
                min_value=50.0,
                max_value=300.0,
                value=100.0,
                step=10.0,
                key=f"{prefix}_rho_min_method2",
                help="Minimum density value for the enhanced formula"
            )
        
        with col2:
            rho_max = st.number_input(
                "Maximum Density (ρ_max) kg/m³:",
                min_value=300.0,
                max_value=700.0,
                value=500.0,
                step=10.0,
                key=f"{prefix}_rho_max_method2",
                help="Maximum density value for the enhanced formula"
            )
        
        st.markdown("### 📊 Calculate & Fit Seasonal γ Curve")
        
        # Initialize session state for curve fitting
        if f'{prefix}_fitted_params_method2' not in st.session_state:
            st.session_state[f'{prefix}_fitted_params_method2'] = {}
        
        # Button to calculate and fit gamma curve
        if st.button("🧮 Calculate & Fit Seasonal γ Curve", key=f"{prefix}_calc_fit_method2"):
            # Calculate discrete gamma values first
            literature_densities = {
                1: 180, 2: 220, 3: 280, 4: 350, 5: 420, 6: 400,
                7: 400, 8: 400, 9: 350, 10: 280, 11: 220, 12: 200
            }
            
            default_fsc_values = {
                1: 0.6, 2: 0.7, 3: 0.65, 4: 0.5, 5: 0.3, 6: 0.1,
                7: 0.05, 8: 0.05, 9: 0.2, 10: 0.4, 11: 0.5, 12: 0.55
            }
            
            monthly_gammas = []
            for month in range(1, 13):
                gamma = calculate_monthly_gamma_inverse_fit(
                    literature_densities[month],
                    default_fsc_values[month],
                    rho_min,
                    rho_max
                )
                monthly_gammas.append(gamma)
            
            # Fit sinusoidal curve
            alpha, beta, phi, r_squared = fit_sinusoidal_gamma_curve(monthly_gammas)
            
            fitted_params = {
                'alpha': alpha,
                'beta': beta,
                'phi': phi,
                'r_squared': r_squared,
                'monthly_gammas': monthly_gammas
            }
            
            st.session_state[f'{prefix}_fitted_params_method2'] = fitted_params
            st.success(f"✅ Sinusoidal curve fitted successfully! R² = {r_squared:.3f}")
        
        # Display fitted parameters and visualization
        if st.session_state[f'{prefix}_fitted_params_method2']:
            fitted_params = st.session_state[f'{prefix}_fitted_params_method2']
            
            # Display fitted parameters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("α (Offset)", f"{fitted_params['alpha']:.3f}")
            with col2:
                st.metric("β (Amplitude)", f"{fitted_params['beta']:.3f}")
            with col3:
                st.metric("φ (Phase)", f"{fitted_params['phi']:.3f}")
            with col4:
                st.metric("R²", f"{fitted_params['r_squared']:.3f}")
            
            # Create visualization if plotly is available
            if go is not None:
                months = np.arange(1, 13)
                fitted_curve = [sinusoidal_gamma_function(m, fitted_params['alpha'], 
                                                        fitted_params['beta'], fitted_params['phi']) 
                               for m in months]
                
                fig = go.Figure()
                
                # Add discrete gamma values
                fig.add_trace(go.Scatter(
                    x=months,
                    y=fitted_params['monthly_gammas'],
                    mode='markers',
                    name='Discrete γ values',
                    marker=dict(size=10, color='blue')
                ))
                
                # Add fitted curve
                fig.add_trace(go.Scatter(
                    x=months,
                    y=fitted_curve,
                    mode='lines',
                    name='Fitted sinusoidal curve',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title='Seasonal γ Variation: Discrete vs Fitted Curve',
                    xaxis_title='Month',
                    yaxis_title='Gamma (γ)',
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
            
            gamma_params = {
                'alpha': fitted_params['alpha'],
                'beta': fitted_params['beta'],
                'phi': fitted_params['phi']
            }
        else:
            gamma_params = {}
        
        density_params = {
            'method': 'enhanced_method2',
            'rho_min': rho_min,
            'rho_max': rho_max,
            'gamma_params': gamma_params
        }
    
    return density_method, density_params

def run_comparison(uploaded_tiff, bounds, year, month, comp_scale, density_method, density_params):
    """Run the comparison analysis"""
    
    with st.spinner("Running comparison analysis..."):
        try:
            # Read reference data using temporary file approach
            import tempfile
            import os
            
            # Create temporary file for reference data
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_file.write(uploaded_tiff.read())
                tmp_path = tmp_file.name
            
            # Reset file pointer if needed
            if hasattr(uploaded_tiff, 'seek'):
                uploaded_tiff.seek(0)
            
            try:
                with rasterio.open(tmp_path) as src:
                    reference_data = src.read(1)
                    reference_transform = src.transform
                    reference_crs = src.crs
                
                st.info(f"📊 Reference: {reference_data.shape} pixels at ~{abs(reference_transform[0]):.0f}m resolution")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            # Create calculator with appropriate parameters based on density method
            if density_params['method'] == 'original':
                calculator = SnowDepthCalculator(
                    scale=comp_scale,
                    use_seasonal_density=density_params.get('use_seasonal_density', True),
                    custom_density_value=density_params.get('custom_density_value'),
                    custom_monthly_densities=density_params.get('custom_monthly_densities'),
                    density_method="original"
                )
                
                # Show density information for original method
                density_value = get_seasonal_snow_density(
                    month, 
                    density_params.get('use_seasonal_density', True),
                    density_params.get('custom_density_value'),
                    density_params.get('custom_monthly_densities')
                )
                
                # Determine density type for display
                if density_params.get('custom_density_value') is not None:
                    density_type = "Custom Constant"
                elif density_params.get('custom_monthly_densities') is not None:
                    density_type = "Custom Monthly"
                elif density_params.get('use_seasonal_density', True):
                    density_type = "Default Seasonal"
                else:
                    density_type = "Default Constant"
                
                st.info(f"🏔️ Using {density_type} density: {density_value:.0f} kg/m³ for {datetime.date(2000, month, 1).strftime('%B')}")
                
            else:
                calculator = SnowDepthCalculator(
                    scale=comp_scale,
                    density_method=density_params['method'],
                    rho_min=density_params['rho_min'],
                    rho_max=density_params['rho_max'],
                    gamma_params=density_params['gamma_params']
                )
                
                # Show enhanced method information
                if density_params['method'] == 'enhanced_method1':
                    gamma_val = density_params['gamma_params'].get(f'gamma_{month}', 1.0)
                    st.info(f"🔬 Using Enhanced Method 1: γ = {gamma_val:.3f}, ρ_min = {density_params['rho_min']:.0f}, ρ_max = {density_params['rho_max']:.0f} kg/m³ for {datetime.date(2000, month, 1).strftime('%B')}")
                elif density_params['method'] == 'enhanced_method2':
                    alpha = density_params['gamma_params'].get('alpha', 1.0)
                    beta = density_params['gamma_params'].get('beta', 0.0)
                    phi = density_params['gamma_params'].get('phi', 1.0)
                    gamma_val = sinusoidal_gamma_function(month, alpha, beta, phi)
                    st.info(f"📈 Using Enhanced Method 2: γ(t) = {gamma_val:.3f}, ρ_min = {density_params['rho_min']:.0f}, ρ_max = {density_params['rho_max']:.0f} kg/m³ for {datetime.date(2000, month, 1).strftime('%B')}")
            
            algorithm_data, algorithm_transform, algorithm_crs = download_algorithm_data(
                calculator, bounds, year, month
            )
            
            if algorithm_data is None:
                st.error("Failed to download algorithm data")
                return
            
            # Resample to common grid
            ref_resampled, alg_resampled, common_transform = resample_to_common_grid(
                reference_data, algorithm_data, 
                reference_transform, algorithm_transform,
                reference_crs, algorithm_crs,
                target_resolution=comp_scale
            )
            
            if ref_resampled is None or alg_resampled is None:
                st.error("Failed to resample data to common grid")
                return
            
            # Calculate metrics
            metrics = calculate_comparison_metrics(ref_resampled, alg_resampled)
            
            if metrics is None:
                st.error("No valid data found for comparison")
                return
            
            # Store results in session state for display below
            st.session_state.comparison_results = {
                'ref_resampled': ref_resampled,
                'alg_resampled': alg_resampled,
                'metrics': metrics,
                'year': year,
                'month': month,
                'bounds': bounds,
                'resolution': comp_scale,
                'density_method': density_method,
                'density_params': density_params
            }
            
            st.success("✅ Comparison completed! Results displayed below.")
            
        except Exception as e:
            st.error(f"❌ Error during comparison: {str(e)}")
            st.exception(e)

def display_comparison_results():
    """Display comparison results from session state"""
    if 'comparison_results' not in st.session_state:
        return
        
    results = st.session_state.comparison_results
    ref_resampled = results['ref_resampled']
    alg_resampled = results['alg_resampled']
    metrics = results['metrics']
    
    # Full width container for results
    st.markdown("---")
    st.header("🔍 Comparison Results")
    
    # Metrics display in full width
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Correlation (R)", f"{metrics['correlation']:.3f}")
        st.metric("RMSE (m)", f"{metrics['rmse']:.3f}")
    
    with col2:
        st.metric("R² Score", f"{metrics['r_squared']:.3f}")
        st.metric("MAE (m)", f"{metrics['mae']:.3f}")
    
    with col3:
        st.metric("Bias (m)", f"{metrics['bias']:.3f}")
        st.metric("Relative Bias (%)", f"{metrics['relative_bias']:.1f}")
    
    with col4:
        st.metric("Valid Pixels", f"{metrics['valid_pixels']:,}")
        st.metric("Coverage (%)", f"{metrics['valid_percentage']:.1f}")
    
    # Statistical summary
    with st.expander("📈 Statistical Summary"):
        summary_df = pd.DataFrame({
            'Dataset': ['Reference', 'Algorithm'],
            'Mean (m)': [metrics['reference_mean'], metrics['algorithm_mean']],
            'Std Dev (m)': [metrics['reference_std'], metrics['algorithm_std']],
            'Min (m)': [np.nanmin(ref_resampled), np.nanmin(alg_resampled)],
            'Max (m)': [np.nanmax(ref_resampled), np.nanmax(alg_resampled)]
        })
        st.dataframe(summary_df, width="stretch")
    
    # Full width visualization
    if go is not None:
        fig = create_comparison_plots(ref_resampled, alg_resampled, metrics)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Install plotly for enhanced visualizations: `pip install plotly`")
    
    # Download section
    st.subheader("📥 Download Results")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    comparison_data = {
        'metrics': convert_numpy_types(metrics),
        'reference_stats': {
            'mean': float(metrics['reference_mean']),
            'std': float(metrics['reference_std']),
            'shape': list(ref_resampled.shape)
        },
        'algorithm_stats': {
            'mean': float(metrics['algorithm_mean']),
            'std': float(metrics['algorithm_std']),
            'shape': list(alg_resampled.shape)
        },
        'comparison_info': {
            'year': int(results['year']),
            'month': int(results['month']),
            'resolution': int(results['resolution']),
            'bounds': [float(b) for b in results['bounds']]
        }
    }
    
    comparison_json = json.dumps(comparison_data, indent=2)
    
    st.download_button(
        label="📥 Download Comparison Results (JSON)",
        data=comparison_json,
        file_name=f"snow_depth_comparison_{results['year']}_{results['month']:02d}.json",
        mime="application/json",
        width="stretch"
    )

def main():
    st.set_page_config(
        page_title="Snow Depth Calculator & Comparison Tool",
        page_icon="❄️",
        layout="wide"
    )
    
    st.title("❄️ Snow Depth Calculator & Comparison Tool")
    st.markdown("Calculate monthly snow depth using MODIS and GLDAS data from Google Earth Engine, with algorithm comparison capabilities")
    
    # Authentication Section
    st.header("🔐 Authentication")
    
    # Check if already authenticated
    ee_initialized = False
    if 'ee_initialized' in st.session_state:
        ee_initialized = st.session_state.ee_initialized
    
    if not ee_initialized:
        st.info("Please authenticate with Google Earth Engine to continue")
        
        # Create authentication form
        with st.form("auth_form"):
            st.subheader("Google Earth Engine Authentication")
            
            # Project ID input
            project_id = st.text_input(
                "Google Cloud Project ID *",
                placeholder="your-project-id",
                help="Enter your Google Cloud Project ID that has Earth Engine enabled"
            )
            
            # Credentials file upload
            credentials_file = st.file_uploader(
                "Upload Earth Engine Credentials File *",
                type=None,  # Accept any file type
                help="Upload your Earth Engine credentials file. This file may have no extension. See instructions below for how to obtain this file."
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Authenticate", type="primary")
            
            if submitted:
                if not project_id:
                    st.error("Please provide a Google Cloud Project ID")
                elif not credentials_file:
                    st.error("Please upload your credentials file")
                else:
                    # Read credentials file
                    credentials_content = credentials_file.read().decode('utf-8')
                    
                    # Attempt authentication
                    with st.spinner("Authenticating with Google Earth Engine..."):
                        if initialize_earth_engine(project_id, credentials_content):
                            st.success("✅ Successfully authenticated with Google Earth Engine!")
                            st.session_state.ee_initialized = True
                            st.session_state.project_id = project_id
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please check your credentials and try again.")
        
        # Instructions for getting credentials
        with st.expander("📋 How to get your credentials file"):
            st.markdown("""
            ### Step 1: Get Google Earth Engine Access
            1. Sign up for Google Earth Engine at [https://earthengine.google.com/](https://earthengine.google.com/)
            2. Wait for approval (this can take a few days for new users)
            
            ### Step 2: Create a Google Cloud Project
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the Earth Engine API for your project
            4. Note down your Project ID
            
            ### Step 3: Get Credentials File
            
            **Option A: Service Account Key (Recommended for applications)**
            1. In Google Cloud Console, go to "IAM & Admin" → "Service Accounts"
            2. Create a new service account or select an existing one
            3. Add the "Earth Engine Resource Viewer" role
            4. Create a JSON key for the service account
            5. Download the JSON file - this is your credentials file
            
            **Option B: User Credentials (For personal use)**
            1. Install Earth Engine CLI: `pip install earthengine-api`
            2. Run: `earthengine authenticate`
            3. Follow the authentication flow in your browser
            4. Find the credentials file at:
               - **Linux/Mac**: `~/.config/earthengine/credentials`
               - **Windows**: `C:\\Users\\[username]\\.config\\earthengine\\credentials`
            5. Upload this file
            
            ### Troubleshooting
            - Make sure your project has Earth Engine API enabled
            - Verify you have the necessary permissions
            - For service accounts, ensure the Earth Engine Resource Viewer role is assigned
            """)
        
        st.divider()
        st.markdown("*Complete authentication above to access the Snow Depth Calculator*")
        return
    
    # Show authenticated status
    st.success(f"✅ Authenticated with project: `{st.session_state.get('project_id', 'Unknown')}`")
    
    # Add logout button
    if st.button("🔓 Logout", help="Clear authentication and start over"):
        for key in ['ee_initialized', 'project_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Main application content starts here
    st.header("📊 Snow Depth Analysis & Comparison Tool")
    
    # Global sidebar for common parameters
    st.sidebar.header("🌍 Global Parameters")
    
    # Global Density settings - available for both modes
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏔️ Snow Density Settings")
    
    with st.sidebar:
        density_method, density_params = create_density_settings_ui("global")
    
    st.sidebar.markdown("---")
    
    # Mode selector
    mode = st.radio(
        "Choose mode:",
        ["📊 Monthly Analysis", "🔍 Algorithm Comparison"],
        horizontal=True,
        help="Select Monthly Analysis to process time series data, or Algorithm Comparison to validate against reference data"
    )
    
    if mode == "📊 Monthly Analysis":
        st.subheader("Monthly Snow Depth Analysis")
        
        # Sidebar for mode-specific parameters
        st.sidebar.header("📊 Monthly Analysis Parameters")
        
        # Date inputs
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.date(2020, 1, 1),
                max_value=datetime.date.today()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.date(2023, 12, 31),
                max_value=datetime.date.today()
            )
        
        # Scale input
        scale = st.sidebar.number_input(
            "Resolution (meters)",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            help="Pixel resolution in meters. Lower values = higher resolution but longer processing time."
        )
        
        # Validate dates
        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date")
            st.stop()
        
        # Generate month list and show info
        months_list = generate_month_list(start_date, end_date)
        st.sidebar.info(f"📅 **Total months to process:** {len(months_list)}")
        st.sidebar.markdown(f"**Date range:** {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Define Study Area")
            
            # AOI input method
            aoi_method = st.radio(
                "Choose method to define Area of Interest:",
                ["Draw on Map", "Upload JSON File"]
            )
            
            aoi_geom = None
            
            if aoi_method == "Draw on Map":
                st.markdown("**Instructions:** Use the drawing tools to create a polygon on the map")
                
                # Create folium map
                m = folium.Map(location=[40.0, -100.0], zoom_start=4)
                
                # Add drawing tools
                from folium.plugins import Draw
                draw = Draw(
                    export=True,
                    filename='data.geojson',
                    position='topleft',
                    draw_options={
                        'polyline': False,
                        'rectangle': True,
                        'polygon': True,
                        'circle': False,
                        'marker': False,
                        'circlemarker': False,
                    },
                    edit_options={'edit': False}
                )
                draw.add_to(m)
                
                # Display map
                map_data = st_folium(m, width=700, height=500)
                
                # Process drawn features
                if map_data['all_drawings']:
                    if len(map_data['all_drawings']) > 0:
                        # Get the last drawn feature
                        feature = map_data['all_drawings'][-1]
                        if feature['geometry']['type'] in ['Polygon', 'Rectangle']:
                            coords = feature['geometry']['coordinates'][0]
                            
                            # Convert to ee.Geometry
                            if feature['geometry']['type'] == 'Rectangle':
                                # For rectangles, we need to handle the coordinate format
                                lons = [coord[0] for coord in coords]
                                lats = [coord[1] for coord in coords]
                                aoi_geom = ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])
                            else:
                                aoi_geom = ee.Geometry.Polygon(coords)
                            
                            st.success("✅ Area of Interest defined!")
                        else:
                            st.warning("Please draw a polygon or rectangle")
            
            else:  # Upload JSON File
                uploaded_file = st.file_uploader(
                    "Upload GeoJSON file",
                    type=['json', 'geojson'],
                    help="Upload a GeoJSON file containing your area of interest"
                )
                
                if uploaded_file is not None:
                    try:
                        geojson_data = json.load(uploaded_file)
                        
                        # Extract geometry
                        if 'features' in geojson_data:
                            # FeatureCollection
                            feature = geojson_data['features'][0]
                            geometry = feature['geometry']
                        elif 'geometry' in geojson_data:
                            # Feature
                            geometry = geojson_data['geometry']
                        else:
                            # Geometry
                            geometry = geojson_data
                        
                        # Convert to ee.Geometry
                        aoi_geom = ee.Geometry(geometry)
                        st.success("✅ GeoJSON file uploaded successfully!")
                        
                        # Display uploaded area on map
                        bounds = aoi_geom.bounds().getInfo()
                        coords = bounds['coordinates'][0]
                        lons = [coord[0] for coord in coords]
                        lats = [coord[1] for coord in coords]
                        center_lat = (min(lats) + max(lats)) / 2
                        center_lon = (min(lons) + max(lons)) / 2
                        
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
                        folium.GeoJson(geojson_data).add_to(m)
                        st_folium(m, width=700, height=300)
                        
                    except Exception as e:
                        st.error(f"Error processing GeoJSON file: {str(e)}")
        
        with col2:
            st.subheader("Processing")
            
            if aoi_geom is not None:
                st.success("Ready to process!")
                
                # Show area info
                try:
                    area_km2 = aoi_geom.area().divide(1000000).getInfo()
                    st.metric("Area", f"{area_km2:.2f} km²")
                except:
                    st.info("Area calculation not available")
                
                # Show number of files that will be generated
                st.metric("Files to generate", f"{len(months_list)} GeoTIFF files")
                
                # Processing button
                if st.button("🚀 Generate Monthly Snow Depth", type="primary"):
                    # Create calculator with appropriate parameters based on density method
                    if density_params['method'] == 'original':
                        calculator = SnowDepthCalculator(
                            scale=scale,
                            use_seasonal_density=density_params['use_seasonal_density'],
                            custom_density_value=density_params['custom_density_value'],
                            custom_monthly_densities=density_params['custom_monthly_densities'],
                            density_method="original"
                        )
                    else:
                        calculator = SnowDepthCalculator(
                            scale=scale,
                            density_method=density_params['method'],
                            rho_min=density_params['rho_min'],
                            rho_max=density_params['rho_max'],
                            gamma_params=density_params['gamma_params']
                        )
                    
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Process and create zip
                        with st.spinner("Processing monthly data..."):
                            zip_data = process_monthly_data(
                                calculator, aoi_geom, months_list, progress_bar, status_text
                            )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        # Provide download button
                        filename = f"snow_depth_monthly_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}_{scale}m.zip"
                        st.download_button(
                            label="📥 Download Monthly Snow Depth Data (ZIP)",
                            data=zip_data,
                            file_name=filename,
                            mime="application/zip"
                        )
                        
                        st.success(f"Monthly snow depth calculation completed! Generated {len(months_list)} files.")
                        
                        # Show what's in the zip
                        with st.expander("📋 Files included in the ZIP"):
                            for year, month in months_list:
                                month_name = datetime.date(year, month, 1).strftime('%Y_%m')
                                st.text(f"snow_depth_{month_name}.tif")
                        
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        st.info("Try reducing the area size, date range, or increasing the resolution scale.")
            
            else:
                st.info("Please define an Area of Interest first")
                
    else:  # Comparison mode
        st.subheader("🔍 Algorithm Comparison Mode")
        
        # Sidebar for comparison-specific parameters
        st.sidebar.header("🔍 Comparison Parameters")
        
        comparison_year = st.sidebar.number_input(
            "Year",
            min_value=2000,
            max_value=2024,
            value=2020,
            help="Year for the comparison data"
        )
        
        comparison_month = st.sidebar.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: datetime.date(2000, x, 1).strftime('%B'),
            index=0,
            help="Month for the comparison data"
        )
        
        comp_scale = st.sidebar.selectbox(
            "Resolution (meters)",
            options=[250, 500, 1000],
            index=1,
            help="Resolution for comparison (lower = more accurate but slower)"
        )
        
        # Upload section - full width
        st.markdown("### Upload Reference Data")
        
        # File upload
        uploaded_tiff = st.file_uploader(
            "Upload Reference Snow Depth TIFF File",
            type=['tif', 'tiff'],
            help="Upload a GeoTIFF file containing reference snow depth data for comparison"
        )
        
        bounds = None
        if uploaded_tiff is not None:
            # Extract bounds from the uploaded file
            bounds = extract_bounds_from_tiff(uploaded_tiff)
            
            if bounds is not None:
                st.success("✅ Reference TIFF file uploaded successfully!")
                st.info(f"**Bounds:** {bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}")
                
                # Show current density information based on method
                if density_params['method'] == 'original':
                    density_value = get_seasonal_snow_density(
                        comparison_month, 
                        density_params.get('use_seasonal_density', True),
                        density_params.get('custom_density_value'),
                        density_params.get('custom_monthly_densities')
                    )
                    
                    # Determine density type for display
                    if density_params.get('custom_density_value') is not None:
                        density_type = "Custom Constant"
                    elif density_params.get('custom_monthly_densities') is not None:
                        density_type = "Custom Monthly"
                    elif density_params.get('use_seasonal_density', True):
                        density_type = "Default Seasonal"
                    else:
                        density_type = "Default Constant"
                    
                    density_info = f"{density_type} - {density_value:.0f} kg/m³"
                
                elif density_params['method'] == 'enhanced_method1':
                    density_type = "Enhanced Method 1"
                    gamma_val = density_params['gamma_params'].get(f'gamma_{comparison_month}', 1.0)
                    density_info = f"{density_type} - γ = {gamma_val:.3f}, ρ_min = {density_params['rho_min']:.0f}, ρ_max = {density_params['rho_max']:.0f} kg/m³"
                
                elif density_params['method'] == 'enhanced_method2':
                    density_type = "Enhanced Method 2"
                    alpha = density_params['gamma_params'].get('alpha', 1.0)
                    beta = density_params['gamma_params'].get('beta', 0.0)
                    phi = density_params['gamma_params'].get('phi', 1.0)
                    gamma_val = sinusoidal_gamma_function(comparison_month, alpha, beta, phi)
                    density_info = f"{density_type} - γ(t) = {gamma_val:.3f}, ρ_min = {density_params['rho_min']:.0f}, ρ_max = {density_params['rho_max']:.0f} kg/m³"
                
                else:
                    density_type = "Unknown"
                    density_info = "Unknown density method"
                
                st.info(f"🏔️ Current density setting: {density_info} for {datetime.date(2000, comparison_month, 1).strftime('%B')}")
                
                # Run comparison button - full width
                st.markdown("---")
                if st.button("🔍 Run Comparison", type="primary", width="stretch"):
                    run_comparison(
                        uploaded_tiff, bounds, comparison_year, comparison_month, comp_scale,
                        density_method, density_params
                    )
        else:
            st.info("📤 Please upload a reference TIFF file to begin comparison")
    
    # Display comparison results if they exist
    display_comparison_results()
    
    # Information section
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool provides **monthly snow depth analysis** and **algorithm comparison** capabilities:
        
        ## 📊 Monthly Analysis Mode
        
        **Data Sources:**
        - **MODIS Snow Cover** (MOD10A1): Normalized Difference Snow Index
        - **GLDAS Snow Water Equivalent**: NASA Global Land Data Assimilation System
        
        **Algorithm:**
        1. For each month in the selected date range:
           - Calculate Fractional Snow Cover (FSC) from MODIS NDSI
           - Estimate snow density based on FSC
           - Combine GLDAS SWE with FSC to get fine-resolution SWE
           - Calculate snow depth = SWE / snow density
        2. Package all monthly GeoTIFF files into a single ZIP archive
        
        **Output:**
        - One GeoTIFF file per month in the date range
        - Files are named: `snow_depth_YYYY_MM.tif`
        - All files packaged in a ZIP archive for download
        
        ## 🔍 Comparison Mode
        
        **Purpose:**
        - Validate algorithm performance against reference data
        - Compare snow depth estimates with ground truth measurements
        - Assess accuracy metrics and visualize differences
        
        **Features:**
        - Upload reference TIFF files for comparison
        - Automatic spatial alignment and resampling
        - Comprehensive metrics: R², RMSE, MAE, bias, correlation
        - Interactive visualization with scatter plots and difference maps
        - Statistical analysis and downloadable results
        
        **Notes:**
        - Large areas are automatically chunked to avoid memory limits
        - Processing time depends on area size, number of months, and resolution
        - Authentication with Google Earth Engine is required
        - Each monthly calculation uses the mean values for that specific month
        - Comparison mode requires reference data in GeoTIFF format
        """)

if __name__ == "__main__":
    main()