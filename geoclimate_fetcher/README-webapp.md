# GeoClimate Fetcher Web Application

This web application provides a user-friendly interface for accessing and downloading climate and geospatial data from Google Earth Engine. It's built using Streamlit and the GeoClimate Fetcher library.

## Features

- Google Earth Engine authentication
- Interactive map for area of interest selection with drawing tools
- Dataset browsing and selection
- Band selection for each dataset
- Time range selection for temporal datasets
- Customizable download options
- Support for multiple file formats (GeoTIFF, NetCDF, CSV)

## Prerequisites

- Python 3.8 or higher
- Google Earth Engine account
- Google Cloud Project with Earth Engine API enabled
- GeoClimate Fetcher package installed

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/geoclimate-fetcher.git
   cd geoclimate-fetcher
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements-web.txt
   ```

5. Install the GeoClimate Fetcher package in development mode:
   ```
   pip install -e .
   ```

## Running the Application

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```
   
   Or use the provided helper script:
   ```
   python run_webapp.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Follow the steps in the application:
   - Authenticate with Google Earth Engine (you'll need your Google Cloud Project ID)
   - Select an area of interest (draw on map, upload GeoJSON, or enter coordinates)
   - Choose a dataset
   - Select bands
   - Specify a time range (if applicable)
   - Configure and download your data

## Google Earth Engine Authentication

The application requires authentication with Google Earth Engine. You will need:

1. A Google Earth Engine account (sign up at [https://earthengine.google.com/](https://earthengine.google.com/))
2. A Google Cloud Project with Earth Engine API enabled
3. Your Google Cloud Project ID

When you first run the application, you'll be prompted to enter your Google Cloud Project ID. You can also optionally use service account authentication if you have a service account set up for Earth Engine.

The application can save your credentials locally for convenience.

## Area of Interest Selection

The application offers three methods for selecting your area of interest:

1. **Draw on Map**: Use the drawing tools in the top right corner of the map to draw a rectangle or polygon around your area of interest.
2. **Upload GeoJSON**: Upload a GeoJSON file containing your area of interest.
3. **Enter Coordinates**: Manually enter the coordinates of a bounding box, with a live preview of the selected area.

## Troubleshooting

- If you encounter authentication issues, make sure you have a valid Google Earth Engine account and have completed the authentication process.
- Ensure your Google Cloud Project has the Earth Engine API enabled.
- For large areas or high-resolution data, downloads may take a significant amount of time.
- If you experience memory issues, try selecting a smaller area of interest or fewer bands.
- If the map drawing tools don't work properly, try using the coordinate input method instead.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 