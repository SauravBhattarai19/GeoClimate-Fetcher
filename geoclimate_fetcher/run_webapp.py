#!/usr/bin/env python
"""
Run the GeoClimate Fetcher web application
"""
import os
import subprocess
import sys
import time

def main():
    """Run the Streamlit application"""
    print("Starting GeoClimate Fetcher Web Application...")
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed.")
    except ImportError:
        print("Streamlit is not installed. Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-web.txt"])
    
    # Check for geemap specifically since it's critical
    try:
        import geemap
        print("geemap is installed.")
    except ImportError:
        print("geemap is not installed. Installing geemap...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geemap"])
        print("Waiting for installation to complete...")
        time.sleep(2)  # Give a moment for installation to complete
    
    # Run the Streamlit application
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    print("Running command:", " ".join(streamlit_cmd))
    subprocess.run(streamlit_cmd)

if __name__ == "__main__":
    main() 