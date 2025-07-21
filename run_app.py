#!/usr/bin/env python3
"""
Simple launcher script for the GeoClimate Fetcher Streamlit app
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'earthengine-api',
        'geemap',
        'folium',
        'streamlit-folium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements-streamlit.txt")
        return False
    
    return True

def setup_environment():
    """Set up the environment"""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Create downloads directory if it doesn't exist
    downloads_dir = current_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    return True

def main():
    """Main launcher function"""
    print("🌍 GeoClimate Fetcher - Starting Application...")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Setup environment
    print("⚙️  Setting up environment...")
    if not setup_environment():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Launching Streamlit application...")
    print("=" * 50)
    print("📍 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 