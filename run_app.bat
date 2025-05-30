@echo off
echo 🌍 GeoClimate Fetcher - Starting Application...
echo ==================================================

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ⚙️ Setting up environment...
if not exist "downloads" mkdir downloads

echo ✅ All checks passed!
echo 🚀 Launching Streamlit application...
echo ==================================================
echo 📍 The app will open in your browser at: http://localhost:8501
echo ⏹️ Press Ctrl+C to stop the application
echo ==================================================

python run_app.py

pause 