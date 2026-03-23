@echo off
title GeoClimate Fetcher
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Please run install-windows.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
streamlit run app.py
pause
