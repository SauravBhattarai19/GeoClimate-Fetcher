@echo off
setlocal enabledelayedexpansion
title GeoClimate Fetcher - Installer
color 0A

echo.
echo ========================================================
echo    GeoClimate Intelligence Platform - Windows Installer
echo ========================================================
echo.
echo This will set up everything you need to run GeoClimate
echo Fetcher on your computer. Just sit back and follow any
echo prompts that appear.
echo.
echo ========================================================
echo.

REM --- Navigate to this script's directory ---
cd /d "%~dp0"

REM ========================================
REM  STEP 1: Check / Install Git
REM ========================================
echo [Step 1/6] Checking for Git...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo    Git is not installed. Attempting to install via winget...
    where winget >nul 2>nul
    if !errorlevel! neq 0 (
        echo.
        echo    ERROR: winget is not available on this system.
        echo    Please install Git manually from: https://git-scm.com/download/win
        echo    After installing Git, close this window and double-click this file again.
        echo.
        pause
        exit /b 1
    )
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    if !errorlevel! neq 0 (
        echo.
        echo    ERROR: Git installation failed.
        echo    Please install Git manually from: https://git-scm.com/download/win
        echo    After installing, close this window and double-click this file again.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo    Git was just installed. Your system PATH needs to refresh.
    echo    Please CLOSE this window and DOUBLE-CLICK this file again.
    echo.
    pause
    exit /b 0
) else (
    echo    Git is already installed.
)

REM ========================================
REM  STEP 2: Clone repository if needed
REM ========================================
echo.
echo [Step 2/6] Checking for project files...
if exist "%~dp0app.py" (
    echo    Project files found. Skipping clone.
) else if exist "%~dp0GeoClimate-Fetcher\app.py" (
    echo    Found existing GeoClimate-Fetcher folder. Using it.
    cd /d "%~dp0GeoClimate-Fetcher"
) else (
    echo    Project files not found. Cloning repository...
    if exist "%~dp0GeoClimate-Fetcher" (
        echo    Note: GeoClimate-Fetcher folder exists but appears incomplete.
        echo    Removing it and cloning fresh...
        rmdir /s /q "%~dp0GeoClimate-Fetcher"
    )
    git clone https://github.com/SauravBhattarai19/GeoClimate-Fetcher.git "%~dp0GeoClimate-Fetcher"
    if !errorlevel! neq 0 (
        echo.
        echo    ERROR: Failed to clone repository. Check your internet connection.
        echo.
        pause
        exit /b 1
    )
    cd /d "%~dp0GeoClimate-Fetcher"
    echo    Repository cloned successfully.
)

REM ========================================
REM  STEP 3: Check / Install Python 3.11+
REM ========================================
echo.
echo [Step 3/6] Checking for Python 3.11+...

set "PYTHON_CMD="

REM Try 'python' first
python --version >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
    for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
        if %%a geq 3 if %%b geq 11 set "PYTHON_CMD=python"
    )
)

REM Try 'python3' if python didn't work
if not defined PYTHON_CMD (
    python3 --version >nul 2>nul
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do set "PYVER=%%v"
        for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
            if %%a geq 3 if %%b geq 11 set "PYTHON_CMD=python3"
        )
    )
)

if not defined PYTHON_CMD (
    echo    Python 3.11+ not found. Attempting to install via winget...
    where winget >nul 2>nul
    if !errorlevel! neq 0 (
        echo.
        echo    ERROR: winget is not available on this system.
        echo    Please install Python 3.11+ from: https://www.python.org/downloads/
        echo    IMPORTANT: Check "Add Python to PATH" during installation!
        echo    After installing, close this window and double-click this file again.
        echo.
        pause
        exit /b 1
    )
    winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
    if !errorlevel! neq 0 (
        echo.
        echo    ERROR: Python installation failed.
        echo    Please install Python 3.11+ from: https://www.python.org/downloads/
        echo    IMPORTANT: Check "Add Python to PATH" during installation!
        echo    After installing, close this window and double-click this file again.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo    Python was just installed. Your system PATH needs to refresh.
    echo    Please CLOSE this window and DOUBLE-CLICK this file again.
    echo.
    pause
    exit /b 0
) else (
    echo    Found !PYTHON_CMD! (!PYVER!)
)

REM ========================================
REM  STEP 4: Create virtual environment
REM ========================================
echo.
echo [Step 4/6] Setting up virtual environment...
if exist ".venv\Scripts\activate.bat" (
    echo    Virtual environment already exists.
) else (
    echo    Creating virtual environment...
    !PYTHON_CMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo.
        echo    ERROR: Failed to create virtual environment.
        echo    Try running: !PYTHON_CMD! -m venv .venv
        echo.
        pause
        exit /b 1
    )
    echo    Virtual environment created.
)

REM Activate venv
call .venv\Scripts\activate.bat

REM ========================================
REM  STEP 5: Install dependencies
REM ========================================
echo.
echo [Step 5/6] Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo    Warning: pip upgrade failed, continuing anyway...
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo    ERROR: Dependency installation failed.
    echo    Try running manually:
    echo      .venv\Scripts\activate.bat
    echo      pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo    Dependencies installed successfully.

REM ========================================
REM  STEP 6: Earth Engine Authentication
REM ========================================
echo.
echo [Step 6/6] Google Earth Engine Authentication
echo.
echo ========================================================
echo  IMPORTANT: A browser window will open.
echo  Log in with the Google account you used to sign up
echo  for Google Earth Engine.
echo.
echo  Don't have an account? Sign up FREE at:
echo  https://earthengine.google.com/signup/
echo.
echo  You can skip this step by pressing Ctrl+C and
echo  authenticate later through the app.
echo ========================================================
echo.

REM Check if already authenticated
if exist "%USERPROFILE%\.config\earthengine\credentials" (
    echo    Earth Engine credentials found. Skipping authentication.
    echo    (Delete %USERPROFILE%\.config\earthengine\credentials to re-authenticate)
) else (
    earthengine authenticate
    if !errorlevel! neq 0 (
        echo.
        echo    Warning: Earth Engine authentication was not completed.
        echo    You can authenticate later by running: earthengine authenticate
        echo    Or upload credentials through the app interface.
    ) else (
        echo    Earth Engine authentication successful!
    )
)

REM ========================================
REM  CREATE DESKTOP LAUNCHER
REM ========================================
echo.
echo Creating Desktop launcher...

set "DESKTOP=%USERPROFILE%\Desktop"
set "APP_DIR=%cd%"
set "LAUNCHER=%DESKTOP%\GeoClimate Fetcher.bat"

(
    echo @echo off
    echo title GeoClimate Fetcher
    echo cd /d "%APP_DIR%"
    echo call .venv\Scripts\activate.bat
    echo streamlit run app.py
    echo pause
) > "%LAUNCHER%"

echo.
echo ========================================================
echo    Setup complete! Launching GeoClimate Fetcher...
echo    The app will open in your web browser shortly.
echo.
echo    A launcher has been saved to your Desktop:
echo    %LAUNCHER%
echo.
echo    Just double-click it anytime to start the app!
echo ========================================================
echo.

streamlit run app.py

pause
