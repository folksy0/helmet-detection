@echo off
echo.
echo ============================================
echo   🛡️ HELMET DETECTION SYSTEM - ULTRA FAST
echo ============================================
echo.
echo Starting the helmet detection application...
echo Target: 30+ FPS Real-time Detection
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import streamlit, cv2, ultralytics, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 📦 Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install packages
        pause
        exit /b 1
    )
)

echo.
echo ✅ All dependencies are ready
echo 🚀 Starting Streamlit application...
echo.
echo 🌐 The app will open in your default browser
echo 📍 URL: http://localhost:8501
echo.
echo 💡 Tips for best performance:
echo    - Ensure good lighting
echo    - Keep 1-3 meters distance from camera
echo    - Close other heavy applications
echo.

REM Start the Streamlit app
streamlit run app.py

echo.
echo 👋 Application closed. Press any key to exit...
pause >nul
