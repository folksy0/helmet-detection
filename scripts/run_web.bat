@echo off
echo ========================================
echo    🛡️ HELMET DETECTION WEB DASHBOARD
echo    Professional Safety Monitoring System
echo ========================================
echo.

echo 🔧 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

echo 📦 Installing dependencies...
pip install -r requirements_web.txt

if errorlevel 1 (
    echo ❌ Error installing dependencies.
    echo 💡 Try: pip install --upgrade pip
    pause
    exit /b 1
)

echo ✅ Dependencies installed!
echo.

echo 🚀 Starting Helmet Detection Web Application...
echo 📱 Dashboard will be available at: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

python web_app.py

pause