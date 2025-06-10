# Helmet Detection Web Dashboard Launcher
# Professional Safety Monitoring System

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   🛡️ HELMET DETECTION WEB DASHBOARD" -ForegroundColor Green
Write-Host "   Professional Safety Monitoring System" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "🔧 Checking Python installation..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    Write-Host "📥 Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Blue
try {
    pip install -r requirements_web.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
    } else {
        throw "Installation failed"
    }
} catch {
    Write-Host "❌ Error installing dependencies." -ForegroundColor Red
    Write-Host "💡 Try: pip install --upgrade pip" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Start the web application
Write-Host "🚀 Starting Helmet Detection Web Application..." -ForegroundColor Green
Write-Host "📱 Dashboard will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    python web_app.py
} catch {
    Write-Host "❌ Error starting the application: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host ""
    Read-Host "Press Enter to exit"
}