# Helmet Detection System - Quick Launcher
# Professional Safety Monitoring Dashboard

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  🛡️ HELMET DETECTION SYSTEM - QUICK LAUNCHER" -ForegroundColor Green
Write-Host "  Professional Safety Monitoring Dashboard" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "🚀 Launching Helmet Detection Web Application..." -ForegroundColor Blue
Write-Host "📱 Dashboard will open at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""

try {
    python scripts\launcher_web.py
} catch {
    Write-Host "❌ Error launching application: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running: python scripts\launcher_web.py" -ForegroundColor Yellow
} finally {
    Read-Host "Press Enter to exit"
}
