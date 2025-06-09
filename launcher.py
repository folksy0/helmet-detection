#!/usr/bin/env python3
"""
🛡️ HELMET DETECTION SYSTEM - LAUNCHER
=====================================
Simple launcher script for the helmet detection application.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'opencv-python',
        'ultralytics', 
        'pillow',
        'numpy',
        'torch',
        'torchvision'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("📦 Installing dependencies from requirements.txt...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        return result.returncode == 0
    else:
        print("❌ requirements.txt not found")
        return False

def main():
    """Main launcher function."""
    print("=" * 50)
    print("🛡️  HELMET DETECTION SYSTEM - ULTRA FAST")
    print("=" * 50)
    print()
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    print("✅ Python version check passed")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            input("Press Enter to exit...")
            return
        print("✅ Dependencies installed successfully")
    else:
        print("✅ All dependencies are available")
    
    print()
    print("🚀 Starting Helmet Detection Application...")
    print("🌐 App will open at: http://localhost:8501")
    print()
    print("💡 Performance Tips:")
    print("   - Ensure good lighting")
    print("   - Keep 1-3 meters from camera")
    print("   - Close other heavy applications")
    print("   - Target FPS: 30+")
    print()
    
    # Launch Streamlit app
    app_file = Path(__file__).parent / "app.py"
    if not app_file.exists():
        print("❌ app.py not found")
        input("Press Enter to exit...")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
