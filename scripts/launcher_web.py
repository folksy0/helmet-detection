#!/usr/bin/env python3
"""
🚀 HELMET DETECTION WEB LAUNCHER
===============================
Simple launcher script for the helmet detection web dashboard
Checks system requirements and launches the Flask application

Author: Muhammad Zein
Version: 1.0
"""

import sys
import subprocess
import pkg_resources
import webbrowser
import time
import os
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 50)
    print("🛡️  HELMET DETECTION WEB DASHBOARD")
    print("   Professional Safety Monitoring System")
    print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔧 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible!")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Checking dependencies...")
    
    requirements_file = Path("requirements_web.txt")
    if not requirements_file.exists():
        print("❌ requirements_web.txt not found")
        return False
    
    try:
        # Check if packages are already installed
        with open(requirements_file, 'r') as f:
            requirements = f.read().splitlines()
        
        missing_packages = []
        for requirement in requirements:
            if requirement.strip() and not requirement.startswith('#'):
                package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0]
                try:
                    pkg_resources.get_distribution(package_name)
                except pkg_resources.DistributionNotFound:
                    missing_packages.append(requirement)
        
        if missing_packages:
            print(f"📥 Installing {len(missing_packages)} missing packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"
            ])
        
        print("✅ All dependencies are ready!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 Try: pip install --upgrade pip")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_model_file():
    """Check if the trained model exists"""
    print("\n🤖 Checking AI model...")
    
    model_path = Path("runs/detect/train/weights/best.pt")
    if not model_path.exists():
        print("⚠️  Warning: Trained model not found at runs/detect/train/weights/best.pt")
        print("   The application will try to load the model, but detection may not work")
        print("   Please ensure you have trained the model first")
        return False
    
    print("✅ AI model found!")
    return True

def launch_web_app():
    """Launch the Flask web application"""
    print("\n🚀 Starting Helmet Detection Web Application...")
    print("📱 Dashboard will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    # Launch the web app
    try:
        # Wait a moment then open browser
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Import and run the Flask app
        from web_app import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except ImportError as e:
        print(f"❌ Error importing web application: {e}")
        print("   Make sure web_app.py is in the current directory")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check system requirements
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    if not install_requirements():
        input("Press Enter to exit...")
        return
    
    # Check model (warning only)
    check_model_file()
    
    # Launch application
    try:
        launch_web_app()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()