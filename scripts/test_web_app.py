#!/usr/bin/env python3
"""
🧪 HELMET DETECTION WEB APP TEST
===============================
Quick test script to verify the web application setup

Author: Muhammad Zein
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'flask', 'cv2', 'numpy', 'PIL', 'ultralytics'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'web_app.py',
        'templates/index.html',
        'requirements_web.txt',
        'runs/detect/train/weights/best.pt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_web_app():
    """Test if the Flask app can be created"""
    print("\n🌐 Testing Flask app...")
    
    try:
        from web_app import app, detector
        print("✅ Flask app created successfully")
        
        # Test if detector was initialized
        if detector.model is not None:
            print("✅ YOLO model loaded successfully")
        else:
            print("⚠️  YOLO model not loaded (this may cause detection issues)")
            
        return True
    except Exception as e:
        print(f"❌ Error creating Flask app: {e}")
        return False

def test_static_files():
    """Test if static files are accessible"""
    print("\n🎨 Testing static files...")
    
    static_dirs = [
        'static/assets/compiled/css',
        'static/assets/compiled/js',
        'static/assets/compiled/svg'
    ]
    
    missing_dirs = []
    
    for dir_path in static_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️  {dir_path} - Missing (may affect UI)")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) < len(static_dirs)  # Allow some missing

def main():
    """Run all tests"""
    print("🛡️ HELMET DETECTION WEB APP TEST")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Flask App Test", test_web_app),
        ("Static Files Test", test_static_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 40)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready to launch the web app!")
        print("\n🚀 To start the application, run:")
        print("   python web_app.py")
        print("   OR")
        print("   python launcher_web.py")
        print("\n📱 Then open: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if passed >= 2:  # At least imports and flask app work
            print("💡 The app might still work, but with limited functionality.")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
