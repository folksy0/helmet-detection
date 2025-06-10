# 🛡️ Helmet Detection System - Professional Safety Monitoring

**Advanced AI-Powered Helmet Detection with Modern Web Dashboard**

A comprehensive helmet detection system built with YOLOv8 and Flask, featuring a professional web interface with real-time monitoring capabilities.

---

## 📋 Project Overview

This system provides:
- **🎯 High-Accuracy Detection** - YOLOv8-based helmet detection model
- **🌐 Professional Web Interface** - Modern admin dashboard with Mazer template
- **📹 Real-Time Monitoring** - Live webcam detection at 30+ FPS
- **📊 Advanced Analytics** - Detection statistics and performance metrics
- **🚀 Optimized Performance** - Multi-scale detection with smart filtering

---

## 🏗️ Project Structure

```
📁 helmetDetection/
├── 🧠 AI & Training
│   ├── bikehelmetdetection-yolov8n-training.ipynb  # Model training notebook
│   ├── helmet-dataset/                              # Training dataset
│   └── runs/detect/                                 # Trained models
│
├── 🌐 Web Application
│   ├── web_app.py                                   # Main Flask application
│   ├── templates/index.html                         # Web dashboard template
│   ├── static/                                     # CSS, JS, assets
│   └── requirements_web.txt                        # Python dependencies
│
├── 🛠️ Scripts & Tools
│   ├── scripts/launcher_web.py                     # Python launcher
│   ├── scripts/run_web.ps1                         # PowerShell launcher
│   ├── scripts/run_web.bat                         # Batch launcher
│   └── scripts/test_web_app.py                     # System tests
│
└── 📚 Documentation
    ├── docs/README.md                              # Main documentation
    └── docs/README_WEB.md                          # Web app guide
```

---

## 🚀 Quick Start

### **Option 1: PowerShell (Recommended for Windows)**
```powershell
# Navigate to project directory
cd "C:\Users\Muhammad Zein\Documents\KuliyeahProject\helmetDetection"

# Run the application
.\scripts\run_web.ps1
```

### **Option 2: Python Launcher**
```bash
# Run the Python launcher
python scripts\launcher_web.py
```

### **Option 3: Direct Launch**
```bash
# Install dependencies
pip install -r requirements_web.txt

# Start the web application
python web_app.py
```

### **Access the Dashboard**
Open your browser and navigate to: **http://localhost:5000**

---

## 🎯 Key Features

### **🔍 Detection Capabilities**
- **Multi-Scale Analysis** - Processes images at multiple resolutions for maximum accuracy
- **Real-Time Processing** - Ultra-fast webcam detection optimized for 30+ FPS
- **Smart Filtering** - AI-powered confidence boosting and noise reduction
- **Dual Mode Operation** - High-accuracy for images, ultra-fast for live video

### **📱 Professional Web Interface**
- **Modern Dashboard** - Built with Mazer admin template
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- **Real-Time Statistics** - Live detection counts and performance metrics
- **Interactive Controls** - Easy-to-use upload and webcam controls
- **Safety Alerts** - Instant warnings for non-compliance detection

### **⚡ Performance Optimization**
- **Adaptive Processing** - Automatically adjusts detection parameters
- **Background Threading** - Non-blocking webcam processing
- **Memory Efficient** - Optimized for continuous operation
- **GPU Acceleration** - Supports CUDA when available

---

## 📊 System Requirements

### **Minimum Requirements**
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **CPU**: Intel i3 or equivalent
- **Storage**: 2GB free space

### **Recommended Requirements**
- **RAM**: 8GB or more
- **CPU**: Intel i5 or equivalent
- **GPU**: NVIDIA GTX 1050+ (for GPU acceleration)
- **Webcam**: 720p or higher resolution

---

## 🛠️ Installation & Setup

### **1. Prerequisites**
Ensure you have Python 3.8+ installed:
```bash
python --version
```

### **2. Install Dependencies**
```bash
pip install -r requirements_web.txt
```

### **3. Verify Model**
Ensure the trained model exists:
```
runs/detect/train/weights/best.pt
```

### **4. Run System Test**
```bash
python scripts\test_web_app.py
```

### **5. Launch Application**
Use any of the quick start methods above.

---

## 💡 Usage Guide

### **📤 Image Detection (High Accuracy Mode)**
1. Click the **"📤 High-Accuracy Image Detection"** card
2. Upload an image via drag-and-drop or file selection
3. Click **"🔍 Analyze Image"** to process
4. View results with bounding boxes and confidence scores

### **🎥 Live Webcam Detection (Ultra-Fast Mode)**
1. Navigate to **"🎥 Ultra-Fast Live Detection"** section
2. Click **"🚀 Start Webcam"** to begin monitoring
3. Monitor real-time detection with FPS display
4. Use **"⏹️ Stop"** to end the session

### **📊 Analytics & Statistics**
- View real-time detection counts in dashboard cards
- Monitor system performance with FPS metrics
- Receive safety alerts for non-compliance
- Reset statistics using the reset button

---

## ⚙️ Configuration

### **Detection Settings**
- **Image Mode**: 30% confidence threshold (high accuracy)
- **Webcam Mode**: 70% confidence threshold (optimized speed)
- **Processing Sizes**: 320px (webcam), 640-832px (images)
- **Frame Rate**: Target 30+ FPS for webcam

### **Performance Tuning**
The system automatically optimizes based on mode:

**🎯 High-Accuracy Mode (Images):**
- Multi-scale detection (640px, 832px)
- Advanced preprocessing (CLAHE, bilateral filtering)
- Smart NMS filtering
- Domain knowledge confidence boosting

**⚡ Ultra-Fast Mode (Webcam):**
- Single-scale detection (320px)
- Minimal preprocessing
- Optimized for real-time performance
- Frame skipping when needed

---

## 🔧 API Documentation

The Flask application provides RESTful endpoints:

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | Main dashboard page |
| `/upload_image` | POST | Analyze uploaded image |
| `/start_webcam` | GET | Start webcam detection |
| `/stop_webcam` | GET | Stop webcam detection |
| `/webcam_feed` | GET | Live video stream |
| `/webcam_stats` | GET | Current detection stats |

---

## 🐛 Troubleshooting

### **Common Issues & Solutions**

#### **"Model not found" Error**
```bash
# Verify model file exists
ls runs/detect/train/weights/best.pt

# If missing, run the training notebook first
```

#### **Webcam Issues**
- Ensure no other applications are using the camera
- Grant camera permissions in your browser
- Try refreshing the web page
- Check browser console for detailed errors

#### **Performance Issues**
- Close resource-intensive applications
- Ensure adequate lighting for better detection
- Consider reducing browser zoom level
- Verify hardware acceleration is enabled

#### **Installation Problems**
```bash
# Update pip and reinstall
pip install --upgrade pip
pip install -r requirements_web.txt --force-reinstall
```

---

## 🧪 Testing

Run the comprehensive test suite:
```bash
python scripts\test_web_app.py
```

This will verify:
- ✅ Python environment and dependencies
- ✅ File structure integrity
- ✅ Flask application functionality
- ✅ Model loading capability

---

## 📈 Performance Metrics

### **Detection Speed**
- **Image Processing**: 0.5-2 seconds (depending on size)
- **Webcam FPS**: 30+ (excellent), 25+ (good), 15+ (acceptable)
- **Memory Usage**: ~500MB-1GB during operation
- **CPU Usage**: 20-60% (varies with detection frequency)

### **Accuracy Metrics**
- **Precision**: High accuracy with multi-scale detection
- **Recall**: Optimized for safety-critical applications
- **Confidence Thresholds**: Adjustable per use case

---

## 🛡️ Safety & Privacy

### **Data Security**
- **Local Processing**: All detection happens on your device
- **No Data Storage**: Images are not saved or transmitted
- **Privacy First**: No external data transmission
- **Secure Access**: Local network only (localhost)

### **Safety Features**
- **Real-Time Alerts**: Instant warnings for safety violations
- **Visual Indicators**: Clear color-coded detection results
- **Compliance Monitoring**: Continuous safety status tracking

---

## 🤝 Contributing

This project is part of the academic research initiative for workplace safety monitoring. 

### **Development Setup**
```bash
# Clone and setup development environment
git clone [repository]
cd helmetDetection
pip install -r requirements_web.txt

# Run in development mode
python web_app.py
```

---

## 📄 License

This project is developed for academic and research purposes as part of the KuliyeahProject safety monitoring initiative.

---

## 👨‍💻 Author

**Muhammad Zein**  
📧 Email: [Your Email]  
🎓 Institution: [Your Institution]  
📅 Project: Helmet Detection System v4.0  
🗓️ Last Updated: June 2025

---

## 🎯 Next Steps

1. **Launch the Application**: Use any of the quick start methods
2. **Access Dashboard**: Navigate to http://localhost:5000
3. **Test Detection**: Try both image upload and webcam modes
4. **Monitor Performance**: Check FPS and accuracy metrics
5. **Explore Features**: Use all dashboard functionalities

---

### 🚀 **Ready to Start?**

Choose your preferred launch method and start monitoring safety compliance with our professional helmet detection system!

**Stay Safe! 🛡️**
