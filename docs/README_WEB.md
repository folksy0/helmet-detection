# 🛡️ Helmet Detection Web Dashboard

**Professional Safety Monitoring System with Modern Web Interface**

A powerful web-based helmet detection system that replaces the original Streamlit interface with a professional admin dashboard using the Mazer template. Features real-time webcam monitoring, high-accuracy image analysis, and comprehensive safety analytics.

## ✨ Features

### 🎯 **Detection Capabilities**
- **High-Accuracy Image Analysis** - Upload images for detailed helmet detection
- **Ultra-Fast Webcam Monitoring** - Real-time detection at 30+ FPS
- **Multi-Scale Detection** - Advanced preprocessing for better accuracy
- **Smart Filtering** - AI-powered noise reduction and confidence boosting

### 📊 **Professional Dashboard**
- **Modern Admin Interface** - Built with Mazer template
- **Real-Time Statistics** - Live detection counts and performance metrics
- **Performance Monitoring** - FPS tracking and optimization tips
- **Responsive Design** - Works on desktop, tablet, and mobile

### 🚀 **Performance Optimized**
- **Ultra-Fast Mode** - Optimized for 30+ FPS webcam detection
- **Smart Processing** - Adaptive resolution and frame skipping
- **Background Threading** - Non-blocking webcam processing
- **Memory Efficient** - Optimized for continuous operation

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- Webcam (for live detection)
- Trained YOLO model (`runs/detect/train/weights/best.pt`)

### **Quick Start**

#### **Option 1: Using PowerShell (Recommended for Windows)**
```powershell
# Navigate to project directory
cd "C:\Users\Muhammad Zein\Documents\KuliyeahProject\helmetDetection"

# Run the PowerShell launcher
.\run_web.ps1
```

#### **Option 2: Using Batch File**
```batch
# Double-click run_web.bat or run in Command Prompt
run_web.bat
```

#### **Option 3: Using Python Launcher**
```bash
# Navigate to project directory
cd "C:\Users\Muhammad Zein\Documents\KuliyeahProject\helmetDetection"

# Run the Python launcher
python launcher_web.py
```

#### **Option 4: Manual Setup**
```bash
# Install dependencies
pip install -r requirements_web.txt

# Start the web application
python web_app.py
```

### **Access the Dashboard**
After starting the application, open your web browser and go to:
```
http://localhost:5000
```

## 📱 How to Use

### **1. Image Detection (High Accuracy)**
1. Click on the **"📤 High-Accuracy Image Detection"** card
2. Upload an image by clicking the upload area or dragging & dropping
3. Click **"🔍 Analyze Image"** to process
4. View detection results with bounding boxes and confidence scores

### **2. Live Webcam Detection (Ultra-Fast)**
1. Go to the **"🎥 Ultra-Fast Live Detection"** section
2. Click **"🚀 Start Webcam"** to begin real-time monitoring
3. View live detection feed with FPS counter
4. Click **"⏹️ Stop"** to end the session

### **3. Statistics & Analytics**
- **Real-time Stats** - View detection counts in the dashboard cards
- **Performance Metrics** - Monitor FPS and system performance
- **Safety Alerts** - Get warnings when people without helmets are detected
- **Reset Statistics** - Clear all counts using the reset button

## 🎛️ Configuration

### **Detection Settings**
- **Image Detection Confidence**: 30% (high accuracy)
- **Webcam Detection Confidence**: 70% (optimized for speed)
- **Webcam Resolution**: 640x480 (optimized for performance)
- **Processing Size**: 320px for webcam, 640-832px for images

### **Performance Optimization**
The system automatically optimizes based on mode:

**Image Mode (High Accuracy):**
- Multi-scale detection (640px, 832px)
- Advanced preprocessing (CLAHE, bilateral filtering)
- Smart filtering and NMS
- Domain knowledge boosting

**Webcam Mode (Ultra-Fast):**
- Single-scale detection (320px)
- Minimal preprocessing
- Optimized for 30+ FPS
- Frame skipping for performance

## 📁 Project Structure

```
helmetDetection/
├── web_app.py              # Main Flask application
├── launcher_web.py         # Python launcher script
├── run_web.ps1            # PowerShell launcher
├── run_web.bat            # Batch launcher
├── requirements_web.txt    # Web app dependencies
├── templates/
│   └── index.html         # Main dashboard template
├── static/                # Static assets (CSS, JS, images)
│   └── assets/
├── runs/detect/train/weights/
│   └── best.pt           # Trained YOLO model
└── README_WEB.md         # This file
```

## 🔧 API Endpoints

The Flask application provides the following endpoints:

- `GET /` - Main dashboard page
- `POST /upload_image` - Upload and analyze image
- `GET /start_webcam` - Start webcam detection
- `GET /stop_webcam` - Stop webcam detection
- `GET /webcam_feed` - Live webcam stream
- `GET /webcam_stats` - Current detection statistics

## 🎨 UI Features

### **Dashboard Components**
- **Statistics Cards** - Real-time detection counts with animated counters
- **Performance Monitor** - FPS display with color-coded status
- **Upload Area** - Drag & drop interface with visual feedback
- **Live Stream** - Real-time webcam feed with overlay detection
- **Alert System** - Safety warnings and status notifications

### **Responsive Design**
- **Desktop** - Full dashboard with sidebar navigation
- **Tablet** - Optimized layout with collapsible sidebar
- **Mobile** - Touch-friendly interface with bottom navigation

### **Theme Support**
- **Light Theme** - Clean, professional appearance
- **Dark Theme** - Eye-friendly for low-light environments
- **Auto Switch** - Automatic theme detection

## 🚀 Performance Tips

### **For Maximum FPS:**
- ✅ **Good Lighting** - Helps faster detection
- ✅ **Optimal Distance** - 1-3 meters from camera
- ✅ **Simple Background** - Reduces processing noise
- ✅ **Close Other Apps** - Free up CPU/memory
- ✅ **Use Chrome/Edge** - Better WebRTC performance

### **System Requirements:**
- **Minimum**: 4GB RAM, Intel i3 or equivalent
- **Recommended**: 8GB RAM, Intel i5 or equivalent, dedicated GPU
- **Optimal**: 16GB RAM, Intel i7 or equivalent, NVIDIA GTX 1050+

## 🛡️ Safety Features

### **Detection Alerts**
- **Visual Warnings** - Red alerts for people without helmets
- **Audio Notifications** - Optional sound alerts (browser-dependent)
- **Real-time Status** - Live safety compliance monitoring

### **Data Privacy**
- **Local Processing** - All detection happens on your device
- **No Data Storage** - Images are not saved or transmitted
- **Secure Connection** - Local network only (localhost)

## 🐛 Troubleshooting

### **Common Issues:**

#### **"Model not found" Error**
```bash
# Make sure the trained model exists
ls runs/detect/train/weights/best.pt

# If missing, train the model first using the Jupyter notebook
```

#### **Webcam Not Working**
- Check if camera is being used by another application
- Grant camera permissions in browser
- Try refreshing the page
- Check browser console for errors

#### **Low FPS Performance**
- Close other applications using CPU/memory
- Ensure good lighting conditions
- Try reducing browser zoom level
- Check if GPU acceleration is enabled

#### **Installation Issues**
```bash
# Update pip first
pip install --upgrade pip

# Install with specific versions
pip install -r requirements_web.txt --force-reinstall
```

## 📊 Technical Specifications

### **AI Model**
- **Architecture**: YOLOv8n (Nano) - Optimized for speed
- **Classes**: 2 (with helmet, without helmet)
- **Input Size**: Variable (320px for webcam, 640-832px for images)
- **Framework**: Ultralytics PyTorch

### **Web Technology**
- **Backend**: Flask 2.3.3
- **Frontend**: HTML5, CSS3, JavaScript ES6
- **Template**: Mazer Admin Dashboard
- **Streaming**: WebRTC/HTTP streaming
- **Icons**: Bootstrap Icons, Iconly

### **Performance Metrics**
- **Webcam FPS**: 30+ (target), 25+ (good), 15+ (acceptable)
- **Image Processing**: 0.5-2 seconds depending on image size
- **Memory Usage**: ~500MB-1GB during operation
- **CPU Usage**: 20-60% depending on detection frequency

## 🤝 Contributing

This project is part of the KuliyeahProject helmet detection system. The web interface enhances the original Streamlit application with a professional dashboard experience.

### **Development Setup**
```bash
# Clone/navigate to project
cd helmetDetection

# Install development dependencies
pip install -r requirements_web.txt

# Run in development mode
python web_app.py
```

## 📝 License

This project is part of the academic KuliyeahProject for safety monitoring and helmet detection research.

## 👨‍💻 Author

**Muhammad Zein**  
Project: Helmet Detection System  
Version: 4.0 (Web Dashboard)  
Date: June 2025

---

### 🚀 **Ready to Start?**

Run the application using any of the launcher scripts and access the professional dashboard at `http://localhost:5000`

**Stay Safe! 🛡️**
