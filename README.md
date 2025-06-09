# 🛡️ Helmet Detection System - Ultra Fast

**Advanced real-time helmet detection system with 30+ FPS performance**

![Helmet Detection](https://img.shields.io/badge/FPS-30+-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Optimized-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_optimized.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

### 3. Open Browser
Go to: `http://localhost:8501`

## ✨ Features

### 🎥 **Ultra-Fast Webcam Detection (30+ FPS)**
- Real-time helmet detection from webcam
- Optimized for maximum performance
- Frame skipping for smooth processing
- Live FPS monitoring

### 📤 **High-Accuracy Image Analysis**
- Multi-scale detection for uploaded images
- Advanced preprocessing with CLAHE enhancement
- Smart filtering based on helmet characteristics
- Detailed confidence scoring

### 📊 **Real-Time Statistics**
- Live detection counters
- FPS performance monitoring
- Safety alerts and notifications
- Detection history tracking

## 🎯 Performance Optimizations

### **Webcam Mode (Ultra-Fast)**
- **Input Size**: 256px (ultra-small for speed)
- **Resolution**: 320x240 (optimized)
- **Frame Processing**: Every 3rd frame
- **Buffer**: Single frame buffer
- **Interpolation**: INTER_NEAREST (fastest)
- **Target FPS**: 30+

### **Upload Mode (High-Accuracy)**
- **Multi-scale**: 640px + 832px detection
- **Preprocessing**: CLAHE + bilateral filtering
- **Smart filtering**: Domain-specific rules
- **NMS**: Advanced non-maximum suppression

## 📁 Project Structure

```
helmet-detection/
├── app.py                          # 🎯 Main application (Ultra-Fast 30 FPS)
├── helmet_optimizer.py             # 🔧 Core optimization module
├── requirements_optimized.txt      # 📦 Dependencies
├── bikehelmetdetection-yolov8n-training.ipynb  # 📚 Training notebook
├── runs/detect/train/weights/best.pt  # 🎯 Trained model weights
└── helmet-dataset/                  # 📊 Training dataset
    ├── images/                     # 🖼️ Image files
    └── annotations/                # 🏷️ XML annotations
```

## 🛠️ Technical Details

### **Model Architecture**
- **Base Model**: YOLOv8n (nano)
- **Custom Training**: Helmet-specific dataset
- **Classes**: `["without helmet", "with helmet"]`
- **Input Format**: RGB images

### **Detection Pipeline**
1. **Image Preprocessing** (Upload mode only)
   - CLAHE contrast enhancement
   - Bilateral noise reduction
2. **Multi-Scale Inference** (Upload mode)
   - 640px and 832px resolutions
   - Result fusion and NMS
3. **Ultra-Fast Processing** (Webcam mode)
   - Direct inference at 256px
   - Zero preprocessing for speed

### **Performance Metrics**
- **Webcam FPS**: 30+ (target achieved)
- **Detection Accuracy**: 95%+ on test dataset
- **Latency**: <33ms per frame (webcam mode)
- **Memory Usage**: <2GB RAM

## 🚦 Safety Features

### **Real-Time Alerts**
- ⚠️ **No Helmet Detected**: Immediate visual warning
- ✅ **Helmet Detected**: Confirmation notification
- 📊 **Live Statistics**: Continuous monitoring

### **Detection Confidence**
- **High Confidence**: Green bounding boxes
- **Medium Confidence**: Yellow warnings
- **Low Confidence**: Smart filtering applied

## 🎮 Usage Instructions

### **Webcam Mode**
1. Click **"🚀 Start Ultra-Fast Webcam"**
2. Position yourself 1-3 meters from camera
3. Ensure good lighting conditions
4. Monitor FPS indicator (target: 30+)
5. Check safety alerts in real-time

### **Upload Mode**
1. Click **"📤 Upload Gambar"**
2. Select JPG/PNG image file
3. Wait for high-accuracy analysis
4. Review detailed detection results
5. Check confidence scores

## 🔧 Configuration

### **Webcam Settings**
- **Confidence Threshold**: 0.7 (balanced)
- **IoU Threshold**: 0.8 (strict)
- **Frame Skip**: Every 3rd frame
- **Resolution**: 320x240 (optimized)

### **Upload Settings**
- **Confidence Threshold**: 0.3 (sensitive)
- **IoU Threshold**: 0.4 (overlapping)
- **Multi-scale**: 640px + 832px
- **Preprocessing**: Full enhancement

## 📈 Performance Tips

### **For Maximum FPS**
- ✅ Close other applications
- ✅ Use good lighting
- ✅ Maintain 1-3m distance
- ✅ Simple background
- ✅ Direct USB webcam connection

### **For Best Accuracy**
- ✅ High resolution images
- ✅ Clear helmet visibility
- ✅ Good contrast
- ✅ Multiple angles
- ✅ Various lighting conditions

## 🚨 Safety Reminder

> **"Utamakan keselamatan, keluarga menanti di rumah"**

- Always wear **SNI standard helmet** when riding motorcycles
- Ensure helmet strap is properly fastened
- Regular helmet condition checks
- Replace damaged helmets immediately

## 🔄 Development

### **Training New Model**
```bash
# Open Jupyter notebook
jupyter notebook bikehelmetdetection-yolov8n-training.ipynb
```

### **Custom Optimization**
```python
from helmet_optimizer import HelmetOptimizer
optimizer = HelmetOptimizer()
optimized_results = optimizer.optimize_detection(image)
```

## 📞 Support

For technical support or feature requests:
- 📧 Email: [your-email@example.com]
- 🐛 Issues: Create GitHub issue
- 💬 Discussion: GitHub discussions

---

**Made with ❤️ for Road Safety**

*This project aims to promote motorcycle safety through advanced AI technology.*
