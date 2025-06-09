"""
🛡️ HELMET DETECTION SYSTEM - ULTRA FAST 30 FPS
===============================================
Advanced helmet detection with optimized performance for real-time webcam monitoring.

Features:
- ⚡ Ultra-fast 30+ FPS webcam detection
- 🎯 High accuracy image analysis
- 📊 Real-time statistics
- 🚀 Optimized for performance

Author: Muhammad Zein
Project: Helmet Detection System
Version: 3.0 (Ultra-Fast)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

st.set_page_config(
    page_title="🛡️ Helmet Detection - Ultra Fast", 
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== ULTRA-FAST MODEL LOADING ==========
@st.cache_resource
def load_optimized_model():
    """Load model dengan konfigurasi optimal untuk FPS tinggi"""
    try:
        model = YOLO('./runs/detect/train/weights/best.pt')
        model.model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def get_class_names():
    return ["without helmet", "with helmet"]

# Ultra-Fast detection class untuk 30 FPS
class HelmetDetectionApp:
    def __init__(self):
        self.model = load_optimized_model()
        self.class_names = get_class_names()
        self.prev_detections = None
        
    def preprocess_image(self, image):
        """Advanced preprocessing untuk upload gambar"""
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
            
        # Enhance contrast dan brightness
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        return enhanced
    
    def apply_smart_filtering(self, boxes, scores, classes):
        """Apply smart filtering berdasarkan domain knowledge"""
        if boxes is None or len(boxes) == 0:
            return [], [], []
            
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []
        
        for box, score, cls in zip(boxes, scores, classes):
            # Calculate box properties
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Filter based on helmet characteristics
            if area < 400:  # Too small
                continue
                
            if aspect_ratio < 0.2 or aspect_ratio > 4.0:  # Unusual ratio
                score *= 0.8  # Reduce confidence
                
            # Boost confidence for well-formed detections
            if 0.7 <= aspect_ratio <= 2.0 and area > 1500:
                score = min(score * 1.1, 1.0)
                
            # Position-based adjustment
            y_center = (box[1] + box[3]) / 2
            img_height = 480  # Assume standard height
            
            # Helm biasanya di bagian atas
            if y_center < img_height * 0.4:
                score = min(score * 1.05, 1.0)
            
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_classes.append(cls)
            
        return filtered_boxes, filtered_scores, filtered_classes
    
    def detect_with_optimization(self, image, conf_threshold=0.25):
        """Optimized detection untuk upload gambar (akurasi tinggi)"""
        if self.model is None:
            return [], 0, 0
            
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        # Multi-scale detection untuk akurasi lebih baik
        scales = [640, 832]  # Multiple input sizes
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for scale in scales:
            # Resize untuk scale yang berbeda
            h, w = processed_img.shape[:2]
            scale_factor = scale / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            if new_h != h or new_w != w:
                resized_img = cv2.resize(processed_img, (new_w, new_h))
            else:
                resized_img = processed_img
            
            # Detection
            results = self.model(resized_img, verbose=False, conf=conf_threshold, iou=0.5)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                # Scale boxes back jika diperlukan
                if scale_factor != 1.0:
                    boxes = boxes / scale_factor
                
                all_boxes.extend(boxes)
                all_scores.extend(scores)
                all_classes.extend(classes)
        
        # Apply smart filtering
        if all_boxes:
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            all_classes = np.array(all_classes)
            
            # Advanced NMS
            from torchvision.ops import nms
            import torch
            
            keep_indices = nms(
                torch.tensor(all_boxes, dtype=torch.float32),
                torch.tensor(all_scores, dtype=torch.float32),
                iou_threshold=0.4
            )
            
            final_boxes = all_boxes[keep_indices.numpy()]
            final_scores = all_scores[keep_indices.numpy()]
            final_classes = all_classes[keep_indices.numpy()]
            
            # Apply smart filtering
            final_boxes, final_scores, final_classes = self.apply_smart_filtering(
                final_boxes, final_scores, final_classes
            )
        else:
            final_boxes, final_scores, final_classes = [], [], []
        
        # Count detections
        count_helmet = sum(1 for cls in final_classes if int(cls) == 1)
        count_nohelmet = sum(1 for cls in final_classes if int(cls) == 0)
        
        # Store for tracking
        self.prev_detections = list(zip(final_boxes, final_scores, final_classes))
        
        return list(zip(final_boxes, final_scores, final_classes)), count_helmet, count_nohelmet
    
    def detect_ultra_fast_webcam(self, image, conf_threshold=0.7):
        """Ultra-fast detection untuk 30+ FPS webcam"""
        if self.model is None:
            return [], 0, 0
            
        # Zero preprocessing untuk maksimal speed
        img = image
        
        # Ultra-small resize (256px) untuk speed maksimal
        h, w = img.shape[:2]
        target_size = 256  # Even smaller for maximum speed
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
        
        # Lightning detection dengan parameter minimal
        results = self.model(
            img, 
            verbose=False, 
            conf=conf_threshold, 
            iou=0.8, 
            imgsz=256,  # Very small input size
            device='cpu'
        )
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Scale boxes back jika diresize
            if max(h, w) > target_size:
                scale_back = max(h, w) / target_size
                boxes = boxes * scale_back
            
            # Zero filtering untuk maksimal speed
            detections = list(zip(boxes, scores, classes))
        
        # Count detections
        count_helmet = sum(1 for _, _, cls in detections if int(cls) == 1)
        count_nohelmet = sum(1 for _, _, cls in detections if int(cls) == 0)
        
        return detections, count_helmet, count_nohelmet

# Initialize detection app
detector = HelmetDetectionApp()
model = detector.model
class_names = detector.class_names

# Minimal CSS untuk performance
st.markdown("""
    <style>
    .header-title {
        color: #1f2937;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .header-desc {
        color: #6b7280;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .performance-info {
        background: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .fps-display {
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .himbauan {
        background: rgba(239, 68, 68, 0.1);
        color: #991b1b;
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 10px;
        padding: 15px;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .success {
        background: rgba(34, 197, 94, 0.1);
        color: #065f46;
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 10px;
        padding: 15px;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-title">🛡 Ultra-Fast Helmet Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-desc">Sistem deteksi helm real-time dengan target 30 FPS</div>', unsafe_allow_html=True)

# Performance info
st.markdown("""
<div class="performance-info">
    <strong>🚀 Ultra-Fast Mode Active:</strong><br>
    • Image Size: 256px (ultra-optimized)<br>
    • Preprocessing: Disabled for webcam<br>
    • Target FPS: 30+<br>
    • Frame Skip: Enabled
</div>
""", unsafe_allow_html=True)

# Session state initialization
if 'total' not in st.session_state:
    st.session_state['total'] = 0
if 'helmet' not in st.session_state:
    st.session_state['helmet'] = 0
if 'nohelmet' not in st.session_state:
    st.session_state['nohelmet'] = 0
if 'webcam_on' not in st.session_state:
    st.session_state['webcam_on'] = False
if 'fps_list' not in st.session_state:
    st.session_state['fps_list'] = []

def update_stats(helmet_count, nohelmet_count):
    st.session_state['total'] += helmet_count + nohelmet_count
    st.session_state['helmet'] += helmet_count
    st.session_state['nohelmet'] += nohelmet_count

def reset_stats():
    st.session_state['total'] = 0
    st.session_state['helmet'] = 0
    st.session_state['nohelmet'] = 0
    st.session_state['fps_list'] = []

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Gambar (Akurasi Tinggi)")
    uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="upload")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        # Use optimized detection untuk akurasi tinggi
        detections, count_helmet, count_nohelmet = detector.detect_with_optimization(
            img_np, conf_threshold=0.3
        )
        
        himbauan = count_nohelmet > 0
        
        # Draw results with better visualization
        result_img = img_np.copy()
        for detection in detections:
            box, score, cls = detection
            x1, y1, x2, y2 = map(int, box)
            label = int(cls)
            confidence = float(score)
            
            # Better colors and styling
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            thickness = 3
            
            # Draw rectangle
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label_text = f"{class_names[label]} ({confidence:.2f})"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for text
            cv2.rectangle(result_img, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(result_img, label_text, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        st.image(result_img, caption="Hasil Deteksi (High Accuracy)", channels="RGB")
        
        if himbauan:
            st.markdown('<div class="himbauan">⚠️ Ada orang yang tidak menggunakan helm!</div>', unsafe_allow_html=True)
        if count_helmet > 0:
            st.markdown(f'<div class="success">✅ {count_helmet} orang terdeteksi menggunakan helm.</div>', unsafe_allow_html=True)
        if count_nohelmet > 0:
            st.markdown(f'<div class="himbauan">❌ {count_nohelmet} orang tidak memakai helm!</div>', unsafe_allow_html=True)
        update_stats(count_helmet, count_nohelmet)

with col2:
    st.markdown("### 🎥 Ultra-Fast Webcam (30+ FPS)")
    
    # FPS display
    fps_placeholder = st.empty()
    avg_fps = np.mean(st.session_state['fps_list']) if st.session_state['fps_list'] else 0
    fps_color = "🟢" if avg_fps >= 25 else "🟡" if avg_fps >= 15 else "🔴"
    fps_placeholder.markdown(f'<div class="fps-display">{fps_color} Current FPS: {avg_fps:.1f} / Target: 30+</div>', unsafe_allow_html=True)
    
    start_webcam = st.button("🚀 Start Ultra-Fast Webcam", key="start_webcam", type="primary")
    stop_webcam = st.button("⏹️ Stop Webcam", key="stop_webcam", type="secondary")
    
    himbauan_webcam = st.empty()
    frame_placeholder = st.empty()
    
    if start_webcam:
        st.session_state['webcam_on'] = True
    if stop_webcam:
        st.session_state['webcam_on'] = False
    
    if st.session_state['webcam_on']:
        cap = cv2.VideoCapture(0)
        
        # Optimal webcam settings untuk FPS tinggi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Very small resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimum buffer
        
        frame_count = 0
        detections = []  # Initialize detections
        
        while cap.isOpened() and st.session_state['webcam_on']:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam tidak terdeteksi.")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process detection every 3rd frame untuk maintain FPS
            if frame_count % 3 == 0:
                detections, count_helmet, count_nohelmet = detector.detect_ultra_fast_webcam(
                    frame, conf_threshold=0.7
                )
            
            # Draw results dengan minimal processing
            for detection in detections:
                box, score, cls = detection
                x1, y1, x2, y2 = map(int, box)
                label = int(cls)
                
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                
                # Minimal drawing untuk speed
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_names[label], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Calculate FPS
            frame_end = time.time()
            fps = 1.0 / (frame_end - frame_start) if (frame_end - frame_start) > 0 else 0
            
            # Update FPS list (keep last 20 values)
            st.session_state['fps_list'].append(fps)
            if len(st.session_state['fps_list']) > 20:
                st.session_state['fps_list'].pop(0)
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update Streamlit display with very small size
            frame_placeholder.image(frame, channels="BGR", width=320)
            
            # Update FPS metric every 10 frames
            if frame_count % 10 == 0:
                avg_fps = np.mean(st.session_state['fps_list'][-10:]) if len(st.session_state['fps_list']) >= 10 else avg_fps
                fps_color = "🟢" if avg_fps >= 25 else "🟡" if avg_fps >= 15 else "🔴"
                fps_placeholder.markdown(f'<div class="fps-display">{fps_color} Current FPS: {avg_fps:.1f} / Target: 30+</div>', unsafe_allow_html=True)
            
            # Status update setiap 15 frame
            if frame_count % 15 == 0:
                if count_nohelmet > 0:
                    himbauan_webcam.markdown('<div class="himbauan">⚠️ Ada yang tidak pakai helm!</div>', unsafe_allow_html=True)
                elif count_helmet > 0:
                    himbauan_webcam.markdown('<div class="success">✅ Semua pakai helm</div>', unsafe_allow_html=True)
                else:
                    himbauan_webcam.markdown('<div style="color: #3b82f6;">👀 Scanning...</div>', unsafe_allow_html=True)
                
                # Update stats
                update_stats(count_helmet, count_nohelmet)
            
            frame_count += 1
            
            # Break condition
            if stop_webcam or not st.session_state['webcam_on']:
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Statistics
st.markdown("### 📊 Statistik & Performance")
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Total Deteksi", st.session_state['total'])
with col_b:
    st.metric("Pakai Helm", st.session_state['helmet'])
with col_c:
    st.metric("Tanpa Helm", st.session_state['nohelmet'])
with col_d:
    avg_fps_total = np.mean(st.session_state['fps_list']) if st.session_state['fps_list'] else 0
    fps_status = "🟢 Excellent" if avg_fps_total >= 25 else "🟡 Good" if avg_fps_total >= 15 else "🔴 Slow"
    st.metric("Avg FPS", f"{avg_fps_total:.1f}", fps_status)

if st.button("🔄 Reset All Stats", type="secondary"):
    reset_stats()
    st.rerun()

# Performance tips
st.markdown("""
### 🎯 Tips untuk FPS Optimal:
- **Pencahayaan baik** - membantu deteksi lebih cepat
- **Jarak optimal** - 1-3 meter dari kamera  
- **Background simple** - mengurangi noise
- **Tutup aplikasi lain** - untuk CPU/memory maksimal
- **Resolusi kecil** - webcam 320x240 untuk speed maksimal
""")

# Safety reminder
st.markdown("""
### 🚦 Himbauan Keselamatan:
- Selalu gunakan helm standar SNI saat berkendara motor
- Pastikan tali helm terpasang dengan benar
- Utamakan keselamatan, keluarga menanti di rumah
""")
