"""
🛡️ HELMET DETECTION WEB APPLICATION
==================================
Flask-based web application with Mazer admin template integration
Replaces Streamlit interface with professional admin dashboard

Author: Muhammad Zein
Project: Helmet Detection System
Version: 4.0 (Web Dashboard)
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from PIL import Image
import base64
import io
import time
from ultralytics import YOLO
import threading
import json

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Global variables for webcam
camera = None
detection_active = False
frame_data = {'frame': None, 'detections': [], 'stats': {'helmet': 0, 'no_helmet': 0, 'total': 0, 'fps': 0}}

class HelmetDetectionWeb:
    def __init__(self):
        self.model = None
        self.class_names = ["without helmet", "with helmet"]
        self.load_model()
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO('./runs/detect/train/weights/best.pt')
            self.model.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect_image(self, image, conf_threshold=0.3):
        """Detect helmets in uploaded image with high accuracy"""
        if self.model is None:
            return [], 0, 0
            
        # Preprocess image
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
            
        # Enhance image quality
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # Multi-scale detection
        scales = [640, 832]
        all_detections = []
        
        for scale in scales:
            h, w = enhanced.shape[:2]
            scale_factor = scale / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            if new_h != h or new_w != w:
                resized_img = cv2.resize(enhanced, (new_w, new_h))
            else:
                resized_img = enhanced
            
            results = self.model(resized_img, verbose=False, conf=conf_threshold, iou=0.5)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                if scale_factor != 1.0:
                    boxes = boxes / scale_factor
                
                for box, score, cls in zip(boxes, scores, classes):
                    all_detections.append({
                        'box': box.tolist(),
                        'confidence': float(score),
                        'class': int(cls),
                        'label': self.class_names[int(cls)]
                    })
        
        # Apply NMS and filtering
        if all_detections:
            # Simple NMS based on IoU
            filtered_detections = self.apply_nms(all_detections, iou_threshold=0.4)
        else:
            filtered_detections = []
        
        # Count detections
        count_helmet = sum(1 for det in filtered_detections if det['class'] == 1)
        count_nohelmet = sum(1 for det in filtered_detections if det['class'] == 0)
        
        return filtered_detections, count_helmet, count_nohelmet
    
    def detect_webcam_frame(self, frame, conf_threshold=0.7):
        """Fast detection for webcam frames"""
        if self.model is None:
            return [], 0, 0
            
        # Resize for speed
        h, w = frame.shape[:2]
        target_size = 320
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Lightning fast detection
        results = self.model(frame, verbose=False, conf=conf_threshold, iou=0.8, imgsz=320)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Scale boxes back if resized
            if max(h, w) > target_size:
                scale_back = max(h, w) / target_size
                boxes = boxes * scale_back
            
            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    'box': box.tolist(),
                    'confidence': float(score),
                    'class': int(cls),
                    'label': self.class_names[int(cls)]
                })
        
        count_helmet = sum(1 for det in detections if det['class'] == 1)
        count_nohelmet = sum(1 for det in detections if det['class'] == 0)
        
        return detections, count_helmet, count_nohelmet
    
    def apply_nms(self, detections, iou_threshold=0.4):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, det in enumerate(detections):
            if i == 0:
                keep.append(det)
                continue
                
            # Check IoU with all kept detections
            should_keep = True
            for kept_det in keep:
                if self.calculate_iou(det['box'], kept_det['box']) > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det)
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

# Initialize detector
detector = HelmetDetectionWeb()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        img_np = np.array(image)
        
        # Detect helmets
        detections, count_helmet, count_nohelmet = detector.detect_image(img_np, conf_threshold=0.3)
        
        # Draw detections on image
        result_img = img_np.copy()
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = map(int, box)
            label = detection['class']
            confidence = detection['confidence']
            
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
            
            label_text = f"{detection['label']} ({confidence:.2f})"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_img, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(result_img, label_text, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert result image to base64
        result_pil = Image.fromarray(result_img)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_base64}",
            'detections': detections,
            'stats': {
                'helmet': count_helmet,
                'no_helmet': count_nohelmet,
                'total': count_helmet + count_nohelmet
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def webcam_thread():
    """Background thread for webcam processing"""
    global camera, detection_active, frame_data
    
    while True:
        if detection_active and camera is not None:
            ret, frame = camera.read()
            if ret:
                frame_start = time.time()
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Detect helmets
                detections, count_helmet, count_nohelmet = detector.detect_webcam_frame(frame, conf_threshold=0.7)
                
                # Draw detections
                for detection in detections:
                    box = detection['box']
                    x1, y1, x2, y2 = map(int, box)
                    label = detection['class']
                    
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, detection['label'], (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Calculate FPS
                frame_end = time.time()
                fps = 1.0 / (frame_end - frame_start) if (frame_end - frame_start) > 0 else 0
                
                # Update global frame data
                frame_data['frame'] = frame
                frame_data['detections'] = detections
                frame_data['stats'] = {
                    'helmet': count_helmet,
                    'no_helmet': count_nohelmet,
                    'total': count_helmet + count_nohelmet,
                    'fps': fps
                }
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/start_webcam')
def start_webcam():
    """Start webcam detection"""
    global camera, detection_active
    
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        detection_active = True
        return jsonify({'success': True, 'message': 'Webcam started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_webcam')
def stop_webcam():
    """Stop webcam detection"""
    global detection_active
    detection_active = False
    return jsonify({'success': True, 'message': 'Webcam stopped'})

@app.route('/webcam_feed')
def webcam_feed():
    """Stream webcam frames"""
    def generate():
        while True:
            if detection_active and frame_data['frame'] is not None:
                frame = frame_data['frame']
                
                # Add FPS display
                fps = frame_data['stats']['fps']
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_stats')
def webcam_stats():
    """Get current webcam detection stats"""
    return jsonify(frame_data['stats'])

if __name__ == '__main__':
    # Start webcam thread
    webcam_thread_obj = threading.Thread(target=webcam_thread)
    webcam_thread_obj.daemon = True
    webcam_thread_obj.start()
    
    print("🚀 Starting Helmet Detection Web Application...")
    print("📱 Access the dashboard at: http://localhost:5000")
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
