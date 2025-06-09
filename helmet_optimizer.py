# ========== OPTIMIZED HELMET DETECTION MODULE ==========
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
import time

class HelmetDetectionOptimizer:
    """
    Advanced helmet detection system dengan multiple optimization techniques
    untuk meningkatkan akurasi dan mengurangi false positive/negative
    """
    
    def __init__(self, model_path: str = './runs/detect/train/weights/best.pt'):
        """
        Initialize optimizer dengan model path
        
        Args:
            model_path: Path ke model YOLO yang sudah ditraining
        """
        try:
            self.model = YOLO(model_path)
            self.model.model.eval()  # Set ke evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        self.class_names = ["without helmet", "with helmet"]
        self.prev_detections = []
        self.detection_history = []
        
        # Konfigurasi optimasi
        self.conf_threshold = 0.25
        self.iou_threshold = 0.5
        self.min_box_area = 400
        self.max_aspect_ratio = 4.0
        self.min_aspect_ratio = 0.2
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image enhancement untuk meningkatkan kualitas deteksi
        
        Args:
            image: Input image dalam format BGR atau RGB
            
        Returns:
            Enhanced image
        """
        if image is None or image.size == 0:
            return image
            
        # Convert ke RGB jika diperlukan
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR format dari OpenCV
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image.copy()
            
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 2. Bilateral filter untuk noise reduction sambil preserve edges
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # 3. Optional: Gamma correction untuk brightness
        gamma = 1.2
        enhanced = np.power(enhanced/255.0, gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced
    
    def validate_detection(self, box: np.ndarray, score: float, class_id: int) -> Tuple[bool, float]:
        """
        Validasi detection berdasarkan domain knowledge helm
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            score: Confidence score
            class_id: Class ID (0=without helmet, 1=with helmet)
            
        Returns:
            Tuple of (is_valid, adjusted_score)
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Filter berdasarkan ukuran minimum
        if area < self.min_box_area:
            return False, score
            
        # Filter berdasarkan aspect ratio
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            # Reduce confidence untuk unusual ratios
            score *= 0.7
            
        # Boost confidence untuk deteksi yang well-formed
        if 0.7 <= aspect_ratio <= 2.0 and area > 1500:
            score = min(score * 1.1, 1.0)
            
        # Position-based scoring
        # Helm biasanya di bagian atas frame
        y_center = (y1 + y2) / 2
        img_height = 480  # Assume standard height
        
        if y_center < img_height * 0.4:  # Upper 40% of image
            score = min(score * 1.05, 1.0)
        elif y_center > img_height * 0.8:  # Lower 20% of image
            score *= 0.9  # Slightly reduce confidence
            
        return True, score
    
    def multi_scale_detection(self, image: np.ndarray, scales: List[int] = [640, 832]) -> List[dict]:
        """
        Multi-scale detection untuk meningkatkan akurasi
        
        Args:
            image: Input image
            scales: List of scales untuk detection
            
        Returns:
            List of detection results
        """
        if self.model is None:
            return []
            
        all_detections = []
        original_shape = image.shape[:2]
        
        for scale in scales:
            # Resize image ke scale yang diinginkan
            h, w = original_shape
            scale_factor = scale / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            if scale_factor != 1.0:
                resized_img = cv2.resize(image, (new_w, new_h))
            else:
                resized_img = image
                
            # Run detection
            try:
                results = self.model(resized_img, verbose=False, 
                                   conf=self.conf_threshold, 
                                   iou=self.iou_threshold)
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    # Scale boxes back to original size
                    if scale_factor != 1.0:
                        boxes = boxes / scale_factor
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        is_valid, adjusted_score = self.validate_detection(box, score, int(cls))
                        
                        if is_valid and adjusted_score > self.conf_threshold:
                            all_detections.append({
                                'box': box,
                                'score': adjusted_score,
                                'class': int(cls),
                                'scale': scale
                            })
                            
            except Exception as e:
                print(f"Detection error at scale {scale}: {e}")
                continue
                
        return all_detections
    
    def advanced_nms(self, detections: List[dict]) -> List[dict]:
        """
        Advanced Non-Maximum Suppression dengan class-aware filtering
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
            
        # Separate by class
        class_detections = {0: [], 1: []}
        for det in detections:
            class_detections[det['class']].append(det)
            
        final_detections = []
        
        for class_id, class_dets in class_detections.items():
            if not class_dets:
                continue
                
            # Convert to tensors
            boxes = torch.tensor([det['box'] for det in class_dets], dtype=torch.float32)
            scores = torch.tensor([det['score'] for det in class_dets], dtype=torch.float32)
            
            # Apply NMS
            from torchvision.ops import nms
            keep_indices = nms(boxes, scores, self.iou_threshold)
            
            # Add kept detections
            for idx in keep_indices:
                final_detections.append(class_dets[idx])
                
        return final_detections
    
    def temporal_smoothing(self, current_detections: List[dict]) -> List[dict]:
        """
        Temporal smoothing untuk konsistensi detection antar frame
        
        Args:
            current_detections: Current frame detections
            
        Returns:
            Smoothed detections
        """
        if not self.prev_detections:
            self.prev_detections = current_detections
            return current_detections
            
        smoothed_detections = []
        tracking_threshold = 0.5
        
        for curr_det in current_detections:
            curr_box = curr_det['box']
            best_match = None
            best_iou = 0
            
            # Find best match dengan previous detections
            for prev_det in self.prev_detections:
                prev_box = prev_det['box']
                iou = self.calculate_iou(curr_box, prev_box)
                
                if iou > best_iou and iou > tracking_threshold and curr_det['class'] == prev_det['class']:
                    best_iou = iou
                    best_match = prev_det
                    
            if best_match:
                # Smooth detection dengan weighted average
                alpha = 0.3  # Weight untuk previous detection
                
                # Smooth bounding box
                smoothed_box = alpha * best_match['box'] + (1 - alpha) * curr_det['box']
                
                # Smooth confidence
                smoothed_score = alpha * best_match['score'] + (1 - alpha) * curr_det['score']
                
                smoothed_det = curr_det.copy()
                smoothed_det['box'] = smoothed_box
                smoothed_det['score'] = smoothed_score
                smoothed_detections.append(smoothed_det)
            else:
                smoothed_detections.append(curr_det)
                
        self.prev_detections = smoothed_detections
        return smoothed_detections
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union
        
        Args:
            box1, box2: Bounding boxes [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_optimized(self, image: np.ndarray, conf_threshold: float = 0.3, enable_temporal: bool = True) -> Tuple[List[dict], int, int]:
        """
        Main optimized detection function
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            enable_temporal: Enable temporal smoothing
            
        Returns:
            Tuple of (detections, helmet_count, no_helmet_count)
        """
        if self.model is None:
            return [], 0, 0
            
        # 1. Enhance image
        enhanced_img = self.enhance_image(image)
        
        # 2. Multi-scale detection
        detections = self.multi_scale_detection(enhanced_img)
        
        # 3. Advanced NMS
        filtered_detections = self.advanced_nms(detections)
        
        # 4. Temporal smoothing (opsional)
        if enable_temporal:
            final_detections = self.temporal_smoothing(filtered_detections)
        else:
            final_detections = filtered_detections
            
        # 5. Count results
        helmet_count = sum(1 for det in final_detections if det['class'] == 1)
        no_helmet_count = sum(1 for det in final_detections if det['class'] == 0)
        
        return final_detections, helmet_count, no_helmet_count
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection results pada image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image dengan bounding boxes
        """
        result_img = image.copy()
        
        for det in detections:
            box = det['box']
            score = det['score']
            class_id = det['class']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Color coding
            color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # Green for helmet, Red for no helmet
            thickness = 3
            
            # Draw rectangle
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label_text = f"{self.class_names[class_id]} ({score:.2f})"
            
            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(result_img, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(result_img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return result_img
