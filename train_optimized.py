# ========== OPTIMIZED TRAINING SCRIPT ==========
# Script untuk training YOLO model dengan konfigurasi optimal untuk helmet detection

import os
import shutil
import random
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import yaml
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Disable wandb jika tidak digunakan
os.environ['WANDB_DISABLED'] = 'true'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class HelmetTrainingOptimizer:
    """Optimized training pipeline untuk helmet detection"""
    
    def __init__(self, dataset_path='./helmet-dataset'):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, "images")
        self.annotations_dir = os.path.join(dataset_path, "annotations")
        
        # Output directories
        self.working_dir = './output'
        self.labels_dir = os.path.join(self.working_dir, "labels")
        self.train_img_dir = os.path.join(self.working_dir, "train", "images")
        self.train_labels_dir = os.path.join(self.working_dir, "train", "labels")
        self.val_img_dir = os.path.join(self.working_dir, "val", "images")
        self.val_labels_dir = os.path.join(self.working_dir, "val", "labels")
        
        # Create directories
        self.setup_directories()
        
    def setup_directories(self):
        """Setup semua directory yang diperlukan"""
        dirs = [self.working_dir, self.labels_dir, self.train_img_dir, 
                self.train_labels_dir, self.val_img_dir, self.val_labels_dir]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            
    def parse_xml(self, xml_file):
        """Parse XML annotation file"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract image info
        image_name = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        labels_and_bboxes = []
        
        # Extract objects
        for obj in root.findall('object'):
            label = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            labels_and_bboxes.append((label, (xmin, ymin, xmax, ymax)))
            
        return image_name, (width, height), labels_and_bboxes
    
    def convert_to_yolo_format(self, label, bbox, img_width, img_height):
        """Convert Pascal VOC format ke YOLO format"""
        xmin, ymin, xmax, ymax = bbox
        
        # Convert ke YOLO format (normalized)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Class mapping
        class_id = 1 if label == 'With Helmet' else 0
        
        return class_id, x_center, y_center, width, height
    
    def analyze_dataset(self):
        """Analyze dataset quality dan distribution"""
        print("🔍 Analyzing dataset...")
        
        class_counts = {'With Helmet': 0, 'Without Helmet': 0}
        image_sizes = []
        bbox_sizes = []
        problematic_files = []
        
        xml_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            try:
                xml_path = os.path.join(self.annotations_dir, xml_file)
                img_name, img_size, labels_bboxes = self.parse_xml(xml_path)
                
                # Check if corresponding image exists
                img_path = os.path.join(self.images_dir, img_name)
                if not os.path.exists(img_path):
                    problematic_files.append(f"Missing image: {img_name}")
                    continue
                    
                image_sizes.append(img_size)
                
                for label, bbox in labels_bboxes:
                    class_counts[label] = class_counts.get(label, 0) + 1
                    
                    # Check bbox validity
                    xmin, ymin, xmax, ymax = bbox
                    if xmax <= xmin or ymax <= ymin:
                        problematic_files.append(f"Invalid bbox in {xml_file}")
                        
                    bbox_sizes.append((xmax-xmin, ymax-ymin))
                    
            except Exception as e:
                problematic_files.append(f"Error parsing {xml_file}: {e}")
                
        # Print analysis
        print(f"📊 Dataset Analysis:")
        print(f"   Total annotations: {len(xml_files)}")
        print(f"   Class distribution: {class_counts}")
        print(f"   Image sizes: {len(set(image_sizes))} unique sizes")
        
        if problematic_files:
            print(f"⚠️  Found {len(problematic_files)} issues:")
            for issue in problematic_files[:5]:  # Show first 5
                print(f"     - {issue}")
                
        return class_counts, problematic_files
    
    def create_optimized_labels(self):
        """Convert annotations ke YOLO format dengan optimasi"""
        print("🔄 Converting annotations to YOLO format...")
        
        converted_count = 0
        error_count = 0
        
        for xml_file in os.listdir(self.annotations_dir):
            if not xml_file.endswith('.xml'):
                continue
                
            try:
                xml_path = os.path.join(self.annotations_dir, xml_file)
                img_name, img_size, labels_bboxes = self.parse_xml(xml_path)
                
                # Check if image exists
                img_path = os.path.join(self.images_dir, img_name)
                if not os.path.exists(img_path):
                    continue
                    
                # Create YOLO label file
                label_file = xml_file.replace('.xml', '.txt')
                label_path = os.path.join(self.labels_dir, label_file)
                
                with open(label_path, 'w') as f:
                    for label, bbox in labels_bboxes:
                        class_id, x_center, y_center, width, height = self.convert_to_yolo_format(
                            label, bbox, img_size[0], img_size[1]
                        )
                        
                        # Validate YOLO format
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 < width <= 1 and 0 < height <= 1):
                            print(f"Warning: Invalid YOLO coordinates in {xml_file}")
                            continue
                            
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                converted_count += 1
                
            except Exception as e:
                print(f"Error converting {xml_file}: {e}")
                error_count += 1
                
        print(f"✅ Converted {converted_count} files, {error_count} errors")
        
    def create_balanced_split(self, train_ratio=0.8, stratify=True):
        """Create balanced train/val split"""
        print("🎯 Creating balanced train/validation split...")
        
        # Get all image files that have corresponding labels
        valid_files = []
        class_labels = []
        
        for img_file in os.listdir(self.images_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_file):
                # Read label file to determine dominant class
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Use first class as representative (could be improved)
                        first_class = int(lines[0].split()[0])
                        valid_files.append(base_name)
                        class_labels.append(first_class)
                        
        if stratify and len(set(class_labels)) > 1:
            # Stratified split
            train_files, val_files = train_test_split(
                valid_files, 
                test_size=1-train_ratio, 
                stratify=class_labels,
                random_state=42
            )
        else:
            # Random split
            random.seed(42)
            random.shuffle(valid_files)
            split_idx = int(len(valid_files) * train_ratio)
            train_files = valid_files[:split_idx]
            val_files = valid_files[split_idx:]
            
        # Copy files
        self._copy_files(train_files, 'train')
        self._copy_files(val_files, 'val')
        
        print(f"📂 Split complete: {len(train_files)} train, {len(val_files)} val")
        
    def _copy_files(self, files, split_type):
        """Copy files ke train atau val directory"""
        img_dest = self.train_img_dir if split_type == 'train' else self.val_img_dir
        label_dest = self.train_labels_dir if split_type == 'train' else self.val_labels_dir
        
        for base_name in files:
            # Find image file (handle different extensions)
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_img = os.path.join(self.images_dir, f"{base_name}{ext}")
                if os.path.exists(potential_img):
                    img_file = potential_img
                    break
                    
            if img_file:
                # Copy image
                shutil.copy2(img_file, img_dest)
                
                # Copy label
                label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
                if os.path.exists(label_file):
                    shutil.copy2(label_file, label_dest)
                    
    def create_config_file(self):
        """Create optimized config file untuk training"""
        config_path = os.path.join(self.working_dir, 'config_optimized.yaml')
        
        config_content = {
            'path': os.path.abspath(self.working_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                0: 'without helmet',
                1: 'with helmet'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False)
            
        print(f"📝 Config file created: {config_path}")
        return config_path
    
    def train_optimized_model(self, model_size='s', epochs=150, batch_size=16):
        """Train model dengan konfigurasi optimal"""
        print(f"🚀 Starting optimized training (YOLOv8{model_size})...")
        
        # Load pre-trained model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Create config
        config_path = self.create_config_file()
        
        # Optimized training parameters
        training_args = {
            'data': config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': 640,
            'patience': 30,
            'save_period': 25,
            'plots': True,
            'device': 'auto',
            
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Data augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            
            # Loss weights
            'cls': 0.5,
            'box': 7.5,
            'dfl': 1.5,
            
            # Confidence and IoU
            'conf': 0.001,
            'iou': 0.7,
            
            # Additional optimizations
            'dropout': 0.15,
            'val': True,
            'save': True,
            'cache': True,
            'workers': 8,
            'project': 'runs/detect',
            'name': 'helmet_optimized',
        }
        
        # Start training
        results = model.train(**training_args)
        
        print("✅ Training completed!")
        print(f"📊 Best model saved to: runs/detect/helmet_optimized/weights/best.pt")
        
        return results
    
    def validate_model(self, model_path=None):
        """Validate trained model"""
        if model_path is None:
            model_path = 'runs/detect/helmet_optimized/weights/best.pt'
            
        print(f"🔍 Validating model: {model_path}")
        
        model = YOLO(model_path)
        results = model.val()
        
        print("📊 Validation Results:")
        print(f"   mAP50: {results.box.map50:.4f}")
        print(f"   mAP50-95: {results.box.map:.4f}")
        
        return results

def main():
    """Main training pipeline"""
    print("🛡️ Helmet Detection - Optimized Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = HelmetTrainingOptimizer()
    
    # Step 1: Analyze dataset
    class_counts, issues = trainer.analyze_dataset()
    
    # Step 2: Convert labels
    trainer.create_optimized_labels()
    
    # Step 3: Create split
    trainer.create_balanced_split()
    
    # Step 4: Train model
    results = trainer.train_optimized_model(
        model_size='s',  # Bisa diganti ke 'm' atau 'l' untuk model lebih besar
        epochs=150,
        batch_size=16
    )
    
    # Step 5: Validate
    validation_results = trainer.validate_model()
    
    print("\n🎉 Optimized training pipeline completed!")
    print("\n📋 Next steps:")
    print("   1. Run app_optimized.py untuk testing")
    print("   2. Check validation metrics dalam runs/detect/helmet_optimized/")
    print("   3. Gunakan helmet_optimizer.py untuk advanced inference")

if __name__ == "__main__":
    main()
