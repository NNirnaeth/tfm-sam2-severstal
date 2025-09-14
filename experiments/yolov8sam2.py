import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import decode_bitmap_to_mask, compute_iou
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

class YOLO2SAM2Pipeline:
    def __init__(self, yolo_model_path, sam2_config, sam2_base, sam2_checkpoint, device="cuda"):
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Load SAM2 model
        self.sam2_model = build_sam2(sam2_config, sam2_base, device="cpu")
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.sam2_model.load_state_dict(torch.load(sam2_checkpoint, map_location="cpu"))
        self.sam2_model.to(device)
        
        self.device = device
        
    def detect_with_yolo(self, image, conf_threshold=0.25, iou_threshold=0.7):
        """Run YOLO detection and return bounding boxes"""
        results = self.yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls
                    })
        return detections
    
    def generate_sam2_prompts(self, detections, image_shape):
        """Generate optimal prompts for SAM2 from YOLO detections"""
        h, w = image_shape[:2]
        points = []
        point_labels = []
        boxes = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Skip low confidence detections
            if conf < 0.1:  # Lower threshold to see more detections
                continue
                
            # Generate center point for each detection
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            points.append([cx, cy])
            point_labels.append(1)
            
            # Add bounding box as prompt
            boxes.append([x1, y1, x2, y2])
        
        return np.array(points), np.array(point_labels), np.array(boxes) if boxes else None
    
    def segment_with_sam2(self, image, points, point_labels, boxes=None):
        """Run SAM2 segmentation with the given prompts"""
        with torch.no_grad():
            self.predictor.set_image(image)
            
            if len(points) > 0 and boxes is not None and len(boxes) > 0:
                # Use both points and boxes - they should have same number of prompts
                assert len(points) == len(boxes), f"Points ({len(points)}) and boxes ({len(boxes)}) must have same count"
                
                # Process each detection separately to avoid tensor size issues
                all_masks = []
                all_scores = []
                
                for i in range(len(points)):
                    # Single point and single box
                    single_point = points[i:i+1]
                    single_label = point_labels[i:i+1]
                    single_box = boxes[i:i+1]
                    
                    masks, scores, _ = self.predictor.predict(
                        point_coords=single_point,
                        point_labels=single_label,
                        box=single_box,
                        multimask_output=True
                    )
                    
                    # Take the best mask for this detection
                    best_mask_idx = np.argmax(scores)
                    all_masks.append(masks[best_mask_idx])
                    all_scores.append(scores[best_mask_idx])
                
                return np.array(all_masks), np.array(all_scores)
                
            elif len(points) > 0:
                # Use only points
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=True
                )
            else:
                # No prompts available
                return np.array([]), np.array([])
        
        return masks, scores
    
    def combine_masks(self, masks, scores, image_shape, overlap_threshold=0.3):
        """Combine multiple masks intelligently"""
        if len(masks) == 0:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Sort by confidence score
        sorted_indices = np.argsort(scores.flatten())[::-1]
        sorted_masks = masks[sorted_indices]
        
        # Initialize final segmentation map
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(seg_map, dtype=bool)
        
        for i, mask in enumerate(sorted_masks):
            mask_bool = mask.astype(bool)
            
            # Check overlap with existing masks
            overlap_ratio = (mask_bool & occupancy_mask).sum() / (mask_bool.sum() + 1e-6)
            
            if overlap_ratio < overlap_threshold:
                # Add this mask
                mask_bool[occupancy_mask] = False  # Remove overlapping areas
                seg_map[mask_bool] = i + 1
                occupancy_mask[mask_bool] = True
        
        return seg_map
    
    def process_image(self, image_path, gt_annotation_path=None):
        """Process a single image through the full pipeline"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, None, None
        
        image = image[..., ::-1].copy()  # BGR to RGB
        
        # Step 1: YOLO detection
        detections = self.detect_with_yolo(image, conf_threshold=0.1, iou_threshold=0.7)
        
        if not detections:
            return np.zeros(image.shape[:2], dtype=np.uint8), [], None
        
        # Step 2: Generate SAM2 prompts
        points, point_labels, boxes = self.generate_sam2_prompts(detections, image.shape)
        
        if len(points) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8), [], None
        
        # Step 3: SAM2 segmentation
        masks, scores = self.segment_with_sam2(image, points, point_labels, boxes)
        
        # Step 4: Combine masks
        final_mask = self.combine_masks(masks, scores, image.shape)
        
        return final_mask, detections, scores
    
    def evaluate_on_dataset(self, image_dir, annotation_dir, output_dir=None):
        """Evaluate the pipeline on a full dataset"""
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
        all_ious = []
        results = []
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_dir, img_file)
            ann_path = os.path.join(annotation_dir, img_file + '.json')
            
            # Process image
            pred_mask, detections, scores = self.process_image(img_path)
            
            if pred_mask is None:
                continue
            
            # Load ground truth if available
            gt_mask = None
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, 'r') as f:
                        gt_data = json.load(f)
                    
                    h, w = pred_mask.shape
                    masks_gt = [decode_bitmap_to_mask(obj['bitmap']['data'])
                               for obj in gt_data["objects"] if obj.get("geometryType") == "bitmap"]
                    origins = [obj["bitmap"]["origin"] for obj in gt_data["objects"] if obj.get("geometryType") == "bitmap"]
                    
                    gt_mask = np.zeros((h, w), dtype=np.uint8)
                    for m, (x0, y0) in zip(masks_gt, origins):
                        gt_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = np.maximum(
                            gt_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m)
                except:
                    gt_mask = None
            
            # Calculate IoU if ground truth available
            iou = None
            if gt_mask is not None:
                iou = compute_iou(pred_mask > 0, gt_mask > 0)
                all_ious.append(iou)
            
            # Save results
            result = {
                'image': img_file,
                'detections': len(detections),
                'iou': iou,
                'mean_score': np.mean(scores) if scores is not None and len(scores) > 0 else 0
            }
            results.append(result)
            
            # Save visualization if output_dir provided
            if output_dir:
                # Load image for visualization
                img_path = os.path.join(image_dir, img_file)
                if os.path.exists(img_path):
                    vis_image = cv2.imread(img_path)[..., ::-1].copy()  # BGR to RGB
                    self.save_visualization(vis_image, pred_mask, detections, output_dir, img_file)
        
        # Print evaluation metrics
        if all_ious:
            mean_iou = np.mean(all_ious)
            print(f"\n--- Evaluation Results ---")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"Images processed: {len(results)}")
            print(f"Average detections per image: {np.mean([r['detections'] for r in results]):.2f}")
            
            for threshold in [0.5, 0.75, 0.9]:
                acc = np.mean([iou >= threshold for iou in all_ious])
                print(f"IoU@{int(threshold*100)}: {acc*100:.2f}%")
        
        return results, all_ious
    
    def save_visualization(self, image, pred_mask, detections, output_dir, filename):
        """Save visualization of results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image with detections
        axes[1].imshow(image)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-5, f'{conf:.2f}', color='red', fontsize=8)
        axes[1].set_title('YOLO Detections')
        axes[1].axis('off')
        
        # Segmentation result
        axes[2].imshow(pred_mask, cmap='viridis')
        axes[2].set_title('SAM2 Segmentation')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}_results.png'), dpi=150, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Configuration
    YOLO_MODEL = "yolov8n.pt"  # or path to your trained model
    SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
    SAM2_BASE = "models/sam2_base_models/sam2_hiera_large.pt"
    SAM2_CKPT = "models/severstal_updated/sam2_large_7ksteps_lr1e3_val500_best_step3000_iou0.18.torch"
    
    # Initialize pipeline
    pipeline = YOLO2SAM2Pipeline(
        yolo_model_path=YOLO_MODEL,
        sam2_config=SAM2_CONFIG,
        sam2_base=SAM2_BASE,
        sam2_checkpoint=SAM2_CKPT
    )
    
    # Process dataset
    IMG_DIR = "datasets/severstal/test_split/img"
    ANN_DIR = "datasets/severstal/test_split/ann"
    OUTPUT_DIR = "results/yolo2sam2_evaluation"
    
    results, ious = pipeline.evaluate_on_dataset(IMG_DIR, ANN_DIR, OUTPUT_DIR) 