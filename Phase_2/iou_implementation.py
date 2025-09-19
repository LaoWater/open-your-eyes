import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Union
import torch

class IoUCalculator:
    """
    Intersection over Union (IoU) calculator for object detection evaluation
    
    Real-world application: Evaluate how well our restaurant safety detector
    predicts bounding boxes around hands, gloves, and safety violations
    """
    
    def __init__(self):
        self.epsilon = 1e-6  # Prevent division by zero
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format where (x1,y1) is top-left, (x2,y2) is bottom-right
        
        Returns:
            IoU score between 0 and 1
            
        Real-world example:
        - box1: Ground truth hand location in restaurant image
        - box2: Our model's prediction of hand location
        - IoU = 0.7 means 70% overlap (usually considered good detection)
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        # The intersection is the overlapping rectangle
        x1_int = max(x1_1, x1_2)  # Left edge of intersection
        y1_int = max(y1_1, y1_2)  # Top edge of intersection
        x2_int = min(x2_1, x2_2)  # Right edge of intersection
        y2_int = min(y2_1, y2_2)  # Bottom edge of intersection
        
        # Calculate intersection area
        # If boxes don't overlap, intersection area = 0
        intersection_width = max(0, x2_int - x1_int)
        intersection_height = max(0, y2_int - y1_int)
        intersection_area = intersection_width * intersection_height
        
        # Calculate individual box areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        # Union = Area1 + Area2 - Intersection (avoid double counting)
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        if union_area <= self.epsilon:
            return 0.0
        
        iou = intersection_area / union_area
        return iou
    
    def calculate_iou_batch(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Vectorized IoU calculation for multiple boxes
        Essential for efficient training and evaluation
        
        Args:
            boxes1: [N, 4] tensor of ground truth boxes
            boxes2: [M, 4] tensor of predicted boxes
            
        Returns:
            [N, M] tensor of IoU scores between all pairs
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
        
        # Calculate intersection coordinates
        x1_int = torch.max(boxes1[..., 0], boxes2[..., 0])
        y1_int = torch.max(boxes1[..., 1], boxes2[..., 1])
        x2_int = torch.min(boxes1[..., 2], boxes2[..., 2])
        y2_int = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        # Calculate intersection area
        intersection_area = torch.clamp(x2_int - x1_int, min=0) * \
                          torch.clamp(y2_int - y1_int, min=0)
        
        # Calculate individual areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Calculate union and IoU
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / (union_area + self.epsilon)
        
        return iou
    
    def calculate_giou(self, box1: List[float], box2: List[float]) -> float:
        """
        Generalized IoU (GIoU) - Better metric that penalizes non-overlapping boxes
        
        Real-world benefit: Regular IoU is 0 for non-overlapping boxes, giving no gradient
        GIoU provides meaningful gradients even when boxes don't overlap
        """
        # Regular IoU calculation
        iou = self.calculate_iou(box1, box2)
        
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate enclosing box (smallest box containing both boxes)
        x1_enc = min(x1_1, x1_2)
        y1_enc = min(y1_1, y1_2)
        x2_enc = max(x2_1, x2_2)
        y2_enc = max(y2_1, y2_2)
        
        # Areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        enclosing_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
        union_area = area1 + area2 - self.calculate_intersection_area(box1, box2)
        
        # GIoU formula
        if enclosing_area <= self.epsilon:
            return iou
        
        giou = iou - (enclosing_area - union_area) / enclosing_area
        return giou
    
    def calculate_intersection_area(self, box1: List[float], box2: List[float]) -> float:
        """Helper function to calculate intersection area"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        return max(0, x2_int - x1_int) * max(0, y2_int - y1_int)

class RestaurantDetectionEvaluator:
    """
    Complete evaluation system for restaurant safety detection
    Real-world application: Measure how well our model detects safety violations
    """
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75, 0.9]):
        self.iou_calculator = IoUCalculator()
        self.iou_thresholds = iou_thresholds
        
        # Restaurant-specific class mapping
        self.classes = {
            0: 'hand_with_gloves',
            1: 'hand_without_gloves', 
            2: 'food_item',
            3: 'utensil',
            4: 'safety_violation'
        }
    
    def evaluate_single_image(self, 
                            predictions: List[Dict], 
                            ground_truths: List[Dict]) -> Dict[str, float]:
        """
        Evaluate predictions for a single restaurant image
        
        Args:
            predictions: [{'bbox': [x1,y1,x2,y2], 'class': int, 'confidence': float}, ...]
            ground_truths: [{'bbox': [x1,y1,x2,y2], 'class': int}, ...]
        
        Returns:
            Evaluation metrics for this image
        """
        results = {}
        
        for iou_threshold in self.iou_thresholds:
            tp, fp, fn = self._calculate_tp_fp_fn(predictions, ground_truths, iou_threshold)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[f'precision@{iou_threshold}'] = precision
            results[f'recall@{iou_threshold}'] = recall
            results[f'f1@{iou_threshold}'] = f1_score
        
        return results
    
    def _calculate_tp_fp_fn(self, predictions: List[Dict], ground_truths: List[Dict], 
                           iou_threshold: float) -> Tuple[int, int, int]:
        """
        Calculate True Positives, False Positives, False Negatives
        
        Real-world interpretation:
        - TP: Model correctly detected a safety violation
        - FP: Model falsely flagged safe behavior as violation  
        - FN: Model missed an actual safety violation
        """
        # Sort predictions by confidence (highest first)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = 0  # True positives
        fp = 0  # False positives
        matched_gts = set()  # Track which ground truths are matched
        
        # Check each prediction
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gts:
                    continue  # Already matched
                    
                # Must be same class
                if pred['class'] != gt['class']:
                    continue
                
                # Calculate IoU
                iou = self.iou_calculator.calculate_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gts.add(best_gt_idx)
            else:
                fp += 1
        
        # False negatives = unmatched ground truths
        fn = len(ground_truths) - len(matched_gts)
        
        return tp, fp, fn
    
    def calculate_map(self, all_predictions: Dict[str, List[Dict]], 
                     all_ground_truths: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP) - the gold standard metric
        
        Real-world significance:
        - mAP@0.5 = 0.8 means model is 80% accurate at detecting safety issues
        - Used to compare different models and track improvement over time
        - Industry standard: mAP@0.5 > 0.7 is considered good for safety applications
        """
        aps_per_class = {}
        
        for class_id in self.classes.keys():
            class_name = self.classes[class_id]
            
            # Collect all predictions and GTs for this class
            class_predictions = []
            class_ground_truths = []
            
            for image_id in all_predictions.keys():
                # Filter by class
                img_preds = [p for p in all_predictions[image_id] if p['class'] == class_id]
                img_gts = [g for g in all_ground_truths[image_id] if g['class'] == class_id]
                
                # Add image_id to track which image each detection belongs to
                for p in img_preds:
                    p['image_id'] = image_id
                for g in img_gts:
                    g['image_id'] = image_id
                
                class_predictions.extend(img_preds)
                class_ground_truths.extend(img_gts)
            
            # Calculate AP for this class
            if len(class_ground_truths) == 0:
                aps_per_class[class_name] = 0.0
                continue
                
            ap = self._calculate_average_precision(class_predictions, class_ground_truths)
            aps_per_class[class_name] = ap
        
        # Calculate overall mAP
        map_score = np.mean(list(aps_per_class.values()))
        
        result = {'mAP': map_score}
        result.update(aps_per_class)
        
        return result
    
    def _calculate_average_precision(self, predictions: List[Dict], 
                                   ground_truths: List[Dict]) -> float:
        """
        Calculate Average Precision for a single class
        AP = Area under the Precision-Recall curve
        """
        if not predictions:
            return 0.0
        
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Track matches
        tp = []
        fp = []
        matched_gts = {}  # image_id -> set of matched GT indices
        
        for pred in predictions:
            image_id = pred['image_id']
            if image_id not in matched_gts:
                matched_gts[image_id] = set()
            
            # Get ground truths for this image
            image_gts = [g for g in ground_truths if g['image_id'] == image_id]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(image_gts):
                if gt_idx in matched_gts[image_id]:
                    continue
                    
                iou = self.iou_calculator.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if TP or FP
            if best_iou >= 0.5 and best_gt_idx != -1:  # Using IoU threshold of 0.5
                tp.append(1)
                fp.append(0)
                matched_gts[image_id].add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)
        
        # Convert to cumulative
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp / (tp + fp)
        recall = tp / len(ground_truths)
        
        # Calculate AP using 11-point interpolation
        ap = self._calculate_ap_11_point(precision, recall)
        
        return ap
    
    def _calculate_ap_11_point(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """11-point interpolation for Average Precision"""
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        return ap

# Visualization tools for understanding IoU
class IoUVisualizer:
    """
    Visualize IoU calculations to understand model performance
    Critical for debugging and explaining model behavior to stakeholders
    """
    
    def __init__(self):
        self.iou_calculator = IoUCalculator()
    
    def plot_iou_example(self, box1: List[float], box2: List[float], 
                        title: str = "IoU Calculation Example"):
        """
        Visualize IoU calculation between two boxes
        
        Real-world use: Show restaurant managers how detection accuracy is measured
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Calculate IoU
        iou_score = self.iou_calculator.calculate_iou(box1, box2)
        
        # Create rectangles
        rect1 = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1], 
                                 linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, 
                                 label='Ground Truth (Hand)')
        rect2 = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1], 
                                 linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, 
                                 label='Prediction (Hand)')
        
        # Add rectangles to plot
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        
        # Calculate and show intersection
        x1_int = max(box1[0], box2[0])
        y1_int = max(box1[1], box2[1])
        x2_int = min(box1[2], box2[2])
        y2_int = min(box1[3], box2[3])
        
        if x2_int > x1_int and y2_int > y1_int:
            rect_int = patches.Rectangle((x1_int, y1_int), x2_int-x1_int, y2_int-y1_int, 
                                       linewidth=2, edgecolor='green', facecolor='green', 
                                       alpha=0.6, label='Intersection')
            ax.add_patch(rect_int)
        
        # Set axis properties
        all_coords = box1 + box2
        margin = 20
        ax.set_xlim(min(all_coords[::2]) - margin, max(all_coords[2::2]) + margin)
        ax.set_ylim(min(all_coords[1::2]) - margin, max(all_coords[3::2]) + margin)
        
        # Add labels and title
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        ax.set_title(f'{title}\nIoU Score: {iou_score:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add IoU interpretation
        if iou_score >= 0.7:
            interpretation = "Excellent Detection ✅"
        elif iou_score >= 0.5:
            interpretation = "Good Detection ✓"
        elif iou_score >= 0.3:
            interpretation = "Moderate Detection ⚠️"
        else:
            interpretation = "Poor Detection ❌"
        
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return iou_score
    
    def plot_iou_sensitivity_analysis(self):
        """
        Show how IoU changes with box displacement
        Helps understand model sensitivity to localization errors
        """
        # Fixed ground truth box
        gt_box = [100, 100, 200, 200]
        
        # Test different displacements
        displacements = range(0, 101, 5)  # 0 to 100 pixels
        ious = []
        
        for disp in displacements:
            # Displace predicted box
            pred_box = [100 + disp, 100, 200 + disp, 200]
            iou = self.iou_calculator.calculate_iou(gt_box, pred_box)
            ious.append(iou)
        
        plt.figure(figsize=(10, 6))
        plt.plot(displacements, ious, 'b-', linewidth=2, marker='o')
        plt.axhline(y=0.5, color='r', linestyle='--', label='IoU = 0.5 (Good threshold)')
        plt.axhline(y=0.7, color='g', linestyle='--', label='IoU = 0.7 (Excellent threshold)')
        
        plt.xlabel('Horizontal Displacement (pixels)')
        plt.ylabel('IoU Score')
        plt.title('IoU Sensitivity to Localization Errors')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add real-world context
        plt.text(50, 0.3, 'Real-world insight:\nSmall localization errors\nstill give reasonable IoU', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
                fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def create_detection_report(self, predictions: List[Dict], ground_truths: List[Dict], 
                              image_path: str = None):
        """
        Create a comprehensive detection report with visualizations
        Real-world use: Generate reports for restaurant safety audits
        """
        evaluator = RestaurantDetectionEvaluator()
        
        # Calculate metrics
        metrics = evaluator.evaluate_single_image(predictions, ground_truths)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Detection visualization
        if image_path:
            try:
                import cv2
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax1.imshow(image)
            except:
                ax1.set_xlim(0, 640)
                ax1.set_ylim(480, 0)  # Flip y-axis for image coordinates
        else:
            ax1.set_xlim(0, 640)
            ax1.set_ylim(480, 0)
        
        # Draw ground truth boxes (green)
        for gt in ground_truths:
            x1, y1, x2, y2 = gt['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none', 
                                   label='Ground Truth')
            ax1.add_patch(rect)
        
        # Draw prediction boxes (red)
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none', 
                                   linestyle='--', label='Prediction')
            ax1.add_patch(rect)
            
            # Add confidence score
            ax1.text(x1, y1-5, f'{pred["confidence"]:.2f}', 
                    color='red', fontweight='bold')
        
        ax1.set_title('Detection Results')
        ax1.legend()
        
        # Right plot: Metrics bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax2.bar(range(len(metrics)), metric_values, 
                      color=['blue', 'red', 'green'] * (len(metrics)//3 + 1))
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metric_names, rotation=45)
        ax2.set_ylabel('Score')
        ax2.set_title('Detection Metrics')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed report
        print("\n" + "="*50)
        print("RESTAURANT SAFETY DETECTION REPORT")
        print("="*50)
        print(f"Total Predictions: {len(predictions)}")
        print(f"Total Ground Truths: {len(ground_truths)}")
        print("\nDetailed Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Business interpretation
        avg_precision = np.mean([v for k, v in metrics.items() if 'precision' in k])
        avg_recall = np.mean([v for k, v in metrics.items() if 'recall' in k])
        
        print(f"\nBUSINESS IMPACT:")
        print(f"  Precision: {avg_precision:.1%} - Accuracy of safety alerts")
        print(f"  Recall: {avg_recall:.1%} - Coverage of actual violations")
        
        if avg_precision >= 0.8 and avg_recall >= 0.7:
            print("  ✅ Model performance: EXCELLENT - Ready for production")
        elif avg_precision >= 0.6 and avg_recall >= 0.5:
            print("  ⚠️  Model performance: GOOD - Consider more training")
        else:
            print("  ❌ Model performance: NEEDS IMPROVEMENT")

# Real-world usage examples
def demonstrate_iou_concepts():
    """
    Demonstrate IoU concepts with real restaurant safety examples
    """
    print("🎯 IoU Implementation for Restaurant Safety Detection")
    print("=" * 60)
    
    iou_calc = IoUCalculator()
    visualizer = IoUVisualizer()
    
    # Example 1: Perfect detection
    print("\nExample 1: Perfect Detection")
    gt_box = [100, 50, 200, 150]  # Ground truth: hand location
    pred_box = [100, 50, 200, 150]  # Perfect prediction
    
    iou_perfect = iou_calc.calculate_iou(gt_box, pred_box)
    print(f"Perfect overlap IoU: {iou_perfect:.3f}")
    
    # Example 2: Good detection with slight offset
    print("\nExample 2: Good Detection (Slight Offset)")
    pred_box_offset = [105, 55, 205, 155]  # 5-pixel offset
    iou_good = iou_calc.calculate_iou(gt_box, pred_box_offset)
    print(f"Good detection IoU: {iou_good:.3f}")
    
    # Example 3: Poor detection
    print("\nExample 3: Poor Detection")
    pred_box_poor = [150, 100, 250, 200]  # Significant offset
    iou_poor = iou_calc.calculate_iou(gt_box, pred_box_poor)
    print(f"Poor detection IoU: {iou_poor:.3f}")
    
    # Example 4: Non-overlapping boxes
    print("\nExample 4: Complete Miss")
    pred_box_miss = [300, 300, 400, 400]  # No overlap
    iou_miss = iou_calc.calculate_iou(gt_box, pred_box_miss)
    print(f"Complete miss IoU: {iou_miss:.3f}")
    
    # Visualize the examples
    print("\n📊 Visualizing IoU Examples...")
    visualizer.plot_iou_example(gt_box, pred_box_offset, 
                               "Restaurant Hand Detection - Good Case")
    
    # Demonstrate batch processing
    print("\n⚡ Batch IoU Calculation (Production Speed)")
    gt_boxes = torch.tensor([[100, 50, 200, 150], [300, 200, 400, 300]])
    pred_boxes = torch.tensor([[105, 55, 205, 155], [295, 195, 395, 295], [150, 100, 250, 200]])
    
    batch_ious = iou_calc.calculate_iou_batch(gt_boxes, pred_boxes)
    print("Batch IoU matrix:")
    print(batch_ious)
    print("Shape: [num_ground_truths, num_predictions]")
    
    # Real-world evaluation example
    print("\n🏭 Real-world Evaluation Example")
    
    # Simulate restaurant detection results
    predictions = [
        {'bbox': [105, 55, 205, 155], 'class': 1, 'confidence': 0.85},  # Hand without gloves
        {'bbox': [295, 195, 395, 295], 'class': 0, 'confidence': 0.72},  # Hand with gloves
        {'bbox': [500, 400, 600, 500], 'class': 2, 'confidence': 0.65}   # Food item
    ]
    
    ground_truths = [
        {'bbox': [100, 50, 200, 150], 'class': 1},  # Hand without gloves
        {'bbox': [300, 200, 400, 300], 'class': 0}   # Hand with gloves
    ]
    
    # Evaluate
    evaluator = RestaurantDetectionEvaluator()
    results = evaluator.evaluate_single_image(predictions, ground_truths)
    
    print("Detection Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.3f}")
    
    # Create visual report
    visualizer.create_detection_report(predictions, ground_truths)

if __name__ == "__main__":
    demonstrate_iou_concepts()