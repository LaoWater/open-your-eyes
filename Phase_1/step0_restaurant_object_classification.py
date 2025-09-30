import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import json

class RestaurantSafetyDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Faster R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Complete COCO classes (91 classes)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Custom safety rules
        self.safety_rules = {
            'hands_near_food': {'requires_gloves': True, 'min_distance': 50},
            'handling_utensils': {'requires_gloves': True},
            'cash_handling': {'requires_gloves': False, 'wash_hands_after': True}
        }
    
    def preprocess_image(self, image):
        """Convert image for model input"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Convert to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image_pil).to(self.device)
    
    def detect_objects(self, image):
        """Detect all objects in image"""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image).unsqueeze(0)
            predictions = self.model(input_tensor)
            
            return predictions[0]
    
    def analyze_safety_violations(self, image, confidence_threshold=0.5):
        """Main safety analysis function"""
        # Step 1: Detect all objects
        detections = self.detect_objects(image)
        
        # Step 2: Filter confident detections
        confident_detections = self.filter_confident_detections(
            detections, confidence_threshold
        )
        
        # Step 3: Apply safety logic
        violations = self.apply_safety_rules(confident_detections, image)
        
        return {
            'detections': confident_detections,
            'violations': violations,
            'safety_score': self.calculate_safety_score(violations)
        }
    
    def filter_confident_detections(self, detections, threshold):
        """Filter detections above confidence threshold"""
        boxes = detections['boxes'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        
        confident_objects = []
        
        for i, score in enumerate(scores):
            if score > threshold:
                label_idx = int(labels[i])
                # Safety check for label index
                if 0 <= label_idx < len(self.coco_classes):
                    confident_objects.append({
                        'bbox': boxes[i],
                        'label': self.coco_classes[label_idx],
                        'confidence': float(score),
                        'center': self.get_bbox_center(boxes[i])
                    })
        
        return confident_objects
    
    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def apply_safety_rules(self, detections, image):
        """Apply restaurant safety rules"""
        violations = []
        
        # Find people in the image
        people = [d for d in detections if d['label'] == 'person']
        
        for person in people:
            person_violations = self.check_person_safety(person, detections, image)
            violations.extend(person_violations)
        
        return violations
    
    def check_person_safety(self, person, all_detections, image):
        """Check safety for a specific person"""
        violations = []
        person_bbox = person['bbox']
        
        # Extract hand regions
        hand_regions = self.estimate_hand_locations(person_bbox, image)
        
        for hand_region in hand_regions:
            # Check if gloves are present
            has_gloves = self.detect_gloves_in_region(hand_region, image)
            
            # Check if hands are near food/prep areas
            near_food = self.check_proximity_to_food(hand_region, all_detections)
            
            if near_food and not has_gloves:
                violations.append({
                    'type': 'missing_gloves_near_food',
                    'severity': 'high',
                    'location': hand_region,
                    'person_id': id(person),
                    'description': 'Hands near food without gloves'
                })
        
        return violations
    
    def estimate_hand_locations(self, person_bbox, image):
        """Simplified hand detection"""
        x1, y1, x2, y2 = person_bbox
        person_width = x2 - x1
        person_height = y2 - y1
        
        # Estimate hand locations (rough approximation)
        left_hand = [x1, y1 + person_height * 0.3, x1 + person_width * 0.2, y1 + person_height * 0.7]
        right_hand = [x2 - person_width * 0.2, y1 + person_height * 0.3, x2, y1 + person_height * 0.7]
        
        return [left_hand, right_hand]
    
    def detect_gloves_in_region(self, hand_region, image):
        """Detect gloves in hand region - simplified color-based detection"""
        # Extract hand region from image
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in hand_region]
        
        # Clamp coordinates to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        hand_crop = image[y1:y2, x1:x2]
        
        if hand_crop.size == 0:
            return False
            
        # Simple color-based detection (blue/white gloves)
        hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common glove colors
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        glove_pixels = cv2.countNonZero(blue_mask) + cv2.countNonZero(white_mask)
        total_pixels = hand_crop.shape[0] * hand_crop.shape[1]
        
        # If >30% of hand region is glove color, assume gloves
        return (glove_pixels / total_pixels) > 0.3 if total_pixels > 0 else False
    
    def check_proximity_to_food(self, hand_region, all_detections):
        """Check if hands are near food items"""
        hand_center = self.get_bbox_center(hand_region)
        
        # Food-related objects (from COCO classes)
        food_items = ['bowl', 'cup', 'fork', 'knife', 'spoon', 'banana', 'apple', 
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'bottle', 'wine glass', 'dining table']
        
        for detection in all_detections:
            if detection['label'] in food_items:
                food_center = detection['center']
                distance = np.sqrt((hand_center[0] - food_center[0])**2 + 
                                 (hand_center[1] - food_center[1])**2)
                
                if distance < 100:  # pixels - tune this threshold
                    return True
        
        return False
    
    def calculate_safety_score(self, violations):
        """Calculate overall safety score"""
        if not violations:
            return 100.0
        
        penalty_per_violation = {'high': 30, 'medium': 15, 'low': 5}
        total_penalty = sum(penalty_per_violation.get(v['severity'], 10) for v in violations)
        
        return max(0, 100 - total_penalty)
    
    def visualize_results(self, image, analysis_results):
        """Draw detections and violations on image"""
        vis_image = image.copy()
        
        # Draw object detections
        for detection in analysis_results['detections']:
            bbox = detection['bbox'].astype(int)
            label = f"{detection['label']}: {detection['confidence']:.2f}"
            
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(vis_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw violations
        for violation in analysis_results['violations']:
            bbox = [int(x) for x in violation['location']]
            color = (0, 0, 255) if violation['severity'] == 'high' else (0, 165, 255)
            
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            cv2.putText(vis_image, violation['type'], (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add safety score
        score = analysis_results['safety_score']
        score_color = (0, 255, 0) if score > 80 else (0, 165, 255) if score > 60 else (0, 0, 255)
        cv2.putText(vis_image, f"Safety Score: {score:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
        
        return vis_image

# Real-time monitoring example
def monitor_restaurant_safety():
    detector = RestaurantSafetyDetector()
    cap = cv2.VideoCapture(0)  # Your camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Restaurant Safety Monitor Active")
    print("Press 'q' to quit, 's' to save current analysis")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Analyze every 5th frame (performance optimization)
        if frame_count % 5 == 0:
            try:
                analysis = detector.analyze_safety_violations(frame)
                vis_frame = detector.visualize_results(frame, analysis)
                
                # Log violations
                if analysis['violations']:
                    print(f"VIOLATIONS DETECTED: {len(analysis['violations'])}")
                    for v in analysis['violations']:
                        print(f"  - {v['type']}: {v['description']}")
            except Exception as e:
                print(f"Error during analysis: {e}")
                vis_frame = frame
        else:
            vis_frame = frame
        
        cv2.imshow('Restaurant Safety Monitor', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save analysis to file
            timestamp = cv2.getTickCount()
            cv2.imwrite(f'safety_check_{timestamp}.jpg', vis_frame)
            
            try:
                analysis = detector.analyze_safety_violations(frame)
                with open(f'safety_report_{timestamp}.json', 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                
                print(f"Analysis saved! Safety Score: {analysis['safety_score']:.1f}")
            except Exception as e:
                print(f"Error saving analysis: {e}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_restaurant_safety()