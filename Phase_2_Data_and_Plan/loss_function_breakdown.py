import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class DetectionLossExplained:
    """
    Complete breakdown of object detection loss functions
    
    Why multi-part loss?
    Object detection has 3 distinct tasks:
    1. "Is there an object here?" (Objectness)
    2. "What type of object is it?" (Classification) 
    3. "Where exactly is it?" (Localization)
    
    Each needs its own loss function!
    """
    
    def __init__(self, num_classes=5, safety_weights=None):
        self.num_classes = num_classes
        
        # Restaurant safety classes with business criticality
        self.classes = {
            0: {'name': 'hand_with_gloves', 'criticality': 'low'},      # Safe - low weight
            1: {'name': 'hand_without_gloves', 'criticality': 'high'},  # CRITICAL - high weight
            2: {'name': 'food_contamination', 'criticality': 'high'},   # CRITICAL - high weight  
            3: {'name': 'utensil_misuse', 'criticality': 'medium'},     # Important - medium weight
            4: {'name': 'background', 'criticality': 'low'}             # Background - low weight
        }
        
        # Safety-based class weights (higher = more important)
        if safety_weights is None:
            self.class_weights = self.calculate_safety_weights()
        else:
            self.class_weights = safety_weights
        
        print("🏥 Safety-Critical Class Weights:")
        for class_id, info in self.classes.items():
            weight = self.class_weights[class_id]
            print(f"  {info['name']}: {weight:.1f}x weight ({info['criticality']} risk)")
    
    def calculate_safety_weights(self):
        """
        Calculate class weights based on business criticality
        
        Logic:
        - High criticality (safety violations): 3-5x weight
        - Medium criticality (quality issues): 2x weight  
        - Low criticality (normal operations): 1x weight
        
        Why? Missing a safety violation is much worse than a false alarm
        """
        weights = torch.ones(self.num_classes)
        
        for class_id, info in self.classes.items():
            if info['criticality'] == 'high':
                weights[class_id] = 4.0  # 4x penalty for missing critical violations
            elif info['criticality'] == 'medium':
                weights[class_id] = 2.0  # 2x penalty for quality issues
            else:
                weights[class_id] = 1.0  # Normal weight
        
        return weights
    
    def objectness_loss(self, pred_objectness, target_objectness):
        """
        LOSS COMPONENT 1: Objectness Loss
        
        Question: "Is there any object in this region?"
        - Binary classification problem
        - Pred: [batch_size, num_anchors] - probability of object presence
        - Target: [batch_size, num_anchors] - 1 if object present, 0 if background
        
        Why needed? Many anchor boxes contain no objects - need to learn this
        """
        
        # Use Binary Cross Entropy with Logits (numerically stable)
        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_objectness, target_objectness.float()
        )
        
        return objectness_loss
    
    def classification_loss(self, pred_classes, target_classes, valid_mask):
        """
        LOSS COMPONENT 2: Classification Loss
        
        Question: "What type of object is this?" (only for regions that contain objects)
        - Multi-class classification problem
        - Pred: [batch_size, num_anchors, num_classes] - class probabilities
        - Target: [batch_size, num_anchors] - ground truth class IDs
        
        Key: Only compute for positive samples (where objects actually exist)
        """
        
        # Only compute loss for positive samples (valid_mask = True where objects exist)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_classes.device)
        
        # Extract predictions for positive samples
        pos_pred_classes = pred_classes[valid_mask]  # [num_positive, num_classes]
        pos_target_classes = target_classes[valid_mask]  # [num_positive]
        
        # Apply safety-critical class weights
        weights = self.class_weights.to(pred_classes.device)
        
        # Weighted Cross Entropy Loss
        classification_loss = F.cross_entropy(
            pos_pred_classes, 
            pos_target_classes,
            weight=weights
        )
        
        return classification_loss
    
    def localization_loss(self, pred_boxes, target_boxes, valid_mask):
        """
        LOSS COMPONENT 3: Localization Loss (Bounding Box Regression)
        
        Question: "Where exactly is the object?"
        - Regression problem: predict (x, y, width, height) coordinates
        - Pred: [batch_size, num_anchors, 4] - predicted box coordinates
        - Target: [batch_size, num_anchors, 4] - ground truth box coordinates
        
        Why Smooth L1? Less sensitive to outliers than L2, more stable training
        """
        
        # Only compute for positive samples
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        pos_pred_boxes = pred_boxes[valid_mask]  # [num_positive, 4]
        pos_target_boxes = target_boxes[valid_mask]  # [num_positive, 4]
        
        # Smooth L1 Loss (also called Huber Loss)
        # L1 loss for small errors, L2 loss for large errors
        localization_loss = F.smooth_l1_loss(pos_pred_boxes, pos_target_boxes)
        
        return localization_loss
    
    def compute_total_loss(self, predictions, targets, loss_weights=None):
        """
        COMBINE ALL LOSSES with proper weighting
        
        Args:
            predictions: dict with 'objectness', 'classes', 'boxes'
            targets: dict with 'objectness', 'classes', 'boxes'  
            loss_weights: dict with weights for each loss component
        """
        
        if loss_weights is None:
            # Default weights - these are hyperparameters you tune!
            loss_weights = {
                'objectness': 1.0,     # Binary classification weight
                'classification': 2.0,  # Class prediction weight (more important)
                'localization': 1.0     # Box regression weight
            }
        
        # Extract predictions and targets
        pred_objectness = predictions['objectness']  # [batch, anchors]
        pred_classes = predictions['classes']        # [batch, anchors, classes] 
        pred_boxes = predictions['boxes']            # [batch, anchors, 4]
        
        target_objectness = targets['objectness']
        target_classes = targets['classes']
        target_boxes = targets['boxes']
        
        # Valid mask: where objects actually exist (objectness = 1)
        valid_mask = target_objectness > 0.5
        
        # Compute individual losses
        obj_loss = self.objectness_loss(pred_objectness, target_objectness)
        cls_loss = self.classification_loss(pred_classes, target_classes, valid_mask)
        loc_loss = self.localization_loss(pred_boxes, target_boxes, valid_mask)
        
        # Weighted total loss
        total_loss = (
            loss_weights['objectness'] * obj_loss +
            loss_weights['classification'] * cls_loss + 
            loss_weights['localization'] * loc_loss
        )
        
        # Return detailed breakdown for monitoring
        loss_info = {
            'total_loss': total_loss,
            'objectness_loss': obj_loss,
            'classification_loss': cls_loss, 
            'localization_loss': loc_loss,
            'num_positive_samples': valid_mask.sum().item()
        }
        
        return total_loss, loss_info

class SafetyCriticalWeighting:
    """
    Advanced techniques for handling safety-critical class imbalance
    """
    
    def __init__(self):
        # Restaurant safety violation frequencies (from real data)
        self.class_frequencies = {
            'hand_with_gloves': 0.60,      # 60% of detections (common, safe)
            'hand_without_gloves': 0.05,   # 5% of detections (rare, CRITICAL)
            'food_contamination': 0.02,    # 2% of detections (very rare, CRITICAL)
            'utensil_misuse': 0.10,        # 10% of detections (uncommon, important)
            'background': 0.23             # 23% background regions
        }
        
    def inverse_frequency_weights(self):
        """
        Method 1: Inverse Frequency Weighting
        
        Logic: Rare classes get higher weights
        Formula: weight = 1 / frequency
        
        Problem: Can make very rare classes dominate training
        """
        weights = {}
        for class_name, freq in self.class_frequencies.items():
            weights[class_name] = 1.0 / freq
            
        # Normalize weights so they sum to num_classes
        total_weight = sum(weights.values())
        num_classes = len(weights)
        
        normalized_weights = {
            class_name: (weight / total_weight) * num_classes
            for class_name, weight in weights.items()
        }
        
        return normalized_weights
    
    def effective_number_weights(self, beta=0.99):
        """
        Method 2: Effective Number of Samples
        
        Logic: Account for overlap between samples
        Formula: weight = (1 - beta) / (1 - beta^n)
        
        Better than inverse frequency - prevents extreme weights
        """
        weights = {}
        
        # Simulate sample counts (in real scenario, count actual samples)
        total_samples = 10000
        
        for class_name, freq in self.class_frequencies.items():
            n_samples = int(total_samples * freq)
            effective_num = (1 - beta) / (1 - beta**n_samples)
            weights[class_name] = 1.0 / effective_num
            
        return weights
    
    def focal_loss_alpha_weights(self):
        """
        Method 3: Focal Loss Alpha Weighting
        
        Logic: Combine frequency weighting with focus on hard examples
        Alpha controls class balance, gamma controls hard example focus
        """
        # Alpha values - higher for rare, critical classes
        alpha_weights = {
            'hand_with_gloves': 0.25,      # Common, safe
            'hand_without_gloves': 0.75,   # Rare, critical  
            'food_contamination': 0.85,    # Very rare, critical
            'utensil_misuse': 0.60,        # Uncommon, important
            'background': 0.20             # Common, background
        }
        
        return alpha_weights
    
    def business_impact_weights(self):
        """
        Method 4: Business Impact Weighting
        
        Logic: Weight based on actual business consequences
        - Cost of missing violation
        - Cost of false alarm
        - Regulatory requirements
        """
        
        # Business impact analysis
        impact_costs = {
            'hand_with_gloves': {
                'miss_cost': 0,      # No cost - it's safe
                'false_alarm_cost': 50,  # Minor staff interruption
                'weight': 1.0
            },
            'hand_without_gloves': {
                'miss_cost': 5000,   # Health code violation fine
                'false_alarm_cost': 100,  # Staff retraining
                'weight': 8.0        # Very high weight
            },
            'food_contamination': {
                'miss_cost': 50000,  # Foodborne illness lawsuit
                'false_alarm_cost': 200,  # Food waste
                'weight': 10.0       # Highest weight
            },
            'utensil_misuse': {
                'miss_cost': 1000,   # Minor violation
                'false_alarm_cost': 75,   # Workflow disruption  
                'weight': 3.0        # Moderate weight
            },
            'background': {
                'miss_cost': 0,
                'false_alarm_cost': 0,
                'weight': 0.5        # Low weight
            }
        }
        
        return {k: v['weight'] for k, v in impact_costs.items()}

class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance
    
    Key insight: Focus learning on hard examples, down-weight easy examples
    Formula: FL(p) = -α(1-p)^γ * log(p)
    
    - α (alpha): Class balance weight
    - γ (gamma): Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [N, C] class logits
            targets: [N] class labels
        """
        
        # Convert logits to probabilities
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t in the paper
        
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Demonstration and Visualization
def demonstrate_loss_components():
    """
    Demonstrate how different loss components work together
    """
    print("🔍 OBJECT DETECTION LOSS BREAKDOWN")
    print("=" * 50)
    
    # Initialize loss calculator
    loss_calculator = DetectionLossExplained(num_classes=5)
    
    # Create synthetic batch data
    batch_size, num_anchors = 4, 100
    
    # Simulate predictions (what model outputs)
    predictions = {
        'objectness': torch.randn(batch_size, num_anchors),  # Raw logits
        'classes': torch.randn(batch_size, num_anchors, 5),  # Class logits
        'boxes': torch.randn(batch_size, num_anchors, 4)     # Box coordinates
    }
    
    # Simulate targets (ground truth)
    targets = {
        'objectness': torch.randint(0, 2, (batch_size, num_anchors)).float(),
        'classes': torch.randint(0, 5, (batch_size, num_anchors)),
        'boxes': torch.randn(batch_size, num_anchors, 4)
    }
    
    # Compute loss
    total_loss, loss_breakdown = loss_calculator.compute_total_loss(predictions, targets)
    
    print("📊 Loss Component Analysis:")
    print(f"  Total Loss: {total_loss:.4f}")
    print(f"  ├── Objectness Loss: {loss_breakdown['objectness_loss']:.4f}")
    print(f"  ├── Classification Loss: {loss_breakdown['classification_loss']:.4f}")
    print(f"  └── Localization Loss: {loss_breakdown['localization_loss']:.4f}")
    print(f"  Positive samples: {loss_breakdown['num_positive_samples']}")
    
    # Show effect of different loss weights
    print(f"\n⚖️  Effect of Different Loss Weighting:")
    
    weight_scenarios = [
        {'name': 'Balanced', 'weights': {'objectness': 1.0, 'classification': 1.0, 'localization': 1.0}},
        {'name': 'Class-focused', 'weights': {'objectness': 0.5, 'classification': 3.0, 'localization': 1.0}},
        {'name': 'Localization-focused', 'weights': {'objectness': 0.5, 'classification': 1.0, 'localization': 3.0}},
    ]
    
    for scenario in weight_scenarios:
        total_loss, _ = loss_calculator.compute_total_loss(
            predictions, targets, scenario['weights']
        )
        print(f"  {scenario['name']}: {total_loss:.4f}")
    
    # Demonstrate safety-critical weighting
    print(f"\n🚨 Safety-Critical Class Weighting Impact:")
    
    safety_weighter = SafetyCriticalWeighting()
    
    # Show different weighting strategies
    strategies = {
        'Inverse Frequency': safety_weighter.inverse_frequency_weights(),
        'Effective Number': safety_weighter.effective_number_weights(),
        'Business Impact': safety_weighter.business_impact_weights()
    }
    
    for strategy_name, weights in strategies.items():
        print(f"\n  {strategy_name} Strategy:")
        for class_name, weight in weights.items():
            if 'gloves' in class_name or 'contamination' in class_name:
                print(f"    {class_name}: {weight:.2f}x weight")

def visualize_loss_curves():
    """
    Visualize how different loss components behave during training
    """
    
    # Simulate training progression
    epochs = np.arange(1, 101)
    
    # Typical loss curves (decreasing with different rates)
    objectness_loss = 2.0 * np.exp(-epochs / 30) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    classification_loss = 3.0 * np.exp(-epochs / 25) + 0.2 + np.random.normal(0, 0.03, len(epochs))
    localization_loss = 1.5 * np.exp(-epochs / 35) + 0.3 + np.random.normal(0, 0.02, len(epochs))
    
    total_loss = objectness_loss + classification_loss + localization_loss
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual loss components
    plt.subplot(2, 2, 1)
    plt.plot(epochs, objectness_loss, label='Objectness Loss', color='blue', alpha=0.7)
    plt.plot(epochs, classification_loss, label='Classification Loss', color='red', alpha=0.7)
    plt.plot(epochs, localization_loss, label='Localization Loss', color='green', alpha=0.7)
    plt.plot(epochs, total_loss, label='Total Loss', color='black', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show effect of class weighting
    plt.subplot(2, 2, 2)
    
    # Simulate different weighting scenarios
    uniform_weights = np.ones(5)
    safety_weights = np.array([1.0, 4.0, 5.0, 2.0, 1.0])  # High weight for violations
    
    classes = ['Hand+Gloves', 'Hand-Gloves', 'Contamination', 'Utensil Misuse', 'Background']
    x_pos = np.arange(len(classes))
    
    plt.bar(x_pos - 0.2, uniform_weights, 0.4, label='Uniform Weights', alpha=0.7)
    plt.bar(x_pos + 0.2, safety_weights, 0.4, label='Safety-Critical Weights', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Loss Weight')
    plt.title('Class Weight Comparison')
    plt.xticks(x_pos, classes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show focal loss effect
    plt.subplot(2, 2, 3)
    
    # Focal loss vs standard cross-entropy
    confidence = np.linspace(0.01, 0.99, 100)
    
    # Standard cross-entropy: -log(p)
    ce_loss = -np.log(confidence)
    
    # Focal loss with different gamma values
    focal_gamma_1 = -(1 - confidence)**1 * np.log(confidence)
    focal_gamma_2 = -(1 - confidence)**2 * np.log(confidence) 
    focal_gamma_5 = -(1 - confidence)**5 * np.log(confidence)
    
    plt.plot(confidence, ce_loss, label='Cross Entropy', linewidth=2)
    plt.plot(confidence, focal_gamma_1, label='Focal Loss (γ=1)', alpha=0.8)
    plt.plot(confidence, focal_gamma_2, label='Focal Loss (γ=2)', alpha=0.8)
    plt.plot(confidence, focal_gamma_5, label='Focal Loss (γ=5)', alpha=0.8)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Loss Value')
    plt.title('Focal Loss vs Cross Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Show business impact weighting
    plt.subplot(2, 2, 4)
    
    violation_types = ['Hand-Gloves', 'Contamination', 'Utensil Misuse']
    miss_costs = [5000, 50000, 1000]  # Cost of missing each violation
    false_alarm_costs = [100, 200, 75]  # Cost of false alarm
    
    x_pos = np.arange(len(violation_types))
    width = 0.35
    
    plt.bar(x_pos - width/2, miss_costs, width, label='Miss Cost ($)', alpha=0.8)
    plt.bar(x_pos + width/2, false_alarm_costs, width, label='False Alarm Cost ($)', alpha=0.8)
    plt.xlabel('Violation Type')
    plt.ylabel('Cost ($)')
    plt.title('Business Impact of Detection Errors')
    plt.xticks(x_pos, violation_types, rotation=45)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📈 Key Insights from Loss Analysis:")
    print("1. Classification loss typically dominates early training")
    print("2. Localization loss is harder to optimize (geometric reasoning)")  
    print("3. Safety-critical weighting prevents model from ignoring rare violations")
    print("4. Focal loss helps with extreme class imbalance")
    print("5. Business impact weighting aligns model with real costs")

if __name__ == "__main__":
    demonstrate_loss_components()
    visualize_loss_curves()