# Kitchen Safety Detection Dataset Preparation Guide

# We hit a bottleneck with the impossibility to gather quality Dataset
# Let's change context/purpose to where we can use some already existing dataset.

# Anyway - in real world implementation for restaurant it'd go like this:

Returning to a solid foundation like YOLO or DETR and focusing on dataset preparation is the smart move. Your research notes show that "clean data and thoughtful training methodology trump raw architectural complexity" - this is spot on.

Let me break down a systematic approach to dataset preparation for kitchen safety detection:

## Dataset Preparation Strategy

### 1. Define Your Safety Classes First

Before collecting any images, clearly define what you want to detect:

**Core Safety Categories:**
* Hand protection: `gloves_on`, `gloves_off`, `bare_hands_danger`
* Object handling: `knife_proper_grip`, `knife_unsafe`, `hot_surface_contact`
* Spills/hazards: `liquid_spill`, `grease_fire`, `cluttered_workspace`
* PPE compliance: `apron_worn`, `hair_covered`, `proper_footwear`

### 2. Data Collection Approaches

**Option A: Staged Data Collection** (Recommended for starting)
* Set up controlled kitchen scenarios
* Have actors demonstrate both safe and unsafe behaviors
* Capture multiple angles, lighting conditions
* Pro: High control, balanced classes, clear labeling
* Con: May not capture real-world variability

**Option B: Real Kitchen Footage**
* Partner with restaurants/culinary schools
* Record actual kitchen work (with permission)
* Pro: Real-world scenarios, natural behaviors
* Con: Class imbalance, harder to label, privacy concerns

**Option C: Synthetic + Real Hybrid**
* Start with staged data for baseline
* Gradually incorporate real footage
* Use data augmentation heavily

### 3. Annotation Strategy

**Bounding Box Annotations:**

```
image_001.jpg → gloves_on: [x1,y1,x2,y2], knife_proper: [x1,y1,x2,y2]
image_002.jpg → bare_hands_danger: [x1,y1,x2,y2], hot_surface: [x1,y1,x2,y2]
```

**Tools to Consider:**
* **LabelImg** (simple, free)
* **CVAT** (more advanced, supports teams)
* **Roboflow** (cloud-based, good for YOLO format export)

### 4. Data Quality Guidelines

**Image Quality Standards:**
* Resolution: 640x640 minimum (YOLO standard)
* Lighting: Varied conditions (bright kitchen, dim corners)
* Angles: Multiple viewpoints of same scenario
* Occlusion: Partial hand/object visibility

**Annotation Quality:**
* Tight bounding boxes around objects
* Consistent labeling across annotators
* Edge cases clearly defined (e.g., glove partially visible)

### 5. Dataset Size Targets

**Minimum Viable Dataset:**
* 500-1000 images per class
* 80/10/10 train/val/test split
* Balanced representation across conditions

**Production-Ready Dataset:**
* 2000+ images per class
* Multiple annotators for quality control
* Extensive validation set for real-world scenarios

### 6. Quick Start Approach

Since you're researching, here's a rapid prototyping path:

1. **Week 1**: Collect 200 staged images (10 scenarios × 20 variations)
2. **Week 2**: Annotate using LabelImg, export to YOLO format
3. **Week 3**: Train YOLOv8 baseline, evaluate performance
4. **Week 4**: Identify failure cases, collect targeted additional data

### 7. Data Augmentation Strategy

```python
# Example augmentation pipeline
transforms = [
    RandomBrightnessContrast(p=0.5),
    RandomRotate90(p=0.3),
    HorizontalFlip(p=0.5),
    Cutout(max_h_size=50, max_w_size=50, p=0.3),
    GaussNoise(p=0.2)
]
```

### 8. Validation Methodology

**Key Metrics to Track:**
* mAP@0.5 (standard object detection metric)
* Per-class precision/recall (identify weak classes)
* Confusion matrices (understand misclassification patterns)
* Real-world accuracy on held-out kitchen footage





### Pursuing a task from existing dataset - we find 2 pathways (vboth dataset and approaches dependent on context)

Existing Datasets for Safety/Object Detection
1. Open Images V7

What it has: 9M images, 600 classes including kitchen items
Relevant classes: Hand, Glove, Kitchen knife, Cooking spray, Oven
Advantage: More diverse, real-world images
Challenge: More noise, inconsistent quality

2. Epic-Kitchens Dataset

What it has: 100 hours of kitchen activities, first-person view
Perfect for: Hand-object interactions, kitchen tool usage
Unique value: Temporal sequences showing proper/improper handling
Format: Video frames + object bounding boxes

Recommendation:
Start with Open Images V7 because:

✅ Immediate access to 9M images
✅ Pre-existing bounding box annotations
✅ No video file management
✅ Direct YOLO training pipeline
✅ Can have a working model in 2-3 hours

Epic-Kitchens is better for:

Advanced temporal analysis
Understanding action sequences
More sophisticated safety context
Research-grade benchmarking