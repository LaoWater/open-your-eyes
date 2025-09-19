# Visual Models Research - Kitchen Safety Detection

## Project Context

This document chronicles our research journey exploring various visual models for developing an intelligent kitchen safety detection system. The goal is to identify and classify safety hazards in kitchen environments, including proper glove usage, dangerous object handling, and general safety compliance monitoring.

## Key Takeaways for Restaurant Safety Detection

### 1. COCO-based Detectors (YOLOv8, DETR)
- **Advantages**: Fast and easy baseline implementation
- **Limitations**: Won't detect gloves or subtle kitchen hazards without fine-tuning
- **Use Case**: Quick prototyping and general object detection foundation

### 2. Open-vocabulary Models (Grounding DINO / OWL-ViT)
- **Advantages**: Can detect *any object you describe* (e.g., "hand wearing glove") without explicit training
- **Limitations**: Slower inference, may require GPU for real-time applications
- **Use Case**: Flexible detection of custom safety-related objects

### 3. Hybrid Strategy (Recommended Approach)
- **Stage 1**: Fast detector (YOLO) → identify "human hands, objects of interest"
- **Stage 2**: Specialized classifier (fine-tuned ResNet/EfficientNet) → "gloves on/off, dangerous object classification"
- **Optional Stage 3**: Importance scoring → only analyze relevant regions (AGI-style foveated attention)

## Research Journey

### Phase 1: YOLO Exploration

**Initial Enthusiasm**: YOLO showed promise, especially with model scaling (larger variants). The architecture's elegance and speed made it an attractive baseline.

**Implementation**: Tested both real-time detection and single snapshot processing:
- Real-time detection from live camera feed
- Static image analysis from local files

**Results**: Disappointing performance
- Model defaulted to highest confidence predictions from given labels
- Extremely poor accuracy for kitchen-specific safety scenarios
- Baseline architecture limitations became apparent

**Key Insight**: The foundation architecture is sound, but clean, domain-specific data and proper training methodology are crucial. *The true art lies in the data preparation and training process.*

### Phase 2: SAM + CLIP Integration (Current Focus)

**Architectural Shift**: Moving from real-time constraints to intelligent snapshot analysis

**Core Philosophy**: 
> "Live video feed doesn't need every frame analyzed. Snapshots every few seconds mirror human monitoring behavior."

**SAM (Segment Anything Model)**:
- **Function**: Universal object segmentation - finds and masks every object in snapshots
- **Identifies**: Knives, spilled liquids, hands, utensils, containers, etc.
- **Optimization Strategies**:
  - Use smaller SAM backbone (vit_b or vit_l instead of vit_h)
  - Reduce image resolution (--max_dim parameter)
  - Increase snapshot intervals (5-15 seconds)

**CLIP (Contrastive Language-Image Pre-training)**:
- **Function**: Semantic understanding - scores each mask against custom safety tags
- **Process**: "What is this thing?" evaluation for each segmented object
- **Efficiency**: Only compute CLIP on promising masks, not entire image pixels

### Technical Challenges

**Hardware Limitations**: 
- MAC system bottleneck preventing proper SAM_vit_h loading and handling
- **Solution**: Migration to PC environment
- **Resource**: SAM model download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Future Directions

### Immediate Next Steps
1. Complete SAM + CLIP integration on PC environment
2. Develop custom safety tag taxonomy for kitchen environments
3. Create evaluation pipeline for safety detection accuracy

### Long-term Strategy
1. **Data Collection**: Build comprehensive dataset of kitchen safety scenarios
2. **Fine-tuning**: Adapt models for specific safety detection tasks
3. **Real-world Testing**: Deploy in actual kitchen environments for validation
4. **Performance Optimization**: Balance accuracy with computational efficiency

## Architectural Philosophy

The evolution from real-time detection to intelligent snapshot analysis reflects a deeper understanding of the problem domain. Human safety monitors don't analyze every millisecond of activity - they observe, assess, and respond to meaningful events. Our system should mirror this cognitive approach, focusing computational resources on the most relevant moments and objects.

**Core Principle**: *Clean data and thoughtful training methodology trump raw architectural complexity.*


------


next steps could be:

A working prototype combining CLIP + Grounding DINO:

DINO → finds object boxes
CLIP → scores each box against your safety prompts → outputs “gloves on/off, dirty floor, fire hazard”