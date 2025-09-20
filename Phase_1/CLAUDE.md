# CLAUDE.md - Open Your Eyes: Universal Video Intelligence Platform

## Project Overview

**Mission**: Building a next-generation platform that transforms raw video streams from any camera into structured, actionable insights across any domain.

**Vision**: Create a general-purpose AI vision system that can understand and analyze any scene - from indoor rooms to outdoor environments, people interactions, object detection, and behavioral analysis.

## Phase 1: Universal Visual Intelligence Foundation

### Architecture Evolution

Evolving from domain-specific detection to a general-purpose visual understanding system:

#### 1. Domain-Specific Approach (Initial Restaurant Focus)
- **Learning**: Specialized models perform well but lack transferability
- **Challenge**: Each new use case required complete retraining
- **Insight**: Need for universal foundation models

#### 2. **Current Universal Architecture: SAM + CLIP Pipeline**
- **Philosophy**: Zero-shot understanding of any visual scene
- **Components**:
  - **SAM (Segment Anything Model)**: Universal object segmentation for any domain
  - **CLIP**: Open-vocabulary classification for any object or concept
- **Advantages**:
  - Works on any scene: rooms, outdoor, people, objects, activities
  - No domain-specific training required
  - Infinitely extensible through natural language prompts

#### 3. Testing Methodology: Phone Camera Exploration
- **Approach**: Use phone camera to test diverse scenarios
- **Environments**: Indoor rooms, outdoor scenes, people interactions, object arrangements
- **Goal**: Build robust general-purpose understanding

### Universal Capabilities Being Developed

**Scene Understanding**:
- 🏠 Indoor environments (rooms, furniture, layouts)
- 🌳 Outdoor scenes (nature, urban, weather conditions)
- 👥 People analysis (pose, clothing, activities, interactions)
- 📦 Object detection (any object via natural language)
- 🎯 Activity recognition (eating, working, walking, etc.)
- 🔍 Anomaly detection (unusual patterns or behaviors)

**Cross-Domain Applications**:
- Security monitoring (any environment)
- Inventory management (any items)
- Behavioral analysis (any activities)
- Quality control (any process)
- Safety compliance (any industry)
- Content creation (scene understanding)

## Technical Architecture

### Universal Pipeline (SAM + CLIP)

```
Any Video Stream (Phone, Webcam, Security Camera)
     ↓
Snapshot Capture (Adaptive intervals)
     ↓
SAM Universal Segmentation (Finds all objects/regions)
     ↓
CLIP Open-Vocabulary Classification (Understands anything via text)
     ↓
General Intelligence Engine (Context-aware reasoning)
     ↓
Structured Insights & Actions
```

### Key Files and Functions

**Universal Pipeline Scripts**:
- `step6_CLIP_with_segmentation_and_boxes.py` - Core SAM+CLIP implementation
- `step4_CLIP_out_of_realtime.py` - CLIP baseline for any image
- `step0_restaurant_object_classification.py` - Framework adaptable to any domain

**Experimental Scripts**:
- `step3_yolo_flavors.py` - Baseline object detection comparison
- `test_mac_cameras.py` - Camera integration for any device

### Universal Model Requirements

**SAM Models** (Segment Everything):
- `sam_vit_h_4b8939.pth` (Best quality, ~2.5GB)
- Works on any image: indoor, outdoor, people, objects
- Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

**CLIP Models** (Understand Everything):
- ViT-B/32 (Fast, good for exploration)
- ViT-L/14 (Best understanding, slower)
- Can classify anything you can describe in words

## Development Guidelines

### Universal Testing with Phone Camera

**Quick Start - Test Anything**:
```bash
# Point phone camera at any scene
python step4_CLIP_out_of_realtime.py
```

**Full Universal Pipeline**:
```bash
python step6_CLIP_with_segmentation_and_boxes.py --cam 0 --interval 3
```

### Example Universal Prompts

**Indoor Scenes**:
```python
text_prompts = [
    "person sitting on couch",
    "messy room",
    "clean organized space",
    "electronic devices",
    "books on shelf",
    "plants in room",
    "natural lighting",
    "artificial lighting"
]
```

**Outdoor Scenes**:
```python
text_prompts = [
    "trees and nature",
    "urban buildings",
    "cars on street",
    "people walking",
    "sunny weather",
    "cloudy sky",
    "bicycle parked",
    "street signs"
]
```

**People Analysis**:
```python
text_prompts = [
    "person wearing casual clothes",
    "person in formal attire",
    "person exercising",
    "person eating food",
    "person using phone",
    "group of people talking",
    "person sitting alone",
    "person working on laptop"
]
```

**General Objects**:
```python
text_prompts = [
    "food items",
    "technology devices",
    "furniture pieces",
    "clothing items",
    "sports equipment",
    "art and decorations",
    "cleaning supplies",
    "vehicles"
]
```

## Path to General Intelligence

### Stage 1: Universal Foundation (Current)
- ✅ SAM + CLIP integration working
- ✅ Phone camera testing across diverse scenes
- 🔄 Building comprehensive prompt libraries
- 🔄 Performance optimization for real-time use

### Stage 2: Context Understanding
- **Goal**: System understands relationships between objects
- **Approach**: Multi-object reasoning, spatial relationships
- **Example**: "Person cooking in kitchen" vs "Person eating in dining room"

### Stage 3: Temporal Intelligence
- **Goal**: Understand activities and changes over time
- **Approach**: Multi-frame analysis, action recognition
- **Example**: "Person walking into room" vs "Person leaving room"

### Stage 4: Adaptive Learning
- **Goal**: System improves through exposure to new scenarios
- **Approach**: Few-shot learning, prompt optimization
- **Example**: Better understanding of user's specific environment

## Universal Testing Protocol

### Phone Camera Experiments

**Scenario Coverage**:
1. **Indoor Spaces**: Living room, kitchen, bedroom, office, bathroom
2. **Outdoor Environments**: Park, street, garden, parking lot, beach
3. **People Activities**: Eating, working, exercising, socializing, sleeping
4. **Object Interactions**: Cooking, cleaning, reading, gaming, crafting
5. **Lighting Conditions**: Natural, artificial, dim, bright, mixed
6. **Weather/Time**: Morning, afternoon, evening, sunny, cloudy, rain

**Testing Method**:
```bash
# Start universal monitoring
python step6_CLIP_with_segmentation_and_boxes.py --cam 0 --interval 5

# Test different prompts for same scene
# Modify text_prompts in the script for your specific test case
```

### Performance Optimization

**For General Use**:
- Adaptive snapshot intervals based on scene change detection
- Dynamic prompt selection based on detected objects
- Efficient processing for mobile/edge deployment
- Cross-platform compatibility (phone, laptop, desktop)

**Hardware Scaling**:
- **Phone**: CLIP-only mode for lightweight analysis
- **Laptop**: Full SAM + CLIP for detailed understanding
- **Desktop/Server**: Multi-camera, real-time processing
- **Edge**: Optimized models for IoT deployment

## Real-World Applications

### Immediate Use Cases
- **Home Monitoring**: General security and activity tracking
- **Content Creation**: Automatic scene description and tagging
- **Accessibility**: Scene description for visually impaired
- **Productivity**: Activity tracking and environment optimization
- **Research**: Human behavior studies across environments

### Scalable Applications
- **Smart City**: Traffic, crowd, and infrastructure monitoring
- **Retail**: Customer behavior and inventory management
- **Healthcare**: Patient monitoring and safety compliance
- **Education**: Learning environment optimization
- **Manufacturing**: Quality control and safety monitoring

## Success Metrics for General Intelligence

### Technical Benchmarks
- **Universal Accuracy**: >85% correct classification across all domains
- **Zero-Shot Performance**: Handle new scenarios without retraining
- **Real-Time Processing**: <3 seconds per analysis cycle
- **Cross-Domain Stability**: Consistent performance indoor/outdoor/people

### User Experience
- **Ease of Use**: Point camera, get instant understanding
- **Customization**: Natural language queries for any use case
- **Reliability**: Works in various lighting and environmental conditions
- **Scalability**: From single phone to multi-camera systems

## Quick Start - Test Your World

```bash
# Basic universal understanding
python static_Images_VisualModels/step4_CLIP_out_of_realtime.py

# Full scene analysis
python static_Images_VisualModels/step6_CLIP_with_segmentation_and_boxes.py --cam 0

# Test camera setup
python static_Images_VisualModels/test_mac_cameras.py
```

**Experiment Ideas**:
1. Point camera at your room - see what it detects
2. Go outside - test urban vs nature scenes
3. Film people activities - analyze behaviors
4. Test different lighting conditions
5. Try with objects, pets, food, electronics

This platform is the foundation for turning any camera into an intelligent observer that understands the world like a human would - but with the consistency and scalability of AI.