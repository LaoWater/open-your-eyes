# Open Your Eyes — Universal Video Intelligence Exploration

> Building toward a general-purpose vision system that can turn raw video streams into structured, contextual intelligence across any environment.

---

## Why This Exists

- **Mission:** Grow from domain-specific restaurant safety monitoring into a universal visual intelligence platform.
- **Vision:** Combine the speed of classic computer-vision models with the adaptability of open-vocabulary and multimodal LLM workflows.
- **Approach:** Iterate through exploratory phases—baseline detectors, data strategy, production hardening, and alternative paradigms—to map the road to human-level perception.

---

## Repository Tour

```
Phase_1/
  ├── step0_restaurant_object_classification.py     # Faster R-CNN baseline with glove heuristics
  ├── step1_object_detection_baseline.py            # Live ResNet50 COCO demo
  ├── step2_Yolo.py / step3_yolo_flavors.py         # YOLOv8 speed/quality explorations
  ├── step7_DETR.py                                 # Transformer-based detection experiment
  ├── static_Images_VisualModels/                   # CLIP + SAM pipelines and learnings
  └── CLAUDE.md / phase1_takeaways.MD               # Narrative of the universal SAM+CLIP direction

Phase_2_Data_and_Plan/
  ├── Strategic_approach_to_data_and_model_design.md  # Data-first strategy & loss design
  ├── step1_open_images_v7.py / step2_epic-kitchens.py # Dataset harvesting pipelines
  ├── cnn_from_scratch.py / data_augmentation_pipeline.py # From-scratch training exploration
  └── production_inference_speed_optimization.py        # Deployment mindset and tooling

Phase_3_base_architecture_and_pipelines/
  └── …                                                 # Optimization and deployment helpers (shared with Phase 2)

Phase_4_a_step_back_into_computer_vision/
  └── hand_control.py                                   # Hand-tracking → OS control experimentation

alternative_approaches.md
llm_vision_architecture.md
```

---

## Quick Start

### 1. Environment

```bash
# Python >= 3.10 recommended
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or CUDA/MPS build
pip install ultralytics opencv-python pillow matplotlib numpy scikit-image
pip install git+https://github.com/openai/CLIP.git
pip install albumentations transformers onnxruntime pynput mediapipe psutil GPUtil
```

> **Apple Silicon notes:** `torch.backends.mps` is leveraged where possible. SAM models often exceed M2 memory limits; fall back to CPU or run on a discrete GPU.

### 2. Model Assets

- **YOLOv8 weights:** already included (`Phase_1/yolov8*.pt`).
- **SAM checkpoints:** download to `Phase_1/static_Images_VisualModels/models/`  
  - `sam_vit_b_01ec64.pth` (lighter)  
  - `sam_vit_h_4b8939.pth` (highest quality, 2.5 GB) from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

---

## Phase Highlights

### Phase 1 – Live Perception & Open Vocabulary

- `Phase_1/step2_Yolo.py:1` → YOLOv8 nano baseline; demonstrates real-time feel vs. accuracy trade-offs.
- `Phase_1/step7_DETR.py:1` → DETR transformer experiment; fewer detections but more deliberate outputs.
- `Phase_1/static_Images_VisualModels/step6_CLIP_with_SAM_segmentation_and_boxes.py:1` → SAM + CLIP pipeline capturing zero-shot understanding by segmenting snapshots and prompting with natural language.
- `Phase_1/static_Images_VisualModels/step6b_light_CLIP_w_seg.py:1` → Optimized variant (SMALL SAM, mask filtering, crop resizing) to survive on CPU/MPS.

**Insights:**  
`Phase_1/CLAUDE.md:1` chronicles the shift from “detect everything, every frame” to “sample strategically + reason deeply.”

### Phase 2 – Data, Training, and Evaluation Strategy

- `Phase_2_Data_and_Plan/Strategic_approach_to_data_and_model_design.md:1` captures the philosophy of multi-part losses, class weighting, and staged data collection to beat safety-violation sparsity.
- `Phase_2_Data_and_Plan/step1_open_images_v7.py:1` + `step2_epic-kitchens.py:1` automate harvesting open datasets, filtering kitchen-relevant samples, and converting to YOLO format.
- `Phase_2_Data_and_Plan/cnn_from_scratch.py:1` builds a glove/no-glove classifier from scratch with thoughtful initialization, schedulers, and visualization hooks.
- `Phase_2_Data_and_Plan/data_augmentation_pipeline.py:1` codifies restaurant-specific augmentations (steam, lighting, motion, occlusion) for resilient models.
- `Phase_2_Data_and_Plan/production_inference_speed_optimization.py:1` documents TorchScript, quantization, ONNX, and TensorRT conversion for sub-50 ms inference targets.

### Phase 3 – Toward Production

- Focuses on reusing the optimization utilities, multi-threaded inference queues, and resource monitoring aimed at deployment readiness.

### Phase 4 – Broader Vision Interfaces

- `Phase_4_a_step_back_into_computer_vision/hand_control.py:1` explores MediaPipe hand tracking to control OS inputs—evidence of expanding into embodied interfaces.

### Research Notes & Strategy Docs

- `alternative_approaches.md:1` ⟶ Vision models vs. multimodal LLMs vs. hybrid architectures; performance/compute matrix.
- `llm_vision_architecture.md:1` ⟶ How multimodal LLMs “tokenize” vision and why hybrid CV + LLM pipelines matter.

---

## Running the Explorations

| Goal | Command |
|------|---------|
| COCO baseline (laggy but simple) | `python Phase_1/step1_object_detection_baseline.py` |
| YOLOv8 real-time demo | `python Phase_1/step2_Yolo.py` |
| DETR transformer test | `python Phase_1/step7_DETR.py` |
| CLIP snapshot classification | `python Phase_1/static_Images_VisualModels/step5_CLIP_Webcam_snaps.py` |
| Full SAM + CLIP pipeline | `python Phase_1/static_Images_VisualModels/step6_CLIP_with_SAM_segmentation_and_boxes.py --cam 0 --interval 5 --sam_checkpoint models/sam_vit_b_01ec64.pth` |
| Camera availability probe | `python Phase_1/static_Images_VisualModels/test_mac_cameras.py` |
| Hand-gesture → OS control | `python Phase_4_a_step_back_into_computer_vision/hand_control.py` |

> **Tip:** For heavier pipelines, adjust snapshot intervals (`--interval`), mask sizes, or run on a GPU workstation.

---

## Data Strategy Recap

1. **Intentional staging:** Violation scenarios must be oversampled to avoid “always safe” predictions.
2. **Multi-loss weighting:** Safety-critical classes carry 4–20× penalties to enforce recall where it matters most.
3. **Augment toward reality:** Steam, glare, occlusions, and motion blur are simulated to match kitchen conditions.
4. **Dataset loaders:** Automated scripts pull from OpenImages and Epic-Kitchens to bootstrap model training.

---

## Hardware & Performance Notes

- **SAM vit_h** needs ≥ 8 GB GPU RAM; use `vit_b` for CPU/MPS prototyping.
- **Live demos** on M2 hardware benefit from smaller YOLO variants and reduced frame sizes.
- **Production pipeline** aims for <50 ms inference via quantization, ONNX/TensorRT, and async batching.

---

## Roadmap & Next Steps

1. **Hybrid Reasoning:** Combine fast YOLO/SAM detections with LLM-based natural-language reasoning (see `alternative_approaches.md:1`).
2. **Temporal Intelligence:** Extend snapshot analysis to sequences for activity recognition (temporal transformers, VideoMAE).
3. **Edge Deployment:** Package optimized models (TensorRT/ONNX) for edge devices; add health monitoring.
4. **Adaptive Learning:** Introduce feedback loops, prompt libraries, and few-shot adapters to customize per site.
5. **Evaluation Harness:** Build unified metrics + dashboards for cross-domain validation.

---

## Acknowledgements

This project stands on the shoulders of PyTorch, Ultralytics YOLO, Meta’s Segment Anything, OpenAI CLIP, Mediapipe, and the broader multimodal research community. The markdown narratives capture the developer’s reflections, strategic pivots, and the emotional arc of watching neural networks “open their eyes.”

