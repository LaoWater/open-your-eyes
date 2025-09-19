#!/usr/bin/env python3
"""
Lightweight CLIP+SAM kitchen labeling pipeline.
- Uses smaller SAM backbone (vit_b)
- Filters tiny masks
- Resizes crops for fast CLIP inference
- Takes snapshots every few seconds
"""

# Hit some mps vs cpu on MAC issues and error in converting to float64 in tensor calculations
# SAM tries to create a float64 tensor on MPS, and Apple’s MPS backend doesn’t support float64. It’s a known limitation for some PyTorch operations on M2/M1 chips.
# Optional: Can leave CLIP on MPS (faster) but run SAM fully on CPU. 

import time, cv2, numpy as np
from PIL import Image
import torch, clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------- Settings --------
SNAPSHOT_INTERVAL = 5          # seconds between snapshots
MAX_DIM = 512                  # max size for mask crops
MIN_MASK_SIZE = 20             # ignore tiny masks
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"  # smaller SAM, still 300MB
DEVICE = "cpu" if torch.backends.mps.is_available() else "cpu"
sam_device = "cpu"
clip_device = "mps"  # optional


# Kitchen tags
TEXT_PROMPTS = [
    "clean countertop", "dirty countertop", "spilled liquid on floor",
    "knife on counter", "raw meat on cutting board", "open flame",
    "chef wearing gloves", "chef not wearing gloves", "hand washing",
    "stacked dishes", "cluttered workspace", "hot pot", "plate with food",
]

# --------------------------
def load_models():
    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    # SAM
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return clip_model, clip_preprocess, mask_generator

def masks_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def draw_mask(image, mask, bbox, label, color=(0,255,0), alpha=0.4):
    colored = np.zeros_like(image, dtype=np.uint8)
    colored[:] = color
    mask3 = np.stack([mask]*3, axis=-1)
    image = np.where(mask3, cv2.addWeighted(image, 1-alpha, colored, alpha, 0), image)
    if bbox:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
        cv2.putText(image, label, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def main():
    # Models
    clip_model, clip_preprocess, mask_generator = load_models()
    text_tokens = clip.tokenize(TEXT_PROMPTS).to(DEVICE)

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    last_snapshot = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Live Feed", frame)
        now = time.time()
        if now - last_snapshot >= SNAPSHOT_INTERVAL:
            last_snapshot = now
            print("=== Snapshot ===")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Run SAM (fast, smaller backbone)
            masks = mask_generator.generate(rgb)
            print(f"Found {len(masks)} masks")

            # Only process meaningful masks
            results = []
            for m in masks:
                mask = m["segmentation"]
                bbox = masks_to_bbox(mask)
                if bbox is None: continue
                x1,y1,x2,y2 = bbox
                if x2-x1 < MIN_MASK_SIZE or y2-y1 < MIN_MASK_SIZE: continue

                crop = rgb[y1:y2, x1:x2]
                crop_pil = Image.fromarray(crop)
                if max(crop_pil.size) > MAX_DIM:
                    scale = MAX_DIM / max(crop_pil.size)
                    new_size = (int(crop_pil.size[0]*scale), int(crop_pil.size[1]*scale))
                    crop_pil = crop_pil.resize(new_size, Image.LANCZOS)

                # CLIP inference - choose cpu or mps device
                image_input = clip_preprocess(crop_pil).unsqueeze(0).to(clip_device)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(image_input)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    txt_feat = clip_model.encode_text(text_tokens)
                    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                    sims = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]

                top_idx = sims.argmax()
                label = TEXT_PROMPTS[top_idx]
                score = sims[top_idx]
                results.append({"bbox": bbox, "mask": mask, "label": label, "score": score})

            # Draw results
            annotated = frame.copy()
            for r in results:
                color = tuple(int(x) for x in np.random.randint(50,255,3))
                annotated = draw_mask(annotated, r["mask"], r["bbox"], f"{r['label']}:{r['score']:.2f}", color=color)

            cv2.imshow("Annotated", annotated)
            fname = f"snapshot_{int(now)}.jpg"
            cv2.imwrite(fname, annotated)
            print("Saved:", fname)
            print("Top labels:", [r['label'] for r in results])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
