# Shit gets real.
# We enter a technique which we want to use a strong, base model to segment the picture into objects
# Then, for each "object" or "Kaya(cluster of relevant objects)" we can perform a much accurate, cheaper inference.

# Overview:
# grab snapshots from webcam every few seconds,
# run SAM to segment the image into regions,
# run CLIP on each region to score a set of kitchen/safety tags,
# draws labeled boxes/masks on the image and prints results.


# Hit MAC bottleneck here - it cannot properly load and handle SAM_vit_h
# Crashed M2.
# Can try Lighther model for the 'Feel' if want to.
# Move to PC - download sam model from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

"""
clip_sam_pipeline.py
Snapshot -> SAM segmentation -> CLIP scoring -> Draw masks/boxes + labels.

Usage:
python step6_CLIP_with_segmentation_and_boxes.py --cam 1 --interval 5 --sam_checkpoint models/sam_vit_h.pth
"""

import argparse
import time
import cv2
import numpy as np
from PIL import Image
import torch
import clip

# SAM imports
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Utilities
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device)  # change to larger model if needed
    model.eval()
    return model, preprocess

def load_sam(sam_checkpoint, device):
    # choose appropriate SAM backbone: "vit_h" recommended for accuracy (heavier)
    sam_type = "vit_h"
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # automatic mask generator for zero-shot segmentation
    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam, mask_generator

def masks_to_bboxes(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [int(x1), int(y1), int(x2), int(y2)]

def draw_mask_and_label(image_bgr, mask, label, score, bbox=None, color=(0,255,0), alpha=0.4):
    # overlay mask
    colored = np.zeros_like(image_bgr, dtype=np.uint8)
    colored[:] = color
    mask3 = np.stack([mask]*3, axis=-1).astype(np.uint8)
    image_bgr = np.where(mask3, cv2.addWeighted(image_bgr, 1-alpha, colored, alpha, 0), image_bgr)
    # draw bbox
    if bbox:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(image_bgr, (x1,y1), (x2,y2), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(image_bgr, text, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr

def main(args):
    device = get_device()
    print("Using device:", device)

    # Load CLIP
    clip_model, clip_preprocess = load_clip(device)
    print("Loaded CLIP ViT-B/32")

    # Load SAM
    print("Loading SAM checkpoint:", args.sam_checkpoint)
    sam, mask_generator = load_sam(args.sam_checkpoint, device)
    print("Loaded SAM")

    # Define your kitchen / safety tags (edit as you wish)
    text_prompts = [
        "clean countertop",
        "dirty countertop",
        "spilled liquid on floor",
        "knife on counter",
        "raw meat on cutting board",
        "open flame",
        "chef wearing gloves",
        "chef not wearing gloves",
        "hand washing",
        "stacked dishes",
        "cluttered workspace",
        "hot pot",
        "plate with food",
        "customers in kitchen area",
    ]
    text_tokens = clip.tokenize(text_prompts).to(device)

    # Open camera (auto-find if not provided)
    cam_idx = args.cam
    if cam_idx is None:
        found = None
        for i in range(5):
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                found = i
                cap_test.release()
                break
        if found is None:
            print("No camera found.")
            return
        cam_idx = found
    print("Using camera index:", cam_idx)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Failed to open camera", cam_idx)
        return

    last_snapshot = 0
    snapshot_interval = args.interval
    scaling_max = args.max_dim  # max dimension for SAM/CLIP cropping performance

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # optional live display
            display = frame.copy()
            cv2.putText(display, f"Press q to quit. Snapshot every {snapshot_interval}s", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Live Feed", display)

            now = time.time()
            if now - last_snapshot >= snapshot_interval:
                last_snapshot = now
                print("=== Snapshot taken ===")

                # Convert frame to RGB PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                # Run SAM automatic mask generator
                # Note: mask_generator expects np array HxWxC (RGB)
                print("Running SAM mask generator (may be heavy on CPU)...")
                masks = mask_generator.generate(rgb)

                print(f"Found {len(masks)} masks")

                # Prepare CLIP cropping and scoring
                image_results = []
                # Precompute image features? We will compute per-crop features
                for idx, m in enumerate(masks):
                    # mask is boolean map in m['segmentation']
                    seg = m["segmentation"]  # numpy bool array HxW
                    bbox = m.get("bbox", None)  # [x,y,w,h]
                    if bbox is None:
                        bbox_coords = masks_to_bboxes(seg)
                    else:
                        x,y,w,h = bbox
                        bbox_coords = [int(x), int(y), int(x+w), int(y+h)]

                    # Skip tiny masks
                    x1,y1,x2,y2 = bbox_coords
                    wbox = x2-x1
                    hbox = y2-y1
                    if wbox < 20 or hbox < 20:
                        continue

                    # Crop region and resize to clip preprocess size while preserving aspect ratio
                    crop = rgb[y1:y2, x1:x2]
                    crop_pil = Image.fromarray(crop)
                    # If huge, shrink
                    max_dim = scaling_max
                    if max(crop_pil.size) > max_dim:
                        scale = max_dim / max(crop_pil.size)
                        new_size = (int(crop_pil.size[0]*scale), int(crop_pil.size[1]*scale))
                        crop_pil = crop_pil.resize(new_size, Image.LANCZOS)

                    # CLIP preprocess: produces tensor
                    image_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)

                    # compute features
                    with torch.no_grad():
                        img_feat = clip_model.encode_image(image_input)
                        img_feat /= img_feat.norm(dim=-1, keepdim=True)
                        txt_feat = clip_model.encode_text(text_tokens)
                        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

                        sims = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]
                    
                    # find top-k tags for this mask
                    topk_idx = sims.argsort()[::-1][:3]
                    top_tags = [(text_prompts[i], float(sims[i])) for i in topk_idx]

                    image_results.append({
                        "idx": idx,
                        "bbox": bbox_coords,
                        "mask": seg,
                        "top_tags": top_tags
                    })

                # Draw results on an annotated frame copy
                annotated = frame.copy()
                for res in image_results:
                    bbox = res["bbox"]
                    seg = res["mask"]
                    # pick color based on top tag
                    top_label, top_score = res["top_tags"][0]
                    color = tuple(int(x) for x in np.random.randint(50,255,size=3).tolist())
                    annotated = draw_mask_and_label(annotated, seg, top_label, top_score, bbox=bbox, color=color, alpha=0.35)

                # Show / Save annotated
                cv2.imshow("Annotated Snapshot", annotated)
                fname = f"snapshot_{int(now)}.jpg"
                cv2.imwrite(fname, annotated)
                print("Saved annotated snapshot:", fname)

                # Print concise result list
                print("Summary of top matches per segment:")
                for i,res in enumerate(image_results):
                    print(f" - segment {i}: bbox={res['bbox']}, top={res['top_tags'][0]} , other={res['top_tags'][1:]}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=None, help="camera index (default: first available)")
    parser.add_argument("--interval", type=float, default=5.0, help="snapshot interval in seconds")
    parser.add_argument("--sam_checkpoint", type=str, default="models/sam_vit_h_4b8939.pth", help="path to SAM checkpoint")
    parser.add_argument("--max_dim", type=int, default=1024, help="max dim for masks/crops (reduces compute)")
    args = parser.parse_args()
    main(args)
