import cv2
import torch
import clip
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define semantic tags
text_prompts = [
    "chef wearing gloves",
    "chef not wearing gloves",
    "dirty floor",
    "clean floor",
    "customers waiting",
    "happy customers eating",
    "fire hazard",
]
text = clip.tokenize(text_prompts).to(device)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

snapshot_interval = 5  # seconds
last_snapshot_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Show live webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Take snapshot every X seconds
    current_time = time.time()
    if current_time - last_snapshot_time >= snapshot_interval:
        last_snapshot_time = current_time

        # Convert frame → PIL
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # CLIP preprocessing
        image_input = preprocess(pil_img).unsqueeze(0).to(device)

        # Run CLIP
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarities.argmax().item()

        print(f"Snapshot → Prediction: {text_prompts[best_idx]} "
              f"(confidence {similarities[0][best_idx]:.2f})")

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
