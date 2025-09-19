# Into Transformers
# Step 7: Testing DETR (DEtection TRansformer)
#
# Experience so far:
# - DETR uses a Transformer backbone instead of the YOLO CNN pipeline.
# - It is SLOWER (lower FPS) and video smoothness declines on Mac M2.
# - BUT: predictions start to feel more "thoughtful" (fewer random boxes, better confidence).
# - Trade-off: In a real restaurant setting, ~5-10 FPS is acceptable since safety events
#   (e.g., knife mishandling, fire hazards) don’t require 30 FPS gaming-level smoothness.
# - The big limitation remains: pretrained on COCO (80 categories only).
#   If our safety-relevant object isn’t in COCO, DETR won’t detect it at all.
#
# This experiment grounds us in the theory–practice trade-off:
#   -> YOLO = fast & responsive, feels “lightweight.”
#   -> DETR = slower but closer to human-like perception (attention-based).
#
# Next steps: compare YOLO vs DETR live, then plan custom fine-tuning for restaurant-specific objects.

import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import time

# Load model + processor (pretrained on COCO, 80 classes)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

# Use MPS (Mac GPU) if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Start webcam (change 1 -> 0 if it doesn’t open the right camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Convert OpenCV BGR frame -> PIL RGB (required by Hugging Face processor)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess image for DETR
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert model outputs to usable boxes (filtered by threshold)
    target_sizes = torch.tensor([pil_img.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    # Draw predicted bounding boxes + labels
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.config.id2label[label.item()]} {score:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show FPS (note: usually lower than YOLO due to transformer overhead)
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("DETR Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
