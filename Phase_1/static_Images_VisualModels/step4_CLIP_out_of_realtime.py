import torch
import clip
from PIL import Image

# Meh.. not too impressive.
# The predictions scores are far from reliable.
# But in this context of restaurant safety, the need for REAL DATA starts to rise.
# We need REAL DATA and context of where the model is going to be used.
# Once the context is clear, we can begin having a clearer picture.
# How will we build our Data set?
# Are we training for usage in a specific restaurant of chain? What is the lighting? The uniforms. The angles of the cameras?


# so CLIP models run locally as well, similar to YOLOv8 - but they run from /cache/torch
# Load model (ViT-B/32 is lightweight, we can try ViT-L/14 later for stronger accuracy)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mpun s.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load an image (snapshot from your restaurant video)
image = preprocess(Image.open("no_gloves.jpeg")).unsqueeze(0).to(device)

# Define your custom restaurant categories
text_prompts = [
    "chef wearing gloves",
    "chef not wearing gloves",
    "dirty floor",
    "clean table",
    "customers waiting",
    "happy customers eating",
]

text = clip.tokenize(text_prompts).to(device)

# Compute similarity
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    best_idx = similarities.argmax().item()

print(f"Prediction: {text_prompts[best_idx]} (confidence {similarities[0][best_idx]:.2f})")
