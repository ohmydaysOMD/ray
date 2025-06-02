import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 32  # Number of embedding dims to use

# === Discretization Helper ===
def discretize(value):
    if value < 0.2:
        return "very_low"
    elif value < 0.4:
        return "low"
    elif value < 0.6:
        return "medium"
    elif value < 0.8:
        return "high"
    else:
        return "very_high"

# === Load RAD-DINO model ===
def load_rad_dino():
    processor = AutoProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino").to(DEVICE)
    model.eval()
    return processor, model

# === Extract discretized prompt from image ===
def extract_rad_dino_prompt(image: Image.Image, processor, model, top_k=TOP_K) -> str:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    arr = embedding.detach().cpu().numpy()
    topk_indices = np.argsort(np.abs(arr))[-top_k:]
    topk_values = arr[topk_indices]

    norm_values = (topk_values - np.min(topk_values)) / (np.ptp(topk_values) + 1e-8)
    semantic_tokens = [discretize(v) for v in norm_values]

    return " ".join(semantic_tokens)
