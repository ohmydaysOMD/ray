import asyncio

# Workaround for "no running event loop" RuntimeError in Streamlit with Python 3.10+
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
import tempfile
import os

# NIH ChestX-ray14 Disease Labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Password protection
def check_password():
    password = st.secrets["auth"]["password"]
    entered = st.text_input("Enter app password", type="password")
    if entered != password:
        st.error("❌ Incorrect password")
        st.stop()

# Function to download model from Google Drive
def download_from_gdrive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

@st.cache_resource(show_spinner=True)
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(LABELS))

    # Download model to temp file
    file_id = st.secrets["google"]["file_id"]
    tmp_dir = tempfile.gettempdir()
    model_path = os.path.join(tmp_dir, "chexnet.pth.tar")

    if not os.path.exists(model_path):
        download_from_gdrive(file_id, model_path)

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Try to get state_dict with common keys
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Clean key prefixes if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        new_state_dict[new_k] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        st.error(f"⚠️ Error loading model weights: {e}")
        st.stop()

    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# --- APP UI ---

st.title("🩻 CheXNet Chest X-ray Diagnosis")
check_password()

st.markdown("Upload a chest X-ray image (JPG or PNG). The model predicts likelihoods for 14 thoracic diseases.")

uploaded_file = st.file_uploader("📁 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    model = load_model()
    tensor = preprocess_image(image)
    predictions = predict(model, tensor)

    st.subheader("🧠 Model Predictions")
    for label, prob in predictions.items():
        if prob > 0.5:
            st.success(f"✅ {label}: {prob:.2f}")
        else:
            st.info(f"{label}: {prob:.2f}")
