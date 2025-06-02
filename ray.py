import streamlit as st
import sys
import logging
import nest_asyncio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
import tempfile
import os

# Apply nest_asyncio at the start
nest_asyncio.apply()

# Disable file watcher and reduce logging noise
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")
logging.getLogger("torch._classes").setLevel(logging.ERROR)

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
    
    with st.spinner("Downloading model..."):
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Use updated model loading without deprecated 'pretrained' parameter
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(LABELS))

        # Download model to temp file
        file_id = st.secrets["google"]["file_id"]
        tmp_dir = tempfile.gettempdir()
        model_path = os.path.join(tmp_dir, "chexnet.pth.tar")

        if not os.path.exists(model_path):
            download_from_gdrive(file_id, model_path)

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        else:
            state_dict = checkpoint

        # Clean key prefixes
        new_state_dict = {k.replace("module.", "").replace("model.", ""): v 
                         for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# --- APP UI ---
def main():
    st.title("🩻 CheXNet Chest X-ray Diagnosis")
    check_password()

    st.markdown("Upload a chest X-ray image (JPG or PNG). The model predicts likelihoods for 14 thoracic diseases.")

    uploaded_file = st.file_uploader("📁 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

            with st.spinner("Analyzing image..."):
                model = load_model()
                tensor = preprocess_image(image)
                predictions = predict(model, tensor)

            st.subheader("🧠 Model Predictions")
            for label, prob in predictions.items():
                if prob > 0.5:
                    st.success(f"✅ {label}: {prob:.2f}")
                else:
                    st.info(f"{label}: {prob:.2f}")
                    
        except Exception as e:
            st.error(f"⚠️ Error processing image: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
