import os
import sys
import logging
import warnings
from PIL import Image
import requests
import tempfile

# Configure warnings and logging before imports
warnings.filterwarnings('ignore')
logging.getLogger("torch._classes").setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Torch imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Import streamlit last
import streamlit as st

# Configure torch
torch.set_grad_enabled(False)

# NIH ChestX-ray14 Disease Labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.device = 'cpu'
    
    def load(self):
        if self.model is None:
            self.model = self._initialize_model()
        return self.model
    
    def _initialize_model(self):
        try:
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
            
            # Download and load weights
            file_id = st.secrets["google"]["file_id"]
            model_path = os.path.join(tempfile.gettempdir(), "chexnet.pth.tar")
            
            if not os.path.exists(model_path):
                self._download_weights(file_id, model_path)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = self._get_state_dict(checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            
            return model
        except Exception as e:
            st.error(f"Failed to initialize model: {str(e)}")
            st.stop()
    
    def _download_weights(self, file_id, destination):
        try:
            URL = "https://docs.google.com/uc?export=download"
            with st.spinner("Downloading model weights..."):
                session = requests.Session()
                response = session.get(URL, params={'id': file_id}, stream=True)
                
                token = next((value for key, value in response.cookies.items() 
                            if key.startswith('download_warning')), None)
                
                if token:
                    response = session.get(URL, 
                                        params={'id': file_id, 'confirm': token}, 
                                        stream=True)
                
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(32768):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            st.error(f"Failed to download model weights: {str(e)}")
            st.stop()
    
    def _get_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            state_dict = (checkpoint.get("state_dict") or 
                         checkpoint.get("model_state_dict") or 
                         checkpoint)
        else:
            state_dict = checkpoint
            
        return {k.replace("module.", "").replace("model.", ""): v 
                for k, v in state_dict.items()}

def check_password():
    try:
        password = st.secrets["auth"]["password"]
        entered = st.text_input("Enter app password", type="password")
        if entered != password:
            st.error("❌ Incorrect password")
            st.stop()
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.stop()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

def predict(model, img_tensor):
    try:
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()

# Initialize model wrapper
model_wrapper = ModelWrapper()

def main():
    try:
        st.title("🩻 CheXNet Chest X-ray Diagnosis")
        check_password()

        st.markdown("Upload a chest X-ray image (JPG or PNG). The model predicts likelihoods for 14 thoracic diseases.")
        
        uploaded_file = st.file_uploader(
            "📁 Upload Chest X-ray",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
            
            # Process image
            with st.spinner("Analyzing image..."):
                model = model_wrapper.load()
                tensor = preprocess_image(image)
                predictions = predict(model, tensor)
            
            # Display results
            st.subheader("🧠 Model Predictions")
            col1, col2 = st.columns(2)
            
            for i, (label, prob) in enumerate(predictions.items()):
                with col1 if i < len(predictions)/2 else col2:
                    if prob > 0.5:
                        st.success(f"✅ {label}: {prob:.2f}")
                    else:
                        st.info(f"{label}: {prob:.2f}")
                        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    # Set environment variables
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Run the application
    main()
