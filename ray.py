import os
import sys
import logging
import warnings
import tempfile
from PIL import Image
import requests

# Configure warnings and logging before imports
warnings.filterwarnings('ignore')
logging.getLogger("torch._classes").setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import streamlit and configure it
import streamlit as st
st.set_option('server.fileWatcherType', 'none')  # Disable file watcher

# Now import torch and related modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Configure torch
torch.set_grad_enabled(False)  # Disable gradients globally

# NIH ChestX-ray14 Disease Labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def check_password():
    """Password protection."""
    try:
        password = st.secrets["auth"]["password"]
        entered = st.text_input("Enter app password", type="password")
        if entered != password:
            st.error("❌ Incorrect password")
            st.stop()
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.stop()

def download_from_gdrive(file_id, destination):
    """Download model from Google Drive with proper error handling."""
    try:
        URL = "https://docs.google.com/uc?export=download"
        with st.spinner("Downloading model..."):
            session = requests.Session()
            response = session.get(URL, params={'id': file_id}, stream=True)
            
            token = next((value for key, value in response.cookies.items() 
                         if key.startswith('download_warning')), None)
            
            if token:
                response = session.get(URL, params={'id': file_id, 'confirm': token}, 
                                    stream=True)
            
            with open(destination, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=True)
def load_model():
    """Load and configure the model with proper error handling."""
    try:
        # Initialize model
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
        
        # Prepare model path
        file_id = st.secrets["google"]["file_id"]
        model_path = os.path.join(tempfile.gettempdir(), "chexnet.pth.tar")
        
        # Download if needed
        if not os.path.exists(model_path):
            download_from_gdrive(file_id, model_path)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        state_dict = (checkpoint.get("state_dict") or 
                     checkpoint.get("model_state_dict") or 
                     checkpoint)
        
        # Clean state dict
        cleaned_state_dict = {
            k.replace("module.", "").replace("model.", ""): v 
            for k, v in state_dict.items()
        }
        
        # Load weights
        model.load_state_dict(cleaned_state_dict)
        model.eval()
        
        return model
    
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def preprocess_image(image):
    """Preprocess image for model input."""
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
    """Make predictions with proper error handling."""
    try:
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()

def main():
    """Main application function."""
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
            with st.spinner("Processing image..."):
                model = load_model()
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
