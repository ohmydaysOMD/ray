import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# NIH ChestX-ray14 Disease Labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(LABELS))

    # Load checkpoint (can be .pth or .pth.tar)
    checkpoint = torch.load("chexnet.pth.tar", map_location=torch.device("cpu"))
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get("state_dict", checkpoint)  # If no 'state_dict' key, use whole checkpoint
    
    # Fix key names if they have 'module.' prefix (common in DataParallel models)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB mean/std for DenseNet
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# UI
st.title("🩻 CheXNet Chest X-ray Diagnosis")
st.markdown("Upload a chest X-ray image (JPG or PNG). The model predicts likelihoods for 14 thoracic diseases.")

uploaded_file = st.file_uploader("📁 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    model = load_model()
    tensor = preprocess_image(image)
    predictions = predict(model, tensor)

    st.subheader("🧠 Model Predictions")
    for label, prob in predictions.items():
        if prob > 0.5:
            st.success(f"✅ {label}: {prob:.2f}")
        else:
            st.info(f"{label}: {prob:.2f}")
