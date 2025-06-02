import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Disease labels (NIH 14)
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Load the CheXNet model
@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(LABELS))
    model.load_state_dict(torch.load("chexnet.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # ImageNet mean/std
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# Predict using model
def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# Streamlit App
st.title("CheXNet Chest X-ray Diagnosis")
st.markdown("Upload a frontal chest X-ray to see predicted findings (14 pathologies).")

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    model = load_model()
    tensor = preprocess_image(image)
    predictions = predict(model, tensor)

    st.subheader("Predicted Conditions")
    for label, prob in predictions.items():
        if prob > 0.5:
            st.markdown(f"✅ **{label}** — `{prob:.2f}`")
        else:
            st.markdown(f"🔹 {label} — `{prob:.2f}`")
