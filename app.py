# app.py

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from monkey_predictor import load_model, predict_species

# Monkey class ID to name mapping
id_to_name = {
    'n0': 'mantled_howler',
    'n1': 'patas_monkey',
    'n2': 'bald_uakari',
    'n3': 'japanese_macaque',
    'n4': 'pygmy_marmoset',
    'n5': 'white_headed_capuchin',
    'n6': 'silvery_marmoset',
    'n7': 'common_squirrel_monkey',
    'n8': 'black_headed_night_monkey',
    'n9': 'nilgiri_langur',
}

class_names = list(id_to_name.keys())

# Define transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load model structure and weights
device = torch.device("cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model = load_model(model, 'model.pth', device)

# Streamlit UI
st.title("🐵 Monkey Species Classifier")
st.write("Upload an image of a monkey to predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save to temporary path
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Predict
    class_id, species_name = predict_species("temp.jpg", model, device, transform, class_names, id_to_name)

    st.success(f"**Predicted Species:** {species_name}")
    st.info(f"Class ID: {class_id}")
