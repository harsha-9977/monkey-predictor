import streamlit as st
from monkey_predictor import load_model, predict_species
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

st.title("🐵 Monkey Species Classifier")
st.write("Upload an image of a monkey to predict its species.")

# Load class ID → name mapping
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
    'n9': 'nilgiri_langur'
}
class_names = list(id_to_name.keys())

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import models
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 monkey classes
model = load_model(model, "model.pth", device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Save temp image
        temp_path = "temp.jpg"
        img.save(temp_path)

        # Predict
        class_id, species_name = predict_species(temp_path, model, device, transform, class_names, id_to_name)

        st.success(f"🌟 Predicted Species: **{species_name.replace('_', ' ').title()}**")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
