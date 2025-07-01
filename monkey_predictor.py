import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_species(image_path, model, device, transform, class_names, id_to_name):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)

    predicted_class = class_names[pred.item()]
    species_name = id_to_name[predicted_class]
    
    return predicted_class, species_name
