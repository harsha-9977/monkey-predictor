https://monkey-predictor-1625.streamlit.app/

# 🐒 Monkey Species Classifier using Transfer Learning

A deep learning web app that classifies monkey images into **10 different species** using a fine-tuned ResNet18 CNN model. Built with PyTorch, trained in Google Colab, and deployed with Streamlit.

---

## 📌 Problem Statement

Identifying monkey species manually is difficult due to their visual similarities. This project uses **transfer learning** on ResNet18 to accurately classify monkey species with limited labeled data.

---

## 📁 Dataset

- **Source:** [Kaggle - 10 Monkey Species](https://www.kaggle.com/datasets/slothkong/10-monkey-species)
- **Structure:** 10 folders (n0 to n9), each for a unique monkey species
- **Images:** ~1,000 training and 300 validation images
- **Labels:** Provided in `monkey_labels.txt`

---

## 🧠 Model Details

| Feature | Description |
|--------|-------------|
| Base model | ResNet18 (pretrained on ImageNet) |
| Layers frozen | All except final fully connected layer |
| Final FC layer | `nn.Linear(512, 10)` |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Accuracy | ~95–99% on validation set |

---

## 🚀 App Workflow

1. User uploads a monkey image
2. Image is resized, normalized, and passed through the model
3. The app returns the **predicted species**

---

## 🖼️ Screenshot

> _You can add a screenshot of your Streamlit app interface here_
> ![Screenshot 2025-07-01 230235](https://github.com/user-attachments/assets/e2ea764a-942e-4ed5-a4b3-527f113406f8)
![Screenshot 2025-07-01 230243](https://github.com/user-attachments/assets/8ea0128b-ca24-4ea0-a627-fdb419dada02)


---

## 🛠️ Tech Stack

- Python
- PyTorch & torchvision
- Streamlit
- Google Colab (training)
- GitHub + Streamlit Cloud (deployment)

---

## 📦 File Structure

```bash
.
├── monkey_classifier.py     # Training script
├── monkey_model.pth         # Saved model weights
├── streamlit_app.py         # Streamlit app
├── requirements.txt         # Dependencies for deployment
├── monkey_labels.txt        # Class label mapping
└── README.md
