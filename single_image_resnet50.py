# single_image_resnet50.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torchvision.datasets import ImageFolder

# Constants
MODEL_PATH = r"D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\best_garbage_classification_model.pth"
TEST_DATA_DIR = r"D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\garbage_split\test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load class names from dataset
dataset = ImageFolder(TEST_DATA_DIR, transform=transform)
class_names = dataset.classes

# Load model once
def get_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# Initialize model globally
model = get_model()

# Predict using a PIL image (used in Gradio or CLI)
def predict_image_pil(image: Image.Image) -> str:
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]
