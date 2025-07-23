# extract_features.py

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import os

# Configuration
DATA_DIR = r'D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\garbage_preprocessed'  # change this to your dataset folder
BATCH_SIZE = 32
FEATURES_FILE = 'features.npy'
LABELS_FILE = 'labels.npy'

# Load pre-trained ResNet50 and remove final classification layer
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
resnet.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(root=DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Extract features
features_list = []
labels_list = []

with torch.no_grad():
    for inputs, labels in loader:
        outputs = resnet(inputs)  # [B, 2048, 1, 1]
        features = outputs.view(outputs.size(0), -1)  # [B, 2048]
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# Combine and save
features = np.concatenate(features_list)
labels = np.concatenate(labels_list)

np.save(FEATURES_FILE, features)
np.save(LABELS_FILE, labels)

print(f"✅ Features saved to {FEATURES_FILE}, Labels saved to {LABELS_FILE}")
