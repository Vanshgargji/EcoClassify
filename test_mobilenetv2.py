import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import pandas as pd

# === Paths and Parameters ===
TEST_DATA_DIR = r"D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\garbage_split\test"
BATCH_SIZE = 32
NUM_CLASSES = 12
MODEL_PATH = r"D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\mobilenetv2_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Load Dataset ===
dataset = ImageFolder(TEST_DATA_DIR, transform=transform)
class_names = dataset.classes

# === Model Setup ===
def get_model():
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# === Evaluation ===
def evaluate_model(model, loader):
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100 * correct / total
    return np.array(all_preds), np.array(all_labels), acc

# === True vs Predicted Plot ===
def plot_prediction_comparison(y_true, y_pred):
    true_count = Counter(y_true)
    pred_count = Counter(y_pred)
    indices = np.arange(len(class_names))
    true_vals = [true_count.get(i, 0) for i in indices]
    pred_vals = [pred_count.get(i, 0) for i in indices]

    plt.figure(figsize=(12, 6))
    width = 0.35
    plt.bar(indices - width/2, true_vals, width=width, label='True')
    plt.bar(indices + width/2, pred_vals, width=width, label='Predicted')
    plt.xticks(indices, class_names, rotation=45)
    plt.ylabel('Count')
    plt.title('True vs Predicted Labels')
    plt.legend()
    plt.tight_layout()
    plt.savefig('true_vs_predicted_mobilenetv2.png')
    plt.close()

# === Confusion Matrix ===
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('test_confusion_matrix_mobilenetv2.png')
    plt.close()

# === Classification Report Heatmap ===
def plot_classification_report(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
    plt.title("Classification Report (Visualized)")
    plt.tight_layout()
    plt.savefig('classification_report_mobilenetv2.png')
    plt.close()

# === Optional Validation Accuracy ===
def validation_accuracy(dataset, model):
    val_size = int(0.2 * len(dataset))
    _, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    _, _, acc = evaluate_model(model, val_loader)
    print(f"\nValidation Accuracy (20% split): {acc:.2f}%")
    return acc

# === Run Evaluation ===
print("=== MobileNetV2 Garbage Classification Evaluation ===")
print(f"Test samples: {len(dataset)}")
print(f"Classes: {class_names}")

test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
model = get_model()
predictions, labels, test_acc = evaluate_model(model, test_loader)

print(f"\nTest Accuracy: {test_acc:.2f}%")
cm = confusion_matrix(labels, predictions)
plot_confusion_matrix(cm, class_names)

print("\nClassification Report:")
report = classification_report(labels, predictions, target_names=class_names)
print(report)
plot_classification_report(labels, predictions)

plot_prediction_comparison(labels, predictions)

val_acc = validation_accuracy(dataset, model)

# === Save Results ===
with open("test_results_mobilenetv2.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Validation Accuracy (20% split): {val_acc:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm, separator=', '))
print("\nEvaluation complete. Results saved.")
