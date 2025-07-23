import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import os

# === Configuration ===
MODEL_PATH = r"D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\random_forest_model.pkl"
RESULTS_DIR = "results_rf"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.txt")

# === Ensure results directory exists ===
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load model ===
model = joblib.load(MODEL_PATH)

# === Load data ===
X = np.load("features.npy")
y = np.load("labels.npy")

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Predict ===
y_pred = model.predict(X_test)

# === Accuracy ===
accuracy = accuracy_score(y_test, y_pred)

# === Cross-validation Accuracy ===
cv_scores = cross_val_score(model, X, y, cv=5)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# === Classification Report ===
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)

# === 1. Confusion Matrix Heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_rf.png"))
plt.close()

# === 2. True vs Predicted Bar Plot ===
true_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()

df_bar = pd.DataFrame({
    "True": true_counts,
    "Predicted": pred_counts
}).fillna(0)

df_bar.plot(kind='bar', figsize=(8, 6))
plt.title("True vs Predicted Class Counts")
plt.ylabel("Count")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "true_vs_predicted_rf.png"))
plt.close()

# === 3. Classification Report Heatmap ===
import seaborn as sns

# Convert classification report into DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Select only precision, recall, f1-score for the heatmap
report_heatmap = report_df[['precision', 'recall', 'f1-score']]

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(report_heatmap.iloc[:-1, :], annot=True, fmt=".2f", cmap='Blues', cbar=True, annot_kws={"size": 10}, linewidths=0.5)
plt.title("Classification Report Heatmap", fontsize=14)
plt.tight_layout()

# Save the heatmap to a file
plt.savefig(os.path.join(RESULTS_DIR, "classification_report_heatmap_rf.png"))
plt.close()


# === 4. Write everything to results.txt ===
with open(RESULTS_FILE, "w") as f:
    f.write("=== Test Accuracy ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    
    f.write("=== Cross-Validation Accuracy (5-fold) ===\n")
    f.write(f"Mean CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}\n\n")
    
    f.write("=== Classification Report ===\n")
    f.write(report_text + "\n")
    
    f.write("=== Confusion Matrix ===\n")
    cm_str = "\n".join(["\t".join(map(str, row)) for row in cm])
    f.write(cm_str + "\n")

print(f"✅ Evaluation complete. All files saved in: '{RESULTS_DIR}/'")
