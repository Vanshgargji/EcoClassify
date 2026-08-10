# ♻️ EcoClassify: AI‑Powered Garbage Classification System

> **An intelligent waste‑segregation system leveraging deep learning to promote sustainability and responsible recycling.**

---

## 📄 Overview

**EcoClassify** is an AI‑powered image classification system that automatically identifies different types of garbage and recommends the correct disposal bin along with recycling guidance. The system is built using **Convolutional Neural Networks (CNNs)** and deployed through a clean, interactive **Gradio web interface**.

The goal of this project is to assist individuals and organizations in improving **waste segregation efficiency**, reducing environmental impact, and encouraging sustainable practices.

---

## 🚀 Key Features

* 🧠 **Garbage Classification** – Classifies waste into **12 distinct categories**
* 🔬 **Multi‑Model Evaluation** – Comparative study of **ResNet50, MobileNetV2, and Random Forest**
* 🌐 **Web Application** – Simple and intuitive UI powered by **Gradio**
* 🗑️ **Smart Recycling Guidance** – Displays **bin color + disposal instructions** for each prediction
* 📈 **High Accuracy** – Final **ResNet50 model achieved 99.08% test accuracy**

---

## 🧠 Model Performance Summary

| Model          | Test Accuracy | Validation Accuracy |
| -------------- | ------------- | ------------------- |
| **ResNet50** ⭐ | **99.08%**    | **99.58%**          |
| MobileNetV2    | 90.81%        | 92.89%              |
| Random Forest  | 85.34%        | N/A                 |

✔️ **ResNet50** was selected for deployment due to its superior and consistent performance.

---

## 🛠️ System Workflow

The project follows a modular and reproducible pipeline:

1. **📂 Data Splitting**
   `split_dataset.py` divides the dataset into:

   * Training: **70%**
   * Validation: **15%**
   * Testing: **15%**

2. **🏋️ Model Training**
   `train_resnet50.py` fine‑tunes a pre‑trained **ResNet50** model on the garbage dataset.

3. **🔍 Feature Extraction (Alternative Path)**

   * `extract_features.py` extracts deep features using ResNet50
   * `train_rf_classifier.py` trains a **Random Forest** classifier on those features

4. **📊 Evaluation**
   Evaluation scripts generate:

   * Accuracy scores
   * Confusion matrices
   * Classification reports

5. **🌐 Web App Deployment**
   `gradio_resnet_app.py` deploys the best‑performing model using **Gradio**.

---

## 💻 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yatharth1511/EcoClassify.git
cd EcoClassify
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Web Application

```bash
python gradio_resnet_app.py
```

---

## 🖼️ Web Application Usage

1. 📤 **Upload an image** of a waste item
2. 🧠 **Model predicts** the garbage category
3. 🗑️ **Recommended bin** is displayed
4. ♻️ **Recycling guidance** is shown for proper disposal

---

## 🗑️ Bin Mapping & Recycling Guidance

| Bin Color         | Waste Categories                      | Description                   |
| ----------------- | ------------------------------------- | ----------------------------- |
| 🟩 **Green Bin**  | Biological, Paper, Cardboard          | Organic & biodegradable waste |
| 🟦 **Blue Bin**   | Plastic, Metal                        | Recyclable materials          |
| ⬜ **White Bin**   | Green‑Glass, Brown‑Glass, White‑Glass | Glass waste                   |
| 🟧 **Orange Bin** | Clothes, Shoes                        | Textile waste                 |
| 🟥 **Red Bin**    | Trash, Battery                        | General & hazardous waste     |

---

## 📈 Detailed Evaluation Results (ResNet50)

**Overall Test Accuracy:** **99%**
Test Samples: **1197**

### 📊 Classification Report

| Class       | Precision | Recall | F1‑Score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| Battery     | 0.99      | 0.99   | 0.99     | 101     |
| Biological  | 1.00      | 1.00   | 1.00     | 101     |
| Brown‑Glass | 1.00      | 0.99   | 0.99     | 92      |
| Cardboard   | 0.98      | 0.96   | 0.97     | 101     |
| Clothes     | 1.00      | 0.99   | 1.00     | 101     |
| Green‑Glass | 1.00      | 0.99   | 0.99     | 95      |
| Metal       | 0.95      | 0.99   | 0.97     | 101     |
| Paper       | 0.99      | 0.99   | 0.99     | 101     |
| Plastic     | 0.99      | 0.99   | 0.99     | 101     |
| Shoes       | 1.00      | 1.00   | 1.00     | 101     |
| Trash       | 1.00      | 1.00   | 1.00     | 101     |
| White‑Glass | 0.99      | 1.00   | 1.00     | 101     |

---

## 📂 Results Directory

The `results_resnet50/` directory contains:

* Confusion matrices
* Classification reports
* Accuracy logs

---

## 🌱 Impact & Future Scope

* Integration with **smart bins**
* Real‑time classification via **mobile camera**
* Dataset expansion for regional waste categories
* Lightweight deployment for edge devices

---

## 🤝 Contributors

* **Yatharth Sharma**
* **Vansh Garg**

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ *If you found this project useful, feel free to star the repository and contribute!*


This is to check that ai review work or not ]



checking garbage is properly classified or not



