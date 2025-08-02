♻️ EcoClassify: An AI-Powered Garbage Classification System
📄 Overview
EcoClassify is an AI-powered web application that uses image recognition to classify garbage into multiple categories. The system is built with Convolutional Neural Networks (CNN) and is deployed using Gradio. This project is designed to help promote efficient waste segregation and environmental sustainability.

🚀 Key Features
Garbage Classification: Classifies waste into 12 distinct categories.

Multiple Model Evaluation: The project explores three different models for garbage classification: ResNet50, MobileNetV2, and a Random Forest Classifier.

Web Application: A user-friendly web interface built with Gradio for easy image-based classification.

Recycling Guidance: Provides a recommended bin color and specific recycling guidance for each classified item.

High Accuracy: The final ResNet50 model achieved a test accuracy of 99.08%, outperforming the other models.

🧠 Model Performance
The project evaluated three different models. The ResNet50 model was chosen for the final application due to its superior performance.

Model

Test Accuracy

Validation Accuracy

ResNet50

99.08%

99.58%

MobileNetV2

90.81%

92.89%

Random Forest

85.34%

N/A

🛠️ How It Works
The system follows a clear and modular workflow:

Data Splitting: The split_dataset.py script divides the raw dataset into training (70%), validation (15%), and testing (15%) sets to ensure a robust evaluation.

Model Training: The train_resnet50.py script fine-tunes a pre-trained ResNet50 model on the training dataset to learn garbage features.

Feature Extraction: An alternative approach uses extract_features.py to get features from ResNet50, which are then used to train a traditional classifier like Random Forest in train_rf_classifier.py.

Evaluation: Scripts like test_resnet50_model.py rigorously evaluate the trained models, generating classification reports, confusion matrices, and accuracy scores.

Web App Deployment: The gradio_resnet_app.py script uses the best-performing ResNet50 model to power a Gradio-based web application.

💻 Getting Started
1. Clone the Repository

git clone https://github.com/vanshgargji/ecoclassify.git
cd vanshgargji-ecoclassify

2. Install Dependencies

Install all necessary libraries from requirements.txt.

pip install -r requirements.txt

3. Run the Web Application

Launch the Gradio interface to start classifying images.

python gradio_resnet_app.py

🖼️ Web Application in Action
The web app is designed to be intuitive. Users upload an image, and the system instantly predicts the garbage class and recommends a bin.

Upload Garbage Image: The user uploads an image of an item to be classified.

Prediction: The model predicts the class (e.g., "plastic," "paper").

Recommended Bin: The app shows an image of the recommended bin.

Recycling Guidance: A message provides specific instructions on how to dispose of the item.

Here's a breakdown of the bin mapping and guidance:

Green Bin: For organic or biodegradable waste, such as biological, paper, and cardboard.

Blue Bin: For recyclable plastic and metal items, such as plastic and metal.

White Bin: For glass waste, such as green-glass, brown-glass, and white-glass.

Orange Bin: For textile waste, such as clothes and shoes.

Red Bin: For general or hazardous waste, such as trash and battery.

📈 Evaluation Results
The results_resnet50 directory contains the detailed performance metrics for the best model. Below is a summary of the high-level results:

Classification Report

              precision    recall  f1-score   support

     battery       0.99      0.99      0.99        101
  biological       1.00      1.00      1.00        101
 brown-glass       1.00      0.99      0.99         92
   cardboard       0.98      0.96      0.97        101
     clothes       1.00      0.99      1.00        101
 green-glass       1.00      0.99      0.99         95
       metal       0.95      0.99      0.97        101
       paper       0.99      0.99      0.99        101
     plastic       0.99      0.99      0.99        101
       shoes       1.00      1.00      1.00        101
       trash       1.00      1.00      1.00        101
 white-glass       0.99      1.00      1.00        101

    accuracy                           0.99       1197
   macro avg       0.99      0.99      0.99       1197
weighted avg       0.99      0.99      0.99       1197

