import gradio as gr
from PIL import Image
import os
from single_image_resnet50 import predict_image_pil

# Folder containing your bin images
BIN_DIR = r"D:\Garbage-Classification-System\Garbage-Classification-System\Garbage-Classification-System\bins"

# Mapping of garbage class to one of 5 realistic bin types
bin_mapping = {
    "biological": "green_bin.png",
    "paper": "green_bin.png",
    "cardboard": "green_bin.png",

    "plastic": "blue_bin.png",
    "metal": "blue_bin.png",

    "green-glass": "white_bin.jpg",
    "brown-glass": "white_bin.jpg",
    "white-glass": "white_bin.jpg",

    "clothes": "orange_bin.jpeg",
    "shoes": "orange_bin.jpeg",

    "trash": "red_bin.jpeg",
    "battery": "red_bin.jpeg"
}

# Messages per bin type
recycle_messages = {
    "green_bin.png": "Use the GREEN bin for organic or biodegradable waste.",
    "blue_bin.png": "Use the BLUE bin for recyclable plastic and metal items.",
    "white_bin.jpg": "Use the WHITE bin for glass waste.",
    "orange_bin.jpeg": "Use the ORANGE bin for textile waste like clothes and shoes.",
    "red_bin.jpeg": "Use the RED bin for general or hazardous waste.",
    "default_bin.png": "Please dispose of this item responsibly."
}

def classify(image: Image.Image):
    # Predict class
    predicted_class = predict_image_pil(image)

    # Determine bin image
    bin_filename = bin_mapping.get(predicted_class, "default_bin.png")
    bin_image_path = os.path.join(BIN_DIR, bin_filename)

    # Get message
    message = recycle_messages.get(bin_filename, "Please dispose of this item responsibly.")

    return f"Predicted class: {predicted_class}", bin_image_path, message

# Define the Gradio interface
interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload Garbage Image"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(type="filepath", label="Recommended Bin", height=256, width=256),
        gr.Text(label="Recycling Guidance")
    ],
    title="EcoClassify ",
    description="Upload an image of waste to classify it and get a recommended bin color for disposal.",
    theme="default",
    css="""
        body, .gradio-container {
            background-color: white !important;
            color: black !important;
        }

        .gr-box, .gr-panel, .gr-group, .gr-column {
            background-color: white !important;
            color: black !important;
            border: none !important;
        }

        .gr-button, .gr-textbox, .gr-image {
            color: black !important;
        }

        h1, h2, h3, h4, h5, h6, p, label {
            color: black !important;
        }
    """
)



if __name__ == "__main__":
    interface.launch()
