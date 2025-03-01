#!/usr/bin/python3

# filepath: /Users/Ferdinand/FarmGuard-1/src/predict.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transformation for input images (must match training)
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load trained model for inference
def load_model(model_path, class_labels):
    num_classes = len(class_labels)

    # Load ResNet18 model
    model = models.resnet18(weights=None)  # No pre-trained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust final layer
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model.to(device)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Define class labels (same as used during training)
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load the model
model_path = 'output/plant_disease_model.pth'  # Adjust this path if needed
model = load_model(model_path, class_labels)

# Ensure the uploads directory exists
uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

# Define route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file to the uploads directory
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Process the uploaded image
        image = Image.open(file_path).convert("RGB")
        image = image_transforms(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]

        return jsonify({'predicted_class': predicted_class})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)