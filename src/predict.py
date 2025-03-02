import openai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import traceback

# ✅ Load environment variables
load_dotenv()

# ✅ Get OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY in .env file.")

# ✅ Set up OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ Define Image Transformations (Must match model training)
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load Model Function
def load_model(model_path, class_labels):
    num_classes = len(class_labels)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# ✅ Define class labels for plants
plant_class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ✅ Define class labels for animals
animal_class_labels = ['(BRD) Bovine Dermatitis Disease healthy lumpy', '(BRD) Bovine Disease Respiratory', '(BRD) Disease Ecthym', 'Contagious Dermatitis lumpy skin', 'Contagious Ecthym', 'Dermatitis', 'Dermatitis Ecthym lumpy skin', 'Ecthym skin', 'Unlabeled', 'healthy', 'healthy lumpy skin', 'lumpy skin', 'test', 'train', 'valid']

# ✅ Load trained models
plant_model_path = 'output/model.pth'
animal_model_path = 'output/animal_model.pth'

plant_model = load_model(plant_model_path, plant_class_labels)
animal_model = load_model(animal_model_path, animal_class_labels)

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# ✅ Function to format disease names
def format_disease_name(predicted_class):
    if "___" in predicted_class:
        return predicted_class.split("___")[1].replace("_", " ")
    return predicted_class.replace("_", " ")

# ✅ /predict_plant: Upload Image & Get Plant Disease Prediction
@app.route('/predict_plant', methods=['POST'])
def predict_plant():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image = Image.open(file).convert("RGB")
        image = image_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = plant_model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)

            if max_prob.item() < 0.7:
                formatted_disease = "Not Sure"
            else:
                raw_prediction = plant_class_labels[predicted.item()]
                formatted_disease = format_disease_name(raw_prediction)

        print(f"🦠 Predicted Disease: {formatted_disease}")
        return jsonify({'predicted_class': formatted_disease})

# ✅ /predict_animal: Upload Image & Get Animal Disease Prediction
@app.route('/predict_animal', methods=['POST'])
def predict_animal():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image = Image.open(file).convert("RGB")
        image = image_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = animal_model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)

            if max_prob.item() < 0.7:
                predicted_class = "Not Sure"
            else:
                predicted_class = animal_class_labels[predicted.item()]

        print(f"🦠 Predicted Disease: {predicted_class}")
        return jsonify({'predicted_class': predicted_class})

# ✅ /get_solution: AI-powered eco-friendly solution
@app.route('/get_solution', methods=['POST'])
def get_solution():
    data = request.get_json()
    disease_name = data.get("disease")

    if not disease_name:
        return jsonify({"error": "No disease provided"}), 400

    try:
        prompt = f"""
        The diagnosed disease is {disease_name}.
        Suggest eco-friendly, sustainable, and organic solutions to manage or cure this disease.
        Consider natural remedies, biological controls, and eco-conscious practices.
        FORMAT RESPONSE AS A NUMBERED LIST WITH MAX 5 POINTS.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in eco-friendly farming and disease management. YOU ONLY RESPOND IN ONE ORDERED LIST WITH MAX 5 POINTS."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )

        return jsonify({"disease": disease_name, "eco_friendly_solution": response.choices[0].message.content.strip()})
    
    except openai.OpenAIError as e:
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500

# ✅ /chat: AI-powered chatbot for follow-up questions
@app.route('/chat', methods=['POST'])
def chat_with_ai():
    try:
        data = request.get_json()
        conversation = data.get("conversation")

        if not conversation or not isinstance(conversation, list):
            return jsonify({"error": "Invalid conversation format"}), 400

        conversation.append({"role": "system", "content": "Keep response concise and to one paragraph."})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=conversation,
            temperature=0.7
        )

        return jsonify({'content': response.choices[0].message.content.strip()})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(port=5000, debug=True)
