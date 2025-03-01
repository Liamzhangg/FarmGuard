import openai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import traceback  # ✅ Import for full error logs

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
print(f"Using device: {device}")

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

    # Load ResNet50 model
    model = models.resnet50(weights=None)  # No pre-trained weights
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for better generalization
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model.to(device)

# ✅ Define class labels (Must match trained model)
class_labels = [
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

# ✅ Load trained model
model_path = 'output/model.pth'
model = load_model(model_path, class_labels)

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# ✅ Function to Get AI-Powered Eco-Friendly Solution
def get_ai_solution(disease_name):
    prompt = f"""
    The plant disease diagnosed is {disease_name}.
    Suggest eco-friendly, sustainable, and organic solutions to manage or cure this disease.
    Consider natural remedies, biological controls, companion planting, and eco-conscious practices. Format your response nicely in point form. Limit yourself to 5 points or remedies.
    """

    try:
        print(f"🔍 Sending request to OpenAI for: {disease_name}")  # ✅ Debugging log

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in plant health and organic farming."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )  # ✅ Log trimmed response

        print("✅ OpenAI Response:", response)  # ✅ Log full OpenAI response  # ✅ Log trimmed
        return response.choices[0].message.content.strip()

    except openai.OpenAIError as e:
        print(f"❌ OpenAI API Error: {e}")
        return f"OpenAI API Error: {e}"
    
def format_disease_name(predicted_class):
    """
    Extracts the disease name from the model output.
    Example: "Apple___Apple_scab" -> "Apple scab"
             "Grape___Black_rot" -> "Black rot"
             "Tomato___Tomato_mosaic_virus" -> "Tomato mosaic virus"
    """
    if "___" in predicted_class:
        return predicted_class.split("___")[1].replace("_", " ")
    return predicted_class.replace("_", " ")  # Fallback in case there's no triple underscore

# ✅ /predict: Upload Image & Get Plant Disease Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Process the uploaded image
        image = Image.open(file).convert("RGB")
        image = image_transforms(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)

            if max_prob.item() < 0.7:
                formatted_disease = "Not Sure"
            else:
                raw_prediction = class_labels[predicted.item()]
                formatted_disease = format_disease_name(raw_prediction)

        return jsonify({'predicted_class': formatted_disease})

# ✅ /get_solution: Get AI-Powered Eco-Friendly Solution
@app.route('/get_solution', methods=['POST'])
def get_eco_friendly_solution():
    print("Received request for eco-friendly solution")
    data = request.get_json()
    disease_name = data.get("disease")

    if not disease_name:
        return jsonify({"error": "No disease provided"}), 400

    solution = get_ai_solution(disease_name)
    return jsonify({"disease": disease_name, "eco_friendly_solution": solution})

# ✅ /chat: Chatbot for Follow-up Questions


@app.route('/chat', methods=['POST'])
def chat_with_ai():
    print("🛠 Received request for chatbot")

    try:
        # Log full request payload
        data = request.get_json()
        print("📝 Incoming Chat Data:", data)

        # Validate input
        if not data or "conversation" not in data:
            print("❌ ERROR: Missing 'conversation' in request data")
            return jsonify({"error": "Invalid conversation data"}), 400

        conversation = data["conversation"]
        
        if not isinstance(conversation, list) or len(conversation) == 0:
            print("❌ ERROR: Conversation must be a non-empty list")
            return jsonify({"error": "Conversation history required"}), 400

        conversation.append({
            "role": "system",
            "content": "RESPOND WITH ONLY 1 PARAGRAPH OR 1 ORDERED LIST!!!"
        })
        # OpenAI API Call
        print("📤 Sending conversation to OpenAI:", conversation)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=conversation,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        print("✅ OpenAI Response:", reply)
        print(response)

        return jsonify({'content': reply})
    
    except Exception as e:
        print(f"❌ SERVER ERROR: {e}")
        traceback.print_exc()  # ✅ Print full error stack trace
        return jsonify({"error": str(e)}), 500


# ✅ Run Flask App
if __name__ == '__main__':
    app.run(port=5000, debug=True)
