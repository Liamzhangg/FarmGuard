import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY in .env file.")

openai.api_key = OPENAI_API_KEY

# Test OpenAI API call
def test_openai():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a plant disease expert."},
                      {"role": "user", "content": "How do I treat Tomato Late Blight in an eco-friendly way?"}],
            temperature=0.7
        )
        print("✅ OpenAI API Response:", response["choices"][0]["message"]["content"])
    except openai.OpenAIError as e:
        print(f"❌ OpenAI API Error: {e}")

test_openai()
