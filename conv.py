import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from environment variable or .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCYJyzf2QXVuYS7XVWhaki9NKzDvgs1WjA")
genai.configure(api_key=api_key)

# Select the model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")  # or "gemini-1.5-pro"

# Generate content
response = model.generate_content("Explain how AI works in a few words")

# Print response safely
if hasattr(response, "text"):
    print(response.text)
else:
    print(response)
