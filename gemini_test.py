import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content(
    "Generate a 4 sentence municipal cleanliness report for a clean street."
)

print(response.text)