from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key loaded:", bool(api_key))
print("API key length:", len(api_key) if api_key else 0)

client = genai.Client(api_key=api_key)

print("\nAvailable models:")

models = list(client.models.list())

print("Number of models:", len(models))

for model in models:
    print(model.name)