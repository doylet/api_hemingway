from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Initialize OpenAI API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY_SMARTSHEET"))
# models = openai.models.list()
# print([model.id for model in models.data])

app = FastAPI(title="Hemingway API", description="Analyze text for readability, adverbs, passive voice, and complexity.")

# Request body model
class TextInput(BaseModel):
    text: str

# GPT-4 Turbo Analysis Function
def analyze_text_with_gpt(text):
    prompt = f"""
    Analyze the following text like Hemingway Editor. Identify:
    - Long or complex sentences (20+ words)
    - Passive voice usage
    - Overuse of adverbs (-ly words)
    - Readability score (Flesch-Kincaid)
    
    Return the response in JSON format with keys: long_sentences, passive_voice, adverbs, readability.

    Text:
    "{text}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
        )

    return response.choices[0].message.content

# API Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hemingway API. Analyze text for readability, adverbs, passive voice, and complexity."}

# API Endpoint
@app.post("/analyze")
def analyze_text(input_text: TextInput):
    analysis = analyze_text_with_gpt(input_text.text)
    return {"analysis": analysis}

