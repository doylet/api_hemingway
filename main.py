from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import redis
import openai
import os
import json

# Load API key from .env
load_dotenv()

# Initialize OpenAI API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY_SMARTSHEET"))
# models = openai.models.list()
# print([model.id for model in models.data])

# Initialize Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

app = FastAPI(title="Hemingway API", description="Analyze text for readability, adverbs, passive voice, and complexity.")

# Request body model
class TextInput(BaseModel):
    text: str

# GPT-4 Turbo Analysis Function
def analyze_text_with_gpt(text):
    cache_key = f"analysis:{hash(text)}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result) # Return cached result if available

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

    result = response.choices[0].message.content

    redis_client.set(cache_key, json.dumps(result), 86400) # Cache result for 24 hour
    return result

# API Endpoint
@app.post("/analyze")
def analyze_text(input_text: TextInput):
    analysis = analyze_text_with_gpt(input_text.text)
    return {"analysis": analysis}


# API Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hemingway API. Analyze text for readability, adverbs, passive voice, and complexity."}