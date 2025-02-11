from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

app = FastAPI(title="Hemingway API", description="Analyze text for readability, adverbs, passive voice, and complexity.")

# ✅ Add CORS Middleware to allow Chrome Extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to your extension ID for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Request body model
class TextInput(BaseModel):
    text: str

# GPT-4 Turbo Analysis Function
def analyze_text_with_gpt(text):
    cache_key = f"analysis:{hash(text)}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result) # Return cached result if available

    # ✅ Improved Prompt
    prompt = f"""
    Analyze the following text and return a JSON response with:
    - "long_sentences": List of sentences over 20 words + rewrite suggestions
    - "passive_voice": List of passive voice sentences + suggestions to make active
    - "adverbs": List of adverbs + whether they weaken the sentence
    - "readability": Flesch-Kincaid readability score + explanation

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