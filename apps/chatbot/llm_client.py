# llm_client.py
import os
from dotenv import load_dotenv
from google import genai

_client = None  # client will be initialized only once

# Load your .env from the specific path
ENV_PATH = os.path.join(os.path.expanduser("~"), "ai-portfolio", ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


def _get_api_key():
    """Get API key directly from .env"""
    return os.getenv("GOOGLE_API_KEY")


def get_client():
    """Initialize GenAI client only when needed"""
    global _client

    if _client is not None:
        return _client

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    _client = genai.Client(api_key=api_key)
    return _client


def generate_llm_response(prompt: str) -> str:
    client = get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


