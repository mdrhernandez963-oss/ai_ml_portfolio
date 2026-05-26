# chatbot_core.py

import json
from nltk.stem import PorterStemmer

# Initialize stemmer
stemmer = PorterStemmer()


def stem_text(text, stemmer=stemmer):
    """
    Stem a text string using NLTK's PorterStemmer.
    Tokenize using simple Python split to avoid NLTK punkt issues.
    """
    # Split on whitespace, remove non-alpha characters
    tokens = [t for t in text.lower().split() if t.isalpha()]
    return " ".join(stemmer.stem(t) for t in tokens)


def load_limited_data(json_path="chatbot/longevity_responses.json"):
    """
    Load and preprocess limited-mode chatbot data.
    Returns:
        stemmed_limited_responses_data (dict)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    stemmed_data = {}
    for topic, info in data.items():
        if topic != "default":
            stemmed_keywords = [stem_text(k) for k in info.get("keywords", [])]
            stemmed_data[topic] = {
                "stemmed_keywords": stemmed_keywords,
                "response": info["response"]
            }
        else:
            stemmed_data["default"] = info

    return stemmed_data


def get_limited_response(user_input, stemmed_data, stemmer=stemmer):
    """
    Return a limited-mode response based on keyword matching.
    """
    stemmed_input_tokens = set(stem_text(user_input, stemmer).split())

    for topic, info in stemmed_data.items():
        for keyword in info.get("stemmed_keywords", []):
            if keyword in stemmed_input_tokens:
                return info["response"]

    # fallback to default response
    return stemmed_data.get("default", {}).get(
        "response", "I'm sorry, I don't have information on that topic."
    )
