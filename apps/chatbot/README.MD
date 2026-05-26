# Longevity Chatbot 🧬🤖

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31.1-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This is a **Longevity Chatbot** demo built with **Python**, **Streamlit**, and **Google Generative AI (GenAI)**.  
It can answer questions about health, longevity, and wellness topics using either **LLM-powered mode** with a Google API key or a **Limited Mode** based on predefined responses in a JSON file.

---

## Features

- **LLM Mode**: Powered by Google GenAI to provide rich, AI-generated responses.
- **Limited Mode**: Fallback mode using curated JSON responses for diet, sleep, exercise, stress, brain health, social connections, mindset, gut health, and more.
- **Automatic `.env` loading**: Reads your API key from a local `.env` file.
- **Query limit**: LLM mode starts with **5 free queries per session**; user can enter a new API key or switch to Limited Mode afterward.
- **Session memory**: Displays conversation history so users can continue the chat without losing context.
- **Streamlit interface**: Clean web interface for chat input and dynamic response display.

---

## Folder Structure

