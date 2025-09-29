from dotenv import load_dotenv
load_dotenv() 

#--- imports --- 
import streamlit as st
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from groq import Groq

#----------------------------------------------------------------------
# --- Configuration ---
#----------------------------------------------------------------------

# Set your Groq API key here or as an environment variable (loaded by dotenv)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please ensure it's set in your .env file as 'GROQ_API_KEY'.")
    st.stop()

# --- GPT-2 Model Name ---
# We will use the standard Hugging Face model identifier for GPT-2
# This allows the 'transformers' library to manage downloading and caching.
GPT2_MODEL_NAME = "gpt2" # You can change this to "gpt2-medium", "gpt2-large", etc.


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Advanced Chatbot", layout="wide")
st.title("Chatbot with Sentiment Analysis for Excel")



# --- Session State Initialization ---
if "messages_gpt2" not in st.session_state:
    st.session_state.messages_gpt2 = []
if "messages_llama" not in st.session_state:
    st.session_state.messages_llama = []
if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = None


#------------------------------------------------------------------------    
# --- Helper Functions ---
#------------------------------------------------------------------------

@st.cache_resource
def load_gpt2_model():
    """Loads the GPT-2 model and tokenizer for text generation."""
    try:
        # The pipeline will automatically download and cache the model
        # if it's not already present in the Hugging Face cache.
        generator = pipeline(
            'text-generation',
            model=GPT2_MODEL_NAME,
            tokenizer=GPT2_MODEL_NAME,
            torch_dtype=torch.float16 # Use float16 for better performance if GPU is available
        )
        st.success(f"GPT-2 model '{GPT2_MODEL_NAME}' loaded successfully!")
        return generator
    except Exception as e:
        st.error(f"Error loading GPT-2 model '{GPT2_MODEL_NAME}': {e}. Check your internet connection for the initial download.")
        st.stop()

@st.cache_resource
def load_sentiment_model():
    """Loads a pre-trained sentiment analysis model."""
    sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    # Move model to GPU if available
    if torch.cuda.is_available():
        sentiment_model.to("cuda")
    st.success("Sentiment analysis model loaded successfully!")
    return sentiment_tokenizer, sentiment_model

# Load models at startup
gpt2_generator = load_gpt2_model()
sentiment_tokenizer, sentiment_model = load_sentiment_model()

def analyze_sentiment(text):
    """Performs sentiment analysis on a given text."""
    # Handle non-string inputs gracefully
    if not isinstance(text, str):
        text = str(text)

    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = outputs.logits[0].softmax(dim=-1)
    # The model outputs scores for negative, neutral, positive
    # We map these to actual labels
    labels = ["Negative", "Neutral", "Positive"]
    sentiment_score = scores.tolist()
    sentiment_label = labels[torch.argmax(scores).item()]
    return sentiment_label, sentiment_score

