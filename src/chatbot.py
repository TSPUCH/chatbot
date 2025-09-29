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


# Now it will correctly load from your .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please ensure it's set in your .env file as 'GROQ_API_KEY'.")
    st.stop()

# Local GPT-2 model path
GPT2_LOCAL_PATH = "C:\\Users\\aalqarawi.t\\.cache\\huggingface\\hub\\models--gpt2"
