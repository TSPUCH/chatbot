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
#------------------------------------------------------------------------
# --- Chat Function for GPT-2 ---
#------------------------------------------------------------------------
def chat_with_gpt2(prompt):
    """Generates a response using the GPT-2 model."""
    response = gpt2_generator(prompt, max_length=200, num_return_sequences=1, truncation=True)
    return response[0]['generated_text']


#-----------------------------------------------------------------------
# --- Chat Function for Groq Llama-3.1 ---
#-----------------------------------------------------------------------
def chat_with_llama(prompt):
    """Generates a response using the Groq Llama-3.1 API."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant", # Ensure this model name is correct and available on Groq
            temperature=0.7,
            max_tokens=200,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Groq API: {e}")
        return "Sorry, I couldn't connect to the Llama model."
    

#------------------------------------------------------------------------
# --- UI Layout ---
#------------------------------------------------------------------------
# Sidebar for file upload and sentiment display
with st.sidebar:
    st.header("Excel Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("Excel file uploaded successfully!")

            default_text_column = next((col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()), df.columns[0])
            text_column = st.selectbox("Select text column for sentiment analysis:", df.columns, index=df.columns.get_loc(default_text_column) if default_text_column in df.columns else 0)


            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    if text_column in df.columns:
                        df['sentiment_label'] = df[text_column].apply(lambda x: analyze_sentiment(x)[0])
                        df['sentiment_scores'] = df[text_column].apply(lambda x: analyze_sentiment(x)[1])
                        st.session_state.sentiment_data = df
                        st.success("Sentiment analysis complete!")
                    else:
                        st.warning(f"Column '{text_column}' not found in the uploaded Excel file.")


            if st.session_state.sentiment_data is not None:
                st.subheader("Sentiment Analysis Results")
                display_cols = [col for col in [text_column, 'sentiment_label'] if col in st.session_state.sentiment_data.columns]
                if display_cols:
                    st.dataframe(st.session_state.sentiment_data[display_cols])

                    sentiment_counts = st.session_state.sentiment_data['sentiment_label'].value_counts()
                    st.bar_chart(sentiment_counts)
                else:
                    st.info("Sentiment data available, but selected text column might have changed.")

        except Exception as e:
            st.error(f"Error processing Excel file: {e}")
            st.session_state.sentiment_data = None
    else:
        st.session_state.sentiment_data = None

        # Main content area with tabs for chatbots
tab1, tab2 = st.tabs(["💬 Chat with GPT-2", "🧠 Chat with Groq Llama-3.1"])

with tab1:
    st.header(f"Chat with GPT-2 ({GPT2_MODEL_NAME})")
    for message in st.session_state.messages_gpt2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Say something to {GPT2_MODEL_NAME}...", key="gpt2_chat_input"):
        st.session_state.messages_gpt2.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"{GPT2_MODEL_NAME} is thinking..."):
                full_response = chat_with_gpt2(prompt)
                st.markdown(full_response)
        st.session_state.messages_gpt2.append({"role": "assistant", "content": full_response})



