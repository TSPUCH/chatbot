from dotenv import load_dotenv
load_dotenv() 

# --- Imports ---
import streamlit as st
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from groq import Groq
import io # For handling in-memory file operations

# --- Configuration ---

# Set your Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please ensure it's set in your .env file as 'GROQ_API_KEY'.")
    st.stop()

# GPT-2 Model Name for text generation
GPT2_MODEL_NAME = "gpt2" # Can be "gpt2-medium", "gpt2-large", etc.

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Advanced Chatbot & Sentiment Analyzer", layout="wide")
st.title("Intelligent Chatbots with Excel Sentiment Analysis")

# --- Session State Initialization ---
# Initialize chat messages for GPT-2
if "messages_gpt2" not in st.session_state:
    st.session_state.messages_gpt2 = []
# Initialize chat messages for Llama
if "messages_llama" not in st.session_state:
    st.session_state.messages_llama = []
# Store sentiment analysis results DataFrame
if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = None
# Store a textual summary of the uploaded file for chatbots
if "file_summary_for_bots" not in st.session_state:
    st.session_state.file_summary_for_bots = ""
# Store the name of the column used for sentiment analysis
if "sentiment_text_column" not in st.session_state:
    st.session_state.sentiment_text_column = ""


# --- Model Loading (Cached) ---
@st.cache_resource
def load_gpt2_model():
    """Loads the GPT-2 text generation model and tokenizer."""
    try:
        generator = pipeline(
            'text-generation',
            model=GPT2_MODEL_NAME,
            tokenizer=GPT2_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Use float16 if GPU, else float32
        )
        st.success(f"GPT-2 model '{GPT2_MODEL_NAME}' loaded successfully!")
        return generator
    except Exception as e:
        st.error(f"Error loading GPT-2 model '{GPT2_MODEL_NAME}': {e}. Check internet for initial download.")
        st.stop()

@st.cache_resource
def load_sentiment_model():
    """Loads a pre-trained sentiment analysis model (RoBERTa-base)."""
    sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    if torch.cuda.is_available():
        sentiment_model.to("cuda") # Move model to GPU if available
    st.success("Sentiment analysis model loaded successfully!")
    return sentiment_tokenizer, sentiment_model

# Load models at application startup
gpt2_generator = load_gpt2_model()
sentiment_tokenizer, sentiment_model = load_sentiment_model()

# --- Sentiment Analysis Function ---
def analyze_sentiment(text):
    """Performs sentiment analysis on a given text using the loaded model."""
    if not isinstance(text, str): # Ensure input is a string
        text = str(text)

    # Prepare input for the model
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()} # Move inputs to GPU

    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = sentiment_model(**inputs)
    
    scores = outputs.logits[0].softmax(dim=-1) # Get probabilities
    labels = ["Negative", "Neutral", "Positive"] # Model's output labels
    
    sentiment_label = labels[torch.argmax(scores).item()]
    sentiment_score = scores.tolist() # Convert scores to a list
    return sentiment_label, sentiment_score

# --- File Processing and Bot Context Generation ---
def generate_file_summary_for_bots(df, text_column):
    """
    Generates a text summary of the uploaded DataFrame for the chatbots.
    This includes schema, basic statistics, and a few sample rows.
    """
    summary_parts = []
    summary_parts.append("--- EXCEL FILE CONTEXT ---\n")
    summary_parts.append("An Excel file has been uploaded with the following columns:\n")
    summary_parts.append(", ".join(df.columns.tolist()) + "\n")
    summary_parts.append(f"The primary column for sentiment analysis is '{text_column}'.\n")
    
    # Add key column descriptions if available in the DataFrame
    key_columns = {
        'review_content': 'Full text of the user review.',
        'review_title': 'Short title or summary of the review.',
        'rating': 'Numerical rating, typically 1-5, indicating user satisfaction.',
        'product_name': 'Name of the product being reviewed.',
        'category': 'Category of the product.',
        'user_id': 'Unique identifier for the user.',
        'rating_count': 'Number of ratings for a particular item/product.'
    }
    
    summary_parts.append("\nKey columns and their descriptions:\n")
    for col, desc in key_columns.items():
        if col in df.columns:
            summary_parts.append(f"- {col}: {desc}\n")

    summary_parts.append("\nDataset Statistics:\n")
    summary_parts.append(f"- Number of rows: {len(df)}\n")
    
    # Add sentiment distribution if already analyzed
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts(normalize=True) * 100
        summary_parts.append("\nOverall Sentiment Distribution:\n")
        for label, percentage in sentiment_counts.items():
            summary_parts.append(f"- {label}: {percentage:.2f}%\n")

        
        summary_parts.append("\n--- DETAILED SENTIMENT ANALYTICS ---\n")
        
        # Average rating by sentiment (if 'rating' column exists)
        if 'rating' in df.columns:
            avg_rating_by_sentiment = df.groupby('sentiment_label')['rating'].mean()
            if not avg_rating_by_sentiment.empty:
                summary_parts.append("\nAverage Rating by Sentiment:\n")
                summary_parts.append(avg_rating_by_sentiment.to_markdown() + "\n")

        # Top/Bottom N products/categories by sentiment (if relevant columns exist)
        if 'product_name' in df.columns:
            positive_products = df[df['sentiment_label'] == 'Positive']['product_name'].value_counts().head(3)
            negative_products = df[df['sentiment_label'] == 'Negative']['product_name'].value_counts().head(3)
            if not positive_products.empty:
                summary_parts.append("\nTop 3 Products with Most Positive Reviews:\n")
                summary_parts.append(positive_products.to_markdown() + "\n")
            if not negative_products.empty:
                summary_parts.append("\nTop 3 Products with Most Negative Reviews:\n")
                summary_parts.append(negative_products.to_markdown() + "\n")

        if 'category' in df.columns:
            positive_categories = df[df['sentiment_label'] == 'Positive']['category'].value_counts().head(3)
            negative_categories = df[df['sentiment_label'] == 'Negative']['category'].value_counts().head(3)
            if not positive_categories.empty:
                summary_parts.append("\nTop 3 Categories with Most Positive Reviews:\n")
                summary_parts.append(positive_categories.to_markdown() + "\n")
            if not negative_categories.empty:
                summary_parts.append("\nTop 3 Categories with Most Negative Reviews:\n")
                summary_parts.append(negative_categories.to_markdown() + "\n")

    # Add a sample of the data (first few rows) for context
    if not df.empty:
        summary_parts.append("\nHere are the first 5 rows of the data (truncated for brevity):\n")
        # Select important columns to show in sample
        display_cols = [col for col in ['review_content', 'review_title', 'rating', 'sentiment_label'] if col in df.columns]
        if not display_cols: # Fallback if none of the specific columns exist
            display_cols = df.columns[:5].tolist() # Take first 5 available columns

        sample_df = df[display_cols].head(5).copy()
        # Truncate long text for display
        for col in ['review_content', 'review_title']:
            if col in sample_df.columns:
                sample_df[col] = sample_df[col].astype(str).apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
        
        # Convert sample_df to markdown table for LLM
        summary_parts.append(sample_df.to_markdown(index=False) + "\n")
    
    summary_parts.append("--- END EXCEL FILE CONTEXT ---\n")
    return "".join(summary_parts)

# --- Chat Function for GPT-2 ---
def chat_with_gpt2(prompt, file_context=""):
    """
    Generates a response using the GPT-2 model, optionally incorporating file context.
    GPT-2 is a completion model, so context is prepended to the prompt.
    """
    full_prompt = file_context + "\n" + prompt if file_context else prompt
    # GPT-2 can sometimes ignore instructions when context is too long or specific
    # We keep it simple for its generation style.
    response = gpt2_generator(
        full_prompt, 
        max_length=max(50, len(full_prompt.split()) + 150), # Ensure response has room beyond prompt
        num_return_sequences=1, 
        truncation=True,
        do_sample=True, # Use sampling for more creative responses
        temperature=0.7 # Add temperature for variety
    )
    # The response often includes the prompt itself. We try to extract only the new generation.
    generated_text = response[0]['generated_text']
    # Attempt to remove the input prompt from the generated text
    if generated_text.startswith(full_prompt):
        return generated_text[len(full_prompt):].strip()
    return generated_text.strip()


# --- Chat Function for Groq Llama-3.1 ---
def chat_with_llama(prompt, file_context=""):
    """
    Generates a response using the Groq Llama-3.1 API, incorporating file context
    via a system message for better instruction following.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        messages = []
        if file_context:
            messages.append({
                "role": "system",
                "content": "You have access to information from an uploaded Excel file. "
                           "Use the provided file context to answer questions about the data, "
                           "sentiment analysis, product reviews, and related statistics. "
                           "If a question requires data not in the summary, state that. "
                           "File Context:\n" + file_context
            })
        
        messages.append({
            "role": "user",
            "content": prompt,
        })

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=500, # Increased max tokens for analytical responses
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Groq API: {e}")
        return "Sorry, I couldn't connect to the Llama model or process your request."

# --- UI Layout ---
# Sidebar for file upload and sentiment analysis trigger
with st.sidebar:
    st.header("Excel Data Uploader")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Read the Excel file into a BytesIO object first to handle it in memory
            # This is robust if Streamlit reruns
            file_buffer = io.BytesIO(uploaded_file.getvalue())
            df = pd.read_excel(file_buffer)
            st.success("Excel file uploaded successfully! Now select a column for analysis.")

            # Identify potential text columns for sentiment analysis
            potential_text_cols = [
                col for col in df.columns if 
                'review_content' in col.lower() or 
                'comment' in col.lower() or 
                'description' in col.lower() or
                'text' in col.lower()
            ]
            default_text_column = potential_text_cols[0] if potential_text_cols else df.columns[0]
            
            # Selectbox for the user to confirm the text column
            st.session_state.sentiment_text_column = st.selectbox(
                "Select the primary text column for sentiment analysis:", 
                df.columns, 
                index=df.columns.get_loc(default_text_column) if default_text_column in df.columns else 0
            )

            if st.button("Analyze Sentiment and Prepare for Bots"):
                with st.spinner("Analyzing sentiment and preparing file context..."):
                    if st.session_state.sentiment_text_column in df.columns:
                        # Perform sentiment analysis
                        sentiment_results = df[st.session_state.sentiment_text_column].apply(lambda x: analyze_sentiment(x))
                        df['sentiment_label'] = [res[0] for res in sentiment_results]
                        df['sentiment_scores'] = [res[1] for res in sentiment_results]
                        st.session_state.sentiment_data = df.copy() # Store the analyzed DataFrame

                        # Generate summary for the bots
                        st.session_state.file_summary_for_bots = generate_file_summary_for_bots(
                            df, st.session_state.sentiment_text_column
                        )
                        st.success("Sentiment analysis complete and file context prepared for chatbots!")
                    else:
                        st.warning(f"Column '{st.session_state.sentiment_text_column}' not found.")
                        st.session_state.sentiment_data = None
                        st.session_state.file_summary_for_bots = ""

        except Exception as e:
            st.error(f"Error processing Excel file: {e}")
            st.session_state.sentiment_data = None
            st.session_state.file_summary_for_bots = ""
    else:
        # Clear data if no file is uploaded
        st.session_state.sentiment_data = None
        st.session_state.file_summary_for_bots = ""
        st.session_state.sentiment_text_column = ""


# Main content area with tabs for chatbots and sentiment results
tab1, tab2 = st.tabs(["💬 Chat with GPT-2", "🧠 Chat with Groq Llama-3.1"])

# --- Display Sentiment Results in Main Page if Available ---
if st.session_state.sentiment_data is not None:
    st.subheader("📊 Sentiment Analysis Overview")
    st.info(f"Analysis performed on column: **`{st.session_state.sentiment_text_column}`**")

    # Display aggregate sentiment counts
    sentiment_counts = st.session_state.sentiment_data['sentiment_label'].value_counts()
    st.write("Sentiment Distribution:")
    st.dataframe(sentiment_counts.reset_index().rename(columns={'index': 'Sentiment', 'sentiment_label': 'Count'}))
    st.bar_chart(sentiment_counts)
    
    # Optionally display a sample of the analyzed data
    st.markdown("---")
    

# --- Display Sentiment Results in Main Page if Available ---
if st.session_state.sentiment_data is not None:
    # ... (existing sentiment display code) ...
    st.markdown("---")
    st.subheader("Sample of Analyzed Reviews")
    # ... (existing sample display code) ...
    st.markdown("---")

    # --- DEBUG: Display Bot Context ---
    st.subheader("🤖 Chatbot File Context (for debugging)")
    if st.session_state.file_summary_for_bots:
        st.expander("Click to see the full context passed to chatbots:")
        st.code(st.session_state.file_summary_for_bots)
    else:
        st.info("No file context generated yet. Please upload a file and run analysis.")
    st.markdown("---")
    # --- END DEBUG SECTION ---
    st.subheader("Sample of Analyzed Reviews")
    display_cols = [
        st.session_state.sentiment_text_column, 
        'sentiment_label', 
        'review_title', 
        'rating', 
        'product_name', 
        'category'
    ]
    # Filter for columns that actually exist in the dataframe
    display_cols_filtered = [col for col in display_cols if col in st.session_state.sentiment_data.columns]
    
    if display_cols_filtered:
        st.dataframe(st.session_state.sentiment_data[display_cols_filtered].head(10))
    else:
        st.info("No specific analysis columns found to display sample data.")
    st.markdown("---")


# --- GPT-2 Chat Tab ---
with tab1:
    st.header(f"Chat with GPT-2 ({GPT2_MODEL_NAME})")
    
    # Display previous messages
    for message in st.session_state.messages_gpt2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for GPT-2
    if prompt := st.chat_input(f"Say something to {GPT2_MODEL_NAME}...", key="gpt2_chat_input"):
        st.session_state.messages_gpt2.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"{GPT2_MODEL_NAME} is thinking..."):
                # Pass file summary to the chatbot function
                full_response = chat_with_gpt2(prompt, st.session_state.file_summary_for_bots)
                st.markdown(full_response)
        st.session_state.messages_gpt2.append({"role": "assistant", "content": full_response})

# --- Llama-3.1 Chat Tab ---
with tab2:
    st.header("Chat with Groq Llama-3.1")
    
    # Display previous messages
    for message in st.session_state.messages_llama:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for Llama
    if prompt := st.chat_input("Say something to Llama-3.1...", key="llama_chat_input"):
        st.session_state.messages_llama.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Llama-3.1 is thinking..."):
                # Pass file summary to the chatbot function
                full_response = chat_with_llama(prompt, st.session_state.file_summary_for_bots)
                st.markdown(full_response)
        st.session_state.messages_llama.append({"role": "assistant", "content": full_response})