# Intelligent Chatbots with Excel Sentiment Analysis

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]

> Unleash the power of AI chatbots and gain instant sentiment insights from your Excel data.

## Table of Contents
- [Intelligent Chatbots with Excel Sentiment Analysis](#intelligent-chatbots-with-excel-sentiment-analysis)
  - [Table of Contents](#table-of-contents)
  - [Visual Demo](#visual-demo)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Data Source & Dictionary](#data-source--dictionary)
  - [Technology Stack](#technology-stack)
  - [Setup and Local Installation](#setup-and-local-installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Visual Demo
<!-- 
     screenshot.
    
-->
Here is an example of the dashboard in action:
`

## Project Overview

This project delivers an advanced web application built with Streamlit, designed to empower users with immediate sentiment analysis of their Excel datasets and interactive engagement with powerful AI chatbots. It seamlessly integrates local large language models (LLMs) like GPT-2 with cloud-based, high-performance models such as Groq's Llama-3.1.

**Problem Solved:** Businesses and individuals often collect vast amounts of textual data (e.g., customer reviews, feedback forms, survey responses) stored in Excel files. Manually extracting sentiment and gleaning insights from this data is time-consuming and prone to human bias. Furthermore, interacting with these insights or exploring data contextually using AI often requires separate tools or advanced technical skills.

**Value Proposition:**
*   **Instant Sentiment Insights:** Upload an Excel file, select a text column, and get an immediate, visual breakdown of sentiment (positive, neutral, negative) across your data.
*   **Intelligent Data Interrogation:** Chatbots are provided with a summary of your uploaded data, enabling them to answer analytical questions about the file's content, sentiment distribution, and key columns. This turns raw data into conversational intelligence.
*   **Dual AI Power:** Choose between a locally-run GPT-2 model for quick text generation or leverage the high-speed and advanced reasoning capabilities of Groq's Llama-3.1 for deeper analytical insights.
*   **User-Friendly Interface:** A clean and intuitive Streamlit interface makes complex NLP and AI interactions accessible to non-technical users.

## Features

*   **Excel File Upload (.xlsx):** Easily upload your data directly through the sidebar.
*   **Configurable Sentiment Analysis:** Select any text column from your uploaded Excel file for sentiment analysis.
*   **Interactive Sentiment Visualizations:** View sentiment distribution via a DataFrame and a bar chart directly on the main page.
*   **Chatbot Integration (GPT-2):** Engage with a locally-run GPT-2 model for general conversation or simple text generation.
*   **Chatbot Integration (Groq Llama-3.1):** Interact with the advanced Llama-3.1 model via the Groq API for faster, more nuanced responses and analytical queries.
*   **Context-Aware Chatbots:** Both GPT-2 and Llama-3.1 are provided with a dynamic summary of your uploaded Excel data, allowing them to answer questions specific to your dataset.
*   **Persistent Chat History:** Your conversations with both chatbots are maintained within the session.
*   **Robust Error Handling:** Clear messages for API key issues, model loading failures, and file processing errors.

## Data Source & Dictionary

While this application is designed to work with *any* user-uploaded Excel file containing text data, a common use case involves customer review datasets.

**Example Data Source (Conceptual):**
For demonstration or testing purposes, you might use publicly available datasets like:
*   [Amazon Product Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Kaggle)
*   [Yelp Reviews Dataset](https://www.yelp.com/dataset)

**Important Columns for Analysis (Expected in uploaded files):**

| Column Name      | Description                                                                                              |
| :--------------- | :------------------------------------------------------------------------------------------------------- |
| `review_content` | The full, detailed text of the user's review or feedback. (Primary for sentiment analysis)                 |
| `review_title`   | A concise title or summary provided by the user for their review.                                        |
| `rating`         | A numerical rating (e.g., 1 to 5 stars) given by the user, indicating their satisfaction.                |
| `product_name`   | The name of the product or service being reviewed.                                                       |
| `category`       | The category to which the product or service belongs.                                                    |
| `user_id`        | A unique identifier for the user who submitted the review.                                               |
| `rating_count`   | (Optional) The total number of ratings received by a specific product or item.                           |

The application automatically identifies and analyzes the `review_content` column (or similar, like `comment`, `text`) for sentiment.

## Technology Stack

*   **Python 3.9+**
*   **Streamlit:** For building the interactive web application.
*   **Pandas:** For data manipulation and Excel file processing.
*   **Hugging Face `transformers`:**
    *   **GPT-2:** For local text generation.
    *   **RoBERTa-base-sentiment (cardiffnlp):** For pre-trained sentiment analysis.
*   **PyTorch:** The underlying deep learning framework for `transformers` models.
*   **Groq API:** For high-speed inference with models like Llama-3.1.
*   **`python-dotenv`:** For managing environment variables (API keys).

## Setup and Local Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/TSPUCH/chatbot.git
cd chatbot.git
```


Secondly, Install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
