from langchain.agents import create_excel_agent
from langchain.llms import GroQ
from dotenv import load_dotenv
import os
import streamlit as st

def main():
    load_dotenv()

    # Load the GroQ API key from the environment variable
    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        print("GROQ_API_KEY is not set")
        exit(1)
    else:
        print("GROQ_API_KEY is set")

    st.set_page_config(page_title="Ask your Excel 📈")
    st.header("Ask your Excel 📈")

    excel_file = st.file_uploader("Upload a Excel file", type="xlsx")
    if excel_file is not None:

        agent = create_excel_agent(
            GroQ(temperature=0), excel_file, verbose=True)

        user_question = st.text_input("Ask a question about your Excel: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()