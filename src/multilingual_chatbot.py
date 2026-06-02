import streamlit as st

st.set_page_config(
    page_title="Multi-Language Chatbot"
)

from langdetect import detect
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.0-flash-lite"
)

st.title("🌍 Multi-Language Chatbot")

st.write(
    """
    Supported Languages:

    • English
    • Tamil
    • Hindi
    • Spanish
    """
)

user_input = st.text_area(
    "Ask anything"
)

if st.button("Submit"):

    if user_input:

        try:

            language = detect(
                user_input
            )

        except:

            language = "en"

        st.subheader(
            "Detected Language"
        )

        st.success(
            language
        )

        prompt = f"""
        Detect the user's language and
        answer in the SAME language.

        User Message:
        {user_input}

        Requirements:
        - Use the same language.
        - Be culturally appropriate.
        - Give a natural response.
        """

        try:

            response = model.generate_content(
                prompt
            )

            st.subheader(
                "Chatbot Response"
            )

            st.write(
                response.text
            )

        except Exception:

            st.error(
                "Gemini quota exceeded. Try again later."
            )