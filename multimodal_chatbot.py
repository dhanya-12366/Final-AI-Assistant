import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# ------------------
# Gemini Setup
# ------------------

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ------------------
# Models
# ------------------

text_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

vision_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ------------------
# UI
# ------------------

st.set_page_config(
    page_title="Multi Modal Chatbot"
)

st.title("🤖 Multi Modal Chatbot")

user_prompt = st.text_input(
    "Enter your question"
)

uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------
# Submit
# ------------------

if st.button("Submit"):

    # TEXT ONLY
    if uploaded_image is None:

        if user_prompt:

            response = text_model.generate_content(
                user_prompt
            )

            st.subheader("Response")

            st.write(
                response.text
            )

    # IMAGE + TEXT
    else:

        image = Image.open(
            uploaded_image
        )

        st.image(
            image,
            caption="Uploaded Image"
        )

        if user_prompt:

            response = vision_model.generate_content(
                [user_prompt, image]
            )

        else:

            response = vision_model.generate_content(
                image
            )

        st.subheader("Response")

        st.write(
            response.text
        )