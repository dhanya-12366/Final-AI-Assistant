import streamlit as st
from textblob import TextBlob

st.set_page_config(
    page_title="Sentiment Analysis Chatbot"
)

st.title("😊 Sentiment Analysis Chatbot")

user_input = st.text_area(
    "Enter your message"
)

if st.button("Analyze Sentiment"):

    if user_input:

        analysis = TextBlob(user_input)

        polarity = analysis.sentiment.polarity

        if polarity > 0:

            sentiment = "Positive 😊"

            response = """
            Thank you for your positive feedback!
            We're glad you're having a good experience.
            """

        elif polarity < 0:

            sentiment = "Negative 😔"

            response = """
            I'm sorry you're experiencing issues.
            Your feedback is important and we'll try to improve.
            """

        else:

            sentiment = "Neutral 😐"

            response = """
            Thank you for sharing your thoughts.
            Let me know if you need any assistance.
            """

        st.subheader("Detected Sentiment")

        st.success(sentiment)

        st.subheader("Chatbot Response")

        st.write(response)