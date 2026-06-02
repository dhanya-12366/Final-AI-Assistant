import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langdetect import detect
from textblob import TextBlob
from PIL import Image
from pypdf import PdfReader
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Final AI Assistant",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stTextArea textarea {
    border-radius: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

.feature-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e293b;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ======================
# GEMINI SETUP
# ======================

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash-lite"
)

# ======================
# DOMAIN EXPERT DB
# ======================

@st.cache_resource
def load_medical_resources():

    medical_index = faiss.read_index(
        "../Medical_QA_Chatbot/medical_vector_db/medical.index"
    )

    with open(
        "../Medical_QA_Chatbot/medical_vector_db/medical.pkl",
        "rb"
    ) as f:

        medical_docs = pickle.load(f)

    return medical_index, medical_docs


medical_index, medical_docs = (
    load_medical_resources()
)
@st.cache_resource
def load_domain_resources():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    import os
    print(os.getcwd())
    index = faiss.read_index(
        "vector_db/arxiv.index"
    )

    with open(
        "vector_db/documents.pkl",
        "rb"
    ) as f:

        documents = pickle.load(f)

    return embedding_model, index, documents


embedding_model, arxiv_index, arxiv_docs = (
    load_domain_resources()
)

# ======================
# SESSION STATE
# ======================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ======================
# TITLE
# ======================
st.markdown("""
# 🤖 Final AI Assistant

### AI Powered Multi-Modal Knowledge Assistant

Supports:
- 🩺 Medical QA
- 📚 Research Papers
- 📄 PDF Analysis
- 🖼️ Image Understanding
- 🌍 Multi-Language Chat
- 😊 Sentiment Analysis

---
""")
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("Languages","100+")

with col2:
    st.metric("Medical Docs",len(medical_docs))

with col3:
    st.metric("Research Papers",len(arxiv_docs))

with col4:
    st.metric("Chat Messages",
              len(st.session_state.chat_history)//2)
# ======================
# MODE
# ======================

mode = st.selectbox(
    "Select Mode",
    [
    "General Chat",
    "Domain Expert",
    "Medical Assistant",
    "PDF Knowledge Base",
    "Image Analysis"
]
)

# ======================
# IMAGE UPLOAD
# ======================

uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if mode == "Image Analysis" and uploaded_image:

    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    if st.button("Analyze Image"):

        try:

            response = model.generate_content(
                [
                    "Describe this image in detail",
                    image
                ]
            )

            st.subheader(
                "Image Analysis"
            )

            st.write(
                response.text
            )

        except Exception as e:

            st.error(
                str(e)
            )

# ======================
# PDF UPLOAD
# ======================

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

pdf_text = ""

if uploaded_pdf:

    try:

        reader = PdfReader(
            uploaded_pdf
        )

        for page in reader.pages:

            text = page.extract_text()

            if text:

                pdf_text += text

        st.success(
            f"PDF Loaded ({len(pdf_text)} characters)"
        )

    except Exception as e:

        st.error(
            str(e)
        )

# ======================
# USER INPUT
# ======================

user_input = st.text_area(
    "Ask me Anything"
)

# ======================
# SEND
# ======================

if st.button("Send"):

    if user_input:

        # ------------------
        # Language
        # ------------------

        try:
            if len(user_input.split()) < 4:
                language = "en"
            else:
                language = detect(user_input)
        except:
            language = "en"

        # ------------------
        # Sentiment
        # ------------------

        polarity = TextBlob(
            user_input
        ).sentiment.polarity

        if polarity > 0:

            sentiment = "Positive 😊"

        elif polarity < 0:

            sentiment = "Negative 😔"

        else:

            sentiment = "Neutral 😐"

        st.subheader(
            "Detected Language"
        )

        st.success(
            language
        )

        st.subheader(
            "Detected Sentiment"
        )

        st.info(
            sentiment
        )

        # ------------------
        # History
        # ------------------

        history = "\n".join(
            st.session_state.chat_history
        )

        # ------------------
        # Context
        # ------------------

        context = ""

        if mode == "PDF Knowledge Base":

            context = pdf_text[:5000]

        elif mode == "Domain Expert":

            query_embedding = embedding_model.encode(
                [user_input]
            )

            query_embedding = np.array(
                query_embedding
            ).astype("float32")

            distances, indices = arxiv_index.search(
                query_embedding,
                k=3
            )

            for idx in indices[0]:

                context += arxiv_docs[idx]
                context += "\n\n"
        elif mode == "Medical Assistant":

             query_embedding = embedding_model.encode(
                [user_input]
             )

             query_embedding = np.array(
               query_embedding
             ).astype("float32")

             distances, indices = medical_index.search(
               query_embedding,
                k=3
             )

             for idx in indices[0]:

              item = medical_docs[idx]

              context += (
                  "Question: "
                   + item["question"]
                   + "\n"
            )

             context += (
               "Answer: "
              + item["answer"]
               + "\n\n"
         )

        # ------------------
        # Prompt
        # ------------------

        prompt = f"""
Previous Conversation:
{history}

Mode:
{mode}

Context:
{context}

User Message:
{user_input}

Detected Language:
{language}

Detected Sentiment:
{sentiment}

Instructions:
- Reply in the SAME language.
- Use the provided context.
- Use PDF context if PDF mode.
- Use research context if Domain Expert mode.
- Be professional and helpful.
- Use medical context if Medical Assistant mode.
"""

        # ------------------
        # Gemini
        # ------------------

        try:

            response = model.generate_content(
                prompt
            )

            answer = response.text

        except Exception as e:

            answer = f"AI Error:\n\n{e}"

        # ------------------
        # Output
        # ------------------

        st.subheader(
            "Assistant Response"
        )

        st.write(
            answer
        )

        if mode == "Domain Expert":

            st.subheader(
                "Retrieved Research Context"
            )

            st.write(
                context[:3000]
            )

        # ------------------
        # Save Chat
        # ------------------

        st.session_state.chat_history.append(
            f"User: {user_input}"
        )

        st.session_state.chat_history.append(
            f"Assistant: {answer}"
        )

# ======================
# SIDEBAR
# ======================

with st.sidebar:

    st.title("⚙️ Assistant Panel")

    st.metric(
    "Messages",
    len(st.session_state.chat_history)//2
    )

    st.markdown("---")

    st.subheader("Chat History")

    for item in st.session_state.chat_history:

        st.write(
            item
        )

    if st.button(
        "Clear History"
    ):

        st.session_state.chat_history = []

        st.rerun()