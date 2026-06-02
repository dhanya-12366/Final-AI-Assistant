import streamlit as st
st.set_page_config(page_title="Medical Q&A Chatbot")

import os
import xml.etree.ElementTree as ET
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# =====================================
# GEMINI SETUP
# =====================================

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "medical_data"
)

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "..",
    "medical_vector_db"
)

os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(
    VECTOR_DIR,
    "medical.index"
)

DATA_PATH = os.path.join(
    VECTOR_DIR,
    "medical.pkl"
)

# =====================================
# EMBEDDING MODEL
# =====================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================
# MEDICAL ENTITIES
# =====================================

medical_entities = [
    "diabetes",
    "cancer",
    "asthma",
    "covid",
    "hypertension",
    "headache",
    "fever",
    "treatment",
    "symptom",
    "disease"
]

# =====================================
# LOAD MEDQUAD
# =====================================

@st.cache_resource
def build_database():

    documents = []

    for root_dir, dirs, files in os.walk(DATA_DIR):

        for file in files:

            if file.endswith(".xml"):

                file_path = os.path.join(
                    root_dir,
                    file
                )

                try:

                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    for qa in root.findall(".//QAPair"):

                        question = ""
                        answer = ""

                        q = qa.find("Question")
                        a = qa.find("Answer")

                        if q is not None:
                            question = q.text or ""

                        if a is not None:
                            answer = a.text or ""

                        if question and answer:

                            documents.append({
                                "question": question,
                                "answer": answer
                            })

                except Exception:
                    pass

    texts = []

    for doc in documents:

        texts.append(
            doc["question"] +
            "\n" +
            doc["answer"]
        )

    embeddings = embedding_model.encode(
        texts
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        DATA_PATH,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )

    return index, documents


# =====================================
# LOAD DATABASE
# =====================================

if (
    os.path.exists(INDEX_PATH)
    and os.path.exists(DATA_PATH)
):

    index = faiss.read_index(
        INDEX_PATH
    )

    with open(
        DATA_PATH,
        "rb"
    ) as f:

        documents = pickle.load(f)

else:

    with st.spinner(
        "Building Medical Database..."
    ):

        index, documents = build_database()

# =====================================
# UI
# =====================================

st.title("🏥 Medical Q&A Chatbot")

question = st.text_input(
    "Ask a medical question"
)

# =====================================
# QUESTION ANSWERING
# =====================================

if question:

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    D, I = index.search(
        query_embedding,
        k=3
    )

    context = ""

    for idx in I[0]:

        context += (
            documents[idx]["question"]
            + "\n"
            + documents[idx]["answer"]
            + "\n\n"
        )

    found_entities = []

    lower_question = question.lower()

    for entity in medical_entities:

        if entity in lower_question:

            found_entities.append(
                entity
            )

    if found_entities:

        st.subheader(
            "Detected Medical Entities"
        )

        st.write(
            ", ".join(found_entities)
        )

    prompt = f"""
    You are a medical assistant.

    Answer ONLY using the
    context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = model.generate_content(
        prompt
    )

    st.subheader(
        "Medical Answer"
    )

    st.write(
        response.text
    )