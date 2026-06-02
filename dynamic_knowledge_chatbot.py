import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Dynamic Knowledge Base Chatbot")

import google.generativeai as genai
import os
import pickle
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==================================
# FOLDERS
# ==================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
CHUNK_PATH = os.path.join(VECTOR_DIR, "chunks.pkl")

# ==================================
# GEMINI CONFIGURATION
# ==================================

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==================================
# EMBEDDING MODEL
# ==================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==================================
# STREAMLIT UI
# ==================================

st.title("📚 Dynamic Knowledge Base Chatbot")

st.write(
    "Upload PDFs and build a dynamic knowledge base."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# ==================================
# PDF PROCESSING
# ==================================

if uploaded_file:

    pdf_path = os.path.join(
        KNOWLEDGE_DIR,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"{uploaded_file.name} uploaded successfully"
    )

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.subheader("Extracted Text Preview")

    st.text_area(
        "Preview",
        text[:2000],
        height=200
    )

    chunk_size = 500

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append({
            "text": text[i:i + chunk_size],
            "source": uploaded_file.name
        })

    st.success(
        f"{len(chunks)} chunks created"
    )

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks]
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    # ==================================
    # LOAD EXISTING DATABASE
    # ==================================

    if os.path.exists(INDEX_PATH):

        index = faiss.read_index(
            INDEX_PATH
        )

        with open(
            CHUNK_PATH,
            "rb"
        ) as f:

            existing_chunks = pickle.load(f)

        index.add(
            embeddings
        )

        existing_chunks.extend(
            chunks
        )

        all_chunks = existing_chunks

        st.info(
            "Existing Knowledge Base Updated"
        )

    else:

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(
            embeddings
        )

        all_chunks = chunks

        st.info(
            "New Knowledge Base Created"
        )

    # ==================================
    # SAVE DATABASE
    # ==================================

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        CHUNK_PATH,
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )

    st.session_state["faiss_index"] = index
    st.session_state["chunks"] = all_chunks

    st.success(
        "Knowledge Base Ready"
    )

# ==================================
# LOAD DATABASE ON APP START
# ==================================

if (
    "faiss_index" not in st.session_state
    and os.path.exists(INDEX_PATH)
):

    st.session_state["faiss_index"] = (
        faiss.read_index(
            INDEX_PATH
        )
    )

    with open(
        CHUNK_PATH,
        "rb"
    ) as f:

        st.session_state["chunks"] = (
            pickle.load(f)
        )

# ==================================
# QUESTION ANSWERING
# ==================================

question = st.text_input(
    "Ask a Question"
)

if question:

    if "faiss_index" not in st.session_state:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        query_embedding = (
            embedding_model.encode(
                [question]
            )
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = (
            st.session_state[
                "faiss_index"
            ].search(
                query_embedding,
                k=5
            )
        )

        context = ""
        sources = set()

        for idx in indices[0]:

            chunk = st.session_state[
                "chunks"
            ][idx]

            # Handle old databases safely
            if isinstance(chunk, dict):

                context += (
                    chunk["text"] + "\n"
                )

                sources.add(
                    chunk["source"]
                )

            else:

                context += (
                    str(chunk) + "\n"
                )

        st.subheader(
            "Retrieved Context"
        )

        st.text_area(
            "Context Used",
            context,
            height=250
        )

        if len(sources) > 0:

            st.subheader(
                "Source Documents"
            )

            for source in sources:

                st.write(
                    f"📄 {source}"
                )

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = model.generate_content(
            prompt
        )

        st.subheader(
            "Answer"
        )

        st.write(
            response.text
        )