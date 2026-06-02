import streamlit as st

st.set_page_config(
    page_title="Computer Science Expert Chatbot",
    layout="wide"
)

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

env_path = os.path.join(
    BASE_DIR,
    "..",
    ".env"
)

load_dotenv(env_path)

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

genai.configure(
    api_key=GOOGLE_API_KEY
)

model = genai.GenerativeModel(
    "gemini-2.5-flash-lite"
)

# =====================================
# VECTOR DATABASE PATHS
# =====================================

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "..",
    "vector_db"
)

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "arxiv.index"
)

DOC_FILE = os.path.join(
    VECTOR_DIR,
    "documents.pkl"
)

# =====================================
# LOAD RESOURCES
# =====================================

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        DOC_FILE,
        "rb"
    ) as f:

        documents = pickle.load(f)

    return (
        embedding_model,
        index,
        documents
    )

embedding_model, index, documents = (
    load_resources()
)

# =====================================
# CHAT MEMORY
# =====================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# =====================================
# HEADER
# =====================================

st.title(
    "📚 Computer Science Domain Expert Chatbot"
)

st.markdown(
    """
    Ask advanced Computer Science questions.

    Features:
    - arXiv Research Retrieval
    - Research Summarization
    - Concept Explanation
    - Follow-up Question Support
    """
)

# =====================================
# USER INPUT
# =====================================

question = st.text_input(
    "Ask a Computer Science Question"
)

# =====================================
# QUESTION HANDLING
# =====================================

if question:

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype(
        "float32"
    )

    distances, indices = index.search(
        query_embedding,
        k=5
    )

    context = ""

    retrieved_papers = []

    for idx in indices[0]:

        paper = documents[idx]

        retrieved_papers.append(
            paper[:500]
        )

        context += paper
        context += "\n\n"

    # =================================
    # SHOW RETRIEVED PAPERS
    # =================================

    st.subheader(
        "📄 Retrieved Research Papers"
    )

    for i, paper in enumerate(
        retrieved_papers,
        start=1
    ):

        with st.expander(
            f"Paper {i}"
        ):

            st.write(
                paper
            )

    # =================================
    # CHAT HISTORY
    # =================================

    history = "\n".join(
        st.session_state.chat_history
    )

    prompt = f"""
    You are a Computer Science Research Expert.

    Previous Conversation:
    {history}

    Research Context:
    {context}

    User Question:
    {question}

    Instructions:

    1. Answer accurately.
    2. Explain complex concepts simply.
    3. Summarize relevant research.
    4. Mention important findings.
    5. Support follow-up discussion.
    """

    try:

        response = model.generate_content(
            prompt
        )

        answer = response.text

    except Exception:

        answer = """
Gemini quota exceeded.

The arXiv papers were successfully retrieved.

Please try again later or use a new Gemini API key.
"""

    st.subheader(
        "🤖 Expert Answer"
    )

    st.write(
        answer
    )

    # =================================
    # CONCEPT VISUALIZATION
    # =================================

    try:

        concept_prompt = f"""
        Extract the key Computer Science concepts
        involved in the question below.

        Question:
        {question}

        Return:
        - Main Concepts
        - Related Topics
        - Important Keywords

        Use bullet points.
        """

        concept_response = (
            model.generate_content(
                concept_prompt
            )
        )

        st.subheader(
            "🧠 Concept Visualization"
        )

        st.write(
            concept_response.text
        )

    except Exception:

        st.subheader(
            "🧠 Concept Visualization"
        )

        st.write(
            """
- Concept extraction unavailable
- Gemini quota exceeded
"""
        )

    # =================================
    # STORE MEMORY
    # =================================

    st.session_state.chat_history.append(
        f"User: {question}"
    )

    st.session_state.chat_history.append(
        f"Assistant: {answer}"
    )

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.header(
        "Conversation History"
    )

    for item in st.session_state.chat_history:

        st.write(item)

    if st.button(
        "Clear History"
    ):

        st.session_state.chat_history = []

        st.rerun()