import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "..",
    "arxiv_data",
    "filtered_cs_papers.json"
)

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "..",
    "vector_db"
)

os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "arxiv.index"
)

DOC_FILE = os.path.join(
    VECTOR_DIR,
    "documents.pkl"
)

documents = []

print("Reading papers...")

with open(DATA_FILE, "r", encoding="utf-8") as f:

    for line in f:

        paper = json.loads(line)

        text = (
            paper["title"]
            + "\n\n"
            + paper["abstract"]
        )

        documents.append(text)

print(f"{len(documents)} papers loaded")

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    batch_size=32
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
    INDEX_FILE
)

with open(
    DOC_FILE,
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

print("Vector Database Created Successfully")