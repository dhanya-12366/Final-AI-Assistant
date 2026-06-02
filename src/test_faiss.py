import faiss
import numpy as np

print("Testing FAISS...")

vectors = np.random.random((10, 384)).astype("float32")

index = faiss.IndexFlatL2(384)

index.add(vectors)

print("FAISS Working")
print("Vectors:", index.ntotal)