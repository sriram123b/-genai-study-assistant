from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def split_text(text, chunk_size=100, overlap=20):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# 1. Original text
# -----------------------------

text = """
Computer networks allow devices to communicate with each other.
TCP provides reliable and ordered delivery of data.
IP is responsible for addressing and routing packets.
DNS translates domain names into IP addresses.
HTTP is used to transfer web resources between clients and servers.
"""


# -----------------------------
# 2. Split text into chunks
# -----------------------------

documents = split_text(text, chunk_size=100, overlap=20)

print("\nChunks:")

for i, chunk in enumerate(documents):
    print(f"\nChunk {i + 1}:")
    print(chunk)


# -----------------------------
# 3. Create embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)

question = "How are website names converted into network addresses?"

question_embedding = model.encode([question])


# -----------------------------
# 4. Create FAISS index
# -----------------------------

document_embeddings = np.array(document_embeddings).astype("float32")

faiss.normalize_L2(document_embeddings)

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)


# -----------------------------
# 5. Search FAISS
# -----------------------------

question_embedding = np.array(question_embedding).astype("float32")

faiss.normalize_L2(question_embedding)

scores, indices = index.search(question_embedding, 2)


# -----------------------------
# 6. Show results
# -----------------------------

print("\nTop 2 relevant chunks:")

for score, index_id in zip(scores[0], indices[0]):
    print(f"{score:.4f} -> {documents[index_id]}")