from sentence_transformers import SentenceTransformer
import faiss
import numpy as np





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
def split_text(text, chunk_size=100, overlap=20):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

chunks = split_text(text, chunk_size=100, overlap=20)

documents = []

for i, chunk in enumerate(chunks):
    documents.append({
        "chunk_id": i,
        "source": "networks.txt",
        "text": chunk
    })

print("\nChunks:")

for i, doc in enumerate(documents):
    print(f"\nChunk {i + 1}:")
    print(doc["text"])

# -----------------------------
# 3. Create embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode([doc["text"] for doc in documents])

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


 # -----------------------------
# 6. Retrieval function
# -----------------------------

def retrieve(question, k=3):
    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    k = min(k, len(documents))

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        doc = documents[idx]

        results.append({
            "score": float(score),
            "chunk_id": doc["chunk_id"],
            "source": doc["source"],
            "text": doc["text"]
        })

    return results


# -----------------------------
# 7. Test retrieval
# -----------------------------

results = retrieve(
    "How are website names converted into network addresses?",
    k=3
)

print("\nRetrieved results:")

for result in results:
    print(f"\nScore: {result['score']:.4f}")
    print(f"Chunk: {result['chunk_id']}")
    print(f"Source: {result['source']}")
    print(f"Text: {result['text']}")