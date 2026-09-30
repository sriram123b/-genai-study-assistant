from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "DNS translates domain names into IP addresses.",
    "HTTP is used to transfer web resources between clients and servers.",
    "TCP provides reliable and ordered delivery of data.",
    "IP is responsible for addressing and routing packets."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)

question = "How are website names converted into network addresses?"
question_embedding = model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    document_embeddings
)[0]

for document, score in zip(documents, similarities):
    print(f"{score:.4f} -> {document}")

best_index = similarities.argmax()

print("\nMost relevant document:")
print(documents[best_index])