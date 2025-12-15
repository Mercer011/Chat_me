import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load conversation data
with open("../data/conversations.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)

# Extract questions
questions = [item["question"] for item in conversations]

# Generate embeddings
embeddings = model.encode(questions)

# Convert embeddings to float32 (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index (cosine similarity via inner product)
index = faiss.IndexFlatIP(dimension)

# Normalize embeddings (required for cosine similarity)
faiss.normalize_L2(embeddings)

# Add embeddings to index
index.add(embeddings)

print("FAISS index created")
print("Total vectors stored:", index.ntotal)

# ----------- SEARCH TEST -----------

query = "Can we go for dinner tommorow?"

# Embed the query
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
faiss.normalize_L2(query_embedding)

# Search top 3 similar questions
k = 3
scores, indices = index.search(query_embedding, k)

print("\nQuery:", query)
print("\nTop similar past questions:")

for rank, idx in enumerate(indices[0]):
    print(f"{rank + 1}. {questions[idx]} (score: {scores[0][rank]:.4f})")
