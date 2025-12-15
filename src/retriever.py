import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load conversations
# -----------------------------
with open("../data/conversations.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)

questions = [item["question"] for item in conversations]
answers = [item["answer"] for item in conversations]

# -----------------------------
# Create embeddings
# -----------------------------
embeddings = model.encode(questions)
embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# -----------------------------
# Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# -----------------------------
# QUERY
# -----------------------------
query = "Are you coming today?"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
faiss.normalize_L2(query_embedding)

# -----------------------------
# SEARCH
# -----------------------------
k = 3
scores, indices = index.search(query_embedding, k)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\nUser message:")
print(query)

print("\nSuggested replies based on memory:\n")

for rank, idx in enumerate(indices[0]):
    print(f"{rank + 1}. Similar question:")
    print(f"   {questions[idx]}")
    print(f"   → Your reply:")
    print(f"   {answers[idx]}")
    print(f"   (similarity score: {scores[0][rank]:.4f})\n")
