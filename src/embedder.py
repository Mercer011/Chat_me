import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("../data/conversations.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)

questions = [item["question"] for item in conversations]

embeddings = model.encode(questions)

print("Number of questions:", len(questions))
print("Embedding shape:", embeddings.shape)
print("\nFirst question:", questions[0])
print("First embedding (first 10 values):")
print(embeddings[0][:10])
