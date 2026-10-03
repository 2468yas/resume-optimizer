from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 3 sentences: resume line, job line that MEANS the same thing, and something random
resume = "Built REST APIs using Flask and PostgreSQL"
job = "Experience developing backend services and databases"
random = "I love cooking pasta on the weekends"

# --- old way: TF-IDF (only cares about matching words) ---
tfidf = TfidfVectorizer().fit_transform([resume, job, random])
print("TF-IDF resume vs job:   ", round(cosine_similarity(tfidf[0], tfidf[1])[0][0], 3))
print("TF-IDF resume vs random:", round(cosine_similarity(tfidf[0], tfidf[2])[0][0], 3))

# --- new way: embeddings (cares about MEANING) ---
model = SentenceTransformer("all-MiniLM-L6-v2")  # small fast model, downloads once
emb = model.encode([resume, job, random])        # turns each sentence into 384 numbers
print("Embed  resume vs job:   ", round(util.cos_sim(emb[0], emb[1]).item(), 3))
print("Embed  resume vs random:", round(util.cos_sim(emb[0], emb[2]).item(), 3))