import subprocess
import json
import numpy as np
import requests
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Set OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your actual key

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print("Error extracting embedding from OpenAI:", e)
        return None

# Connect to Neo4j 
# Adjust URI, user, password by using your own credentials
uri = "neo4j+s://0471dec5.databases.neo4j.io"
user = "neo4j"
password = "ENMMHQy7jHicC8flmOHLuT5Bm5HjEq3RRPTqyLdadbo"

driver = GraphDatabase.driver(uri, auth=(user, password))

# Store sentence and embedding in Neo4j
def store_node(tx, sentence, embedding):
    tx.run("""
        MERGE (s:Sentence {text: $sentence})
        SET s.embedding = $embedding
    """, sentence=sentence, embedding=embedding)

def store_relationship(tx, s1, s2, similarity):
    tx.run("""
        MATCH (a:Sentence {text: $s1}), (b:Sentence {text: $s2})
        MERGE (a)-[:SIMILAR {score: $similarity}]->(b)
    """, s1=s1, s2=s2, similarity=similarity)

#Input & Processing 
sentences = [
    "AI is transforming the world.",
    "Machine learning is a subset of AI.",
    "Bananas are yellow fruits.",
    "Deep learning powers many AI applications."
    "AI can analyze large datasets quickly.",
    "Natural language processing enables machines to understand human language."
]

with driver.session() as session:
    embeddings = []
    for s in sentences:
        emb = get_embedding(s)
        if emb:
            embeddings.append((s, emb))
            session.write_transaction(store_node, s, emb)

    # Create relationships based on cosine similarity
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            s1, e1 = embeddings[i]
            s2, e2 = embeddings[j]
            sim = cosine_similarity([e1], [e2])[0][0]
            if sim > 0.8:  #default threshold given.
                session.write_transaction(store_relationship, s1, s2, float(sim))

# Query: Find most similar sentence to a new input 
new_sentence = "Natural language processing is a key area of AI."
new_embedding = get_embedding(new_sentence)

with driver.session() as session:
    results = session.run("""
        MATCH (s:Sentence)
        RETURN s.text AS text, s.embedding AS embedding
    """)
    texts, emb_vectors = [], []
    for record in results:
        texts.append(record["text"])
        emb_vectors.append(record["embedding"])

    sims = cosine_similarity([new_embedding], emb_vectors)[0]
    top_idx = np.argmax(sims)
    print(f"\n Input: {new_sentence}")
    print(f" Most Similar: {texts[top_idx]} (Score: {sims[top_idx]:.3f})")
