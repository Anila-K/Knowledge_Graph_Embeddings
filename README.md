# Knowledge Graph Builder with OpenAI Embeddings

This project demonstrates how to build a simple **knowledge graph** using **OpenAI’s `text-embedding-ada-002`** model and **Neo4j** to:

- Generate semantic embeddings for input sentences
- Store them as graph nodes
- Create similarity-based relationships between them
- Perform semantic similarity queries

---

## Features

- Accepts a list of text inputs
- Generates embeddings using OpenAI's `text-embedding-ada-002`
- Stores sentences and embeddings in **Neo4j**
- Connects similar sentences using cosine similarity
- Queries for the most semantically similar node to new input

---

## File Overview

| File | Description |
|------|-------------|
| `Knowledge_Graph_Embeddings.py` | Main script to run the pipeline end-to-end |

---

## Requirements

- Python 3.6+
- OpenAI Python SDK
- Neo4j (Aura Free Tier or local instance)
- scikit-learn
- numpy
- requests

Install all dependencies:

```bash
pip install openai neo4j scikit-learn numpy requests
