# learning_insights.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from pinecone import Pinecone

from models import QueryLog
from ragPipeline import RAGPipeline   
from llmModels import LLM

import os
from dotenv import load_dotenv
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

# -----------------------------
# Fetch recent queries per space
# -----------------------------
def fetch_space_queries(db: Session, space_id: int, limit: int = 200):
    return (
        db.query(QueryLog)
        .filter(QueryLog.space_id == space_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )


# -----------------------------
# Cluster query embeddings
# -----------------------------
def cluster_query_embeddings(query_logs, k=5):
    embeddings = np.array([q.query_embedding for q in query_logs])

    if len(embeddings) < k:
        k = max(1, len(embeddings))

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(embeddings)

    return model, labels, embeddings


# -----------------------------
# Fetch document embeddings from Pinecone
# -----------------------------
def fetch_document_embeddings(space_id: int, limit: int = 500):
    """
    Connects to Pinecone to retrieve embeddings for documents in a space.
    This fix bypasses the RAGPipeline class attribute error.
    """
    # 1. Initialize Pinecone using environment variables
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)

    # 2. Use the standard embedding dimension (e.g., 1536 for OpenAI models)
    embedding_dim = 1024
    dummy_vector = [0.0] * embedding_dim

    # 3. Query using metadata filter for the space_id
    res = index.query(
        vector=dummy_vector,
        filter={"space_id": {"$eq": space_id}},
        top_k=limit,
        include_values=True
    )

    return [m["values"] for m in res.get("matches", [])]


# -----------------------------
# Compute coverage score
# -----------------------------
def compute_coverage(cluster_center, document_embeddings):
    if not document_embeddings:
        return 0.0

    sims = cosine_similarity(
        cluster_center.reshape(1, -1),
        np.array(document_embeddings)
    )

    return float(sims.max())


# -----------------------------
# Summarize learning gap (LLM)
# -----------------------------
def summarize_gap(sample_queries: list[str]):
    llm = LLM(llm_model=LLM_MODEL, max_messages=0)

    prompt = f"""
    The following trainee questions show a common confusion:

    {sample_queries}

    Summarize the core technical concept trainees are struggling with.
    Answer in one short phrase.
    """

    return llm.invoke(prompt)["content"].strip()
