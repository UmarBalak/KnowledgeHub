import numpy as np
import logging
import os
import json
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from llmModels import LLM
from models import QueryLog 

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LearningInsights")

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

def fetch_space_queries(db: Session, space_id: int, limit: int = 200):
    return (
        db.query(QueryLog)
        .filter(QueryLog.space_id == space_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    
# -----------------------------
# 1. Cluster Queries (Intent Discovery)
# -----------------------------
def get_query_clusters(query_logs, k=5):
    """
    Groups queries into K clusters.
    Returns clustered_queries dictionary.
    """
    logger.info(f"STEP 1: Starting Query Clustering. Total logs received: {len(query_logs)}")
    
    if not query_logs:
        logger.warning("STEP 1 SKIPPED: No query logs provided.")
        return {}

    try:
        # Extract embeddings
        embeddings = np.array([q.query_embedding for q in query_logs])
        
        # Handle cases with fewer queries than k
        num_samples = len(embeddings)
        actual_k = min(k, num_samples)
        
        if actual_k < 1:
            return {}

        kmeans = KMeans(n_clusters=actual_k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group actual query objects by cluster label
        clustered_queries = {i: [] for i in range(actual_k)}
        for idx, label in enumerate(labels):
            clustered_queries[label].append(query_logs[idx])
        
        return clustered_queries

    except Exception as e:
        logger.error(f"STEP 1 FAILED: Error during clustering: {str(e)}", exc_info=True)
        return {}

# -----------------------------
# 2. Check Coverage (Robust Sampling)
# -----------------------------
def analyze_cluster_coverage(clustered_queries, space_id, rag_pipeline, threshold=0.75):
    """
    Checks coverage by testing ACTUAL queries from each cluster against Pinecone.
    """
    logger.info(f"STEP 2: Starting Coverage Analysis for Space ID: {space_id}")
    
    if not clustered_queries:
        return 0, []

    gaps_indices = []
    covered_clusters_count = 0
    
    try:
        index = rag_pipeline.pcIndex.Index(rag_pipeline.index_name)

        for label, queries in clustered_queries.items():
            # STRATEGY: Sample the top 3 most recent queries in this cluster
            # This avoids checking 1000s of queries but is better than checking 1 centroid
            samples = queries[:3] 
            
            passing_samples = 0
            
            logger.info(f"--- Analyzing Cluster {label} (checking {len(samples)} samples) ---")

            for q in samples:
                # Use the ACTUAL embedding from the user's query
                # This ensures we test exactly what the user asked
                vector_list = q.query_embedding
                
                res = index.query(
                    vector=vector_list,
                    filter={"space_id": {"$eq": space_id}},
                    top_k=1, 
                    include_values=False
                )
                
                matches = res.get("matches", [])
                if matches:
                    score = matches[0].get("score", 0.0)
                    if score >= threshold:
                        passing_samples += 1
            
            # DECISION LOGIC:
            # If more than 50% of the sample queries in this cluster failed the threshold,
            # we consider the ENTIRE topic a "Learning Gap".
            pass_rate = passing_samples / len(samples)
            
            if pass_rate >= 0.5:
                logger.info(f"   > ✅ Cluster {label} COVERED (Pass Rate: {pass_rate:.2f})")
                covered_clusters_count += 1
            else:
                logger.info(f"   > ❌ Cluster {label} is a GAP (Pass Rate: {pass_rate:.2f})")
                gaps_indices.append(label)

        # Calculate final coverage score
        total_clusters = len(clustered_queries)
        coverage_score = int((covered_clusters_count / total_clusters) * 100) if total_clusters > 0 else 0
        
        return coverage_score, gaps_indices

    except Exception as e:
        logger.error(f"STEP 2 FAILED: Pinecone check error: {str(e)}", exc_info=True)
        return 0, []

# -----------------------------
# 3. Summarize Gaps (LLM)
# -----------------------------
def summarize_gap_topics(gap_indices, clustered_queries):
    """
    Summarizes the missing topics using the LLM.
    """
    if not gap_indices:
        return []

    llm = LLM(llm_model=LLM_MODEL, max_messages=0)
    summaries = []

    for idx in gap_indices:
        queries = clustered_queries.get(idx, [])
        
        # Use the text of the queries we actually found
        sample_texts = [q.query_text for q in queries[:5]]
        
        prompt = f"""
        The following are specific user questions that our knowledge base failed to answer confidently (low similarity score matches):
        
        {sample_texts}
        
        Analyze these questions and identify the MISSING knowledge topic.
        Return ONLY a short, 3-5 word label (e.g., "React Hook Form Validation", "Azure Blob Storage Authentication").
        """
        
        try:
            response = llm.invoke(prompt)
            summary = response.get("content", "").strip().replace('"', '').replace("'", "")
            summaries.append(summary)
        except Exception:
            summaries.append("Unidentified Topic")

    return summaries