import numpy as np
import logging
import os
import json
from sklearn.cluster import KMeans
from dotenv import load_dotenv

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
    Groups queries into K clusters and returns the centroids and grouped queries.
    """
    logger.info(f"STEP 1: Starting Query Clustering. Total logs received: {len(query_logs)}")
    
    if not query_logs:
        logger.warning("STEP 1 SKIPPED: No query logs provided.")
        return [], []

    try:
        # Extract embeddings
        embeddings = np.array([q.query_embedding for q in query_logs])
        logger.info(f"STEP 1: Extracted embeddings with shape: {embeddings.shape}")

        # Handle cases with fewer queries than k
        num_samples = len(embeddings)
        actual_k = min(k, num_samples)
        
        if actual_k < 1:
            logger.warning("STEP 1: Not enough samples to cluster.")
            return [], []

        logger.info(f"STEP 1: Running KMeans with k={actual_k}")
        kmeans = KMeans(n_clusters=actual_k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        logger.info(f"STEP 1: Clustering complete. Generated {len(centroids)} centroids.")

        # Group actual query objects by cluster label
        clustered_queries = {i: [] for i in range(actual_k)}
        for idx, label in enumerate(labels):
            clustered_queries[label].append(query_logs[idx])
        
        # Debug log for cluster sizes
        for label, queries in clustered_queries.items():
            logger.debug(f"  > Cluster {label}: {len(queries)} queries")

        return centroids, clustered_queries

    except Exception as e:
        logger.error(f"STEP 1 FAILED: Error during clustering: {str(e)}", exc_info=True)
        return [], []

# -----------------------------
# 2. Check Coverage (Pinecone Interaction)
# -----------------------------
def analyze_cluster_coverage(centroids, space_id, rag_pipeline, threshold=0.75):
    """
    Queries Pinecone with centroids to check if documents exist for these concepts.
    """
    logger.info(f"STEP 2: Starting Coverage Analysis for Space ID: {space_id}")
    
    if len(centroids) == 0:
        logger.warning("STEP 2 SKIPPED: No centroids to analyze.")
        return 0, []

    gaps_indices = []
    covered_count = 0
    
    try:
        # Access the raw index from your pipeline
        logger.info(f"STEP 2: Connecting to Pinecone Index: {rag_pipeline.index_name}")
        index = rag_pipeline.pcIndex.Index(rag_pipeline.index_name)

        for i, centroid in enumerate(centroids):
            logger.info(f"--- Analyzing Cluster {i} ---")
            
            # 1. Prepare Vector
            vector_list = centroid.tolist()
            logger.debug(f"   > Vector Dimension: {len(vector_list)}")

            # 2. Query Pinecone
            logger.info(f"   > Querying Pinecone for Cluster {i}...")
            res = index.query(
                vector=vector_list,
                filter={"space_id": {"$eq": space_id}}, # Strict filtering
                top_k=1, # We only need to know if ONE good doc exists
                include_values=False
            )
            
            # 3. Analyze Matches
            matches = res.get("matches", [])
            logger.info(f"   > Pinecone returned {len(matches)} matches.")

            if not matches:
                logger.warning(f"   > ❌ NO MATCHES found for Cluster {i}. (Gap Detected)")
                gaps_indices.append(i)
                continue

            top_match = matches[0]
            score = top_match.get("score", 0.0)
            doc_id = top_match.get("metadata", {}).get("doc_id", "unknown")
            
            logger.info(f"   > Top Match: Doc {doc_id} | Score: {score:.4f} | Threshold: {threshold}")

            if score >= threshold:
                logger.info(f"   > ✅ Cluster {i} is COVERED.")
                covered_count += 1
            else:
                logger.info(f"   > ⚠️ Cluster {i} score too low. (Gap Detected)")
                gaps_indices.append(i)

        # Calculate final score
        coverage_score = int((covered_count / len(centroids)) * 100)
        logger.info(f"STEP 2 COMPLETE: Final Coverage Score: {coverage_score}%")
        
        return coverage_score, gaps_indices

    except Exception as e:
        logger.error(f"STEP 2 FAILED: Error interacting with Pinecone: {str(e)}", exc_info=True)
        # Fail safe: return 0 score and empty gaps to prevent crash
        return 0, []

# -----------------------------
# 3. Summarize Gaps (LLM)
# -----------------------------
def summarize_gap_topics(gap_indices, clustered_queries):
    """
    Summarizes the missing topics using the LLM.
    """
    logger.info(f"STEP 3: Starting Gap Summarization. Found {len(gap_indices)} gaps.")
    
    if not gap_indices:
        logger.info("STEP 3: No gaps to summarize.")
        return []

    llm = LLM(llm_model=LLM_MODEL, max_messages=0)
    summaries = []

    for idx in gap_indices:
        logger.info(f"--- Summarizing Gap for Cluster {idx} ---")
        queries = clustered_queries.get(idx, [])
        
        # Safe extraction of text (Handling if you haven't updated DB yet)
        sample_texts = []
        for q in queries[:5]:
            # Try to get query_text, fallback to hash if text missing (for debugging)
            text = getattr(q, "query_text", None)
            if not text:
                logger.warning("   > 'query_text' missing in QueryLog! Using placeholder.")
                text = f"[Hash: {q.query_hash[:8]}...]" 
            sample_texts.append(text)
        
        logger.info(f"   > Sending {len(sample_texts)} samples to LLM: {sample_texts}")

        prompt = f"""
        The following are trainee questions that retrieved NO relevant documentation from our knowledge base:
        
        {sample_texts}
        
        Identify the specific technical topic or concept these users are looking for.
        Return ONLY a short, 3-5 word label (e.g., "SQL Window Functions", "Azure Blob SAS Tokens").
        """
        
        try:
            response = llm.invoke(prompt)
            summary = response.get("content", "").strip()
            # Clean up cleanup quotes
            summary = summary.replace('"', '').replace("'", "")
            
            logger.info(f"   > LLM Output: {summary}")
            summaries.append(summary)
            
        except Exception as e:
            logger.error(f"   > LLM Summary Failed for Cluster {idx}: {str(e)}")
            summaries.append("Unidentified Topic")

    logger.info(f"STEP 3 COMPLETE: Generated {len(summaries)} summaries.")
    return summaries
