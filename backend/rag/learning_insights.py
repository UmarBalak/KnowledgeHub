import numpy as np
from sklearn.cluster import KMeans
from llmModels import LLM
import os
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

# -----------------------------
# 1. Cluster Queries (Intent Discovery)
# -----------------------------
def get_query_clusters(query_logs, k=5):
    """
    Groups queries into K clusters and returns the centroids and grouped queries.
    """
    if not query_logs:
        return [], []

    # Extract embeddings
    embeddings = np.array([q.query_embedding for q in query_logs])
    
    # Handle cases with fewer queries than k
    num_samples = len(embeddings)
    actual_k = min(k, num_samples)
    
    if actual_k < 1:
        return [], []

    kmeans = KMeans(n_clusters=actual_k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Group actual query objects by cluster label
    clustered_queries = {i: [] for i in range(actual_k)}
    for idx, label in enumerate(labels):
        clustered_queries[label].append(query_logs[idx])

    return centroids, clustered_queries

# -----------------------------
# 2. Check Coverage (Inverted Logic)
# -----------------------------
def analyze_cluster_coverage(centroids, space_id, rag_pipeline, threshold=0.75):
    """
    Instead of fetching all docs, we ask Pinecone: 
    'Do you have any content similar to this abstract query centroid?'
    """
    gaps_indices = []
    covered_count = 0
    
    # Access the raw index from your pipeline
    index = rag_pipeline.pcIndex.Index(rag_pipeline.index_name)

    for i, centroid in enumerate(centroids):
        # Query Pinecone with the CENTROID vector
        try:
            res = index.query(
                vector=centroid.tolist(),
                filter={"space_id": space_id},
                top_k=1, # We only need the single best match to determine coverage
                include_values=False
            )
            
            # Check the score of the best match
            matches = res.get("matches", [])
            best_score = matches[0]["score"] if matches else 0.0

            if best_score >= threshold:
                covered_count += 1
            else:
                gaps_indices.append(i) # This cluster index is a gap
                
        except Exception as e:
            print(f"Error checking coverage for cluster {i}: {e}")
            # Assume gap on error to be safe
            gaps_indices.append(i)

    coverage_score = 0
    if len(centroids) > 0:
        coverage_score = int((covered_count / len(centroids)) * 100)

    return coverage_score, gaps_indices

# -----------------------------
# 3. Summarize Gaps
# -----------------------------
def summarize_gap_topics(gap_indices, clustered_queries):
    """
    Sends the actual query text to LLM to summarize the confusion.
    """
    llm = LLM(llm_model=LLM_MODEL, max_messages=0)
    summaries = []

    for idx in gap_indices:
        queries = clustered_queries.get(idx, [])
        
        # FIX: We need 'query_text', currently missing in QueryLog model.
        # Assuming you add it, or temporarily use a placeholder.
        # We take a sample of 5 to avoid overflowing context
        sample_texts = [
            getattr(q, "query_text", "Text unavailable") 
            for q in queries[:5]
        ]
        
        prompt = f"""
        The following are user questions that retrieved NO relevant documentation from our knowledge base:
        
        {sample_texts}
        
        Identify the specific technical topic or concept these users are looking for.
        Return ONLY a short, 3-5 word label (e.g., "SQL Window Functions", "Azure Blob SAS Tokens").
        """
        
        try:
            summary = llm.invoke(prompt)["content"].strip()
            # Clean up quotes if LLM adds them
            summary = summary.replace('"', '').replace("'", "")
            summaries.append(summary)
        except Exception as e:
            summaries.append("Unidentified Topic")

    return summaries
