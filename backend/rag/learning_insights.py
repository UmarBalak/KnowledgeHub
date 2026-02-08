import logging
import os
import json
import numpy as np
from sqlalchemy.orm import Session
from sklearn.cluster import KMeans
from dotenv import load_dotenv

from llmModels import LLM
from models import QueryLog 

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LearningInsights")

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

def fetch_recent_queries(db: Session, space_id: int, limit: int = 300):
    """
    Fetch the most recent queries for analysis.
    Limit to 300 to keep performance snappy.
    """
    return (
        db.query(QueryLog)
        .filter(QueryLog.space_id == space_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )

def identify_gaps_from_logs(query_logs, strong_threshold=0.65, weak_threshold=0.4):
    """
    Analyzes historical logs to determine coverage using LLM-Judge flags ('relevance_status')
    OR 'One Strong OR Two Weak' vector logic for legacy logs.
    
    Logic:
    1. CHECK LLM JUDGE (New Method):
       - FULL / PARTIAL -> Pass (Covered)
       - NONE -> Fail (Gap)
       
    2. FALLBACK (Old Method for legacy logs where status is UNKNOWN):
       - 1 chunk >= strong_threshold
       - OR 2 chunks >= weak_threshold
    """
    gaps = []
    covered_count = 0
    total = len(query_logs)

    if total == 0:
        return 0, []

    for log in query_logs:
        is_covered = False
        
        # --- LEVEL 1: LLM-as-a-Judge (Primary Check) ---
        # Access the new column directly. 
        # getattr defaults to 'UNKNOWN' if for some reason the attribute is missing on an old object
        status = getattr(log, 'relevance_status', 'UNKNOWN')
        
        if status == 'FULL':
            is_covered = True
        elif status == 'PARTIAL':
            is_covered = False
        elif status == 'NONE':
            is_covered = False
            
        # --- LEVEL 2: Vector Score Fallback (Secondary Check) ---
        # Only run this expensive logic if the status is UNKNOWN (legacy logs)
        else:
            try:
                sources = log.sources
                # Handle case where sources might be stored as a string in some DBs
                if isinstance(sources, str):
                    sources = json.loads(sources)
                
                if not isinstance(sources, list):
                    sources = []
            except Exception as e:
                logger.warning(f"Failed to parse sources for log {log.id}: {e}")
                sources = []

            # Extract Scores
            scores = [float(s.get("similarity_score", 0.0)) for s in sources if isinstance(s, dict)]
            
            # Rule A: The "Golden Source" (One high-quality match is enough)
            if any(s >= strong_threshold for s in scores):
                is_covered = True
                
            # Rule B: The "Corroborated Evidence" (Two decent matches are enough)
            elif len([s for s in scores if s >= weak_threshold]) >= 2:
                is_covered = True

        # --- FINAL DECISION ---
        if is_covered:
            covered_count += 1
        else:
            gaps.append(log)

    # Calculate percentage
    coverage_score = int((covered_count / total) * 100) if total > 0 else 0
    logger.info(f"Gap Analysis: {covered_count}/{total} queries covered. Found {len(gaps)} gaps.")
    
    return coverage_score, gaps

def cluster_and_summarize_gaps(gap_logs, n_clusters=5):
    """
    Takes the queries identified as GAPS, clusters them, and names the missing topics using LLM.
    """
    if not gap_logs:
        return []

    try:
        # 1. Extract Embeddings
        # We use the embeddings already stored in the log (no new API costs)
        embeddings = [np.array(log.query_embedding) for log in gap_logs]
        
        # Adjust cluster count if we have fewer gaps than n_clusters
        actual_k = min(n_clusters, len(embeddings))
        if actual_k < 1:
            return []

        # 2. Perform K-Means Clustering
        kmeans = KMeans(n_clusters=actual_k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # 3. Group Text by Cluster
        clustered_texts = {i: [] for i in range(actual_k)}
        for idx, label in enumerate(labels):
            clustered_texts[label].append(gap_logs[idx].query_text)

        # 4. Generate Summaries with LLM
        llm = LLM(llm_model=LLM_MODEL, max_messages=0)
        topic_summaries = []

        for label, texts in clustered_texts.items():
            # Send top 10 examples to LLM
            examples = texts[:10]
            
            prompt = f"""
            The following are user questions where our vector database found NO relevant internal documents (low similarity scores):
            
            {examples}
            
            These questions were likely answered using the AI's general knowledge.
            Analyze them and identify the **Missing Documentation Topic**.
            Note: If the topic is purely "Chit-Chat" (e.g., greetings, jokes), ignore them.
            
            Return ONLY a short, 3-5 word label (e.g., "Project X Deployment", "Python Basics", "HR Leave Policy").
            """
            
            try:
                response = llm.invoke(prompt)
                topic = response.get("content", "").strip().replace('"', '').replace("'", "")
                
                topic_summaries.append({
                    "topic": topic,
                    "gap_count": len(texts), # Number of users asking about this
                    "example_query": examples[0]
                })
            except Exception as e:
                logger.error(f"LLM Summary failed: {e}")
                topic_summaries.append({"topic": "Unidentified Topic", "gap_count": len(texts)})
        
        # Sort by frequency (highest impact gaps first)
        topic_summaries.sort(key=lambda x: x['gap_count'], reverse=True)
        return topic_summaries

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        return []

def get_learning_gap_insights(db: Session, space_id: int):
    """
    Main Orchestrator function called by API.
    """
    # 1. Fetch Logs
    logs = fetch_recent_queries(db, space_id)
    
    # Require at least 5 queries to form a valid insight
    if len(logs) < 5:
        return {
            "coverage_score": 0, 
            "top_learning_gaps": [], 
            "message": "Insufficient data (need 5+ queries)"
        }

    # 2. Analyze Scores (1 Strong OR 2 Weak)
    # Recommended: Strong=0.65 (High confidence), Weak=0.45 (Loose relevance)
    coverage_score, gap_logs = identify_gaps_from_logs(
        logs, 
        strong_threshold=0.65, 
        weak_threshold=0.3
    )

    # 3. Cluster & Label Gaps
    gap_topics = cluster_and_summarize_gaps(gap_logs)

    # Return simplified list for frontend
    return {
        "coverage_score": coverage_score,
        "top_learning_gaps": [t['topic'] for t in gap_topics],
        "details": gap_topics 
    }