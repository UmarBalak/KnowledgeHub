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

def identify_gaps_from_logs(query_logs, strong_threshold=0.65, weak_threshold=0.45):
    """
    Analyzes historical logs to determine coverage using LLM-Judge flags.
    
    Returns:
        full_score (int): % of queries with 'FULL' status.
        partial_score (int): % of queries with 'PARTIAL' status.
        effective_score (int): % of queries with 'FULL' OR 'PARTIAL' status.
        gap_logs (list): List of queries with 'NONE' or 'RELATED' status (Content Gaps).
    """
    gap_logs = []
    count_full = 0
    count_partial = 0
    total = len(query_logs)

    if total == 0:
        return 0, 0, 0, []

    for log in query_logs:
        # Determine the standardized status for this log
        status = 'NONE' # Default
        
        # 1. Try New Column (LLM Judge)
        db_status = getattr(log, 'relevance_status', 'UNKNOWN')
        
        if db_status in ['FULL', 'PARTIAL', 'RELATED', 'NONE']:
            status = db_status
        
        # 2. Legacy Fallback (Vector Scores) if status is UNKNOWN
        else:
            try:
                sources = log.sources
                # Handle case where sources might be stored as a string
                if isinstance(sources, str):
                    sources = json.loads(sources)
                if not isinstance(sources, list):
                    sources = []
            except Exception as e:
                logger.warning(f"Failed to parse sources for log {log.id}: {e}")
                sources = []

            scores = [float(s.get("similarity_score", 0.0)) for s in sources if isinstance(s, dict)]
            
            # Map legacy scores to new statuses
            if any(s >= strong_threshold for s in scores):
                status = 'FULL'
            elif len([s for s in scores if s >= weak_threshold]) >= 2:
                status = 'PARTIAL'
            else:
                status = 'NONE'

        # 3. Aggregate Stats
        if status == 'FULL':
            count_full += 1
        elif status == 'PARTIAL':
            count_partial += 1
        
        # 4. Identify Content Gaps
        # NONE: completely missing. 
        # RELATED: topic exists, but specific answer is missing (High value gap).
        if status in ['NONE', 'RELATED']:
            gap_logs.append(log)

    # Calculate percentages
    if total > 0:
        full_score = int((count_full / total) * 100)
        partial_score = int((count_partial / total) * 100)
        effective_score = full_score + partial_score # Simple addition is safer than re-dividing
    else:
        full_score = 0
        partial_score = 0
        effective_score = 0
    
    logger.info(f"Gap Analysis: Full={full_score}%, Partial={partial_score}%, Effective={effective_score}%. Found {len(gap_logs)} gaps.")
    
    return full_score, partial_score, effective_score, gap_logs

def cluster_and_summarize_gaps(gap_logs, n_clusters=5):
    """
    Takes the queries identified as GAPS, clusters them, and names the missing topics using LLM.
    """
    if not gap_logs:
        return []

    try:
        # 1. Extract Embeddings
        # Ensure we filter out logs that might be missing embeddings
        valid_logs = [log for log in gap_logs if log.query_embedding and len(log.query_embedding) > 0]
        
        if not valid_logs:
            return []

        embeddings = [np.array(log.query_embedding) for log in valid_logs]
        
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
            clustered_texts[label].append(valid_logs[idx].query_text)

        # 4. Generate Summaries with LLM
        llm = LLM(llm_model=LLM_MODEL, max_messages=0)
        topic_summaries = []

        for label, texts in clustered_texts.items():
            # Send top 10 examples to LLM
            examples = texts[:10]
            
            prompt = f"""
            The following are user questions where our system found NO satisfactory answer (Gaps):
            
            {examples}
            
            Analyze them and identify the **Missing Documentation Topic**.
            Note: If the topic is purely "Chit-Chat" (e.g., greetings, jokes), ignore them.
            
            Return ONLY a short, 3-5 word label (e.g., "Project X Deployment", "Python Basics", "HR Leave Policy").
            """
            
            try:
                response = llm.invoke(prompt)
                topic = response.get("content", "").strip().replace('"', '').replace("'", "")
                
                # Filter out pure noise if LLM identifies it
                if "chit-chat" not in topic.lower():
                    topic_summaries.append({
                        "topic": topic,
                        "gap_count": len(texts), 
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
            "full_coverage_score": 0,
            "partial_coverage_score": 0,
            "effective_coverage_score": 0,
            "top_learning_gaps": [], 
            "message": "Insufficient data (need 5+ queries)"
        }

    # 2. Analyze Scores 
    full_score, partial_score, effective_score, gap_logs = identify_gaps_from_logs(
        logs, 
        strong_threshold=0.65, 
        weak_threshold=0.45
    )

    # 3. Cluster & Label Gaps
    gap_topics = cluster_and_summarize_gaps(gap_logs)

    # Return structure matching frontend expectations
    return {
        "full_coverage_score": full_score,       # Strict (Answered completely)
        "partial_coverage_score": partial_score, # Partial (Helpful but incomplete)
        "effective_coverage_score": effective_score, # Total Helpful (Full + Partial)
        "top_learning_gaps": [t['topic'] for t in gap_topics],
        "details": gap_topics 
    }