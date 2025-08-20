## TO CREATE AND DELETE A PINECONE INDEX
# This script demonstrates how to create and delete a Pinecone index using the Pinecone client.

from pinecone import Pinecone   

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Ensure PINECONE_API_KEY is set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBED_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL")

# Initialize Pinecone client with your API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index integrated with a hosted embedding model
index_model = pc.create_index_for_model(
    name="test2",
    cloud="aws",
    region="us-east-1",
    embed={
        "model": EMBED_MODEL,  # example Pinecone hosted model
        "field_map": {"text": "chunk_text"},  # embed field named 'chunk_text'
        "metric": "cosine"
    }
)
print(f"Index created: {index_model.name} at {index_model.host}")

# To delete index
pc.delete_index(name="test1")
print("Index deleted")
