import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.schema import Document

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Ensure PINECONE_API_KEY is set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL")

embeddings = PineconeEmbeddings(model=EMBEDDING_MODEL)

texts = [
    "Apples are a popular fruit known for their sweetness.",
    "python is a high-level programming language.",
    "cricket is a bat-and-ball game played between two teams of eleven players.",
]

# Convert texts to LangChain Document objects
documents = [Document(page_content=text) for text in texts]

# Split documents into chunks for better embedding (optional)
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(documents)

# Connect to Pinecone vector store
try:
    vector_store = PineconeVectorStore.from_documents(
        chunked_docs,
        index_name="knowledgehub",
        embedding=embeddings
    )
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    exit(1)

# Query with similarity search
query = "software development"
results = vector_store.similarity_search(query, k=1)

if not results:
    print("No results found for the query.")
# Print matching results
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")