from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from pineconePipeline import PineconePipeline
from dotenv import load_dotenv
import os
import logging
from blobStorage import download_blob_to_local
from database import get_db
from sqlalchemy.orm import Session
from llmModels import LLM

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    doc_id: str
    file_type: str
    blob_url: str
    chunk_count: Optional[int] = None
    status: str = "pending"

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline that combines:
    1. Document processing and chunking
    2. Vector embeddings generation and storage (Pinecone)
    3. LLM-based query answering (TogetherAI)
    """

    def __init__(self, index_name: str):
        self.index_name = index_name
        self.pinecone_pipeline = PineconePipeline(index_name=index_name)
        self.llm = LLM()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        logger.info(f"Initialized RAGPipeline with index: {index_name}")

    def _load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Helper method to load document based on file type"""
        if file_type == "pdf":
            return PyMuPDFLoader(file_path).load()
        elif file_type == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return [Document(page_content=f.read())]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def process_and_index_document(self, blob_url: str, file_type: str, doc_id: str) -> DocumentMetadata:
        """
        Process a document from Azure Blob Storage and index it in Pinecone.
        Returns metadata about the processing results.
        """
        metadata = DocumentMetadata(doc_id=doc_id, file_type=file_type, blob_url=blob_url)
        file_path = None

        try:
            logger.info(f"Starting processing of document {doc_id}")
            
            # Download and load document
            file_path = download_blob_to_local(blob_url)
            documents = self._load_document(file_path, file_type)
            
            # Split into chunks
            chunked_docs = self.text_splitter.split_documents(documents)
            metadata.chunk_count = len(chunked_docs)
            logger.info(f"Split document into {len(chunked_docs)} chunks")

            # Generate embeddings and store in Pinecone
            self.pinecone_pipeline.create_vector_store(chunked_docs)
            
            metadata.status = "completed"
            logger.info(f"Successfully processed document {doc_id}")

            return metadata

        except Exception as e:
            metadata.status = "failed"
            logger.error(f"Error processing document {doc_id}: {str(e)}", exc_info=True)
            raise

        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")

    def query(self, 
             query_text: str, 
             top_k: int = 3,
             temperature: float = 0.1) -> Dict[str, Any]:
        """
        Perform RAG query with detailed response including sources.
        """
        try:
            if not self.pinecone_pipeline.vector_store:
                raise RuntimeError("Vector store not initialized. Index documents first.")

            # Retrieve relevant documents
            retrieved_docs = self.pinecone_pipeline.vector_store.similarity_search(
                query_text, 
                k=top_k
            )

            # Format context and sources
            context_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = [doc.metadata for doc in retrieved_docs]

            # Create prompt
            prompt = (
                f"Based on the following context, please answer the question. "
                f"If the context doesn't contain relevant information, say so.\n\n"
                f"Context:\n{context_texts}\n\n"
                f"Question: {query_text}\n"
                f"Answer:"
            )

            # Get LLM response
            response = self.llm.invoke(prompt)

            return {
                "answer": response.get("content", "No response generated."),
                "sources": sources,
                "tokens_used": response.get("tokens", {}),
                "context_chunks": len(retrieved_docs)
            }

        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    rag = RAGPipeline(index_name="knowledgehub")
    
    # Index a document
    metadata = rag.process_and_index_document(
        blob_url="your_blob_url",
        file_type="pdf",
        doc_id="doc123"
    )
    
    # Query the system
    result = rag.query("What is software development?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")