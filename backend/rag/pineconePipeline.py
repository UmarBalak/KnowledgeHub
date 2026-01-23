import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.schema import Document
from pinecone import Pinecone  
from pinecone import PineconeException 

load_dotenv()

class PineconePipeline:
    def __init__(self, index_name: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load environment variables for Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL") or os.getenv("PINECONE_EMBEDDING_MODEL")
        
        if not self.pinecone_api_key or not self.embedding_model:
            raise ValueError("PINECONE_API_KEY and EMBEDDING_MODEL environment variables must be set.")
        
        self.embeddings = PineconeEmbeddings(model=self.embedding_model)
        self.vector_store = None
        self.pcIndex = Pinecone(api_key=self.pinecone_api_key)

        self.create_index(self.index_name)

    def prepare_documents(self, texts: list[str]) -> list[Document]:
        """Convert raw text list to LangChain Document objects and chunk them."""
        documents = [Document(page_content=text) for text in texts]
        splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunked_docs = splitter.split_documents(documents)
        return chunked_docs
    
    def create_vector_store(self, documents: list[Document]) -> None:
        """
        Create Pinecone vector store from preprocessed documents.
        This method generates the actual embeddings and stores them in Pinecone.
        """
        try:
            self.vector_store = PineconeVectorStore.from_documents(
                documents,
                index_name=self.index_name,
                embedding=self.embeddings
            )
            print(f"Vector store created with index: {self.index_name}")
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 1) -> list[Document]:
        """Perform similarity search in the vector store with given query."""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Run `create_vector_store` first.")
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def create_index(self, index_name: str):
        """Create Pinecone index if it doesn't exist."""
        # Create an index integrated with a hosted embedding model
        try:
            index_model = self.pcIndex.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": self.embedding_model,  # example Pinecone hosted model
                    "field_map": {"text": "chunk_text"},  # embed field named 'chunk_text'
                    "metric": "cosine"
                }
            )
            print(f"Index created: {index_name}")
        except PineconeException as e:
            if "already exists" in str(e):
                print(f"Index {index_name} already exists.")
            else:
                print(f"Pinecone API error: {e}")
                raise
        except Exception as e:
            print("Error creating Pinecone index:", e)
        
    
    def delete_index(self, index_name: str):
        """Delete a Pinecone index."""
        try:
            self.pcIndex.delete_index(name=index_name)
            print(f"Index {index_name} deleted successfully.")
        except Exception as e:
            print(f"Error deleting index {index_name}: {e}")

    def delete_vectors_by_metadata(self, filter_dict: dict):
        """
        Delete vectors matching the metadata filter.
        Example filter: {"doc_id": "123"}
        """
        try:
            # Get the index object directly to perform delete operations
            index = self.pcIndex.Index(self.index_name)
            
            # Delete by metadata filter
            index.delete(filter=filter_dict)
            print(f"Deleted vectors with filter: {filter_dict}")
            
        except Exception as e:
            print(f"Error deleting vectors from Pinecone: {e}")
            # We raise the error so the main API knows the deletion was incomplete
            raise e


if __name__ == "__main__":
    # Example usage
    texts = [
        "Hello world! This is a test document.",
        "LangChain is a framework for building applications with LLMs.",
        "Apples are a popular fruit known for their sweetness.",
        "python is a high-level programming language.",
        "cricket is a bat-and-ball game played between two teams of eleven players.",
    ]

    pinecone_rag = PineconePipeline("knowledgehub-main")
    chunked_docs = pinecone_rag.prepare_documents(texts)
    pinecone_rag.create_vector_store(chunked_docs)

    # query = "software development"
    # results = pinecone_rag.similarity_search(query, k=1)

    # if not results:
    #     print("No results found for the query.")
    # else:
    #     for i, doc in enumerate(results):
    #         print(f"Result {i + 1}: {doc.page_content}")
