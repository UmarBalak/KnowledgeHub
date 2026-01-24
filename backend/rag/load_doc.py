from llama_parse import LlamaParse
import pymupdf4llm
import pymupdf
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from typing import List
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def parse_pdf_unstructured(file_path: str) -> List[Document]:
    """
    Medium accuracy but slower, good for unstructured pdf.
    Returns text elements merged by page.
    """
    logger.info("Using unstructured parser")
    loader = UnstructuredPDFLoader(
        file_path,
        mode="elements",
        strategy="fast",  # disables OCR
        chunking_strategy="by_title",    
        max_characters=4000,             # Max chunk size
        new_after_n_chars=3800,          # Soft limit
        combine_text_under_n_chars=2000, # Avoid tiny chunks
    )
    
    # Loader returns a list of semantically chunked Documents
    docs = loader.load()

    # Normalize metadata
    for doc in docs:
        doc.metadata["parser"] = "unstructured"
        doc.metadata["source"] = file_path

    logger.info(f"Generated {len(docs)} semantic chunks with unstructured")
    return docs

def llama_parser(file_path: str) -> List[Document]:
    """
    Most accurate but slower and uses external API.
    Returns markdown format.
    """
    logger.info("Using llama_parser")

    # Initialize parser
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",  # Markdown is best for RAG context
        verbose=True
    )

    # 1. Parse the file (returns LlamaIndex documents)
    llama_docs = parser.load_data(file_path)

    full_text = "\n\n".join([doc.text for doc in llama_docs])

    doc = [
        Document(
            page_content=full_text,
            metadata={
                "source": file_path,
                "parser": "llama-parser",
                "strategy": "continuous_markdown"
            }
        )
    ]

    logger.info(f"Parsed document with llama_parser")
    return doc

def parse_pdf4llm(file_path: str) -> List[Document]:
    """
    Faster but lower accuracy.
    Returns markdown format.
    """
    logger.info("Using pdf4llm parser")

    content = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(content)

    clean_text = md_text.replace("\n\n---\n\n", "\n\n")

    # Safety check for JSON escaping issues common in some environments
    if "\\n" in clean_text:
        try:
            import json
            clean_text = json.loads(f'"{clean_text}"')
        except:
            pass # If json load fails, use raw text

    # 3. Return as Single Document (or very large sections)
    # This is "RAG Ready" because it preserves cross-page context.
    doc = [
        Document(
            page_content=clean_text,
            metadata={
                "source": file_path,
                "parser": "pymupdf4llm",
                "strategy": "continuous_markdown"
            }
        )
    ]

    logger.info(f"Parsed document with pdf4llm")
    return doc

def load_document(file_path: str, file_type: str, mode: str = "auto") -> List[Document]:
    """
    Helper method to load document based on file type and parsing mode.

    Args:
        file_path: Path to the document file
        file_type: Type of file ('pdf' or 'txt')
        mode: Parsing mode ('auto', 'llama', 'unstructured', 'balanced', 'fast')

    Returns:
        List of Document objects with page_content and metadata
    """
    if file_type == "pdf":
        if mode == "llama":
            logger.info(f"Loading PDF with llama parser: {file_path}")
            return llama_parser(file_path)

        elif mode == "unstructured":
            logger.info(f"Loading PDF with unstructured parser: {file_path}")
            return parse_pdf_unstructured(file_path)

        elif mode == "pdf4llm":
            logger.info(f"Loading PDF with pdf4llm (balanced): {file_path}")
            return parse_pdf4llm(file_path)

        elif mode == "auto":
            logger.info(f"Loading PDF with PyMuPDFLoader (fast): {file_path}")
            # PyMuPDF (Fast, Lower Quality)
            raw_docs = PyMuPDFLoader(file_path).load()

            merged_text = "\n\n".join([d.page_content for d in raw_docs])
            
            doc = [
                Document(
                    page_content=merged_text, 
                    metadata={
                        "source": file_path, 
                        "parser": "pymupdf_fast",
                        "strategy": "merged_text"
                    }
                )
            ] 
            return doc

        else:
            raise ValueError(f"Unknown parse mode for PDF: {mode}")

    elif file_type == "txt":
        logger.info(f"Loading TXT file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return [Document(
            page_content=content,
            metadata={
                "source": file_path,
                "page": 1,
                "parser": "text_loader"
            }
        )]

    else:
        raise ValueError(f"Unsupported file type: {file_type}")

if __name__ == "__main__":
    # Test with a sample PDF
    test_file = "E:/KnowledgeHub/backend/rag/test/UmarBalakResume7.pdf"

    if os.path.exists(test_file):
        result = load_document(test_file, "pdf", mode="unstructured")

        print(f"\nLoaded {len(result)} pages")
        for i, doc in enumerate(result):
            print(f"\n--- Page {i+1} ---")
            print(f"Content length: {len(doc.page_content)} chars")
            print(f"Metadata: {doc.metadata}")
            print(f"First 200 chars: {doc.page_content[:200]}...")
    else:
        print(f"Test file not found: {test_file}")