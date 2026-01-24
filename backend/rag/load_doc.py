from llama_parse import LlamaParse
import pymupdf4llm
import pymupdf
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
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
        strategy="fast"  # disables OCR
    )
    elements = loader.load()

    # Merge elements by page to match other parsers
    page_map = {}
    for el in elements:
        page = el.metadata.get("page_number", 1)
        page_map.setdefault(page, []).append(el.page_content)

    docs = []
    for page, texts in page_map.items():
        docs.append(
            Document(
                page_content="\n\n".join(texts),
                metadata={
                    "source": file_path,
                    "page": page,
                    "parser": "unstructured"
                }
            )
        )

    logger.info(f"Parsed {len(docs)} pages with unstructured")
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

    # 2. Convert LlamaIndex docs -> LangChain docs
    langchain_docs = []
    for i, doc in enumerate(llama_docs):
        langchain_docs.append(
            Document(
                page_content=doc.text,
                metadata=doc.metadata or {
                    "source": file_path,
                    "page": i + 1,
                    "parser": "llama-parser"
                }
            )
        )

    logger.info(f"Parsed {len(langchain_docs)} pages with llama_parser")
    return langchain_docs

def parse_pdf4llm(file_path: str) -> List[Document]:
    """
    Faster but lower accuracy.
    Returns markdown format.
    """
    logger.info("Using pdf4llm parser")

    content = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(content)

    # Safety: ensure not double-escaped
    if "\\n" in md_text:
        import json
        md_text = json.loads(f'"{md_text}"')

    docs = []
    pages = md_text.split("\n\n---\n\n")  # pymupdf4llm page delimiter

    for i, page_md in enumerate(pages):
        if not page_md.strip():
            continue

        docs.append(
            Document(
                page_content=page_md.strip(),
                metadata={
                    "source": file_path,
                    "page": i + 1,
                    "parser": "pymupdf4llm"
                }
            )
        )

    logger.info(f"Parsed {len(docs)} pages with pdf4llm")
    return docs

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

        elif mode == "balanced":
            logger.info(f"Loading PDF with pdf4llm (balanced): {file_path}")
            return parse_pdf4llm(file_path)

        elif mode == "fast":
            logger.info(f"Loading PDF with PyMuPDFLoader (fast): {file_path}")
            # PyMuPDF (Fast, Lower Quality)
            docs = PyMuPDFLoader(file_path).load()
            # Add parser info to metadata
            for doc in docs:
                doc.metadata["parser"] = "pymupdf"
            return docs

        elif mode == "auto":
            logger.info(f"Loading PDF with auto mode (defaulting to unstructured): {file_path}")
            return parse_pdf_unstructured(file_path)

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