from llama_parse import LlamaParse
import pymupdf.layout # PyMuPDF Layout must be imported as shown and before importing PyMuPDF4LLM to activate PyMuPDF’s layout feature and make it available to PyMuPDF4LLM.
import pymupdf4llm
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

def parse_pdf_unstructured(file_path: str) -> list[Document]:
    '''
    medium accuracy but slower, good for unstructured pdf.
    return text
    '''
    logger.info("Using unstructured")

    loader = UnstructuredPDFLoader(
        file_path,
        mode="elements",
        strategy="fast"   # disables OCR
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

    return docs

def llama_parser(file_path: str):
    '''
    Most accurate but slower and external api
    return md
    '''
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
    return langchain_docs


def parse_pdf4llm(file_path: str):
    '''
    Faster but low accuracy
    return md
    '''

    logger.info("Using pdf4llm")
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

    return docs


def load_document(file_path: str, file_type: str, mode: str = "auto") -> List[Document]:
    """Helper method to load document based on file type"""
    if file_type == "pdf":

        if mode == "llama":
            data = llama_parser(file_path)
            return data

        if mode == "unstructured":
            docs = parse_pdf_unstructured(file_path)
            return docs
        
        if mode == "balanced":
            data = parse_pdf4llm(file_path)
            return data

        if mode == "fast":
            logger.info("Using pymupdfloader")
            # PyMuPDF (Fast, Lower Quality)
            return PyMuPDFLoader(file_path).load()
        
        if mode == "auto":
            docs = parse_pdf_unstructured(file_path)
            return docs

            
    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return [Document(page_content=f.read())]
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    

if __name__ == "__main__":
    result = load_document("E:/KnowledgeHub/backend/rag/test/UmarBalakResume7.pdf", "pdf", unstruc=True)

    for i in result:
        print(i.page_content)

