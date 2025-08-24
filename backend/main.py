# main.py - FastAPI Application Entry Point
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
import os
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import uvicorn

from database import get_db, engine
from models import User, Document, DocumentChunk, ReportedContent, Base
from schemas import (
    UserCreate, UserResponse, Token, DocumentCreate, DocumentResponse,
    DocumentChunkResponse, QueryRequest, QueryResponse, ReportContentRequest
)
from auth import create_access_token, verify_password, get_password_hash, verify_token
from ragPipeline import RAGPipeline
from blobStorage import upload_blob
import uuid

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="ML Knowledge Hub RAG Backend",
    description="A comprehensive RAG application backend with document processing, vector search, and AI-powered Q&A",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(index_name="knowledgehub-main")

# Security
security = HTTPBearer()

# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# Authentication Routes
@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at
    )


@app.post("/auth/login", response_model=Token)
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Authenticate user and return JWT token"""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token({"user_id": user.id})
    return Token(access_token=access_token, token_type="bearer")


# Document Management Routes
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    subject_id: int = Form(default=1),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_types = ["application/pdf", "text/plain"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Read file content
        file_content = await file.read()
        
        # Generate unique filename
        file_extension = "pdf" if file.content_type == "application/pdf" else "txt"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Upload to Azure Blob Storage
        upload_success = upload_blob(file_content, unique_filename)
        if not upload_success:
            raise HTTPException(status_code=500, detail="Failed to upload file to storage")
        
        # Extract text content for database storage
        if file.content_type == "application/pdf":
            # For PDF, we'll store a placeholder - actual content will be extracted during RAG processing
            text_content = f"PDF document: {title}"
        else:
            text_content = file_content.decode('utf-8')
        
        # Create document record
        document = Document(
            title=title,
            text_content=text_content,
            uploader_id=current_user.id,
            subject_id=subject_id,
            file_url=f"{os.getenv('BLOB_SAS_URL')}/{os.getenv('BLOB_CONTAINER_NAME')}/{unique_filename}",
            file_type=file_extension
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Process document with RAG pipeline in background
        try:
            metadata = rag_pipeline.process_and_index_document(
                blob_url=document.file_url,
                file_type=file_extension,
                doc_id=str(document.id)
            )
            
            # Update document status
            document.processing_status = metadata.status
            document.chunk_count = metadata.chunk_count
            db.commit()
            
        except Exception as e:
            # Mark as failed but don't fail the upload
            document.processing_status = "failed"
            document.error_message = str(e)
            db.commit()
        
        return DocumentResponse(
            id=document.id,
            title=document.title,
            uploader_id=document.uploader_id,
            subject_id=document.subject_id,
            uploaded_at=document.uploaded_at,
            processing_status=document.processing_status,
            chunk_count=document.chunk_count,
            file_type=document.file_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    subject_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's documents with pagination"""
    query = db.query(Document).filter(Document.uploader_id == current_user.id)
    
    if subject_id:
        query = query.filter(Document.subject_id == subject_id)
    
    documents = query.order_by(desc(Document.uploaded_at)).offset(skip).limit(limit).all()
    
    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            uploader_id=doc.uploader_id,
            subject_id=doc.subject_id,
            uploaded_at=doc.uploaded_at,
            processing_status=doc.processing_status,
            chunk_count=doc.chunk_count,
            file_type=doc.file_type
        ) for doc in documents
    ]


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific document"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.uploader_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        uploader_id=document.uploader_id,
        subject_id=document.subject_id,
        uploaded_at=document.uploaded_at,
        processing_status=document.processing_status,
        chunk_count=document.chunk_count,
        file_type=document.file_type
    )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document and its chunks"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.uploader_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete chunks first (foreign key constraint)
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    
    # Delete document
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}


# RAG Query Routes
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """Query documents using RAG pipeline"""
    try:
        result = rag_pipeline.query(
            query_text=query_request.query,
            top_k=query_request.top_k,
            temperature=query_request.temperature
        )
        
        return QueryResponse(
            query=query_request.query,
            answer=result["answer"],
            sources=result.get("sources", []),
            context_chunks=result["context_chunks"],
            tokens_used=result.get("tokens_used", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Content Moderation Routes
@app.post("/report-content")
async def report_content(
    report_request: ReportContentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Report inappropriate content"""
    reported_content = ReportedContent(
        document_id=report_request.document_id,
        chunk_id=report_request.chunk_id,
        reported_by_user_id=current_user.id,
        reason=report_request.reason
    )
    db.add(reported_content)
    db.commit()
    
    return {"message": "Content reported successfully"}


# Health Check and System Info
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }


@app.get("/system/info")
async def system_info(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get system information and user stats"""
    user_doc_count = db.query(Document).filter(Document.uploader_id == current_user.id).count()
    
    return {
        "user_documents": user_doc_count,
        "rag_index": rag_pipeline.index_name,
        "supported_formats": ["PDF", "TXT"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
