# schemas.py - Pydantic models for API
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class DocumentCreate(BaseModel):
    title: str
    subject_id: int = 1

class DocumentResponse(BaseModel):
    id: int
    title: str
    uploader_id: int
    subject_id: int
    uploaded_at: datetime
    processing_status: Optional[str] = "pending"
    chunk_count: Optional[int] = None
    file_type: Optional[str] = None
    
    class Config:
        from_attributes = True

class DocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_text: str
    chunk_order: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    temperature: float = 0.1

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    context_chunks: int
    tokens_used: Dict[str, Any]

class ReportContentRequest(BaseModel):
    document_id: Optional[int] = None
    chunk_id: Optional[int] = None
    reason: str
