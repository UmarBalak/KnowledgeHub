from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    documents = relationship("Document", back_populates="uploader")
    reported_content = relationship("ReportedContent", back_populates="reporter")

class Subject(Base):
    __tablename__ = "subjects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    documents = relationship("Document", back_populates="subject")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    text_content = Column(Text, nullable=False)
    file_url = Column(String)  # Azure Blob Storage URL
    file_type = Column(String)  # pdf, txt, etc.
    file_size = Column(Integer)  # File size in bytes
    uploader_id = Column(Integer, ForeignKey("users.id"))
    subject_id = Column(Integer, ForeignKey("subjects.id"), default=1)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    chunk_count = Column(Integer)
    error_message = Column(Text)  # Store error details if processing fails
    
    # Relationships
    uploader = relationship("User", back_populates="documents")
    subject = relationship("Subject", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    reported_content = relationship("ReportedContent", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_text = Column(Text, nullable=False)
    vector_id = Column(String, nullable=True, unique=True)  # Pinecone vector ID
    chunk_order = Column(Integer, nullable=False)
    chunk_hash = Column(String)  # For deduplication
    embedding_model = Column(String)  # Track which model was used
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    reported_content = relationship("ReportedContent", back_populates="chunk")

class ReportedContent(Base):
    __tablename__ = "reported_content"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    reported_by_user_id = Column(Integer, ForeignKey("users.id"))
    reason = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, reviewed, resolved
    reviewer_notes = Column(Text)
    reported_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True))
    
    # Relationships
    document = relationship("Document", back_populates="reported_content")
    chunk = relationship("DocumentChunk", back_populates="reported_content")
    reporter = relationship("User", back_populates="reported_content")

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    tokens_used = Column(Integer)
    response_time = Column(Float)  # Response time in seconds
    context_chunks = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")