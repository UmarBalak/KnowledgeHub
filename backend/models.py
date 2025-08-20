# ml_knowledgehub_backend/models.py
# (Content is identical to previous explanation)
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
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
    documents = relationship("Document", back_populates="uploader")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    text_content = Column(Text, nullable=False)
    uploader_id = Column(Integer, ForeignKey("users.id"))
    subject_id = Column(Integer, default=1)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    uploader = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_text = Column(Text, nullable=False)
    vector_id = Column(String, nullable=True, unique=True)
    chunk_order = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    document = relationship("Document", back_populates="chunks")

class ReportedContent(Base):
    __tablename__ = "reported_content"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    reported_by_user_id = Column(Integer, ForeignKey("users.id"))
    reason = Column(String, nullable=False)
    reported_at = Column(DateTime(timezone=True), server_default=func.now())
    document = relationship("Document")
    chunk = relationship("DocumentChunk")
    reporter = relationship("User")