from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
    Enum,
    func,
)
from sqlalchemy.orm import relationship
import enum
from database import Base


class DisplayModeEnum(str, enum.Enum):
    raw_content = "raw_content"
    generated_answer = "generated_answer"


class Space(Base):
    __tablename__ = "spaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=True)  # Future visibility toggle
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    documents = relationship("Document", back_populates="space")
    memberships = relationship("SpaceMembership", back_populates="space")


class SpaceMembership(Base):
    __tablename__ = "space_memberships"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    space_id = Column(Integer, ForeignKey("spaces.id"), nullable=False)
    role = Column(String, default="member")  # roles like member, admin
    joined_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="memberships")
    space = relationship("Space", back_populates="memberships")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    file_url = Column(String, nullable=False)  # Azure Blob Storage URL - raw document storage 
    file_type = Column(String)  # pdf, txt, etc.
    file_size = Column(Integer)  # File size in bytes
    uploader_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    space_id = Column(Integer, ForeignKey("spaces.id"), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text)  # Store error details if processing fails

    source_name = Column(String, nullable=True)  # e.g., "User Upload", "Semantic Scholar"
    display_mode = Column(Enum(DisplayModeEnum), default=DisplayModeEnum.generated_answer)

    # Relationships
    uploader = relationship("User", back_populates="documents")
    space = relationship("Space", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    reported_content = relationship("ReportedContent", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_order = Column(Integer, nullable=False)
    vector_id = Column(String, nullable=False, unique=True)  # Pinecone vector ID, mandatory to link vector 
    chunk_hash = Column(String)  # For deduplication
    embedding_model = Column(String)  # Embedding model used for this chunk
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")
    reported_content = relationship("ReportedContent", back_populates="chunk")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    tokens_used = Column(Integer)
    response_time = Column(Float)  # seconds
    context_chunks = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")
