from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
    Enum,
    Float,
    func,
    JSON,
    TypeDecorator
)
from sqlalchemy.orm import relationship
import enum
import os
from cryptography.fernet import Fernet
from database import Base  # imports your SQLAlchemy Base

# --- Encryption Setup ---
# Loads key from environment. If missing, it defaults to None (Encryption won't work without it)
CHAT_ENCRYPTION_KEY = os.getenv("CHAT_ENCRYPTION_KEY")
cipher_suite = Fernet(CHAT_ENCRYPTION_KEY) if CHAT_ENCRYPTION_KEY else None

class EncryptedString(TypeDecorator):
    """
    SQLAlchemy Type that encrypts data on the way IN to the DB 
    and decrypts it on the way OUT.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if not value:
            return value
        if not cipher_suite:
            raise ValueError("CHAT_ENCRYPTION_KEY is missing in environment variables!")
        return cipher_suite.encrypt(value.encode('utf-8')).decode('utf-8')

    def process_result_value(self, value, dialect):
        if not value:
            return value
        if not cipher_suite:
            return value
        return cipher_suite.decrypt(value.encode('utf-8')).decode('utf-8')

# --- Enums ---

class SpaceRoleEnum(str, enum.Enum):
    admin = "admin"
    member = "member"

class DisplayModeEnum(str, enum.Enum):
    raw_content = "raw_content"
    generated_answer = "generated_answer"

# --- Models ---

class Space(Base):
    __tablename__ = "spaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships with cascade
    documents = relationship(
        "Document",
        back_populates="space",
        cascade="all, delete-orphan"
    )
    memberships = relationship(
        "SpaceMembership",
        back_populates="space",
        cascade="all, delete-orphan"
    )
    # New Chat Relationship
    chat_messages = relationship(
        "ChatMessage",
        back_populates="space",
        cascade="all, delete-orphan"
    )


class SpaceMembership(Base):  
    __tablename__ = "space_memberships"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False)  # UUID from external Auth service
    space_id = Column(Integer, ForeignKey("spaces.id"), nullable=False)
    role = Column(Enum(SpaceRoleEnum), nullable=False, default=SpaceRoleEnum.member.value)
    joined_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    space = relationship("Space", back_populates="memberships")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), index=True, nullable=False)
    file_url = Column(String(500), nullable=False)  # Azure Blob Storage URL
    file_type = Column(String(50))  # pdf, txt, etc.
    file_size = Column(Integer)  # File size in bytes
    uploader_id = Column(String(36), nullable=False)  # UUID from external Auth service
    space_id = Column(Integer, ForeignKey("spaces.id"), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text)  # Store error details if processing fails

    source_name = Column(String(50), nullable=True)  # e.g., "User Upload", "Semantic Scholar"
    display_mode = Column(Enum(DisplayModeEnum), default=DisplayModeEnum.generated_answer)

    vector_doc_id = Column(String(200), nullable=True, index=True)
    
    # Relationships
    space = relationship("Space", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_order = Column(Integer, nullable=False)
    vector_id = Column(String(100), nullable=False, unique=True)  # Pinecone vector ID
    embedding_model = Column(String(50))  # Embedding model used
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False)
    query_hash = Column(String(64), index=True, nullable=False)  # SHA-256 hash
    query_text = Column(Text, nullable=False)  
    query_embedding = Column(JSON, nullable=False)  # Vector for semantic matching
    response_text = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    tokens_used = Column(JSON)
    response_time = Column(Float)
    context_chunks = Column(Integer)
    space_id = Column(Integer, nullable=False, index=True)  # Important for filtering
    hit_count = Column(Integer, default=1)  # Track cache reuse
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# --- NEW CHAT MODEL ---
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    space_id = Column(Integer, ForeignKey("spaces.id"), nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True) # UUID from Auth
    # This content is encrypted in the DB, but accessible as plain string in Python
    content = Column(EncryptedString, nullable=False) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    space = relationship("Space", back_populates="chat_messages")