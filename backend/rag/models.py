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
)
from sqlalchemy.orm import relationship
import enum

from database import Base  # imports your SQLAlchemy Base


class SpaceRoleEnum(str, enum.Enum):
    admin = "admin"
    member = "member"


class DisplayModeEnum(str, enum.Enum):
    raw_content = "raw_content"
    generated_answer = "generated_answer"


class CohortUserRoleEnum(str, enum.Enum):
    learner = "learner"
    maintainer = "maintainer"


class Cohort(Base):
    __tablename__ = "cohorts"

    id = Column(Integer, primary_key=True, index=True)
    # e.g. "GenC 2026"
    name = Column(String(100), nullable=False, unique=True)
    # e.g. "GENC_2026"
    code = Column(String(50), nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    spaces = relationship(
        "Space",
        back_populates="cohort",
        cascade="all, delete-orphan",
    )
    users = relationship(
        "CohortUser",
        back_populates="cohort",
        cascade="all, delete-orphan",
    )


class CohortUser(Base):
    """
    Links a user to a cohort with a cohort-local role.
    - Trainee: exactly one row pointing to their cohort, role=learner.
    - Maintainer: one row per cohort they mentor, role=maintainer.
    """
    __tablename__ = "cohort_users"

    id = Column(Integer, primary_key=True, index=True)
    cohort_id = Column(Integer, ForeignKey("cohorts.id"), nullable=False)
    user_id = Column(String(36), nullable=False, index=True)
    role = Column(
        Enum(CohortUserRoleEnum),
        nullable=False,
        default=CohortUserRoleEnum.learner,
    )

    cohort = relationship("Cohort", back_populates="users")


class Space(Base):
    __tablename__ = "spaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # NEW: link space to its cohort
    cohort_id = Column(Integer, ForeignKey("cohorts.id"), nullable=False)

    # Relationships with cascade
    cohort = relationship("Cohort", back_populates="spaces")

    documents = relationship(
        "Document",
        back_populates="space",
        cascade="all, delete-orphan",
    )

    memberships = relationship(
        "SpaceMembership",
        back_populates="space",
        cascade="all, delete-orphan",
    )


class SpaceMembership(Base):
    __tablename__ = "space_memberships"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False)  # UUID from external Auth service
    space_id = Column(Integer, ForeignKey("spaces.id"), nullable=False)
    role = Column(
        Enum(SpaceRoleEnum),
        nullable=False,
        default=SpaceRoleEnum.member.value,
    )
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
    display_mode = Column(
        Enum(DisplayModeEnum),
        default=DisplayModeEnum.generated_answer,
    )
    vector_doc_id = Column(String(200), nullable=True, index=True)

    # Relationships
    space = relationship("Space", back_populates="documents")
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


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
    query_embedding = Column(JSON, nullable=False)  # Vector for semantic matching
    response_text = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    tokens_used = Column(JSON)
    response_time = Column(Float)
    context_chunks = Column(Integer)
    space_id = Column(Integer, nullable=False, index=True)  # Important for filtering
    hit_count = Column(Integer, default=1)  # Track cache reuse
    created_at = Column(DateTime(timezone=True), server_default=func.now())
