from ast import Dict

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Query,
    Path,
    Request,
)
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc
from starlette.status import HTTP_401_UNAUTHORIZED
import hashlib
from typing import List, Optional, Any, Dict

import logging
import os
from datetime import datetime

from starlette.responses import JSONResponse, Response

from database import get_db, Base, engine, SessionLocal
from models import (
    Space,
    SpaceMembership,
    Document,
    DocumentChunk,
    QueryLog,
    SpaceRoleEnum,
    DisplayModeEnum,
    Cohort,
    CohortUser,
    CohortUserRoleEnum,
)
from ragPipeline import RAGPipeline
from blobStorage import upload_blob, delete_blob, extract_filename_from_url
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()


def generate_query_hash(query_text: str, space_id: int) -> str:
    """Generate deterministic SHA-256 hash of normalized query"""
    normalized = " ".join(query_text.lower().strip().split())
    content = f"{normalized}:{space_id}"
    return hashlib.sha256(content.encode()).hexdigest()


# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
FRONTEND_URL = os.getenv("FRONTEND_URL")

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
)

rag_pipeline = RAGPipeline(index_name=os.getenv("PINECONE_INDEX_NAME"))

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------


class SpaceStats(BaseModel):
    total_docs: int
    user_docs: int


class UserContext(BaseModel):
    google_id: str
    name: Optional[str] = None
    role: Optional[str] = None  # "member" or "maintainer"


class UserOut(BaseModel):
    id: str  # UUID string from Auth service
    name: Optional[str]


class SpaceOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_public: bool
    created_at: datetime

    class Config:
        from_attributes = True


class SpaceCreateIn(BaseModel):
    name: str = Field(..., max_length=50)
    description: Optional[str]


class DocumentOut(BaseModel):
    id: int
    title: str
    file_url: str
    file_type: Optional[str]
    file_size: Optional[int]
    uploader_id: str  # UUID user id string
    space_id: int
    uploaded_at: datetime
    processing_status: str
    chunk_count: int
    error_message: Optional[str]
    source_name: Optional[str]
    display_mode: DisplayModeEnum

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    temperature: float = 0.0
    space_id: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    context_chunks: int
    tokens_used: Dict[str, Any]


class DocumentResponse(BaseModel):
    id: int
    title: str
    uploader_id: str
    space_id: int
    uploaded_at: datetime
    processing_status: str
    chunk_count: int
    file_type: Optional[str]

    class Config:
        from_attributes = True


class CohortOut(BaseModel):
    id: int
    name: str
    code: str
    is_active: bool

    class Config:
        from_attributes = True


class CohortCreate(BaseModel):
    name: str
    code: str


class CohortUserCreate(BaseModel):
    user_id: str
    role: CohortUserRoleEnum = CohortUserRoleEnum.learner


user_llm_cache: Dict[str, Any] = {}


def get_user_llm(user_id: str):
    if user_id not in user_llm_cache:
        from llmModels import LLM

        user_llm_cache[user_id] = LLM(gpt5=True)
    return user_llm_cache[user_id]


# -----------------------------------------------------------------------------
# Auth + Cohort resolution
# -----------------------------------------------------------------------------


def get_current_user(request: Request) -> UserContext:
    """
    Dependency to extract user info from headers (set by frontend AuthContext).
    """
    user_id = request.headers.get("X-User-Id")
    role = request.headers.get("X-User-Role", "member")

    if not user_id:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing user id")

    return UserContext(google_id=user_id, role=role)


def require_maintainer(user: UserContext):
    if user.role != "maintainer":
        raise HTTPException(status_code=403, detail="Insufficient privileges")


def get_current_cohort(
    request: Request,
    db: Session = Depends(get_db),
    user: UserContext = Depends(get_current_user),
) -> int:
    """
    - Trainee (non-maintainer): must be in exactly one active cohort in CohortUser.
      Backend picks that cohort, no UI.
    - Maintainer: can be in multiple cohorts. Frontend sends X-Cohort-Id.
      Backend validates that mapping exists in CohortUser.
    """
    header_value = request.headers.get("X-Cohort-Id")

    # Maintainer flow: must provide X-Cohort-Id and be mapped to that cohort as maintainer
    if user.role == "maintainer":
        if not header_value:
            raise HTTPException(
                status_code=400,
                detail="Missing X-Cohort-Id header for maintainer",
            )
        try:
            cohort_id = int(header_value)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid X-Cohort-Id header")

        cohort_user = (
            db.query(CohortUser)
            .join(Cohort, Cohort.id == CohortUser.cohort_id)
            .filter(
                CohortUser.user_id == user.google_id,
                CohortUser.cohort_id == cohort_id,
                Cohort.is_active.is_(True),
            )
            .first()
        )
        if not cohort_user:
            raise HTTPException(
                status_code=403,
                detail="Maintainer is not mapped to this cohort",
            )
        return cohort_id

    # Trainee flow: infer exactly one active cohort
    mappings = (
        db.query(CohortUser)
        .join(Cohort, Cohort.id == CohortUser.cohort_id)
        .filter(
            CohortUser.user_id == user.google_id,
            Cohort.is_active.is_(True),
        )
        .all()
    )
    if len(mappings) == 0:
        raise HTTPException(
            status_code=403,
            detail="User is not assigned to any active cohort",
        )
    if len(mappings) > 1:
        # Misconfiguration: a non-maintainer attached to multiple cohorts
        raise HTTPException(
            status_code=500,
            detail="User mapped to multiple cohorts; expected exactly one",
        )
    return mappings[0].cohort_id


def is_user_member(
    db: Session,
    user_id: str,
    space_id: int,
    cohort_id: int,
) -> bool:
    """
    Check membership of a user in a space *within a given cohort*.
    """
    membership = (
        db.query(SpaceMembership)
        .join(Space, Space.id == SpaceMembership.space_id)
        .filter(
            SpaceMembership.user_id == user_id,
            SpaceMembership.space_id == space_id,
            Space.cohort_id == cohort_id,
        )
        .first()
    )
    return membership is not None


def get_space_in_cohort_or_404(
    db: Session,
    space_id: int,
    cohort_id: int,
) -> Space:
    space = (
        db.query(Space)
        .filter(Space.id == space_id, Space.cohort_id == cohort_id)
        .first()
    )
    if not space:
        raise HTTPException(status_code=404, detail="Space not found in this cohort")
    return space


# -----------------------------------------------------------------------------
# Background document processing
# -----------------------------------------------------------------------------


def process_document_background(
    doc_id: int,
    space_id: int,
    file_type: str,
    parse_mode: str = "fast",
):
    """
    Background processing task that stores the enhanced document ID.
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            return

        # Delete existing chunks before re-processing
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id
        ).delete()
        db.commit()

        # Process document and get enhanced doc ID
        metadata, chunked_docs, embedding_model, enhanced_doc_id = (
            rag_pipeline.process_and_index_document(
                blob_url=document.file_url,
                file_type=file_type,
                doc_id=str(document.id),
                space_id=space_id,
                parse_mode=parse_mode,
            )
        )

        # Update Document with status AND enhanced doc ID
        document.processing_status = metadata.status
        document.chunk_count = metadata.chunk_count
        document.vector_doc_id = enhanced_doc_id

        # Save Chunks
        doc_chunks: List[DocumentChunk] = []
        for chunk in chunked_docs:
            meta = chunk.metadata
            doc_chunk = DocumentChunk(
                document_id=document.id,
                chunk_order=meta["chunk_index"],
                vector_id=meta.get("chunk_id"),
                embedding_model=embedding_model,
                created_at=meta.get("created_at"),
            )
            doc_chunks.append(doc_chunk)

        db.add_all(doc_chunks)
        db.commit()

        logging.info(
            f"Document {doc_id} processed successfully: {len(doc_chunks)} chunks stored"
        )
        logging.info(f"Enhanced doc ID: {enhanced_doc_id}")

    except Exception as e:
        logging.error(f"Background processing failed: {e}", exc_info=True)
        db.rollback()
        try:
            document = db.query(Document).filter(Document.id == doc_id).first()
            if document:
                document.processing_status = "failed"
                document.error_message = str(e)
                db.commit()
        except Exception as update_error:
            logging.error(f"Failed to update error status: {update_error}")
            db.rollback()
    finally:
        db.close()


# -----------------------------------------------------------------------------
# System + Cohort endpoints
# -----------------------------------------------------------------------------


@app.get("/", response_model=dict)
async def root():
    return {"message": "RAG Service is running."}


@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}


@app.head("/health")
async def health_check_monitor():
    return Response(status_code=200)


@app.get("/cohorts/me", response_model=List[CohortOut])
async def list_my_cohorts(
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    - Maintainer: all cohorts they are mapped to.
    - Trainee: typically a single cohort, but returns list for visibility.
    """
    q = (
        db.query(Cohort)
        .join(CohortUser, Cohort.id == CohortUser.cohort_id)
        .filter(
            CohortUser.user_id == user.google_id,
            Cohort.is_active.is_(True),
        )
    )
    cohorts = q.all()
    return [CohortOut.model_validate(c) for c in cohorts]


@app.post("/cohorts", response_model=CohortOut)
async def create_cohort(
    data: CohortCreate,
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Simple cohort creation; you can restrict this further if you add an 'admin' role.
    For now reuses 'maintainer' as system admin.
    """
    require_maintainer(user)
    cohort = Cohort(name=data.name, code=data.code, is_active=True)
    db.add(cohort)
    db.commit()
    db.refresh(cohort)
    return CohortOut.model_validate(cohort)


@app.post("/cohorts/{cohort_id}/users")
async def add_user_to_cohort(
    cohort_id: int = Path(...),
    data: CohortUserCreate = Depends(),
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Admin/maintainer endpoint:
    - For trainees: create learner mapping.
    - For mentors: create maintainer mapping.
    """
    require_maintainer(user)

    cohort = db.query(Cohort).filter(Cohort.id == cohort_id).first()
    if not cohort or not cohort.is_active:
        raise HTTPException(status_code=404, detail="Cohort not found or inactive")

    existing = (
        db.query(CohortUser)
        .filter(
            CohortUser.cohort_id == cohort_id,
            CohortUser.user_id == data.user_id,
        )
        .first()
    )
    if existing:
        # Idempotent add
        existing.role = data.role
        db.commit()
        return {"detail": "Updated cohort membership"}

    cu = CohortUser(
        cohort_id=cohort_id,
        user_id=data.user_id,
        role=data.role,
    )
    db.add(cu)
    db.commit()
    return {"detail": "User added to cohort"}


# -----------------------------------------------------------------------------
# Space endpoints (scoped by cohort)
# -----------------------------------------------------------------------------


@app.get("/all-spaces", response_model=List[SpaceOut])
async def list_all_spaces(db: Session = Depends(get_db)):
    # Debug / ops only; not scoped.
    spaces = db.query(Space).all()
    return spaces


@app.get("/spaces", response_model=List[SpaceOut])
async def list_spaces(
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    """
    Spaces visible to the current user inside the active cohort.
    """
    spaces = (
        db.query(Space)
        .join(SpaceMembership, Space.id == SpaceMembership.space_id)
        .filter(
            SpaceMembership.user_id == user.google_id,
            Space.cohort_id == cohort_id,
        )
        .all()
    )
    return spaces


@app.post("/spaces", response_model=SpaceOut)
async def create_space(
    data: SpaceCreateIn,
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    """
    Maintainer creates a space inside the currently active cohort.
    """
    require_maintainer(user)

    space = Space(
        name=data.name,
        description=data.description,
        is_public=True,
        cohort_id=cohort_id,
        created_at=datetime.utcnow(),
    )
    db.add(space)
    db.commit()
    db.refresh(space)

    # Add current user as admin member of this space
    membership = SpaceMembership(
        user_id=user.google_id,
        space_id=space.id,
        role=SpaceRoleEnum.admin.value,
        joined_at=datetime.utcnow(),
    )
    db.add(membership)
    db.commit()

    return space


@app.post("/spaces/{space_id}/join")
async def join_space(
    space_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    # Ensure space exists in this cohort
    get_space_in_cohort_or_404(db, space_id, cohort_id)

    existing = (
        db.query(SpaceMembership)
        .filter(
            SpaceMembership.user_id == user.google_id,
            SpaceMembership.space_id == space_id,
        )
        .first()
    )
    if existing:
        return {"message": "Already a member"}

    membership = SpaceMembership(
        user_id=user.google_id,
        space_id=space_id,
        role=SpaceRoleEnum.member.value,
        joined_at=datetime.utcnow(),
    )
    db.add(membership)
    db.commit()
    return {"message": f"Joined space {space_id}"}


@app.post("/spaces/{space_id}/leave")
async def leave_space(
    space_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    get_space_in_cohort_or_404(db, space_id, cohort_id)

    membership = (
        db.query(SpaceMembership)
        .filter(
            SpaceMembership.user_id == user.google_id,
            SpaceMembership.space_id == space_id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(status_code=404, detail="Not a member")

    db.delete(membership)
    db.commit()
    return {"message": f"Left space {space_id}"}


@app.get("/spaces/{space_id}/members", response_model=List[str])
async def list_space_members(
    space_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    get_space_in_cohort_or_404(db, space_id, cohort_id)

    if not is_user_member(db, user.google_id, space_id, cohort_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    member_ids = (
        db.query(SpaceMembership.user_id)
        .filter(SpaceMembership.space_id == space_id)
        .all()
    )
    return [m[0] for m in member_ids]


# -----------------------------------------------------------------------------
# Document upload / listing / delete (scoped by cohort)
# -----------------------------------------------------------------------------


@app.post("/spaces/{space_id}/documents/upload", response_model=DocumentResponse)
async def upload_document(
    space_id: int = Path(...),
    file: UploadFile = File(...),
    title: str = Form(...),
    parse_mode: str = Form("fast"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    # Ensure membership & cohort
    get_space_in_cohort_or_404(db, space_id, cohort_id)
    if not is_user_member(db, user.google_id, space_id, cohort_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    allowed_types = {
        "text/plain": ("txt", ".txt"),
        "application/pdf": ("pdf", ".pdf"),
    }
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    ALLOWED_PARSE_MODES = {"fast", "accurate", "balanced"}
    if parse_mode not in ALLOWED_PARSE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parse_mode. Allowed: {ALLOWED_PARSE_MODES}",
        )

    file_type, extension = allowed_types[file.content_type]
    file_content = await file.read()
    import uuid

    unique_filename = f"{uuid.uuid4()}{extension}"

    # 2. Upload to Blob (Quick)
    upload_success = upload_blob(file_content, unique_filename)
    if not upload_success:
        raise HTTPException(status_code=500, detail="Failed to upload file")

    # 3. Create DB Record with "pending" status (Quick)
    document = Document(
        title=title,
        file_url=f"{os.getenv('BLOB_SAS_URL')}/{os.getenv('BLOB_CONTAINER_NAME')}/{unique_filename}",
        file_type=file_type,
        file_size=len(file_content),
        uploader_id=user.google_id,
        space_id=space_id,
        uploaded_at=datetime.utcnow(),
        processing_status="pending",
        chunk_count=0,
        source_name="User Upload",
        display_mode=DisplayModeEnum.generated_answer,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # 4. Schedule the Heavy Task (Fire and Forget)
    background_tasks.add_task(
        process_document_background,
        doc_id=document.id,
        space_id=space_id,
        file_type=file_type,
        parse_mode=parse_mode,
    )

    return DocumentResponse.model_validate(document)


@app.get(
    "/spaces/{space_id}/documents",
    response_model=List[DocumentResponse],
)
async def list_documents(
    space_id: int = Path(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    get_space_in_cohort_or_404(db, space_id, cohort_id)
    if not is_user_member(db, user.google_id, space_id, cohort_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    docs = (
        db.query(Document)
        .join(Space, Space.id == Document.space_id)
        .filter(
            Document.space_id == space_id,
            Space.cohort_id == cohort_id,
        )
        .order_by(desc(Document.uploaded_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [DocumentResponse.model_validate(d) for d in docs]


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    doc = (
        db.query(Document)
        .join(Space, Space.id == Document.space_id)
        .filter(
            Document.id == document_id,
            Document.uploader_id == user.google_id,
            Space.cohort_id == cohort_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse.model_validate(doc)


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    # 1. Fetch the document and enforce cohort
    doc = (
        db.query(Document)
        .join(Space, Space.id == Document.space_id)
        .filter(
            Document.id == document_id,
            Document.uploader_id == user.google_id,
            Space.cohort_id == cohort_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 2. Delete from Vector Database (Pinecone)
    try:
        if doc.vector_doc_id:
            rag_pipeline.delete_vectors_by_metadata(
                filter_dict={"document_id": doc.vector_doc_id}
            )
            logging.info(
                f"Deleted vectors for enhanced doc_id: {doc.vector_doc_id}"
            )
        else:
            logging.warning(
                f"No vector_doc_id found for doc {document_id}, trying simple doc_id"
            )
            rag_pipeline.delete_vectors_by_metadata(
                filter_dict={"doc_id": str(document_id)}
            )
    except Exception as e:
        logging.error(f"Failed to delete vectors for doc {document_id}: {e}")

    # 3. Delete from Blob Storage
    try:
        if doc.file_url:
            filename = extract_filename_from_url(doc.file_url)
            delete_blob(filename)
    except Exception as e:
        logging.error(f"Failed to delete blob for doc {document_id}: {e}")

    # 4. Delete from SQL Database
    try:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete()
        db.delete(doc)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database deletion failed: {e}",
        )

    return {"message": "Document deleted successfully"}


# -----------------------------------------------------------------------------
# Query endpoints (scoped by cohort via space_id check)
# -----------------------------------------------------------------------------


@app.post("/spaces/{space_id}/query", response_model=QueryResponse)
async def query_space_documents(
    query_request: QueryRequest,
    space_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    logging.info(
        f"Query request: user={user.google_id}, space={space_id}, "
        f"query='{query_request.query}'"
    )

    # Ensure space is in current cohort and user is member
    get_space_in_cohort_or_404(db, space_id, cohort_id)
    if not is_user_member(db, user.google_id, space_id, cohort_id):
        logging.warning(
            f"Access denied: User {user.google_id} not member of space {space_id}"
        )
        raise HTTPException(status_code=403, detail="Not a space member")

    try:
        import time

        start_time = time.time()

        query_hash = generate_query_hash(query_request.query, space_id)

        # Check cache
        cached_result = rag_pipeline.check_query_cache(
            query_hash=query_hash,
            query_text=query_request.query,
            space_id=space_id,
            db=db,
            similarity_threshold=0.95,
        )
        if cached_result:
            response_time = time.time() - start_time
            logging.info(
                f"Returned cached response in {response_time:.2f}s"
            )
            return QueryResponse(
                query=query_request.query,
                answer=cached_result["answer"],
                sources=cached_result["sources"],
                context_chunks=cached_result["context_chunks"],
                tokens_used=cached_result["tokens_used"],
            )

        # Cache miss → call LLM
        logging.info("Cache miss. Invoking LLM...")
        llm = get_user_llm(user.google_id)
        result = rag_pipeline.query_with_template_method(
            query_text=query_request.query,
            top_k=query_request.top_k,
            temperature=query_request.temperature,
            space_id=space_id,
            llm_override=llm,
        )

        response_time = time.time() - start_time

        # Store in cache
        query_embedding = rag_pipeline.embeddings.embed_query(
            query_request.query
        )
        query_log = QueryLog(
            user_id=user.google_id,
            query_hash=query_hash,
            query_embedding=query_embedding,
            response_text=result.get("answer", ""),
            sources=result.get("sources", []),
            tokens_used=result.get("tokens_used", {}),
            response_time=response_time,
            context_chunks=result.get("context_chunks", 0),
            space_id=space_id,
            hit_count=1,
            created_at=datetime.utcnow(),
        )
        db.add(query_log)
        db.commit()

        logging.info(
            f"Query cached with hash. Response time: {response_time:.2f}s"
        )

        return QueryResponse(
            query=query_request.query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            context_chunks=result.get("context_chunks", 0),
            tokens_used=result.get("tokens_used", {}),
        )

    except Exception as e:
        logging.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# -----------------------------------------------------------------------------
# Stats / space delete (scoped by cohort)
# -----------------------------------------------------------------------------


@app.get("/spaces/{space_id}/stats", response_model=SpaceStats)
async def get_space_stats(
    space_id: int = Path(...),
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    get_space_in_cohort_or_404(db, space_id, cohort_id)
    if not is_user_member(db, user.google_id, space_id, cohort_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    total = db.query(Document).filter(Document.space_id == space_id).count()
    user_count = (
        db.query(Document)
        .filter(
            Document.space_id == space_id,
            Document.uploader_id == user.google_id,
        )
        .count()
    )
    return SpaceStats(total_docs=total, user_docs=user_count)


@app.delete("/spaces/{space_id}")
async def delete_space(
    space_id: int,
    user: UserContext = Depends(get_current_user),
    cohort_id: int = Depends(get_current_cohort),
    db: Session = Depends(get_db),
):
    require_maintainer(user)
    space = get_space_in_cohort_or_404(db, space_id, cohort_id)

    db.delete(space)
    db.commit()
    return {"detail": "Space deleted"}


@app.get("/system/info")
async def system_info(
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_doc_count = (
        db.query(Document)
        .filter(Document.uploader_id == user.google_id)
        .count()
    )
    return {
        "user_documents": user_doc_count,
        "rag_index": rag_pipeline.index_name,
        "supported_formats": ["TXT", "PDF"],
    }
