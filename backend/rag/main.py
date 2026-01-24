from ast import Dict

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Path, Cookie, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from starlette.responses import Response
from starlette.status import HTTP_401_UNAUTHORIZED
import hashlib
from typing import List, Optional, Any, Dict
import logging
import json
from dotenv import load_dotenv
import uuid
import jwt
import os
from datetime import datetime
from starlette.responses import JSONResponse, Response

from database import get_db, Base, engine, SessionLocal
from models import Space, SpaceMembership, Document, DocumentChunk, QueryLog, SpaceRoleEnum, DisplayModeEnum
from ragPipeline import RAGPipeline
from blobStorage import upload_blob, delete_blob, extract_filename_from_url
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
load_dotenv()

def generate_query_hash(query_text: str, space_id: int) -> str:
    """Generate deterministic SHA-256 hash of normalized query"""
    # Normalize: lowercase, strip whitespace, remove extra spaces
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
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
)

security = HTTPBearer()
rag_pipeline = RAGPipeline(index_name="knowledgehub-main")

# Pydantic response/request models
class SpaceStats(BaseModel):
    total_docs: int
    user_docs: int

class UserContext(BaseModel):
    google_id: str
    name: Optional[str] = None
    role: Optional[str] = None

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

user_llm_cache = {}

def get_user_llm(user_id):
    if user_id not in user_llm_cache:
        from llmModels import LLM
        user_llm_cache[user_id] = LLM(gpt5=True)
    return user_llm_cache[user_id]

# Dependency to extract user info from headers (set by frontend AuthContext)
def get_current_user(request: Request) -> UserContext:
    user_id = request.headers.get("X-User-Id")
    role = request.headers.get("X-User-Role", "member")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user id")
    return UserContext(google_id=user_id, role=role)

# Helper to check if user is member of space
def is_user_member(db: Session, user_id: str, space_id: int) -> bool:
    membership = db.query(SpaceMembership).filter(
        SpaceMembership.user_id == user_id,
        SpaceMembership.space_id == space_id
    ).first()
    return membership is not None

def require_maintainer(user: UserContext):
    if user.role != "maintainer":
        raise HTTPException(status_code=403, detail="Insufficient privileges")

def process_document_background(doc_id: int, space_id: int, file_type: str, parse_mode: str = "auto"):
    """
    This runs AFTER the browser gets a response.
    It performs the heavy lifting: chunking, embedding, and indexing.
    FIXED: Uses global chunk index for proper ordering
    """
    db = SessionLocal()  # Create a new independent DB session
    try:
        # Re-fetch the document within this new session
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            return

        # Delete existing chunks before re-processing (idempotency)
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
        db.commit()

        # --- HEAVY RAG PIPELINE START ---
        metadata, chunked_docs, embedding_model = rag_pipeline.process_and_index_document(
            blob_url=document.file_url,
            file_type=file_type,
            doc_id=str(document.id),
            space_id=space_id,
            parse_mode=parse_mode
        )

        # Update Document Status
        document.processing_status = metadata.status
        document.chunk_count = metadata.chunk_count

        # Save Chunks - FIXED: Use global chunk_index from metadata
        doc_chunks = []
        for chunk in chunked_docs:
            meta = chunk.metadata
            doc_chunk = DocumentChunk(
                document_id=document.id,
                chunk_order=meta['chunk_index'],  # Now uses global index from fixed chunking
                vector_id=meta.get('chunk_id'),
                embedding_model=embedding_model,
                created_at=meta.get('created_at')
            )
            doc_chunks.append(doc_chunk)

        db.add_all(doc_chunks)
        db.commit()
        # --- HEAVY RAG PIPELINE END ---

        logging.info(f"Document {doc_id} processed successfully: {len(doc_chunks)} chunks stored")

    except Exception as e:
        logging.error(f"Background processing failed: {e}", exc_info=True)
        db.rollback()  # Add explicit rollback
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

@app.get("/", response_model=dict)
async def root():
    return {"message": "RAG Service is running."}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    return Response(status_code=200)

@app.get("/all-spaces", response_model=List[SpaceOut])
async def list_all_spaces(db: Session = Depends(get_db)):
    spaces = db.query(Space).all()
    return spaces

@app.get("/spaces", response_model=List[SpaceOut])
async def list_spaces(userContext=Depends(get_current_user), db: Session = Depends(get_db)):
    spaces = db.query(Space).join(SpaceMembership, Space.id == SpaceMembership.space_id).filter(
        SpaceMembership.user_id == userContext.google_id
    ).all()
    return spaces


@app.post("/spaces/{space_id}/query", response_model=QueryResponse)
async def query_space_documents(
    query_request: QueryRequest,
    space_id: int = Path(...),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logging.info(
        f"Query request: user={userContext.google_id}, space={space_id}, "
        f"query='{query_request.query}'"
    )
    
    if not is_user_member(db, userContext.google_id, space_id):
        logging.warning(f"Access denied: User {userContext.google_id} not member of space {space_id}")
        raise HTTPException(status_code=403, detail="Not a space member")
    
    try:
        import time
        start_time = time.time()
        
        # Generate query hash for privacy-conscious caching
        query_hash = generate_query_hash(query_request.query, space_id)
        
        # 🔍 CHECK CACHE FIRST (saves tokens and time!)
        cached_result = rag_pipeline.check_query_cache(
            query_hash=query_hash,
            query_text=query_request.query,
            space_id=space_id,
            db=db,
            similarity_threshold=0.95  # Adjust: 0.90-0.98 recommended
        )
        
        if cached_result:
            response_time = time.time() - start_time
            logging.info(f"✅ Returned cached response in {response_time:.2f}s")
            
            return QueryResponse(
                query=query_request.query,
                answer=cached_result["answer"],
                sources=cached_result["sources"],
                context_chunks=cached_result["context_chunks"],
                tokens_used=cached_result["tokens_used"]
            )
        
        # ❌ CACHE MISS - Call LLM
        logging.info("Cache miss. Invoking LLM...")
        llm = get_user_llm(userContext.google_id)
        
        result = rag_pipeline.query_with_template_method(
            query_text=query_request.query,
            top_k=query_request.top_k,
            temperature=query_request.temperature,
            space_id=space_id,
            llm_override=llm
        )
        
        response_time = time.time() - start_time
        
        # 💾 STORE IN CACHE with hash + embedding
        query_embedding = rag_pipeline.embeddings.embed_query(query_request.query)
        
        query_log = QueryLog(
            user_id=userContext.google_id,
            query_hash=query_hash,  # Privacy: hash instead of plain text
            query_embedding=query_embedding,  # For semantic matching
            response_text=result.get("answer", ""),
            sources=result.get("sources", []),  # Store as JSON directly
            tokens_used=result.get("tokens_used", {}),
            response_time=response_time,
            context_chunks=result.get("context_chunks", 0),
            space_id=space_id,
            hit_count=1,
            created_at=datetime.utcnow()
        )
        
        db.add(query_log)
        db.commit()
        
        logging.info(f"✅ Query cached with hash. Response time: {response_time:.2f}s")
        
        return QueryResponse(
            query=query_request.query,
            answer=result.get('answer', ''),
            sources=result.get("sources", []),
            context_chunks=result.get("context_chunks", 0),
            tokens_used=result.get("tokens_used", {}),
        )
        
    except Exception as e:
        logging.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")



@app.post("/spaces", response_model=SpaceOut)
async def create_space(
    data: SpaceCreateIn,
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    require_maintainer(userContext)

    space = Space(
        name=data.name,
        description=data.description,
        is_public=True,
        created_at=datetime.utcnow(),
    )
    db.add(space)
    db.commit()
    db.refresh(space)

    # Add current user as admin member
    membership = SpaceMembership(
        user_id=userContext.google_id,
        space_id=space.id,
        role=SpaceRoleEnum.admin.value,
        joined_at=datetime.utcnow(),
    )
    db.add(membership)
    db.commit()

    return space

@app.post("/spaces/{space_id}/join")
async def join_space(space_id: int = Path(...), userContext=Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if already member
    existing = db.query(SpaceMembership).filter(
        SpaceMembership.user_id == userContext.google_id,
        SpaceMembership.space_id == space_id
    ).first()
    if existing:
        return {"message": "Already a member"}

    membership = SpaceMembership(
        user_id=userContext.google_id,
        space_id=space_id,
        role=SpaceRoleEnum.member.value,
        joined_at=datetime.utcnow()
    )
    db.add(membership)
    db.commit()

    return {"message": f"Joined space {space_id}"}

@app.post("/spaces/{space_id}/leave")
async def leave_space(space_id: int = Path(...), userContext=Depends(get_current_user), db: Session = Depends(get_db)):
    membership = db.query(SpaceMembership).filter(
        SpaceMembership.user_id == userContext.google_id,
        SpaceMembership.space_id == space_id
    ).first()
    if not membership:
        raise HTTPException(status_code=404, detail="Not a member")

    db.delete(membership)
    db.commit()

    return {"message": f"Left space {space_id}"}

@app.get("/spaces/{space_id}/members", response_model=List[str])
async def list_space_members(space_id: int = Path(...), userContext=Depends(get_current_user), db: Session = Depends(get_db)):
    # Enforce member info visibility only if current user is member
    if not is_user_member(db, userContext.google_id, space_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    member_ids = db.query(SpaceMembership.user_id).filter(
        SpaceMembership.space_id == space_id
    ).all()

    return [m[0] for m in member_ids]

@app.post("/spaces/{space_id}/documents/upload", response_model=DocumentResponse)
async def upload_document(
    space_id: int = Path(...),
    file: UploadFile = File(...),
    title: str = Form(...),
    parse_mode: str = Form("auto"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Validation Logic
    if not is_user_member(db, userContext.google_id, space_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    allowed_types = {"text/plain": ("txt", ".txt"), "application/pdf": ("pdf", ".pdf")}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    ALLOWED_PARSE_MODES = {"auto", "unstructured", "pdf4llm", "llama", "fast", "balanced"}
    if parse_mode not in ALLOWED_PARSE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parse_mode. Allowed: {ALLOWED_PARSE_MODES}"
        )

    file_type, extension = allowed_types[file.content_type]
    file_content = await file.read()
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
        uploader_id=userContext.google_id,
        space_id=space_id,
        uploaded_at=datetime.utcnow(),
        processing_status="pending",  # <--- Important: Initial status
        chunk_count=0,
        source_name="User Upload",
        display_mode=DisplayModeEnum.generated_answer
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
        parse_mode=parse_mode
    )

    # 5. Return immediately!
    # The browser receives this response in ~1-2 seconds.
    # The background task continues running on the server.
    return DocumentResponse.model_validate(document)

@app.get("/spaces/{space_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    space_id: int = Path(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not is_user_member(db, userContext.google_id, space_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    docs = db.query(Document).filter(
        Document.space_id == space_id
    ).order_by(desc(Document.uploaded_at)).offset(skip).limit(limit).all()

    return [DocumentResponse.model_validate(d) for d in docs]

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int = Path(...),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.uploader_id == userContext.google_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse.model_validate(doc)

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int = Path(...),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Fetch the document
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.uploader_id == userContext.google_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 2. Delete from Vector Database (Pinecone)
    try:
        rag_pipeline.delete_vectors_by_metadata(filter_dict={"doc_id": str(document_id)})
    except Exception as e:
        logging.error(f"Failed to delete vectors for doc {document_id}: {e}")
        # Proceeding anyway to ensure DB/Blob cleanup

    # 3. Delete from Blob Storage
    try:
        if doc.file_url:
            filename = extract_filename_from_url(doc.file_url)
            delete_blob(filename)
    except Exception as e:
        logging.error(f"Failed to delete blob for doc {document_id}: {e}")

    # 4. Delete from SQL Database
    try:
        # Delete chunks first (foreign key dependency)
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        # Delete document record
        db.delete(doc)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database deletion failed: {e}")

    return {"message": "Document deleted successfully"}

@app.get("/spaces/{space_id}/stats", response_model=SpaceStats)
async def get_space_stats(
    space_id: int = Path(...),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if user is a member
    if not is_user_member(db, userContext.google_id, space_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    # Efficient SQL count queries
    total = db.query(Document).filter(Document.space_id == space_id).count()
    user_count = db.query(Document).filter(
        Document.space_id == space_id,
        Document.uploader_id == userContext.google_id
    ).count()

    return SpaceStats(total_docs=total, user_docs=user_count)

@app.delete("/spaces/{space_id}")
async def delete_space(
    space_id: int,
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_maintainer(userContext)

    space = db.query(Space).filter(Space.id == space_id).first()
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")

    # Optionally check or cascade delete members, documents...
    db.delete(space)
    db.commit()

    return {"detail": "Space deleted"}

@app.get("/system/info")
async def system_info(
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_doc_count = db.query(Document).filter(Document.uploader_id == userContext.google_id).count()
    return {
        "user_documents": user_doc_count,
        "rag_index": rag_pipeline.index_name,
        "supported_formats": ["TXT", "PDF"]  # Updated to include PDF
    }