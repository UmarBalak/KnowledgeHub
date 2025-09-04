from ast import Dict
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Path, Cookie, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from starlette.responses import Response
from starlette.status import HTTP_401_UNAUTHORIZED
from typing import List, Optional, Any, Dict
import logging
import json
from dotenv import load_dotenv
import uuid
import jwt
import os
from datetime import datetime
from starlette.responses import JSONResponse, Response

from database import get_db, Base, engine
from models import Space, SpaceMembership, Document, DocumentChunk, QueryLog, SpaceRoleEnum, DisplayModeEnum
from ragPipeline import RAGPipeline
from blobStorage import upload_blob

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
load_dotenv()

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
    query_request: QueryRequest,  # FastAPI parses JSON body here
    space_id: int = Path(...),
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logging.info(
        f"Query request received: user_id={userContext.google_id}, space_id={space_id}, "
        f"query={query_request.query}, top_k={query_request.top_k}, temperature={query_request.temperature}"
    )

    if not is_user_member(db, userContext.google_id, space_id):
        logging.warning(f"Access denied: User {userContext.google_id} not member of space {space_id}")
        raise HTTPException(status_code=403, detail="Not a space member")

    try:
        # return {
        #         "answer": answer,
        #         "sources": [source["metadata"] for source in sources],  # For backward compatibility
        #         "enhanced_sources": enhanced_sources,  # New enhanced sources
        #         "tokens_used": tokens_used,
        #         "context_chunks": len(retrieved_docs),
        #         "query_text": query_text,
        #         "response_metadata": {
        #             "top_k": top_k,
        #             "temperature": temperature,
        #             "include_context": include_context,
        #             "total_chunks_found": len(retrieved_docs)
        #         }
        #     }

        result = rag_pipeline.query(
            query_text=query_request.query,
            top_k=query_request.top_k,
            temperature=query_request.temperature,
            space_id=space_id
        )
        logging.info(f"Query result: answer length={len(result.get('answer', ''))}, tokens_used={result.get('tokens_used', 0)}")
        # class QueryLog(Base):
        #     __tablename__ = "query_logs"

        #     id = Column(Integer, primary_key=True, index=True)
        #     user_id = Column(String(36), nullable=False)  # UUID from external Auth service
        #     query_text = Column(Text, nullable=False)
        #     response_text = Column(Text)
        #     sources = Column(JSON, nullable=True)
        #     tokens_used = Column(Integer)
        #     response_time = Column(Float)  # seconds
        #     context_chunks = Column(Integer)
        #     created_at = Column(DateTime(timezone=True), server_default=func.now())

        # Record query log in DB
        query_log = QueryLog(
            user_id=userContext.google_id,
            query_text=query_request.query,
            response_text=result.get("answer", ""),
            sources=json.dumps(result.get("sources", [])),  # serialize to str here
            tokens_used=result.get("tokens_used", 0),
            response_time=result.get("response_time", 0.0),
            context_chunks=result.get("context_chunks"),
            created_at=datetime.utcnow(),
        )
        db.add(query_log)
        db.commit()
        logging.info("QueryLog stored in db")

        # class QueryResponse(BaseModel):
        #     query: str
        #     answer: str
        #     sources: List[str] = []
        #     context_chunks: List[str] = []
        #     tokens_used: int

        return QueryResponse(
            query=query_request.query,
            answer=result.get('answer', ''),
            sources=result.get("sources", []),
            context_chunks=result.get("context_chunks"),
            tokens_used=result.get("tokens_used", 0),
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
    userContext=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not is_user_member(db, userContext.google_id, space_id):
        raise HTTPException(status_code=403, detail="Not a space member")

    if file.content_type != "text/plain":
        raise HTTPException(status_code=400, detail="Only text/plain files are supported for now")

    file_content = await file.read()
    unique_filename = f"{uuid.uuid4()}.txt"

    upload_success = upload_blob(file_content, unique_filename)
    if not upload_success:
        raise HTTPException(status_code=500, detail="Failed to upload file to storage")

    document = Document(
        title=title,
        file_url=f"{os.getenv('BLOB_SAS_URL')}/{os.getenv('BLOB_CONTAINER_NAME')}/{unique_filename}",
        file_type="txt",
        file_size=len(file_content),
        uploader_id=userContext.google_id,
        space_id=space_id,
        uploaded_at=datetime.utcnow(),
        processing_status="pending",
        chunk_count=0,
        error_message=None,
        source_name="User Upload",
        display_mode=DisplayModeEnum.generated_answer
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        metadata = rag_pipeline.process_and_index_document(
            blob_url=document.file_url,
            file_type="txt",
            doc_id=str(document.id)
        )
        document.processing_status = metadata.status
        document.chunk_count = metadata.chunk_count
        db.commit()
    except Exception as e:
        document.processing_status = "failed"
        document.error_message = str(e)
        db.commit()

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
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.uploader_id == userContext.google_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    db.delete(doc)
    db.commit()
    return {"message": "Document deleted successfully"}



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
        "supported_formats": ["TXT"]  # Currently only plain text supported
    }
