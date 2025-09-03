
from database import SessionLocal, Base, engine
from models import Space, SpaceMembership, Document, DocumentChunk, QueryLog
import uuid

# Drop all tables
### Base.metadata.drop_all(bind=engine)

# Drop a single table
### ModelName.__table__.drop(bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Generate fake UUIDs for external user references
user1_id = str(uuid.uuid4())
user2_id = str(uuid.uuid4())

with SessionLocal() as session:
    try:
        # Insert a Space
        space = Space(name="AI Research", description="Shared space for AI papers", is_public=True)
        session.add(space)
        session.commit()
        session.refresh(space)

        # Insert SpaceMemberships
        membership1 = SpaceMembership(user_id=user1_id, space_id=space.id, role="admin")
        membership2 = SpaceMembership(user_id=user2_id, space_id=space.id, role="member")
        session.add_all([membership1, membership2])
        session.commit()

        # Insert a Document
        doc = Document(
            title="Federated Learning Paper",
            file_url="https://example.blob.core.windows.net/docs/federated_learning.pdf",
            file_type="pdf",
            file_size=204800,
            uploader_id=user1_id,
            space_id=space.id,
            processing_status="completed",
            chunk_count=2,
            source_name="User Upload"
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)

        # Insert DocumentChunks
        chunk1 = DocumentChunk(
            document_id=doc.id,
            chunk_order=1,
            vector_id="vec_001",
            embedding_model="text-embedding-ada"
        )
        chunk2 = DocumentChunk(
            document_id=doc.id,
            chunk_order=2,
            vector_id="vec_002",
            embedding_model="text-embedding-ada"
        )
        session.add_all([chunk1, chunk2])
        session.commit()

        # Insert QueryLog
        log = QueryLog(
            user_id=user2_id,
            query_text="What is federated learning?",
            response_text="Federated learning is a collaborative machine learning technique...",
            tokens_used=128,
            response_time=1.2,
            context_chunks=2,
        )
        session.add(log)
        session.commit()

        print("✅ Seed data inserted successfully")

    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
