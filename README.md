# RAG-Based Collaborative Learning Platform (Backend)

## Overview

This repository contains the **backend service** for the **RAG-based collaborative learning platform** built for **CTS GenC trainees**.

The backend is responsible for:
- Authentication and authorization
- Space and membership management
- Document ingestion and processing
- RAG-based querying
- Semantic query caching

---

## High-Level Architecture

Frontend -> 
FastAPI -> 
(PostgreSQL ── Pinecone ── Azure Blob Storage) -> 
Azure OpenAI / TogetherAI


---

## Core Components

| Component | Responsibility |
|--------|---------------|
| FastAPI | API layer, auth enforcement, background tasks |
| PostgreSQL | Metadata, memberships, query cache |
| Azure Blob Storage | Raw documents and normalized text |
| Pinecone | Vector embeddings |
| LangChain | Chunking, prompting, embeddings |
| Azure OpenAI / TogetherAI | LLM inference |
| RAGPipeline | End-to-end RAG orchestration |

---

## Authentication

### Authentication Strategy

- Google OAuth is handled **inside this backend**
- After authentication, a JWT is issued
- JWT is stored in an **HTTP-only, secure cookie**
- Frontend never accesses or stores tokens

---

### OAuth Flow

1. Frontend redirects to:
   - GET /auth/google/guestlogin
2. User authenticates with Google
3. Google redirects to:
   - GET /auth/google/guestcallback
4. Backend:
    - Validates Google identity
    - Applies access rules
    - Creates or updates user record
    - Issues JWT
    - Sets secure cookie
5. User is redirected to frontend `/home`

---

### Access Rules

A user is allowed to log in if:
- Email belongs to the configured college/company domain, or
- Email exists in the `TestUser` allowlist

---

### Auth Endpoints

| Endpoint | Purpose |
|--------|--------|
| `/auth/google/guestlogin` | Start Google OAuth |
| `/auth/google/guestcallback` | OAuth callback |
| `/auth/status` | Check authentication status |
| `/auth/logout` | Logout |

---

## Authorization Model

Authorization is enforced using:
- Space membership
- User role

### Roles

| Role | Capabilities |
|----|-------------|
| maintainer | Upload/delete documents, create spaces |
| user | Query documents and chat |

---

## Complete System Flow (Backend Perspective)

This section describes the **full backend execution path** for all major operations.

---

### 1. Authentication and Session Flow

1. OAuth callback is received from Google
2. Backend validates access rules
3. User record is created or updated in database
4. JWT is generated with user metadata
5. JWT is stored in secure cookie
6. Subsequent requests validate JWT on every call

---

### 2. Space Access Flow

1. Client calls `GET /spaces`
2. Backend fetches spaces where user is a member
3. Membership is validated on every space-scoped request

---

### 3. Document Upload and Ingestion Flow

1. Maintainer uploads PDF or TXT document
2. File is stored in Azure Blob Storage
3. SQL document record is created with status `pending`
4. Background task is triggered
5. Parser is selected (fast, balanced, accurate)
6. Text is extracted and normalized
7. Enhanced document ID is generated
8. Content is chunked (1000 chars, 100 overlap)
9. Embeddings are generated
10. Vectors are stored in Pinecone with `space_id`
11. Chunk metadata is persisted in SQL
12. Document status is updated to `completed`

---

### 4. RAG Query Execution Flow

1. User submits query to `/spaces/{space_id}/query`
2. Backend validates space membership
3. Query hash is generated
4. Query cache is checked:
- Exact hash match
- Semantic similarity match
5. If cache hit:
- Cached answer is returned
6. If cache miss:
- Relevant chunks retrieved from Pinecone
- Prompt constructed using retrieved context
- LLM is invoked
7. Response is cached with embeddings and metadata
8. Answer is returned to client

---

### 5. Chat Memory Flow

- Each user has an in-memory conversation buffer
- Only last 10 messages are retained
- Memory is not persisted
- Memory is reset on refresh or restart

---

### 6. Document Deletion Flow

1. Document deletion request received
2. Vectors deleted from Pinecone using document ID
3. File deleted from Azure Blob Storage
4. SQL records deleted via cascade
5. System remains consistent even on partial failure

---

## PDF Parsing Modes

| Mode | Description |
|----|------------|
| fast | Fast parsing, lower accuracy |
| balanced | Balanced speed and accuracy |
| accurate | LlamaParser (external API, highest accuracy) |

---

## RAG Query Execution Flow

1. Validate space membership
2. Generate deterministic query hash
3. Check query cache:
   - Exact hash match
   - Semantic similarity match
4. If cache hit, return cached response
5. If cache miss:
   - Retrieve vectors from Pinecone
   - Build prompt
   - Invoke LLM
6. Cache response for future reuse

---

## Query Cache Design

- Query text is never stored
- Cache uses:
- SHA-256 hash
- Query embeddings
- Enables privacy-preserving semantic reuse

---

## Chat Memory Model

| Property | Value |
|-------|------|
| Scope | Per user |
| Storage | In-memory |
| Retention | Last 10 messages |
| Persistence | None |

---

## Document Deletion Workflow

Deletion is executed in three phases:

1. Delete vectors from Pinecone using document ID
2. Delete file from Azure Blob Storage
3. Cascade delete SQL records

Fallback logic exists for legacy documents.

---

## Core API Endpoints

| Endpoint | Purpose |
|--------|--------|
| `POST /spaces` | Create a space |
| `POST /spaces/{id}/documents/upload` | Upload document |
| `POST /spaces/{id}/query` | RAG-based chat |
| `GET /spaces/{id}/documents` | List documents |
| `DELETE /documents/{id}` | Delete document |
| `GET /spaces/{id}/stats` | Space analytics |

---

