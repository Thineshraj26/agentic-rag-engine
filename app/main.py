import os
import shutil
import uuid
from typing import List, Optional
import psycopg
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.core.graph import build_graph
from app.retrieval.store import (
    DATABASE_URL,
    create_document_record
)
from app.retrieval.tasks import process_pdf_document

load_dotenv()

app = FastAPI(
    title="Agentic RAG Engine API",
    description="Multi-tenant Agentic RAG service with async PDF ingestion and stateful retrieval",
    version="1.0.0"
)

# Compile agent graph once
agent = build_graph()

UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==========================================
# Pydantic Schemas
# ==========================================

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the Model Context Protocol?")

class DocumentSource(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[DocumentSource]
    rewritten_iterations: int
    used_retrieval: bool

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    error_message: Optional[str] = None


# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Accepts PDF uploads, stores file temporarily, and queues async ingestion."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    doc_id = str(uuid.uuid4())
    temp_file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    # Save uploaded file to disk
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Insert initial tracking record into PostgreSQL
    create_document_record(doc_id=doc_id, filename=file.filename)

    # 2. Dispatch non-blocking background job to Celery via Redis
    process_pdf_document.delay(
        doc_id=doc_id,
        file_path=temp_file_path,
        filename=file.filename
    )

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status="pending",
        message="Document uploaded successfully. Processing started in background."
    )


@app.get("/api/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str):
    """Check ingestion status and chunk count of an uploaded document."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, filename, status, chunk_count, error_message
                FROM documents
                WHERE id = %s;
            """, (document_id,))
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Document not found.")

            return DocumentStatusResponse(
                document_id=row[0],
                filename=row[1],
                status=row[2],
                chunk_count=row[3],
                error_message=row[4]
            )


@app.post("/api/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest):
    """Executes the agentic RAG workflow against indexed knowledge."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = {
        "query": payload.query,
        "original_query": payload.query,
        "documents": [],
        "generation": None,
        "loop_count": 0,
        "is_grounded": True,
        "needs_retrieval": False
    }

    try:
        final_state = agent.invoke(initial_state)

        sources = [
            DocumentSource(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                relevance_score=doc.relevance_score
            )
            for doc in final_state.get("documents", [])
        ]

        return QueryResponse(
            query=payload.query,
            response=final_state.get("generation", "No generation produced."),
            sources=sources,
            rewritten_iterations=final_state.get("loop_count", 0),
            used_retrieval=final_state.get("needs_retrieval", False)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))