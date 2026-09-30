import os
import uuid
from celery import Celery
from pypdf import PdfReader
from dotenv import load_dotenv
from app.retrieval.store import (
    update_document_status,
    insert_document_chunks
)

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery("rag_tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Simple sliding window chunker."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += (chunk_size - overlap)
    return chunks


@celery_app.task(name="process_pdf_document", bind=True, max_retries=2)
def process_pdf_document(self, doc_id: str, file_path: str, filename: str):
    """Background task: Reads PDF, chunks text, creates embeddings, updates DB."""
    try:
        update_document_status(doc_id, status="processing")

        reader = PdfReader(file_path)
        all_chunks = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue

            # Split page text into chunks
            text_segments = split_text(page_text, chunk_size=120, overlap=30)
            for seg_idx, segment in enumerate(text_segments):
                chunk_id = f"{doc_id}_p{page_idx + 1}_s{seg_idx}"
                all_chunks.append({
                    "id": chunk_id,
                    "content": segment,
                    "metadata": {
                        "source": filename,
                        "document_id": doc_id,
                        "page": page_idx + 1,
                        "segment": seg_idx
                    }
                })

        # Insert chunks and update status
        insert_document_chunks(doc_id=doc_id, chunks=all_chunks)
        update_document_status(doc_id, status="completed", chunk_count=len(all_chunks))

        # Cleanup uploaded local file
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"status": "success", "chunks_indexed": len(all_chunks)}

    except Exception as exc:
        update_document_status(doc_id, status="failed", error_message=str(exc))
        if os.path.exists(file_path):
            os.remove(file_path)
        raise exc