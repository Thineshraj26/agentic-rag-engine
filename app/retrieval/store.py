import os
import psycopg
from pgvector.psycopg import register_vector
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_db")

# Local FastEmbed engine (384 dimensions)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def init_db():
    """Initializes schema with parent documents table and chunk table."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 1. Parent Documents Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    chunk_count INT DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Drop existing chunks table to apply new schema with document_id
            cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")

            # 2. Document Chunks Table linked by document_id
            cur.execute("""
                CREATE TABLE document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    embedding vector(384),
                    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding 
                ON document_chunks USING hnsw (embedding vector_cosine_ops);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv 
                ON document_chunks USING gin (tsv_content);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id 
                ON document_chunks(document_id);
            """)
    print("Database schema with document tracking initialized.")


def create_document_record(doc_id: str, filename: str):
    """Creates initial record when file upload starts."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (id, filename, status)
                VALUES (%s, %s, 'pending')
                ON CONFLICT (id) DO NOTHING;
            """, (doc_id, filename))


def update_document_status(doc_id: str, status: str, chunk_count: int = 0, error_message: str = None):
    """Updates processing state and chunk statistics."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE documents
                SET status = %s,
                    chunk_count = %s,
                    error_message = %s,
                    updated_at = NOW()
                WHERE id = %s;
            """, (status, chunk_count, error_message, doc_id))


def insert_document_chunks(doc_id: str, chunks: list[dict]):
    """Embeds and inserts chunks for an ingested document."""
    if not chunks:
        return

    contents = [c["content"] for c in chunks]
    vectors = embeddings.embed_documents(contents)

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors):
                cur.execute("""
                    INSERT INTO document_chunks (id, document_id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE 
                    SET content = EXCLUDED.content, 
                        metadata = EXCLUDED.metadata, 
                        embedding = EXCLUDED.embedding;
                """, (
                    chunk["id"],
                    doc_id,
                    chunk["content"],
                    psycopg.types.json.Json(chunk.get("metadata", {})),
                    vec
                ))


def hybrid_search(query: str, top_k: int = 4) -> list[dict]:
    """Runs dense vector similarity search using local query embedding."""
    query_vector = embeddings.embed_query(query)

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vector, query_vector, top_k))
            results = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "score": float(row[3])
                }
                for row in results
            ]